import json
import math
from pathlib import Path
import unittest
from types import SimpleNamespace

import torch

from src.core.axis import Axis
from src.core.registry import get_technique, list_techniques
from src.techniques.turboquant import (
    TurboQuantTechnique,
    get_active_turboquant_cache,
    set_active_turboquant_cache,
)
from src.techniques.turboquant_cache import TurboQuantCache, TurboQuantLayer
from src.techniques.turboquant_ops import (
    CENTROIDS_3BIT,
    CENTROIDS_4BIT,
    THRESHOLDS_3BIT,
    THRESHOLDS_4BIT,
    compute_shannon_phi_metrics,
    dequantize_lloyd_max,
    fwht,
    generate_rademacher_signs,
    get_codebook_tensors,
    ifwht,
    pack_4bit_nibbles,
    pack_value_nibbles,
    qjl_reconstruct,
    qjl_sketch,
    quantize_lloyd_max,
    turboquant_decode_key,
    turboquant_decode_value,
    turboquant_encode_key,
    turboquant_encode_value,
    unpack_4bit_nibbles,
    unpack_value_nibbles,
)


class TestTurboQuantOperations(unittest.TestCase):
    def test_fwht_invertibility_and_orthonormality(self):
        torch.manual_seed(42)
        d = 128
        x = torch.randn(2, 4, 16, d, dtype=torch.float32)

        # Orthonormal forward transform
        w = fwht(x)
        self.assertEqual(w.shape, x.shape)

        # Energy preservation: ||fwht(x)|| == ||x||
        norm_x = torch.norm(x, p=2, dim=-1)
        norm_w = torch.norm(w, p=2, dim=-1)
        self.assertTrue(torch.allclose(norm_x, norm_w, atol=1e-5))

        # Exact self-inverse: ifwht(fwht(x)) == x
        x_rec = ifwht(w)
        self.assertTrue(torch.allclose(x, x_rec, atol=1e-5))

        # Non-power-of-2 dimension must raise ValueError
        with self.assertRaises(ValueError):
            fwht(torch.randn(2, 100))

    def test_beta_lloyd_max_codebooks(self):
        # 4-bit (16 levels) codebook tests
        self.assertEqual(len(CENTROIDS_4BIT), 16)
        self.assertEqual(len(THRESHOLDS_4BIT), 15)
        self.assertTrue(all(c1 < c2 for c1, c2 in zip(CENTROIDS_4BIT[:-1], CENTROIDS_4BIT[1:])))
        self.assertTrue(all(t1 < t2 for t1, t2 in zip(THRESHOLDS_4BIT[:-1], THRESHOLDS_4BIT[1:])))

        # Exact symmetry around 0
        for i in range(8):
            self.assertAlmostEqual(CENTROIDS_4BIT[i], -CENTROIDS_4BIT[15 - i], places=6)
        self.assertAlmostEqual(THRESHOLDS_4BIT[7], 0.0, places=6)

        # 3-bit (8 levels) codebook tests
        self.assertEqual(len(CENTROIDS_3BIT), 8)
        self.assertEqual(len(THRESHOLDS_3BIT), 7)
        self.assertTrue(all(c1 < c2 for c1, c2 in zip(CENTROIDS_3BIT[:-1], CENTROIDS_3BIT[1:])))
        self.assertTrue(all(t1 < t2 for t1, t2 in zip(THRESHOLDS_3BIT[:-1], THRESHOLDS_3BIT[1:])))
        for i in range(4):
            self.assertAlmostEqual(CENTROIDS_3BIT[i], -CENTROIDS_3BIT[7 - i], places=6)
        self.assertAlmostEqual(THRESHOLDS_3BIT[3], 0.0, places=6)

    def test_bit_packing_roundtrip(self):
        torch.manual_seed(42)
        shape = (2, 8, 32, 128)

        # 1. 4-bit nibble pack/unpack
        idx_4bit = torch.randint(0, 16, shape, dtype=torch.uint8)
        packed_4bit = pack_4bit_nibbles(idx_4bit)
        self.assertEqual(packed_4bit.shape, (2, 8, 32, 64))
        self.assertEqual(packed_4bit.dtype, torch.uint8)

        unpacked_4bit = unpack_4bit_nibbles(packed_4bit)
        self.assertTrue(torch.equal(idx_4bit, unpacked_4bit))

        # 2. Value 3-bit index + 1-bit sign pack/unpack
        idx_3bit = torch.randint(0, 8, shape, dtype=torch.uint8)
        signs = torch.randint(0, 2, shape, dtype=torch.float32) * 2.0 - 1.0  # {-1.0, 1.0}
        packed_val = pack_value_nibbles(idx_3bit, signs)
        self.assertEqual(packed_val.shape, (2, 8, 32, 64))
        self.assertEqual(packed_val.dtype, torch.uint8)

        unpacked_idx, unpacked_signs = unpack_value_nibbles(packed_val)
        self.assertTrue(torch.equal(idx_3bit, unpacked_idx))
        self.assertTrue(torch.equal(signs, unpacked_signs))

    def test_qjl_residual_sketch_unbiasedness(self):
        torch.manual_seed(42)
        d = 128
        query = torch.randn(d)
        res = torch.randn(d)
        true_dot = torch.dot(query, res).item()

        estimates = []
        for seed_idx in range(2000):
            s_prime = generate_rademacher_signs((d,), device="cpu", seed=1000 + seed_idx)
            q = qjl_sketch(res, s_prime)
            self.assertTrue(torch.all((q == 1.0) | (q == -1.0)))

            res_norm = torch.norm(res, p=2)
            r_hat = qjl_reconstruct(q, s_prime, res_norm, head_dim=d)
            est_dot = torch.dot(query, r_hat).item()
            estimates.append(est_dot)

        mean_est = sum(estimates) / len(estimates)
        rel_diff = abs(true_dot - mean_est) / abs(true_dot)
        self.assertLess(rel_diff, 0.05)

    def test_asymmetric_key_variance_penalty(self):
        """Verify the literature's core finding: QJL on Keys inflates inner product variance > 2x vs 4-bit scalar."""
        torch.manual_seed(42)
        d = 128
        c16, t16, c8, t8 = get_codebook_tensors("cpu")

        errors_scalar = []
        errors_qjl = []
        trials = 1000

        for trial in range(trials):
            q = torch.randn(d)
            k = torch.randn(d)
            k_norm = torch.norm(k, p=2)
            k_unit = k / k_norm

            s = generate_rademacher_signs((d,), device="cpu", seed=trial)
            k_rot = fwht(s * k_unit)

            # Scheme A: 4-bit scalar
            idx16 = quantize_lloyd_max(k_rot, t16)
            k_rec16_rot = dequantize_lloyd_max(idx16, c16)
            k_rec16 = s * ifwht(k_rec16_rot) * k_norm
            err_scalar = torch.dot(q, k_rec16) - torch.dot(q, k)
            errors_scalar.append(err_scalar.item())

            # Scheme B: 3-bit scalar + 1-bit QJL
            idx8 = quantize_lloyd_max(k_rot, t8)
            k_rec8_rot = dequantize_lloyd_max(idx8, c8)
            res = k_rot - k_rec8_rot
            res_norm = torch.norm(res, p=2)

            s_prime = generate_rademacher_signs((d,), device="cpu", seed=trials + trial)
            q_sketch = qjl_sketch(res, s_prime)
            res_hat = qjl_reconstruct(q_sketch, s_prime, res_norm, head_dim=d)
            k_rec_qjl_rot = k_rec8_rot + res_hat
            k_rec_qjl = s * ifwht(k_rec_qjl_rot) * k_norm
            err_qjl = torch.dot(q, k_rec_qjl) - torch.dot(q, k)
            errors_qjl.append(err_qjl.item())

        var_scalar = torch.tensor(errors_scalar).var().item()
        var_qjl = torch.tensor(errors_qjl).var().item()
        variance_ratio = var_qjl / var_scalar

        # Empirically validates that QJL on Keys inflates variance by > 2.0x
        self.assertGreater(variance_ratio, 2.0)

    def test_shannon_phi_metrics_orthogonality(self):
        # 1. Identical vectors -> 0 SNR error and 0 DIR error
        x = torch.randn(4, 128)
        m_ident = compute_shannon_phi_metrics(x, x)
        self.assertAlmostEqual(m_ident["eps_snr"], 0.0, places=5)
        self.assertAlmostEqual(m_ident["eps_dir"], 0.0, places=5)

        # 2. Scaled vector -> positive SNR error, zero DIR error
        x_scaled = 2.0 * x
        m_scaled = compute_shannon_phi_metrics(x, x_scaled)
        self.assertGreater(m_scaled["eps_snr"], 0.5)
        self.assertAlmostEqual(m_scaled["eps_dir"], 0.0, places=5)

        # 3. Orthogonal vector -> eps_dir == 1.0
        v1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        v2 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        m_orth = compute_shannon_phi_metrics(v1, v2)
        self.assertAlmostEqual(m_orth["eps_dir"], 1.0, places=5)


class TestTurboQuantCache(unittest.TestCase):
    def test_cache_prefill_and_decode_cycle(self):
        torch.manual_seed(42)
        cache = TurboQuantCache(seed=42)

        # Prefill: 32 tokens
        k_prefill = torch.randn(1, 8, 32, 128, dtype=torch.bfloat16)
        v_prefill = torch.randn(1, 8, 32, 128, dtype=torch.bfloat16)
        k_out1, v_out1 = cache.update(k_prefill, v_prefill, layer_idx=0)

        self.assertEqual(k_out1.shape, (1, 8, 32, 128))
        self.assertEqual(v_out1.shape, (1, 8, 32, 128))
        self.assertEqual(k_out1.dtype, torch.bfloat16)
        self.assertEqual(v_out1.dtype, torch.bfloat16)
        self.assertEqual(cache.get_seq_length(0), 32)

        # Decode: 5 sequential 1-token steps
        curr_len = 32
        for _ in range(5):
            k_step = torch.randn(1, 8, 1, 128, dtype=torch.bfloat16)
            v_step = torch.randn(1, 8, 1, 128, dtype=torch.bfloat16)
            k_out_s, v_out_s = cache.update(k_step, v_step, layer_idx=0)
            curr_len += 1
            self.assertEqual(k_out_s.shape, (1, 8, curr_len, 128))
            self.assertEqual(v_out_s.shape, (1, 8, curr_len, 128))
            self.assertEqual(cache.get_seq_length(0), curr_len)

        # Check memory metrics
        phys, theo = cache.get_cache_size_bytes()
        expected_bytes = 1 * 8 * 37 * 140  # 1 batch * 8 heads * 37 tokens * 140 bytes
        self.assertEqual(theo, expected_bytes)
        self.assertEqual(phys, expected_bytes)

        # Check Shannon-phi distortion metrics recorded during prefill
        metrics = cache.get_shannon_phi_metrics()
        self.assertGreater(metrics["k_eps_snr"], 0.0)
        self.assertLess(metrics["k_eps_snr"], 0.05)
        self.assertGreater(metrics["k_eps_dir"], 0.0)
        self.assertLess(metrics["k_eps_dir"], 0.02)
        self.assertGreater(metrics["v_eps_snr"], 0.0)
        self.assertLess(metrics["v_eps_snr"], 0.05)
        self.assertGreater(metrics["v_eps_dir"], 0.0)
        self.assertLess(metrics["v_eps_dir"], 0.02)


class TestTurboQuantTechnique(unittest.TestCase):
    def test_registration_and_axes(self):
        self.assertIn("turboquant", list_techniques())
        technique = get_technique("turboquant")
        self.assertIsInstance(technique, TurboQuantTechnique)
        self.assertEqual(technique.axes, Axis.STRUCTURAL)

    def test_defaults_and_validation(self):
        tech = get_technique("turboquant")
        self.assertEqual(tech.k_bits, 4)
        self.assertEqual(tech.v_bits, 3)
        self.assertEqual(tech.seed, 42)
        self.assertEqual(tech.block_size, 16)

        # Invalid configurations rejected
        with self.assertRaises(ValueError):
            TurboQuantTechnique(k_bits=2)
        with self.assertRaises(ValueError):
            TurboQuantTechnique(v_bits=4)

    def test_extra_metrics_reporting(self):
        tech = TurboQuantTechnique()
        tech.setup()
        factory = tech.pytorch_cache_factory()
        cache = factory()

        # Update cache with dummy tokens
        k = torch.randn(1, 8, 64, 128, dtype=torch.bfloat16)
        v = torch.randn(1, 8, 64, 128, dtype=torch.bfloat16)
        cache.update(k, v, layer_idx=0)

        # Explicitly delete cache to simulate PyTorchRunner.generate() lifecycle
        del cache

        mock_out = SimpleNamespace(prompt_tokens=64, generated_tokens=16)
        metrics = tech.extra_metrics(None, mock_out)

        self.assertEqual(metrics["k_bits"], 4)
        self.assertEqual(metrics["v_bits"], 3)
        self.assertTrue(metrics["qjl_v"])
        self.assertFalse(metrics["qjl_k"])
        self.assertNotIn("compression_ratio", metrics)
        self.assertEqual(metrics["theoretical_bytes_per_token_head"], 140)
        self.assertEqual(metrics["uncompressed_bytes_per_token_head"], 512)
        self.assertGreater(metrics["theoretical_cache_bytes"], 0)
        self.assertGreater(metrics["shannon_k_eps_snr"], 0.0)

        # Test analytical fallback after reset when active cache is absent
        tech.setup()
        fallback_metrics = tech.extra_metrics(None, mock_out)
        self.assertEqual(fallback_metrics["theoretical_cache_bytes"], 8 * 36 * 80 * 140)


def _reference_fwht(x: torch.Tensor) -> torch.Tensor:
    """Local reference radix-2 Sylvester butterfly FWHT in FP32 for AC-1 verification."""
    orig_shape = x.shape
    d = orig_shape[-1]
    x_flat = x.reshape(-1, d).to(dtype=torch.float32)
    h = 1
    while h < d:
        x_flat = x_flat.view(-1, d // (2 * h), 2, h)
        x1 = x_flat[:, :, 0, :]
        x2 = x_flat[:, :, 1, :]
        x_flat = torch.cat([x1 + x2, x1 - x2], dim=2)
        h *= 2
    return x_flat.view(orig_shape) * (1.0 / math.sqrt(d))


class TestTurboQuantHadamardOptimization(unittest.TestCase):
    def test_ac1_fwht_equivalence_shapes_dtypes_noncontiguous_and_empty(self):
        """AC-1: fwht/ifwht match reference butterfly across shapes, dims, dtypes, non-contiguous & empty."""
        torch.manual_seed(42)

        # 1D, 2D, 3D, 4D across d in (1, 2, 128, 256) and dtypes
        for d in (1, 2, 128, 256):
            for shape in ((d,), (4, d), (2, 8, d), (2, 4, 16, d)):
                for dtype in (torch.float32, torch.bfloat16, torch.float16):
                    x = torch.randn(shape, dtype=dtype)
                    expected = _reference_fwht(x)
                    out = fwht(x)

                    self.assertEqual(out.dtype, torch.float32)
                    self.assertEqual(out.shape, x.shape)
                    self.assertTrue(
                        torch.allclose(out, expected, atol=1e-5),
                        f"Mismatch for shape={shape}, dtype={dtype}",
                    )

                    # Energy preservation
                    norm_x = torch.norm(x.float(), p=2, dim=-1)
                    norm_out = torch.norm(out, p=2, dim=-1)
                    self.assertTrue(torch.allclose(norm_x, norm_out, atol=1e-5))

                    # Self-inverse
                    rec = ifwht(out)
                    self.assertEqual(rec.dtype, torch.float32)
                    self.assertTrue(torch.allclose(rec, x.float(), atol=1e-5))

        # Non-contiguous tensor via .transpose(1, 2)
        x_nc = torch.randn(2, 16, 4, 128, dtype=torch.float32).transpose(1, 2)
        self.assertFalse(x_nc.is_contiguous())
        out_nc = fwht(x_nc)
        self.assertEqual(out_nc.shape, x_nc.shape)
        self.assertEqual(out_nc.dtype, torch.float32)
        self.assertTrue(torch.allclose(out_nc, _reference_fwht(x_nc), atol=1e-5))
        self.assertTrue(torch.allclose(ifwht(out_nc), x_nc.float(), atol=1e-5))

        # Empty-sequence tensor (1, 4, 0, 128)
        for dtype in (torch.float32, torch.bfloat16):
            x_empty = torch.empty(1, 4, 0, 128, dtype=dtype)
            out_empty = fwht(x_empty)
            self.assertEqual(out_empty.shape, (1, 4, 0, 128))
            self.assertEqual(out_empty.dtype, torch.float32)
            rec_empty = ifwht(out_empty)
            self.assertEqual(rec_empty.shape, (1, 4, 0, 128))

    def test_ac2_hadamard_caching_and_inference_mode_safety(self):
        """AC-2: get_hadamard_matrix caches normalized device keys and is safe across inference_mode."""
        from src.techniques.turboquant_ops import get_hadamard_matrix

        with torch.inference_mode():
            h_infer = get_hadamard_matrix(128, "cpu")
        h_norm = get_hadamard_matrix(128, torch.device("cpu"))

        self.assertIs(h_infer, h_norm)
        self.assertFalse(h_infer.is_inference())
        self.assertEqual(h_infer.dtype, torch.float32)
        self.assertEqual(h_infer.shape, (128, 128))

        # Autograd compatibility after cache population under inference_mode
        x = torch.randn(2, 4, 128, dtype=torch.float32, requires_grad=True)
        y = fwht(x)
        y.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

    def test_ac3_invalid_hadamard_inputs_raise_value_error(self):
        """AC-3: Invalid dimensions and 0-D scalars raise ValueError in fwht and get_hadamard_matrix."""
        from src.techniques.turboquant_ops import get_hadamard_matrix

        for bad_d in (0, -4, 100, True, False):
            with self.assertRaises(ValueError, msg=f"get_hadamard_matrix should reject d={bad_d!r}"):
                get_hadamard_matrix(bad_d, "cpu")

        with self.assertRaises(ValueError):
            fwht(torch.empty(2, 0, dtype=torch.float32))
        with self.assertRaises(ValueError):
            fwht(torch.randn(2, 100, dtype=torch.float32))
        with self.assertRaises(ValueError):
            fwht(torch.tensor(1.0, dtype=torch.float32))


class TestTurboQuantPreallocatedCache(unittest.TestCase):
    def test_ac4_preallocation_data_ptr_invariance_and_active_slice_sizing(self):
        """AC-4: Pre-allocated 1024 capacity keeps identical data_ptr across decode & active-slice sizing."""
        torch.manual_seed(42)
        layer = TurboQuantLayer(seed=42)

        k_prefill = torch.randn(1, 8, 32, 128, dtype=torch.bfloat16)
        v_prefill = torch.randn(1, 8, 32, 128, dtype=torch.bfloat16)
        layer.update(k_prefill, v_prefill)

        self.assertEqual(layer.capacity, 1024)
        self.assertIsNotNone(layer.key_packed)
        self.assertIsNotNone(layer.key_norms)
        self.assertIsNotNone(layer.value_packed)
        self.assertIsNotNone(layer.value_norms)
        self.assertIsNotNone(layer.value_res_norms)

        ptrs_before = (
            layer.key_packed.data_ptr(),
            layer.key_norms.data_ptr(),
            layer.value_packed.data_ptr(),
            layer.value_norms.data_ptr(),
            layer.value_res_norms.data_ptr(),
        )

        k_out = v_out = None
        for _ in range(5):
            k_step = torch.randn(1, 8, 1, 128, dtype=torch.bfloat16)
            v_step = torch.randn(1, 8, 1, 128, dtype=torch.bfloat16)
            k_out, v_out = layer.update(k_step, v_step)
            ptrs_after = (
                layer.key_packed.data_ptr(),
                layer.key_norms.data_ptr(),
                layer.value_packed.data_ptr(),
                layer.value_norms.data_ptr(),
                layer.value_res_norms.data_ptr(),
            )
            self.assertEqual(ptrs_before, ptrs_after)

        self.assertEqual(layer.capacity, 1024)
        self.assertEqual(layer.get_seq_length(), 37)
        self.assertEqual(k_out.shape, (1, 8, 37, 128))
        self.assertEqual(v_out.shape, (1, 8, 37, 128))
        expected_bytes = 1 * 8 * 37 * 140
        self.assertEqual(layer.get_cache_size_bytes(), (expected_bytes, expected_bytes))

    def test_ac5_next_capacity_schedule(self):
        """AC-5a: _next_capacity doubles up to 16384 then grows linearly by +2048."""
        from src.techniques.turboquant_cache import _next_capacity

        self.assertEqual(_next_capacity(0, 1), 1024)
        self.assertEqual(_next_capacity(1024, 1025), 2048)
        self.assertEqual(_next_capacity(8192, 16384), 16384)
        self.assertEqual(_next_capacity(16384, 16385), 18432)
        self.assertEqual(_next_capacity(32768, 32769), 34816)

    def test_ac5_linear_growth_above_16384_on_cpu(self):
        """AC-5b: Lightweight (1, 1, *, 8) layer grows 16384 -> 18432 -> 32768 -> 34816."""
        torch.manual_seed(42)
        layer = TurboQuantLayer(seed=42)

        k_16k = torch.randn(1, 1, 16384, 8, dtype=torch.bfloat16)
        v_16k = torch.randn(1, 1, 16384, 8, dtype=torch.bfloat16)
        layer.update(k_16k, v_16k)
        self.assertEqual(layer.capacity, 16384)
        self.assertEqual(layer.get_seq_length(), 16384)

        k_1 = torch.randn(1, 1, 1, 8, dtype=torch.bfloat16)
        v_1 = torch.randn(1, 1, 1, 8, dtype=torch.bfloat16)
        layer.update(k_1, v_1)
        self.assertEqual(layer.capacity, 18432)
        self.assertEqual(layer.get_seq_length(), 16385)

        k_rem = torch.randn(1, 1, 32768 - 16385, 8, dtype=torch.bfloat16)
        v_rem = torch.randn(1, 1, 32768 - 16385, 8, dtype=torch.bfloat16)
        layer.update(k_rem, v_rem)
        self.assertEqual(layer.capacity, 32768)
        self.assertEqual(layer.get_seq_length(), 32768)

        layer.update(k_1, v_1)
        self.assertEqual(layer.capacity, 34816)
        self.assertEqual(layer.get_seq_length(), 32769)

    def test_ac5_cross_growth_boundary_numerical_equivalence(self):
        """AC-5c: Incremental 1020 -> 1030 crossing capacity growth matches single-shot 1030 update."""
        torch.manual_seed(42)
        k_all = torch.randn(1, 2, 1030, 128, dtype=torch.bfloat16)
        v_all = torch.randn(1, 2, 1030, 128, dtype=torch.bfloat16)

        layer_inc = TurboQuantLayer(seed=42)
        layer_inc.update(k_all[:, :, :1020, :], v_all[:, :, :1020, :])
        self.assertEqual(layer_inc.capacity, 1024)
        k_inc = v_inc = None
        for t in range(1020, 1030):
            k_inc, v_inc = layer_inc.update(k_all[:, :, t : t + 1, :], v_all[:, :, t : t + 1, :])

        layer_ref = TurboQuantLayer(seed=42)
        k_ref, v_ref = layer_ref.update(k_all, v_all)

        self.assertEqual(layer_inc.capacity, 2048)
        self.assertEqual(layer_ref.capacity, 2048)
        self.assertTrue(torch.equal(k_inc, k_ref))
        self.assertTrue(torch.equal(v_inc, v_ref))

    def test_ac6_empty_zero_token_update(self):
        """AC-6a: seq_len == 0 update on empty layer returns (B, H, 0, D) without initializing buffers."""
        layer = TurboQuantLayer(seed=42)
        k_empty = torch.empty(1, 8, 0, 128, dtype=torch.bfloat16)
        v_empty = torch.empty(1, 8, 0, 128, dtype=torch.bfloat16)

        k_out, v_out = layer.update(k_empty, v_empty)
        self.assertEqual(k_out.shape, (1, 8, 0, 128))
        self.assertEqual(v_out.shape, (1, 8, 0, 128))
        self.assertEqual(k_out.dtype, torch.bfloat16)
        self.assertEqual(v_out.dtype, torch.bfloat16)
        self.assertEqual(layer.total_tokens_seen, 0)
        self.assertEqual(layer.get_cache_size_bytes(), (0, 0))
        self.assertIsNone(layer.keys)
        self.assertIsNone(layer.values)

    def test_ac6_complete_reset_and_reuse_with_different_head_dim(self):
        """AC-6b: cache.reset() clears all state and permits reuse with a different head_dim."""
        torch.manual_seed(42)
        cache = TurboQuantCache(seed=42)
        k1 = torch.randn(1, 8, 32, 128, dtype=torch.bfloat16)
        v1 = torch.randn(1, 8, 32, 128, dtype=torch.bfloat16)
        cache.update(k1, v1, layer_idx=0)

        layer = cache.layers[0]
        self.assertGreater(layer.num_profile_chunks, 0)
        self.assertIsNotNone(layer._s)

        cache.reset()

        self.assertIsNone(layer.key_packed)
        self.assertIsNone(layer.key_norms)
        self.assertIsNone(layer.value_packed)
        self.assertIsNone(layer.value_norms)
        self.assertIsNone(layer.value_res_norms)
        self.assertEqual(layer.capacity, 0)
        self.assertEqual(layer.total_tokens_seen, 0)
        self.assertEqual(layer.get_seq_length(), 0)
        self.assertEqual(cache.get_seq_length(0), 0)
        self.assertFalse(layer.is_initialized)
        self.assertEqual(layer.accum_k_eps_snr, 0.0)
        self.assertEqual(layer.accum_k_eps_dir, 0.0)
        self.assertEqual(layer.accum_v_eps_snr, 0.0)
        self.assertEqual(layer.accum_v_eps_dir, 0.0)
        self.assertEqual(layer.num_profile_chunks, 0)
        self.assertIsNone(layer._s)
        self.assertIsNone(layer._s_prime)

        # Reuse with a different head_dim=64 and num_kv_heads=4
        k2 = torch.randn(1, 4, 16, 64, dtype=torch.bfloat16)
        v2 = torch.randn(1, 4, 16, 64, dtype=torch.bfloat16)
        k2_out, v2_out = cache.update(k2, v2, layer_idx=0)
        self.assertEqual(k2_out.shape, (1, 4, 16, 64))
        self.assertEqual(v2_out.shape, (1, 4, 16, 64))
        self.assertEqual(cache.get_seq_length(0), 16)

    def test_ac6_iter_and_layer_metadata_parity(self):
        """AC-6c: iter(cache) yields decoded (keys, values, None) or (None, None, None), plus metadata parity."""
        torch.manual_seed(42)
        cache = TurboQuantCache(seed=42)
        k = torch.randn(1, 8, 16, 128, dtype=torch.bfloat16)
        v = torch.randn(1, 8, 16, 128, dtype=torch.bfloat16)
        k_out, v_out = cache.update(k, v, layer_idx=0)

        layer = cache.layers[0]
        self.assertFalse(layer.is_croppable)
        self.assertEqual(layer.original_positions, list(range(16)))

        entries = list(cache)
        self.assertEqual(len(entries), 1)
        k_iter, v_iter, extra = entries[0]
        self.assertIsNone(extra)
        self.assertTrue(torch.equal(k_iter, k_out))
        self.assertTrue(torch.equal(v_iter, v_out))

        # After reset, empty layer yields (None, None, None)
        cache.reset()
        self.assertEqual(list(cache), [(None, None, None)])

    def test_ac7_atomic_entry_shape_guards(self):
        """AC-7: Mismatched shapes or non-4D inputs raise ValueError without mutating layer state."""
        torch.manual_seed(42)
        layer = TurboQuantLayer(seed=42)
        k_init = torch.randn(1, 8, 32, 128, dtype=torch.bfloat16)
        v_init = torch.randn(1, 8, 32, 128, dtype=torch.bfloat16)
        layer.update(k_init, v_init)

        orig_tokens = layer.total_tokens_seen
        orig_cap = layer.capacity
        self.assertEqual(orig_tokens, 32)
        self.assertEqual(orig_cap, 1024)

        bad_inputs = [
            ("batch_mismatch", torch.randn(2, 8, 1, 128, dtype=torch.bfloat16), torch.randn(2, 8, 1, 128, dtype=torch.bfloat16)),
            ("head_mismatch", torch.randn(1, 4, 1, 128, dtype=torch.bfloat16), torch.randn(1, 4, 1, 128, dtype=torch.bfloat16)),
            ("dim_mismatch", torch.randn(1, 8, 1, 64, dtype=torch.bfloat16), torch.randn(1, 8, 1, 64, dtype=torch.bfloat16)),
            ("kv_shape_mismatch", torch.randn(1, 8, 1, 128, dtype=torch.bfloat16), torch.randn(1, 8, 2, 128, dtype=torch.bfloat16)),
            ("non_4d", torch.randn(8, 1, 128, dtype=torch.bfloat16), torch.randn(8, 1, 128, dtype=torch.bfloat16)),
        ]
        for label, bad_k, bad_v in bad_inputs:
            with self.assertRaises(ValueError, msg=f"Expected ValueError for {label}"):
                layer.update(bad_k, bad_v)
            self.assertEqual(layer.total_tokens_seen, orig_tokens, msg=f"total_tokens_seen mutated on {label}")
            self.assertEqual(layer.capacity, orig_cap, msg=f"capacity mutated on {label}")


class TestTurboQuantSuiteConfig(unittest.TestCase):
    def test_ac8_fix_turboquant_suite_config(self):
        """AC-8: src/config/suites.json defines fix_turboquant with required evaluation flags."""
        suites_path = Path(__file__).resolve().parents[1] / "src" / "config" / "suites.json"
        with suites_path.open("r", encoding="utf-8") as f:
            suites = json.load(f)

        self.assertIn("fix_turboquant", suites)
        suite = suites["fix_turboquant"]
        self.assertEqual(suite["techniques"], ["turboquant"])
        self.assertIs(suite["force_longmem_ref"], False)
        self.assertIs(suite["skip_hardware_sweeps"], False)
        self.assertIs(suite["run_verify"], True)
        self.assertIs(suite["run_longmem"], True)
        self.assertIs(suite["run_niah"], True)
        self.assertIs(suite["skip_existing"], False)


if __name__ == "__main__":
    unittest.main()
