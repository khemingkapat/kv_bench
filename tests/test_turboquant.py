import math
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
        self.assertAlmostEqual(metrics["compression_ratio"], 3.657, places=2)
        self.assertEqual(metrics["theoretical_bytes_per_token_head"], 140)
        self.assertEqual(metrics["uncompressed_bytes_per_token_head"], 512)
        self.assertGreater(metrics["theoretical_cache_bytes"], 0)
        self.assertGreater(metrics["shannon_k_eps_snr"], 0.0)

        # Test analytical fallback after reset when active cache is absent
        tech.setup()
        fallback_metrics = tech.extra_metrics(None, mock_out)
        self.assertEqual(fallback_metrics["theoretical_cache_bytes"], 8 * 36 * 80 * 140)


if __name__ == "__main__":
    unittest.main()
