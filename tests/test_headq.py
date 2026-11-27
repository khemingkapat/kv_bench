import unittest
from types import SimpleNamespace

import torch

from src.core.axis import Axis
from src.core.diagnostic import inspect_sample_diagnostic
from src.core.pytorch_runner import PyTorchRunner
from src.core.registry import get_technique, list_techniques
from src.techniques.headq import (
    HeadQTechnique,
    compute_gqa_headq_attention,
    compute_key_sidecode,
    compute_query_pca_basis,
    dequantize_key_2bit,
    dequantize_key_sidecode,
    dequantize_value_4bit,
    quantize_key_2bit,
    quantize_value_4bit,
)
from src.techniques.headq_cache import (
    HeadQCache,
    HeadQLayer,
    get_active_headq_cache,
)


class TestHeadQTechnique(unittest.TestCase):
    def test_registration_and_axis(self):
        self.assertIn("headq", list_techniques())
        technique = get_technique("headq")
        self.assertIsInstance(technique, HeadQTechnique)
        self.assertEqual(technique.axes, Axis.STRUCTURAL)
        self.assertFalse(hasattr(technique, "compression_ratio"))

    def test_defaults_and_custom_parameters(self):
        technique = get_technique("headq")
        self.assertEqual(technique.key_bits, 2)
        self.assertEqual(technique.val_bits, 4)
        self.assertEqual(technique.rank, 4)
        self.assertEqual(technique.sidecode_bits, 4)
        self.assertEqual(technique.block_size, 16)

        mock_out = SimpleNamespace(prompt_tokens=1024, generated_tokens=64)
        metrics = technique.extra_metrics(None, mock_out)
        self.assertEqual(metrics["key_bits"], 2)
        self.assertEqual(metrics["val_bits"], 4)
        self.assertEqual(metrics["pca_rank"], 4)
        self.assertTrue(metrics["model_visible_distortion_aware"])

        kwargs = technique.engine_kwargs(context_length=2048, max_tokens=128)
        self.assertEqual(kwargs["block_size"], 16)
        self.assertEqual(kwargs["kv_cache_dtype"], "fp8")

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            HeadQTechnique(key_bits=4)
        with self.assertRaises(ValueError):
            HeadQTechnique(val_bits=0)
        with self.assertRaises(ValueError):
            HeadQTechnique(rank=0)
        with self.assertRaises(ValueError):
            HeadQTechnique(block_size=-1)

    def test_query_pca_basis_orthonormality(self):
        torch.manual_seed(42)
        dim = 64
        rank = 4
        queries = torch.randn(100, dim, dtype=torch.float32)
        basis = compute_query_pca_basis(queries, rank=rank)

        self.assertEqual(basis.shape, (dim, rank))
        identity = torch.eye(rank, dtype=torch.float32)
        inner = torch.matmul(basis.transpose(0, 1), basis)
        self.assertTrue(torch.allclose(inner, identity, atol=1e-5))

    def test_key_2bit_quantization_roundtrip(self):
        torch.manual_seed(100)
        dim = 64
        k = torch.randn(2, 4, 16, dim, dtype=torch.float32)

        packed, scale, dequant = quantize_key_2bit(k)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.shape[-1], dim // 4)

        unpacked = dequantize_key_2bit(packed, scale, original_dim=dim, dtype=torch.float32)
        self.assertEqual(unpacked.shape, k.shape)
        self.assertTrue(torch.allclose(dequant, unpacked, atol=1e-5))

    def test_key_sidecode_projection_roundtrip(self):
        torch.manual_seed(200)
        dim = 64
        rank = 4
        queries = torch.randn(50, dim, dtype=torch.float32)
        basis = compute_query_pca_basis(queries, rank=rank)

        k_residual = torch.randn(2, 4, 16, dim, dtype=torch.float32) * 0.1
        packed_code, scale, dequant_code = compute_key_sidecode(k_residual, basis, sidecode_bits=4)

        self.assertEqual(packed_code.dtype, torch.uint8)
        self.assertEqual(packed_code.shape[-1], rank // 2)

        unpacked_code = dequantize_key_sidecode(packed_code, scale, rank=rank, dtype=torch.float32)
        self.assertEqual(unpacked_code.shape, (2, 4, 16, rank))
        self.assertTrue(torch.allclose(dequant_code, unpacked_code, atol=1e-5))

    def test_value_4bit_quantization_roundtrip(self):
        torch.manual_seed(300)
        dim = 64
        v = torch.randn(2, 4, 16, dim, dtype=torch.float32)

        packed, scale, dequant = quantize_value_4bit(v)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.shape[-1], dim // 2)

        unpacked = dequantize_value_4bit(packed, scale, original_dim=dim, dtype=torch.float32)
        self.assertEqual(unpacked.shape, v.shape)
        self.assertTrue(torch.allclose(dequant, unpacked, atol=1e-5))

    def test_gqa_headq_attention_and_kl_reduction(self):
        batch, seq_len, head_dim = 1, 32, 64
        num_q_heads, num_kv_heads = 8, 2

        torch.manual_seed(400)
        q = torch.randn(batch, num_q_heads, seq_len, head_dim, dtype=torch.float32)
        k = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.float32)
        v = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.float32)

        attn_out, attn_weights, metrics = compute_gqa_headq_attention(q, k, v, rank=4)

        self.assertEqual(attn_out.shape, (batch, num_q_heads, seq_len, head_dim))
        self.assertEqual(attn_weights.shape, (batch, num_q_heads, seq_len, seq_len))
        self.assertTrue(torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1)), atol=1e-4))
        self.assertLess(metrics["attention_kl_corrected"], metrics["attention_kl_coarse"])
        self.assertGreater(metrics["kl_reduction_ratio"], 0.0)

    def test_headq_layer_lifecycle_and_compression(self):
        torch.manual_seed(77)
        layer = HeadQLayer(rank=4, sidecode_bits=4)
        batch, heads, seq_len, dim = 1, 8, 24, 128

        k = torch.randn(batch, seq_len, heads, dim, dtype=torch.bfloat16).transpose(1, 2)
        v = torch.randn(batch, seq_len, heads, dim, dtype=torch.bfloat16).transpose(1, 2)

        out_k, out_v = layer.update(k, v)
        self.assertEqual(out_k.shape, (batch, heads, seq_len, dim))
        self.assertEqual(out_v.shape, (batch, heads, seq_len, dim))
        self.assertEqual(layer.get_seq_length(), 24)

        k_step = torch.randn(batch, heads, 1, dim, dtype=torch.bfloat16)
        v_step = torch.randn(batch, heads, 1, dim, dtype=torch.bfloat16)
        out_k2, _ = layer.update(k_step, v_step)
        self.assertEqual(out_k2.shape, (batch, heads, 25, dim))
        self.assertEqual(layer.original_positions, list(range(25)))

        bf16_bytes = 2 * batch * heads * 25 * dim * 2
        self.assertLess(layer.quantized_bytes, bf16_bytes * 0.4)

    def test_headq_cache_factory_and_pytorch_runner(self):
        technique = get_technique("headq")
        cache_factory = technique.pytorch_cache_factory()
        self.assertIsNotNone(cache_factory)

        cache = cache_factory()
        self.assertIsInstance(cache, HeadQCache)
        self.assertEqual(get_active_headq_cache(), cache)

        k1 = torch.randn(1, 8, 16, 128, dtype=torch.float32)
        v1 = torch.randn(1, 8, 16, 128, dtype=torch.float32)
        cache.update(k1, v1, layer_idx=0)

        diag = inspect_sample_diagnostic(
            cache=cache,
            prompt_text="Test headq diagnostic",
            prompt_token_ids=list(range(16)),
            sample_meta={"id": "hq_diag_0", "task_type": "unit"},
            is_correct=True,
            baseline_correct=True,
            bits_per_scalar=3.1,
        )
        self.assertEqual(diag.overall_retention_ratio, 1.0)
        self.assertEqual(diag.overall_effective_bitwidth, 3.25)

        runner = PyTorchRunner(
            model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
            revision="cdbee75f17c01a7cc42f958dc650907174af0554",
            cache_factory=cache_factory,
            dry_run=True,
        )
        outputs = runner.generate(["Test HeadQ prompt."], max_tokens=16)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(outputs[0].outputs[0].token_ids), 16)


class TestHeadQGateCFixes(unittest.TestCase):
    def test_ac1_per_head_query_pca_basis_and_m_lt_rank(self):
        from src.techniques import headq

        compute_per_head = getattr(headq, "compute_per_head_query_pca_basis", None)
        self.assertIsNotNone(compute_per_head, "compute_per_head_query_pca_basis must be implemented")

        torch.manual_seed(11)
        # Standard GQA case: (1, 32, 64, 128), num_kv_heads=8, rank=4
        q = torch.randn(1, 32, 64, 128, dtype=torch.bfloat16)
        basis = compute_per_head(q, num_kv_heads=8, rank=4)
        self.assertEqual(basis.shape, (1, 8, 128, 4))
        self.assertEqual(basis.dtype, torch.bfloat16)

        eye = torch.eye(4, dtype=torch.float32)
        for h in range(8):
            u = basis[0, h].float()
            err = (u.transpose(0, 1) @ u - eye).abs().max().item()
            self.assertLess(err, 1e-2)  # bfloat16 precision

        # Batch > 1 and head independence: changing head 0's queries must not change head 1's basis
        q_b2 = torch.randn(2, 8, 16, 64, dtype=torch.float32)
        basis_1 = compute_per_head(q_b2, num_kv_heads=4, rank=4)
        q_b2_mod = q_b2.clone()
        q_b2_mod[:, 0:2, :, :] = torch.randn_like(q_b2_mod[:, 0:2, :, :]) * 5.0
        basis_2 = compute_per_head(q_b2_mod, num_kv_heads=4, rank=4)
        self.assertTrue(torch.allclose(basis_1[:, 1:], basis_2[:, 1:], atol=1e-5))
        self.assertFalse(torch.allclose(basis_1[:, 0], basis_2[:, 0], atol=1e-3))

        # M < rank edge case: B=1, num_q_heads=2, num_kv_heads=1, T=1 -> M=2 < rank=4
        q_small = torch.randn(1, 2, 1, 32, dtype=torch.float32)
        basis_small = compute_per_head(q_small, num_kv_heads=1, rank=4)
        self.assertEqual(basis_small.shape, (1, 1, 32, 4))
        u_small = basis_small[0, 0]
        self.assertLess((u_small.transpose(0, 1) @ u_small - eye).abs().max().item(), 1e-4)

        # M == 1 edge case: B=1, num_q_heads=1, num_kv_heads=1, T=1 -> M=1 < rank=4
        q_m1 = torch.randn(1, 1, 1, 32, dtype=torch.float32)
        basis_m1 = compute_per_head(q_m1, num_kv_heads=1, rank=4)
        self.assertEqual(basis_m1.shape, (1, 1, 32, 4))
        u_m1 = basis_m1[0, 0]
        self.assertLess((u_m1.transpose(0, 1) @ u_m1 - eye).abs().max().item(), 1e-4)

    def test_ac2_per_head_pca_input_validation(self):
        from src.techniques import headq

        compute_per_head = getattr(headq, "compute_per_head_query_pca_basis", None)
        self.assertIsNotNone(compute_per_head, "compute_per_head_query_pca_basis must be implemented")

        with self.assertRaises(ValueError):
            compute_per_head(torch.randn(10, 64), num_kv_heads=2, rank=4)  # ndim != 4
        with self.assertRaises(ValueError):
            compute_per_head(torch.randn(1, 8, 16, 64), num_kv_heads=0, rank=4)
        with self.assertRaises(ValueError):
            compute_per_head(torch.randn(1, 8, 16, 64), num_kv_heads=3, rank=4)  # 8 % 3 != 0
        with self.assertRaises(ValueError):
            compute_per_head(torch.randn(1, 8, 16, 64), num_kv_heads=2, rank=0)
        with self.assertRaises(ValueError):
            compute_per_head(torch.randn(1, 8, 16, 16), num_kv_heads=2, rank=32)  # rank > d

    def test_ac3_symmetric_2bit_rms_and_outlier_standardization(self):
        torch.manual_seed(22)
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            k = torch.randn(1, 4, 16, 64, dtype=dtype)
            if dtype == torch.float16:
                k[:, :, 0, 0] = 500.0  # > 255 to test FP16 pow(2) overflow prevention
            packed, scale, dequant = quantize_key_2bit(k)
            self.assertTrue(torch.isfinite(scale).all())
            self.assertTrue(torch.isfinite(dequant).all())
            unpacked = dequantize_key_2bit(packed, scale, original_dim=64, dtype=dtype)
            atol = 1e-5 if dtype == torch.float32 else 0.0
            self.assertTrue(torch.allclose(dequant, unpacked, atol=atol))

        # Verify symmetric 4-level grid {-1.5, -0.5, +0.5, +1.5} * scale (positive and negative symmetric levels)
        k_sym = torch.tensor([[[[-3.0, -1.0, 1.0, 3.0] * 16]]], dtype=torch.float32)
        packed_s, scale_s, dequant_s = quantize_key_2bit(k_sym)
        unique_ratios = torch.unique(torch.round((dequant_s / scale_s) * 2.0) / 2.0).tolist()
        self.assertEqual(sorted(unique_ratios), [-1.5, -0.5, 0.5, 1.5])

        # Outlier channels + persistent DC bias must NOT collapse normal channels in HeadQLayer
        torch.manual_seed(33)
        layer = HeadQLayer(rank=4, sidecode_bits=4)
        k_out = torch.randn(1, 8, 64, 128, dtype=torch.bfloat16)
        # Persistent per-channel DC bias + 100x outlier channels on dims 0..3
        dc_bias = torch.randn(1, 8, 1, 128, dtype=torch.bfloat16) * 5.0
        k_out = k_out + dc_bias
        k_out[:, :, :, :4] *= 100.0
        v_out = torch.randn(1, 8, 64, 128, dtype=torch.bfloat16)

        # Simulate Chunk 0 + SDPA query hook finalization
        layer.update(k_out, v_out)
        q_out = torch.randn(1, 32, 64, 128, dtype=torch.bfloat16)
        layer.finalize_chunk0_from_queries(q_out)
        k_recon = layer.keys
        self.assertIsNotNone(k_recon)

        cos_sim = torch.nn.functional.cosine_similarity(
            k_out.float().reshape(-1, 128), k_recon.float().reshape(-1, 128), dim=-1
        ).mean().item()
        self.assertGreaterEqual(cos_sim, 0.95)
        # Non-outlier channels (4..127) must not collapse to zero
        non_outlier_energy = k_recon[:, :, :, 4:].float().abs().mean().item()
        self.assertGreater(non_outlier_energy, 0.5)

    def test_ac4_ac5_sdpa_hook_deferred_chunk0_and_lifecycle(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from src.techniques.headq_cache import set_active_headq_cache

        technique = HeadQTechnique(rank=4, sidecode_bits=4)
        orig_sdpa_before = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
        self.assertIsNotNone(orig_sdpa_before)

        captured_calls = []

        def spy_sdpa(module, query, key, value, *args, **kwargs):
            captured_calls.append({"key": key.detach().clone(), "module": module})
            return torch.zeros_like(query), None

        ALL_ATTENTION_FUNCTIONS.register("sdpa", spy_sdpa)
        try:
            technique.setup()
            # Idempotent second setup()
            technique.setup()

            sdpa_fn = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
            self.assertTrue(getattr(sdpa_fn, "_is_headq_sdpa_wrapper", False))

            cache = technique.pytorch_cache_factory()()
            torch.manual_seed(44)
            k0 = torch.randn(1, 8, 32, 128, dtype=torch.bfloat16)
            v0 = torch.randn(1, 8, 32, 128, dtype=torch.bfloat16)
            k_base_returned, _ = cache.update(k0, v0, layer_idx=0)
            layer = cache.layers[0]

            # Before SDPA: basis is None, pending residual exists, keys returns k_base
            self.assertIsNone(layer.basis)
            self.assertIsNotNone(layer._pending_chunk0_residual)
            self.assertTrue(torch.allclose(layer.keys, k_base_returned))

            # Fire SDPA hook with post-RoPE query (1, 32, 32, 128)
            q0 = torch.randn(1, 32, 32, 128, dtype=torch.bfloat16)
            mod = SimpleNamespace(layer_idx=0)
            sdpa_fn(mod, q0, k_base_returned, v0)

            self.assertIsNotNone(layer.basis)
            self.assertEqual(layer.basis.shape, (1, 8, 128, 4))
            self.assertEqual(layer.basis.dtype, torch.bfloat16)
            self.assertIsNone(layer._pending_chunk0_residual)
            self.assertIsNotNone(layer.z_packed_buf)
            self.assertEqual(len(captured_calls), 1)
            # Underlying SDPA received PCA-corrected layer.keys, differing from k_base
            passed_key = captured_calls[0]["key"]
            self.assertTrue(torch.allclose(passed_key, layer.keys))
            self.assertFalse(torch.allclose(passed_key, k_base_returned))

            # AC-5: Stale cache isolation & mismatched seq_len guard
            stale_cache = HeadQCache(rank=4, sidecode_bits=4)
            stale_cache.update(k0, v0, layer_idx=0)
            stale_layer = stale_cache.layers[0]
            self.assertIsNone(stale_layer.basis)

            # Mismatched seq_len must not mutate stale_cache or replace key
            other_k = torch.randn(1, 8, 12, 128, dtype=torch.bfloat16)
            other_q = torch.randn(1, 32, 12, 128, dtype=torch.bfloat16)
            other_v = torch.randn(1, 8, 12, 128, dtype=torch.bfloat16)
            sdpa_fn(mod, other_q, other_k, other_v)
            self.assertTrue(torch.allclose(captured_calls[-1]["key"], other_k))
            self.assertIsNone(stale_layer.basis)

            # Calling setup() clears active cache reference so stale_cache is untouched even with matching seq_len
            technique.setup()
            self.assertIsNone(get_active_headq_cache())
            sdpa_fn = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
            sdpa_fn(mod, q0, k0, v0)
            self.assertTrue(torch.allclose(captured_calls[-1]["key"], k0))
            self.assertIsNone(stale_layer.basis)

            # Teardown restores spy_sdpa
            technique.teardown()
            self.assertIs(ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None), spy_sdpa)
        finally:
            technique.teardown()
            set_active_headq_cache(None)
            ALL_ATTENTION_FUNCTIONS.register("sdpa", orig_sdpa_before)

    def test_ac6_standalone_fallback_second_update_with_t0_1(self):
        torch.manual_seed(55)
        layer = HeadQLayer(rank=4, sidecode_bits=4)
        k0 = torch.randn(1, 8, 1, 128, dtype=torch.bfloat16)
        v0 = torch.randn(1, 8, 1, 128, dtype=torch.bfloat16)
        out_k0, _ = layer.update(k0, v0)
        self.assertEqual(out_k0.shape, (1, 8, 1, 128))
        self.assertIsNone(layer.basis)
        self.assertIsNotNone(layer._pending_chunk0_residual)

        k1 = torch.randn(1, 8, 3, 128, dtype=torch.bfloat16)
        v1 = torch.randn(1, 8, 3, 128, dtype=torch.bfloat16)
        out_k1, _ = layer.update(k1, v1)
        self.assertEqual(out_k1.shape, (1, 8, 4, 128))
        self.assertIsNotNone(layer.basis)
        self.assertEqual(layer.basis.shape, (1, 8, 128, 4))
        self.assertIsNone(layer._pending_chunk0_residual)
        self.assertTrue(torch.isfinite(out_k1).all())

    def test_ac7_compute_gqa_headq_attention_with_4d_basis(self):
        from src.techniques import headq

        compute_per_head = getattr(headq, "compute_per_head_query_pca_basis", None)
        self.assertIsNotNone(compute_per_head)

        torch.manual_seed(66)
        q = torch.randn(1, 8, 32, 64, dtype=torch.float32)
        k = torch.randn(1, 2, 32, 64, dtype=torch.float32)
        v = torch.randn(1, 2, 32, 64, dtype=torch.float32)
        basis_4d = compute_per_head(q, num_kv_heads=2, rank=4)
        self.assertEqual(basis_4d.shape, (1, 2, 64, 4))

        attn_out, attn_w, metrics = compute_gqa_headq_attention(q, k, v, basis=basis_4d, rank=4)
        self.assertEqual(attn_out.shape, (1, 8, 32, 64))
        self.assertEqual(attn_w.shape, (1, 8, 32, 32))
        self.assertLess(metrics["attention_kl_corrected"], metrics["attention_kl_coarse"])

    def test_ac8_linear_capacity_growth_above_16k_and_validation(self):
        layer = HeadQLayer(rank=4, sidecode_bits=4)
        layer._ensure_capacity(1, 8, 128, torch.device("cpu"), torch.bfloat16, 16384)
        self.assertEqual(layer.capacity, 16384)

        layer._ensure_capacity(1, 8, 128, torch.device("cpu"), torch.bfloat16, 16385)
        self.assertEqual(layer.capacity, 18432)  # 16384 + 2048, not 32768

        # Mismatched shape without reset raises ValueError
        with self.assertRaises(ValueError):
            layer._ensure_capacity(1, 4, 128, torch.device("cpu"), torch.bfloat16, 18432)

        layer2 = HeadQLayer(rank=4, sidecode_bits=4)
        layer2._ensure_capacity(1, 8, 128, torch.device("cpu"), torch.bfloat16, 20000)
        self.assertEqual(layer2.capacity, 20000)

        # quantized_bytes includes k_mean and k_std
        k = torch.randn(1, 8, 16, 128, dtype=torch.bfloat16)
        v = torch.randn(1, 8, 16, 128, dtype=torch.bfloat16)
        layer3 = HeadQLayer(rank=4, sidecode_bits=4)
        layer3.update(k, v)
        self.assertIsNotNone(layer3.k_mean)
        self.assertIsNotNone(layer3.k_std)
        stats_bytes = layer3.k_mean.nelement() * layer3.k_mean.element_size() + (
            layer3.k_std.nelement() * layer3.k_std.element_size()
        )
        self.assertEqual(stats_bytes, 4096)
        self.assertGreaterEqual(layer3.quantized_bytes, stats_bytes)
        self.assertEqual(HeadQCache().effective_bits_per_scalar, 3.25)

    def test_ac9_fix_headq_suite_config(self):
        import json
        import pathlib

        suites_path = pathlib.Path(__file__).resolve().parents[1] / "src" / "config" / "suites.json"
        with suites_path.open("r", encoding="utf-8") as f:
            suites = json.load(f)

        self.assertIn("fix_headq", suites)
        cfg = suites["fix_headq"]
        self.assertEqual(cfg["techniques"], ["headq"])
        self.assertIs(cfg["force_longmem_ref"], False)
        self.assertIs(cfg["skip_existing"], False)
        self.assertIs(cfg["skip_hardware_sweeps"], False)
        self.assertIs(cfg["run_verify"], True)
        self.assertIs(cfg["run_longmem"], True)
        self.assertIs(cfg["run_niah"], True)
        self.assertEqual(cfg["contexts"], [512, 1024, 2048, 4096, 8192, 16384, 32768])
        self.assertEqual(cfg["eval_contexts"], [32768])
        self.assertEqual(cfg["niah_contexts"], [2048, 4096])


if __name__ == "__main__":
    unittest.main()
