import gc
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.axis import Axis
from src.core.diagnostic import inspect_sample_diagnostic
from src.core.pytorch_runner import PyTorchRunner
from src.core.registry import get_technique, list_techniques
from src.techniques.headq import (
    HeadQTechnique,
    compute_gqa_headq_attention,
    compute_gqa_query_pca_bases,
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
        self.assertGreater(technique.compression_ratio, 4.0)

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
        self.assertEqual(kwargs["kv_cache_dtype"], "auto")

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            HeadQTechnique(key_bits=4)
        with self.assertRaises(ValueError):
            HeadQTechnique(val_bits=0)
        with self.assertRaises(ValueError):
            HeadQTechnique(val_bits=2)
        with self.assertRaises(ValueError):
            HeadQTechnique(rank=0)
        with self.assertRaises(ValueError):
            HeadQTechnique(rank=3)
        with self.assertRaises(ValueError):
            HeadQTechnique(sidecode_bits=2)
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

    def test_gqa_query_pca_bases_per_kv_head(self):
        torch.manual_seed(43)
        q = torch.randn(2, 8, 32, 64, dtype=torch.float32)
        bases = compute_gqa_query_pca_bases(q, num_kv_heads=2, rank=4)
        self.assertEqual(bases.shape, (2, 64, 4))
        identity = torch.eye(4, dtype=torch.float32)
        for h in range(2):
            inner = torch.matmul(bases[h].transpose(0, 1), bases[h])
            self.assertTrue(torch.allclose(inner, identity, atol=1e-5))

    def test_key_2bit_quantization_roundtrip_and_no_positive_clipping(self):
        torch.manual_seed(100)
        dim = 64
        k = torch.randn(2, 4, 16, dim, dtype=torch.float32)

        packed, scale, zero_point, dequant = quantize_key_2bit(k)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.shape[-1], dim // 4)

        unpacked = dequantize_key_2bit(packed, scale, zero_point, original_dim=dim, dtype=torch.float32)
        self.assertEqual(unpacked.shape, k.shape)
        self.assertTrue(torch.allclose(dequant, unpacked, atol=1e-5))

        # Verify positive-heavy asymmetric tensor is not clipped at +amax
        k_pos = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=torch.float32)
        _, _, _, dq_pos = quantize_key_2bit(k_pos)
        self.assertTrue(torch.allclose(dq_pos, k_pos, atol=1e-5))

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
        self.assertGreaterEqual(metrics["value_a2_distortion"], 0.0)

    def test_headq_layer_lifecycle_and_compression(self):
        torch.manual_seed(77)
        basis = compute_gqa_query_pca_bases(
            torch.randn(1, 8, 32, 128, dtype=torch.bfloat16), num_kv_heads=8, rank=4
        )
        layer = HeadQLayer(rank=4, sidecode_bits=4, basis=basis)
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

        # Calibrated basis must survive reset()
        layer.reset()
        self.assertIsNotNone(layer.basis)
        self.assertTrue(layer.is_calibrated)

    def test_telemetry_persists_after_cache_deletion_and_resets_on_setup(self):
        technique = HeadQTechnique()
        technique.setup()

        cache = technique.pytorch_cache_factory()()
        k = torch.randn(1, 2, 16, 64, dtype=torch.float32)
        v = torch.randn(1, 2, 16, 64, dtype=torch.float32)
        cache.update(k, v, layer_idx=0)
        expected_bytes = cache.total_quantized_bytes
        self.assertGreater(expected_bytes, 0)

        del cache
        gc.collect()
        self.assertIsNone(get_active_headq_cache())

        metrics = technique.extra_metrics(None, None)
        self.assertEqual(metrics["quantized_cache_bytes"], expected_bytes)
        self.assertEqual(metrics["head_dim"], 64)

        technique.setup()
        reset_metrics = technique.extra_metrics(None, None)
        self.assertEqual(reset_metrics["quantized_cache_bytes"], 0)

    def test_prepare_runner_captures_post_rope_queries(self):
        class _MockAttn(nn.Module):
            def forward(self, hidden_states):
                bsz, seq, _ = hidden_states.shape
                q = torch.randn(bsz, 4, seq, 32)
                k = torch.randn(bsz, 2, seq, 32)
                v = torch.randn(bsz, 2, seq, 32)
                return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

        class _MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = _MockAttn()

            def forward(self, input_ids, use_cache=False):
                del use_cache
                x = torch.randn(input_ids.shape[0], input_ids.shape[1], 64)
                return self.self_attn(x)

        mock_tokenizer = lambda text, return_tensors, truncation, max_length: {
            "input_ids": torch.ones((1, 24), dtype=torch.long)
        }
        runner = SimpleNamespace(
            model=_MockModel(),
            tokenizer=mock_tokenizer,
            device="cpu",
            dry_run=False,
            model_name_or_path="mock-model",
        )

        technique = HeadQTechnique(rank=4)
        technique.setup()
        technique.prepare_runner(runner)

        self.assertIn(0, technique._calibrated_bases)
        self.assertEqual(technique._calibrated_bases[0].shape, (2, 32, 4))
        metrics = technique.extra_metrics(None, None)
        self.assertTrue(metrics["is_query_calibrated"])
        self.assertEqual(metrics["calibrated_layers"], 1)

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
        self.assertEqual(diag.overall_effective_bitwidth, 3.1)

        runner = PyTorchRunner(
            model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
            revision="cdbee75f17c01a7cc42f958dc650907174af0554",
            cache_factory=cache_factory,
            dry_run=True,
        )
        outputs = runner.generate(["Test HeadQ prompt."], max_tokens=16)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(outputs[0].outputs[0].token_ids), 16)


if __name__ == "__main__":
    unittest.main()
