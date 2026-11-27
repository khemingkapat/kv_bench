import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from src.core.axis import Axis
from src.core.diagnostic import inspect_sample_diagnostic
from src.core.pytorch_runner import PyTorchRunner
from src.core.registry import get_technique, list_techniques
from src.techniques.qserve_cache import (
    QServeKVCache,
    QServeKVLayer,
    get_active_qserve_cache,
)
from src.techniques.qserve_kv4 import (
    compute_gqa_qserve_attention,
    compute_smooth_attention_scales,
    dequantize_kv_int4,
    pack_int4,
    quantize_kv_int4,
    unpack_int4,
)


class TestQServeKV4Technique(unittest.TestCase):
    def test_registration_and_axis(self):
        self.assertIn("qserve_kv4", list_techniques())
        technique = get_technique("qserve_kv4")
        self.assertEqual(technique.axes, Axis.STRUCTURAL)

    def test_defaults_and_custom_overrides(self):
        technique = get_technique("qserve_kv4")
        self.assertEqual(technique.group_size, 32)
        self.assertEqual(technique.alpha, 0.5)
        self.assertEqual(technique.block_size, 16)

        custom = get_technique("qserve_kv4", group_size=64, alpha=0.75, block_size=32)
        self.assertEqual(custom.group_size, 64)
        self.assertEqual(custom.alpha, 0.75)
        self.assertEqual(custom.block_size, 32)

        metrics = custom.extra_metrics(None, None)
        self.assertEqual(metrics["kv_cache_bits"], 4)
        self.assertEqual(metrics["quant_group_size"], 64)
        self.assertEqual(metrics["smooth_attention_alpha"], 0.75)
        self.assertNotIn("compression_ratio_vs_bf16", metrics)

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            get_technique("qserve_kv4", group_size=0)
        with self.assertRaises(ValueError):
            get_technique("qserve_kv4", alpha=-0.1)
        with self.assertRaises(ValueError):
            get_technique("qserve_kv4", alpha=1.5)
        with self.assertRaises(ValueError):
            get_technique("qserve_kv4", block_size=0)

    def test_smooth_attention_mathematical_invariance(self):
        """Verify SmoothAttention scales preserve Q @ K^T while attenuating Key channel outliers."""
        torch.manual_seed(42)
        batch = 2
        num_q_heads = 16   # 4:1 GQA group ratio
        num_kv_heads = 4
        q_len = 8
        kv_len = 32
        head_dim = 64

        q = torch.randn(batch, num_q_heads, q_len, head_dim)
        k = torch.randn(batch, num_kv_heads, kv_len, head_dim)

        # Inject severe outlier into Key channel 5 (50x amplitude)
        k[:, :, :, 5] *= 50.0

        q_smooth, k_smooth = compute_smooth_attention_scales(q, k, alpha=0.5)

        # 1. Verify attention logits Q @ K^T are mathematically invariant across all GQA heads
        group_size = num_q_heads // num_kv_heads
        k_exp = k.repeat_interleave(group_size, dim=1)
        k_smooth_exp = k_smooth.repeat_interleave(group_size, dim=1)

        orig_logits = torch.matmul(q, k_exp.transpose(-2, -1))
        smooth_logits = torch.matmul(q_smooth, k_smooth_exp.transpose(-2, -1))

        self.assertTrue(torch.allclose(orig_logits, smooth_logits, atol=1e-4, rtol=1e-4))

        # 2. Key channel outlier amplitude must be significantly attenuated in k_smooth
        orig_max = k[:, :, :, 5].abs().max().item()
        smooth_max = k_smooth[:, :, :, 5].abs().max().item()
        self.assertLess(smooth_max, orig_max * 0.5)

    def test_pack_and_unpack_int4(self):
        """Verify lossless bit packing and unpacking of INT4 values in [-8, 7]."""
        values = torch.tensor([-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int8)
        packed = pack_int4(values)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.numel(), len(values) // 2)

        unpacked = unpack_int4(packed, original_dim=len(values))
        self.assertTrue(torch.equal(values, unpacked))

    def test_quantize_and_dequantize_kv_int4_fidelity(self):
        """Verify 4-bit quantization compresses memory by 4x and retains >0.99 cosine similarity."""
        torch.manual_seed(42)
        shape = (2, 8, 64, 128)
        tensor = torch.randn(shape, dtype=torch.float32)

        packed, scales = quantize_kv_int4(tensor, group_size=32)
        self.assertEqual(packed.shape, (2, 8, 64, 64))
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(scales.shape, (2, 8, 64, 4, 1))

        dequant = dequantize_kv_int4(packed, scales, original_dtype=torch.float32, group_size=32)
        self.assertEqual(dequant.shape, shape)
        self.assertEqual(dequant.dtype, torch.float32)

        cos_sim = F.cosine_similarity(tensor.flatten(), dequant.flatten(), dim=0).item()
        self.assertGreater(cos_sim, 0.99)

    def test_gqa_qserve_attention_end_to_end(self):
        """Verify full forward attention pass with QServe SmoothAttention + INT4 KV cache."""
        torch.manual_seed(42)
        batch = 1
        num_q_heads = 32   # Qwen3-4B
        num_kv_heads = 8   # 4:1 ratio
        q_len = 1
        kv_len = 128
        head_dim = 128

        q = torch.randn(batch, num_q_heads, q_len, head_dim)
        k = torch.randn(batch, num_kv_heads, kv_len, head_dim)
        v = torch.randn(batch, num_kv_heads, kv_len, head_dim)

        out, weights = compute_gqa_qserve_attention(q, k, v, alpha=0.5, group_size=32)
        self.assertEqual(out.shape, (batch, num_q_heads, q_len, head_dim))
        self.assertEqual(weights.shape, (batch, num_q_heads, q_len, kv_len))

        weight_sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5))

        gqa_ratio = num_q_heads // num_kv_heads
        k_exp = k.repeat_interleave(gqa_ratio, dim=1)
        v_exp = v.repeat_interleave(gqa_ratio, dim=1)
        scores_ref = torch.matmul(q.float(), k_exp.float().transpose(-2, -1)) * (head_dim ** -0.5)
        weights_ref = F.softmax(scores_ref, dim=-1)
        out_ref = torch.matmul(weights_ref, v_exp)

        cos_sim = F.cosine_similarity(out.flatten(), out_ref.flatten(), dim=0).item()
        self.assertGreater(cos_sim, 0.98)

    def test_engine_kwargs(self):
        technique = get_technique("qserve_kv4", group_size=32, block_size=16)
        kwargs = technique.engine_kwargs(context_length=4096, max_tokens=128)
        self.assertEqual(kwargs["block_size"], 16)
        self.assertEqual(kwargs["kv_cache_dtype"], "fp8")

    def test_qserve_layer_prefill_decode_and_non_contiguous(self):
        """Verify QServeKVLayer handles HF-style non-contiguous updates, decode steps, and compression."""
        torch.manual_seed(7)
        layer = QServeKVLayer(group_size=32, alpha=0.5)
        batch, heads, seq_len, dim = 1, 8, 32, 128

        # Simulate HF attention transpose(1, 2) non-contiguous layout + channel outlier
        k_raw = torch.randn(batch, seq_len, heads, dim, dtype=torch.bfloat16).transpose(1, 2)
        v_raw = torch.randn(batch, seq_len, heads, dim, dtype=torch.bfloat16).transpose(1, 2)
        k_raw_mod = k_raw.clone()
        k_raw_mod[:, :, :, 3] *= 25.0

        out_k, out_v = layer.update(k_raw_mod, v_raw)
        self.assertEqual(out_k.shape, (batch, heads, seq_len, dim))
        self.assertEqual(out_v.shape, (batch, heads, seq_len, dim))
        self.assertEqual(layer.get_seq_length(), 32)
        self.assertEqual(layer.original_positions, list(range(32)))

        # Decode append 1 token at a time
        k_step = torch.randn(batch, heads, 1, dim, dtype=torch.bfloat16)
        v_step = torch.randn(batch, heads, 1, dim, dtype=torch.bfloat16)
        out_k2, out_v2 = layer.update(k_step, v_step)
        self.assertEqual(out_k2.shape, (batch, heads, 33, dim))
        self.assertEqual(layer.get_seq_length(), 33)
        self.assertEqual(layer.original_positions, list(range(33)))

        # Verify physical quantized footprint is significantly smaller than BF16
        bf16_bytes = 2 * batch * heads * 33 * dim * 2
        self.assertLess(layer.quantized_bytes, bf16_bytes * 0.4)

        # Verify high cosine fidelity
        k_sim = F.cosine_similarity(k_raw_mod.float().flatten(), out_k2[:, :, :32].float().flatten(), dim=0).item()
        self.assertGreater(k_sim, 0.98)

    def test_qserve_cache_factory_and_diagnostic_contract(self):
        technique = get_technique("qserve_kv4", group_size=32, alpha=0.5)
        factory = technique.pytorch_cache_factory()
        self.assertIsNotNone(factory)

        cache = factory()
        self.assertIsInstance(cache, QServeKVCache)
        self.assertEqual(get_active_qserve_cache(), cache)

        k1 = torch.randn(1, 8, 24, 128, dtype=torch.float32)
        v1 = torch.randn(1, 8, 24, 128, dtype=torch.float32)
        cache.update(k1, v1, layer_idx=0)
        self.assertEqual(cache.get_seq_length(0), 24)
        self.assertEqual(cache.original_positions, list(range(24)))

        diag = inspect_sample_diagnostic(
            cache=cache,
            prompt_text="Test diagnostic integration",
            prompt_token_ids=list(range(24)),
            sample_meta={"id": "qserve_diag_0", "task_type": "unit"},
            is_correct=True,
            baseline_correct=True,
            bits_per_scalar=4.0,
        )
        self.assertEqual(diag.overall_retention_ratio, 1.0)
        self.assertEqual(diag.overall_effective_bitwidth, 4.5)

    def test_pytorch_runner_dry_run_with_qserve_cache(self):
        technique = get_technique("qserve_kv4")
        runner = PyTorchRunner(
            model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
            revision="cdbee75f17c01a7cc42f958dc650907174af0554",
            cache_factory=technique.pytorch_cache_factory(),
            dry_run=True,
        )
        outputs = runner.generate(["Test prompt for QServe KV4."], max_tokens=16)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(outputs[0].outputs[0].token_ids), 16)


class TestQServeKV4GateC(unittest.TestCase):
    def test_compute_new_capacity_doubling_and_linear(self):
        from src.techniques.qserve_cache import _compute_new_capacity

        cases = [
            ((0, 512), 1024),
            ((1024, 1025), 2048),
            ((8192, 9000), 16384),
            ((16384, 16385), 18432),
            ((32768, 32769), 34816),
            ((0, 32768), 32768),
            ((32768, 35000), 36864),
        ]
        for (current, required), expected in cases:
            with self.subTest(current=current, required=required):
                self.assertEqual(_compute_new_capacity(current, required), expected)

    def test_qserve_layer_32k_linear_capacity_growth(self):
        layer = QServeKVLayer(group_size=32, alpha=0.5)
        k1 = torch.zeros((1, 1, 16384, 32), dtype=torch.bfloat16)
        v1 = torch.zeros((1, 1, 16384, 32), dtype=torch.bfloat16)
        layer.update(k1, v1)

        k2 = torch.zeros((1, 1, 16384, 32), dtype=torch.bfloat16)
        v2 = torch.zeros((1, 1, 16384, 32), dtype=torch.bfloat16)
        layer.update(k2, v2)
        self.assertEqual(layer.capacity, 32768)
        self.assertEqual(layer.total_tokens_seen, 32768)

        k3 = torch.zeros((1, 1, 1, 32), dtype=torch.bfloat16)
        v3 = torch.zeros((1, 1, 1, 32), dtype=torch.bfloat16)
        layer.update(k3, v3)
        self.assertEqual(layer.capacity, 34816)
        self.assertNotEqual(layer.capacity, 65536)
        self.assertEqual(layer.total_tokens_seen, 32769)

    def test_unpack_int4_exact_roundtrip_contiguous_and_non_contiguous(self):
        base = torch.arange(-8, 8, dtype=torch.int8).repeat(2, 4, 3, 2)
        # Contiguous round-trip
        packed_contig = pack_int4(base)
        unpacked_contig = unpack_int4(packed_contig, original_dim=base.size(-1))
        self.assertEqual(unpacked_contig.dtype, torch.int8)
        self.assertTrue(torch.equal(unpacked_contig, base))

        # Non-contiguous sliced view round-trip
        non_contig = base[:, ::2, :, :]
        self.assertFalse(non_contig.is_contiguous())
        packed_nc = pack_int4(non_contig)
        packed_nc_view = packed_nc[:, :, ::2, :]
        self.assertFalse(packed_nc_view.is_contiguous())
        expected_nc = non_contig[:, :, ::2, :]
        unpacked_nc = unpack_int4(packed_nc_view, original_dim=expected_nc.size(-1))
        self.assertEqual(unpacked_nc.dtype, torch.int8)
        self.assertTrue(torch.equal(unpacked_nc, expected_nc))

    def test_dequantize_kv_int4_dtypes_non_contiguous_and_immutability(self):
        torch.manual_seed(123)
        raw = torch.randn((2, 4, 6, 64), dtype=torch.float32)
        packed_base, scales_base = quantize_kv_int4(raw, group_size=32)

        for original_dtype in (torch.float32, torch.bfloat16, torch.float16):
            for non_contig in (False, True):
                for mismatch_scales in (False, True):
                    with self.subTest(
                        original_dtype=original_dtype,
                        non_contig=non_contig,
                        mismatch_scales=mismatch_scales,
                    ):
                        packed = packed_base[:, ::2, :, :] if non_contig else packed_base.clone()
                        scales = scales_base[:, ::2, :, :, :] if non_contig else scales_base.clone()
                        if mismatch_scales:
                            alt_dtype = torch.float16 if original_dtype != torch.float16 else torch.float32
                            scales = scales.to(alt_dtype)
                        else:
                            scales = scales.to(original_dtype)

                        packed_snapshot = packed.clone()
                        scales_snapshot = scales.clone()

                        out = dequantize_kv_int4(
                            packed, scales, original_dtype=original_dtype, group_size=32
                        )
                        self.assertEqual(out.dtype, original_dtype)

                        original_dim = packed.size(-1) * 2
                        unpacked = unpack_int4(packed, original_dim=original_dim).to(original_dtype)
                        num_groups = original_dim // 32
                        ref = (
                            unpacked.reshape(*packed.shape[:-1], num_groups, 32)
                            * scales.to(original_dtype)
                        ).reshape(*packed.shape[:-1], original_dim)

                        self.assertTrue(torch.equal(out, ref))
                        self.assertTrue(torch.equal(packed, packed_snapshot))
                        self.assertTrue(torch.equal(scales, scales_snapshot))

    def test_should_use_compile_opt_out_by_default(self):
        from src.techniques.qserve_kv4 import _should_use_compile

        class FakeCudaTensor:
            is_cuda = True

        fake_cuda = FakeCudaTensor()

        env_without = {k: v for k, v in os.environ.items() if k != "QSERVE_ENABLE_COMPILE"}
        with patch.dict(os.environ, env_without, clear=True):
            self.assertFalse(_should_use_compile(fake_cuda))

        with patch.dict(os.environ, {"QSERVE_ENABLE_COMPILE": "0"}):
            self.assertFalse(_should_use_compile(fake_cuda))

    def test_dequantize_compile_fallback_and_no_retry(self):
        from src.techniques.qserve_kv4 import (
            _dequantize_kv_int4_impl,
            _reset_dequant_compile_state,
        )

        _reset_dequant_compile_state()
        self.addCleanup(_reset_dequant_compile_state)

        torch.manual_seed(99)
        x = torch.randn((1, 2, 4, 64), dtype=torch.float32)
        packed, scales = quantize_kv_int4(x, group_size=32)
        expected = _dequantize_kv_int4_impl(packed, scales, torch.float32, 32)

        with (
            patch("src.techniques.qserve_kv4._should_use_compile", return_value=True),
            patch("torch.compile", side_effect=RuntimeError("compile failed")) as mock_compile,
        ):
            with self.assertWarns(RuntimeWarning):
                out1 = dequantize_kv_int4(packed, scales, original_dtype=torch.float32, group_size=32)
            self.assertTrue(torch.equal(out1, expected))
            self.assertEqual(mock_compile.call_count, 1)

            # Second call should not retry failed compilation
            out2 = dequantize_kv_int4(packed, scales, original_dtype=torch.float32, group_size=32)
            self.assertTrue(torch.equal(out2, expected))
            self.assertEqual(mock_compile.call_count, 1)

            # Resetting compile state allows retry
            _reset_dequant_compile_state()
            with self.assertWarns(RuntimeWarning):
                out3 = dequantize_kv_int4(packed, scales, original_dtype=torch.float32, group_size=32)
            self.assertTrue(torch.equal(out3, expected))
            self.assertEqual(mock_compile.call_count, 2)

    def test_qserve_layer_keys_smooth_k_batch_and_immutability(self):
        torch.manual_seed(17)
        for alpha in (0.0, 0.5):
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                with self.subTest(alpha=alpha, dtype=dtype):
                    layer = QServeKVLayer(group_size=32, alpha=alpha)
                    k = torch.randn((2, 2, 8, 64), dtype=dtype)
                    v = torch.randn((2, 2, 8, 64), dtype=dtype)
                    layer.update(k, v)

                    smooth_before = layer.smooth_k.clone() if layer.smooth_k is not None else None
                    k_packed_before = layer.k_packed_buf.clone()
                    k_scales_before = layer.k_scales_buf.clone()

                    keys_out = layer.keys
                    self.assertEqual(keys_out.dtype, dtype)
                    self.assertEqual(keys_out.shape, (2, 2, 8, 64))

                    p_view = layer.k_packed_buf[:, :, : layer.total_tokens_seen, :]
                    s_view = layer.k_scales_buf[:, :, : layer.total_tokens_seen, :, :]
                    ref = dequantize_kv_int4(p_view, s_view, original_dtype=dtype, group_size=32)
                    if layer.smooth_k is not None:
                        ref = (ref * layer.smooth_k).to(dtype)

                    self.assertTrue(torch.allclose(keys_out.float(), ref.float(), atol=1e-4, rtol=1e-4))
                    if smooth_before is not None:
                        self.assertTrue(torch.equal(layer.smooth_k, smooth_before))
                    self.assertTrue(torch.equal(layer.k_packed_buf, k_packed_before))
                    self.assertTrue(torch.equal(layer.k_scales_buf, k_scales_before))

    def test_compute_new_capacity_rejects_invalid_bounds(self):
        from src.techniques.qserve_cache import _compute_new_capacity

        with self.assertRaises(ValueError):
            _compute_new_capacity(0, 0)
        with self.assertRaises(ValueError):
            _compute_new_capacity(0, -10)
        with self.assertRaises(ValueError):
            _compute_new_capacity(-1, 512)

    def test_unpack_int4_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            unpack_int4(torch.zeros((2, 4), dtype=torch.int8), original_dim=8)
        with self.assertRaises(ValueError):
            unpack_int4(torch.tensor(0, dtype=torch.uint8), original_dim=2)
        with self.assertRaises(ValueError):
            unpack_int4(torch.zeros((2, 4), dtype=torch.uint8), original_dim=6)

    def test_dequantize_kv_int4_rejects_invalid_inputs(self):
        valid_packed = torch.zeros((1, 2, 4, 32), dtype=torch.uint8)
        valid_scales = torch.ones((1, 2, 4, 2, 1), dtype=torch.float32)

        # Non-uint8 packed tensor
        with self.assertRaises(ValueError):
            dequantize_kv_int4(
                valid_packed.to(torch.int8), valid_scales, original_dtype=torch.float32, group_size=32
            )
        # 0-D packed tensor
        with self.assertRaises(ValueError):
            dequantize_kv_int4(
                torch.tensor(0, dtype=torch.uint8), valid_scales, original_dtype=torch.float32, group_size=32
            )
        # Non-floating original_dtype
        with self.assertRaises(ValueError):
            dequantize_kv_int4(
                valid_packed, valid_scales, original_dtype=torch.int32, group_size=32
            )
        # group_size <= 0
        with self.assertRaises(ValueError):
            dequantize_kv_int4(
                valid_packed, valid_scales, original_dtype=torch.float32, group_size=0
            )
        # Non-divisible original_dim % group_size != 0
        with self.assertRaises(ValueError):
            dequantize_kv_int4(
                valid_packed, valid_scales, original_dtype=torch.float32, group_size=12
            )
        # Invalid scales shape: missing trailing singleton 1
        with self.assertRaises(ValueError):
            dequantize_kv_int4(
                valid_packed, torch.ones((1, 2, 4, 2), dtype=torch.float32), original_dtype=torch.float32, group_size=32
            )
        # Invalid scales shape: leading dim exceeding packed_tensor
        with self.assertRaises(ValueError):
            dequantize_kv_int4(
                valid_packed, torch.ones((2, 2, 4, 2, 1), dtype=torch.float32), original_dtype=torch.float32, group_size=32
            )

    def test_fix_qserve_suite_config(self):
        suites_path = Path(__file__).resolve().parent.parent / "src" / "config" / "suites.json"
        with suites_path.open("r", encoding="utf-8") as f:
            suites = json.load(f)

        self.assertIn("fix_qserve", suites)
        cfg = suites["fix_qserve"]
        self.assertEqual(cfg["techniques"], ["qserve_kv4"])
        self.assertIs(cfg["force_longmem_ref"], False)
        self.assertIs(cfg["skip_hardware_sweeps"], False)
        self.assertIs(cfg["run_verify"], True)
        self.assertIs(cfg["run_longmem"], True)
        self.assertIs(cfg["run_niah"], True)
        self.assertIs(cfg["skip_existing"], False)
        self.assertEqual(cfg["contexts"], [512, 1024, 2048, 4096, 8192, 16384, 32768])
        self.assertEqual(cfg["eval_contexts"], [32768])
        self.assertEqual(cfg["niah_contexts"], [2048, 4096])


if __name__ == "__main__":
    unittest.main()
