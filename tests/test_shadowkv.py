import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src.core.axis import Axis
from src.core.registry import get_technique, list_techniques
from src.techniques.shadowkv import (
    _ACTIVE_SHADOWKV_CONFIG,
    compute_shadowkv_boundaries,
    compute_shadowkv_paged_block_boundaries,
    estimate_shadowkv_pcie_savings,
)
from src.techniques.shadowkv_cache import (
    ShadowKVCache,
    ShadowKVLayer,
    apply_rope,
    compute_chunk_landmarks,
    compute_rope_cos_sin,
    detect_outlier_chunks,
    get_active_shadowkv_cache,
    invert_rope,
    perform_svd_key_decomposition,
    rotate_half,
    set_active_shadowkv_cache,
)


class TestShadowKVTechnique(unittest.TestCase):
    def test_registration_and_axes(self):
        self.assertIn("shadowkv", list_techniques())
        technique = get_technique("shadowkv")
        self.assertEqual(technique.axes, Axis.SPATIAL | Axis.STRUCTURAL | Axis.TEMPORAL)

    def test_defaults_and_custom_overrides(self):
        technique = get_technique("shadowkv")
        self.assertEqual(technique.svd_rank, 160)
        self.assertEqual(technique.chunk_size, 8)
        self.assertEqual(technique.topk_chunks, 256)
        self.assertEqual(technique.outlier_ratio, 0.003)
        self.assertEqual(technique.temporal_reuse_target, 0.60)
        self.assertEqual(technique.local_window_tokens, 32)
        self.assertEqual(technique.svd_threshold_tokens, 1024)
        self.assertEqual(technique.swap_space_gb, 4)
        self.assertEqual(technique.kv_cache_dtype, "auto")

        custom = get_technique(
            "shadowkv",
            svd_rank=64,
            chunk_size=16,
            topk_chunks=128,
            outlier_ratio=0.005,
            temporal_reuse_target=0.75,
            local_window_tokens=64,
            svd_threshold_tokens=512,
            swap_space_gb=8,
            kv_cache_dtype="fp8",
        )
        self.assertEqual(custom.svd_rank, 64)
        self.assertEqual(custom.chunk_size, 16)
        self.assertEqual(custom.topk_chunks, 128)
        self.assertEqual(custom.outlier_ratio, 0.005)
        self.assertEqual(custom.temporal_reuse_target, 0.75)
        self.assertEqual(custom.local_window_tokens, 64)
        self.assertEqual(custom.svd_threshold_tokens, 512)
        self.assertEqual(custom.swap_space_gb, 8)
        self.assertEqual(custom.kv_cache_dtype, "fp8")

        kwargs = custom.engine_kwargs(context_length=2048, max_tokens=64)
        self.assertEqual(kwargs["swap_space"], 8)
        self.assertEqual(kwargs["kv_cache_dtype"], "fp8")
        self.assertEqual(kwargs["block_size"], 32)

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            get_technique("shadowkv", svd_rank=-1)
        with self.assertRaises(ValueError):
            get_technique("shadowkv", chunk_size=0)
        with self.assertRaises(ValueError):
            get_technique("shadowkv", topk_chunks=-5)
        with self.assertRaises(ValueError):
            get_technique("shadowkv", outlier_ratio=-0.1)
        with self.assertRaises(ValueError):
            get_technique("shadowkv", outlier_ratio=1.5)
        with self.assertRaises(ValueError):
            get_technique("shadowkv", temporal_reuse_target=1.2)
        with self.assertRaises(ValueError):
            get_technique("shadowkv", swap_space_gb=-1)

    def test_boundary_calculations(self):
        # 1. Short sequence fits within dense threshold (512 <= 1024)
        b_short = compute_shadowkv_boundaries(seq_len=512, svd_threshold_tokens=1024)
        self.assertFalse(b_short.is_offloaded)
        self.assertEqual(b_short.active_gpu_key_tokens, 512)
        self.assertEqual(b_short.active_gpu_val_tokens, 512)
        self.assertEqual(b_short.offloaded_cpu_val_tokens, 0)
        self.assertEqual(b_short.theoretical_memory_savings_ratio, 1.0)

        # 2. Long sequence (128K context = 131,072) matching Paper Appendix A.2
        # Paper values: M=1024 (8 KV heads * 128 dim), C=8, K=256, O=48, r=160 -> >6x memory reduction
        b_128k = compute_shadowkv_boundaries(
            seq_len=131072,
            svd_rank=160,
            chunk_size=8,
            topk_chunks=256,
            outlier_chunks=48,
            num_kv_heads=8,
            head_dim=128,
        )
        self.assertTrue(b_128k.is_offloaded)
        self.assertGreaterEqual(b_128k.theoretical_memory_savings_ratio, 6.0)
        self.assertGreater(b_128k.effective_bandwidth_gb_s, 1000.0)
        self.assertGreater(b_128k.offloaded_cpu_val_tokens, 120000)

    def test_paged_block_boundaries(self):
        paged_info = compute_shadowkv_paged_block_boundaries(
            num_computed_tokens=2048,
            block_size=16,
            chunk_size=8,
            outlier_chunks=8,
            local_chunks=4,
            topk_chunks=16,
        )
        self.assertEqual(paged_info.num_computed_tokens, 2048)
        self.assertEqual(paged_info.total_blocks, 128)
        self.assertIn(0, paged_info.sink_blocks)
        self.assertGreater(len(paged_info.local_blocks), 0)
        self.assertGreater(len(paged_info.outlier_blocks), 0)
        self.assertGreater(len(paged_info.candidate_prefetch_blocks), 0)

    def test_pcie_savings_estimation(self):
        savings = estimate_shadowkv_pcie_savings(
            seq_len=32768,
            chunk_size=8,
            topk_chunks=256,
            temporal_hit_rate=0.60,
            num_kv_heads=4,
            head_dim=128,
        )
        self.assertGreater(savings["full_transfer_bytes"], savings["shadowkv_transfer_bytes"])
        self.assertGreater(savings["transfer_latency_saved_ms"], 0.0)
        self.assertGreater(savings["pcie_bandwidth_reduction_factor"], 1.0)


class TestShadowKVMathAndCache(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def tearDown(self):
        set_active_shadowkv_cache(None)

    def test_exact_inverse_rope_fidelity(self):
        seq_len = 64
        head_dim = 128
        b, h = 2, 4
        x = torch.randn(b, h, seq_len, head_dim, dtype=torch.float32)

        cos, sin = compute_rope_cos_sin(
            seq_len=seq_len,
            head_dim=head_dim,
            device=x.device,
            dtype=x.dtype,
            theta=10000.0,
            start_pos=0,
        )

        rotated = apply_rope(x, cos, sin)
        inverted = invert_rope(rotated, cos, sin)

        max_err = (x - inverted).abs().max().item()
        self.assertLess(max_err, 1e-6)

    def test_svd_key_decomposition_accuracy(self):
        b, h_kv, seq_len, d = 2, 4, 256, 128
        keys_pre = torch.randn(b, h_kv, seq_len, d, dtype=torch.float32)
        r = 64

        a, b_mat = perform_svd_key_decomposition(keys_pre, svd_rank=r)
        self.assertEqual(a.shape, (b, h_kv, seq_len, r))
        self.assertEqual(b_mat.shape, (b, h_kv, r, d))

        recon = torch.matmul(a, b_mat)
        # Verify high cosine similarity across reconstructed channels
        cos_sim = F.cosine_similarity(keys_pre.flatten(start_dim=2), recon.flatten(start_dim=2), dim=-1)
        self.assertTrue((cos_sim > 0.85).all())

    def test_chunk_landmarks_and_outlier_detection(self):
        b, h_kv, seq_len, d = 1, 2, 64, 64
        keys = torch.randn(b, h_kv, seq_len, d)

        landmarks, num_chunks, padded_s = compute_chunk_landmarks(keys, chunk_size=8)
        self.assertEqual(num_chunks, 8)
        self.assertEqual(landmarks.shape, (b, h_kv, 8, d))

        # Check that landmark is the exact mean of chunk 0
        expected_c0 = keys[:, :, :8, :].mean(dim=2)
        self.assertTrue(torch.allclose(landmarks[:, :, 0, :], expected_c0, atol=1e-5))

        outliers = detect_outlier_chunks(keys, landmarks, chunk_size=8, outlier_ratio=0.25)
        # Chunk 0 (attention sink) must always be preserved
        self.assertIn(0, outliers)
        self.assertGreaterEqual(len(outliers), 2)

    def test_shadowkv_cache_prefill_and_decode_forward(self):
        cache = ShadowKVCache(
            svd_rank=32,
            chunk_size=8,
            topk_chunks=4,
            outlier_ratio=0.1,
            temporal_reuse_target=0.60,
            svd_threshold_tokens=64,  # Lower threshold for fast unit testing
        )
        self.assertEqual(get_active_shadowkv_cache(), cache)

        b, h_kv, d = 1, 2, 64
        prefill_len = 128

        # 1. Prefill
        k_prefill = torch.randn(b, h_kv, prefill_len, d)
        v_prefill = torch.randn(b, h_kv, prefill_len, d)

        k_out, v_out = cache.update(k_prefill, v_prefill, layer_idx=0)
        layer = cache.layers[0]
        layer.prune()
        self.assertTrue(layer.is_offloaded)
        self.assertEqual(layer.total_tokens_seen, prefill_len)
        self.assertIsNotNone(layer.cpu_values)
        self.assertIsNotNone(layer.gpu_anchor_keys)

        # 2. Decode step (query with 4 heads, GQA group size = 2)
        h_q = 4
        q_decode = torch.randn(b, h_q, 1, d)
        k_decode = torch.randn(b, h_kv, 1, d)
        v_decode = torch.randn(b, h_kv, 1, d)

        cache.update(k_decode, v_decode, layer_idx=0)
        self.assertEqual(layer.total_tokens_seen, prefill_len + 1)

        # 3. Resolve sparse KV with landmark attention
        working_k, working_v = layer.resolve_sparse_kv(q_decode)
        self.assertGreater(working_k.shape[2], 0)
        self.assertEqual(working_k.shape[2], working_v.shape[2])
        self.assertEqual(working_k.shape[-1], d)

        # Verify telemetry
        self.assertGreater(cache.total_bytes_transferred_pcie, 0)
        self.assertGreaterEqual(cache.temporal_hit_rate, 0.0)

    def test_chunked_prefill_accumulation(self):
        cache = ShadowKVCache(svd_threshold_tokens=512)
        b, h_kv, d = 1, 4, 64

        # Chunk 1 (256 tokens)
        k0 = torch.randn(b, h_kv, 256, d)
        v0 = torch.randn(b, h_kv, 256, d)
        out_k0, _ = cache.update(k0, v0, layer_idx=0)
        self.assertEqual(out_k0.shape[2], 256)

        # Chunk 2 (256 tokens) -> total 512 tokens
        k1 = torch.randn(b, h_kv, 256, d)
        v1 = torch.randn(b, h_kv, 256, d)
        out_k1, _ = cache.update(k1, v1, layer_idx=0)
        self.assertEqual(out_k1.shape[2], 512)

        # Chunk 3 (256 tokens) -> total 768 tokens (exceeds threshold)
        k2 = torch.randn(b, h_kv, 256, d)
        v2 = torch.randn(b, h_kv, 256, d)
        out_k2, _ = cache.update(k2, v2, layer_idx=0)
        self.assertEqual(out_k2.shape[2], 768)

        # Trigger prune after prefill
        layer = cache.layers[0]
        layer.prune()
        self.assertTrue(layer.is_offloaded)
        self.assertIsNotNone(layer.cpu_values)
        self.assertEqual(layer.cpu_values.shape[2], 768)

    def test_technique_setup_and_extra_metrics(self):
        technique = get_technique("shadowkv", svd_rank=64, svd_threshold_tokens=128)
        technique.setup()
        self.assertTrue(_ACTIVE_SHADOWKV_CONFIG["enabled"])
        self.assertEqual(_ACTIVE_SHADOWKV_CONFIG["svd_rank"], 64)

        mock_out = SimpleNamespace(prompt_tokens=2048, generated_tokens=32)
        metrics = technique.extra_metrics(None, mock_out)
        self.assertEqual(metrics["svd_rank"], 64)
        self.assertIn("pcie_transfer_bytes", metrics)
        self.assertIn("theoretical_memory_savings_ratio", metrics)
        self.assertIn("effective_bandwidth_gb_s", metrics)
