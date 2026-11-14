import unittest
from types import SimpleNamespace

import torch

from src.core.axis import Axis
from src.core.diagnostic import inspect_sample_diagnostic
from src.core.pytorch_runner import PyTorchRunner
from src.core.registry import get_technique, list_techniques
from src.techniques.arkvale import (
    ArkValeTechnique,
    compute_gqa_arkvale_attention,
    compute_page_aabb_digests,
    score_page_digests_gqa,
    select_and_recall_pages,
)
from src.techniques.arkvale_cache import (
    ArkValeCache,
    ArkValeLayer,
    get_active_arkvale_cache,
)


class TestArkValeTechnique(unittest.TestCase):
    def test_registration_and_axes(self):
        self.assertIn("arkvale", list_techniques())
        technique = get_technique("arkvale")
        self.assertIsInstance(technique, ArkValeTechnique)
        self.assertEqual(technique.axes, Axis.STRUCTURAL | Axis.SPATIAL)
        self.assertEqual(technique.compression_ratio, 4.5)

    def test_defaults_and_custom_parameters(self):
        technique = get_technique("arkvale")
        self.assertEqual(technique.page_size, 16)
        self.assertEqual(technique.active_pages_budget, 128)
        self.assertEqual(technique.exempt_early_layers, 2)
        self.assertEqual(technique.block_size, 16)

        mock_out = SimpleNamespace(prompt_tokens=4096, generated_tokens=64)
        metrics = technique.extra_metrics(None, mock_out)
        self.assertEqual(metrics["page_size"], 16)
        self.assertTrue(metrics["aabb_digests_active"])
        self.assertTrue(metrics["page_recall_supported"])

        kwargs = technique.engine_kwargs(context_length=8192, max_tokens=128)
        self.assertEqual(kwargs["block_size"], 16)
        self.assertEqual(kwargs["swap_space"], 8)

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            ArkValeTechnique(page_size=0)
        with self.assertRaises(ValueError):
            ArkValeTechnique(active_pages_budget=-1)
        with self.assertRaises(ValueError):
            ArkValeTechnique(exempt_early_layers=-1)
        with self.assertRaises(ValueError):
            ArkValeTechnique(block_size=0)

    def test_page_aabb_digests_properties(self):
        torch.manual_seed(42)
        batch, heads, seq_len, dim = 2, 4, 48, 64
        page_size = 16
        k = torch.randn(batch, heads, seq_len, dim, dtype=torch.float32)

        b_min, b_max, num_pages = compute_page_aabb_digests(k, page_size=page_size)
        self.assertEqual(num_pages, 3)
        self.assertEqual(b_min.shape, (batch, heads, 3, dim))
        self.assertEqual(b_max.shape, (batch, heads, 3, dim))
        self.assertTrue((b_max >= b_min).all())

    def test_score_page_digests_gqa_and_select(self):
        batch, seq_len_q, dim = 1, 1, 64
        num_q_heads, num_kv_heads = 8, 2
        seq_len_kv = 64
        page_size = 16

        torch.manual_seed(100)
        q = torch.randn(batch, num_q_heads, seq_len_q, dim, dtype=torch.float32)
        k = torch.randn(batch, num_kv_heads, seq_len_kv, dim, dtype=torch.float32)

        b_min, b_max, num_pages = compute_page_aabb_digests(k, page_size=page_size)
        kv_scores = score_page_digests_gqa(q, b_min, b_max, gqa_ratio=4)

        self.assertEqual(kv_scores.shape, (batch, num_kv_heads, num_pages))

        active_pages, telemetry = select_and_recall_pages(kv_scores, active_pages_budget=2)
        self.assertIn(0, active_pages.tolist())
        self.assertLessEqual(active_pages.numel(), 2)
        self.assertIn("page_sparsity_ratio", telemetry)

    def test_compute_gqa_arkvale_attention(self):
        batch, seq_len_q, head_dim = 1, 1, 64
        num_q_heads, num_kv_heads = 8, 2
        seq_len_kv = 64

        torch.manual_seed(200)
        q = torch.randn(batch, num_q_heads, seq_len_q, head_dim, dtype=torch.float32)
        k = torch.randn(batch, num_kv_heads, seq_len_kv, head_dim, dtype=torch.float32)
        v = torch.randn(batch, num_kv_heads, seq_len_kv, head_dim, dtype=torch.float32)

        attn_out, attn_weights, telemetry = compute_gqa_arkvale_attention(
            q, k, v, page_size=16, active_pages_budget=3
        )

        self.assertEqual(attn_out.shape, (batch, num_q_heads, seq_len_q, head_dim))
        self.assertTrue(torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1)), atol=1e-4))
        self.assertIn("active_tokens", telemetry)
        self.assertTrue(telemetry["zero_copy_eviction_active"])

    def test_arkvale_layer_early_exempt_vs_deep_eviction(self):
        torch.manual_seed(88)
        batch, heads, dim = 1, 4, 64
        # 80 tokens -> 5 pages of size 16; budget is 3 pages
        k_pre = torch.randn(batch, 80, heads, dim, dtype=torch.bfloat16).transpose(1, 2)
        v_pre = torch.randn(batch, 80, heads, dim, dtype=torch.bfloat16).transpose(1, 2)
        k_dec = torch.randn(batch, heads, 1, dim, dtype=torch.bfloat16)
        v_dec = torch.randn(batch, heads, 1, dim, dtype=torch.bfloat16)
        q_dec = torch.randn(batch, heads * 2, 1, dim, dtype=torch.bfloat16)

        # Layer 0 (exempt): stays GPU-resident with zero CPU backup or PCIe traffic, retains all 81 tokens
        l0 = ArkValeLayer(layer_idx=0, page_size=16, active_pages_budget=3, exempt_early_layers=2)
        l0.update(k_pre, v_pre)
        out_k0, _ = l0.update(k_dec, v_dec, query_states=q_dec)
        self.assertEqual(out_k0.shape[-2], 81)
        self.assertIsNone(l0.cpu_k_buf)
        self.assertEqual(l0.total_bytes_transferred_pcie, 0)

        # Layer 3 (deep): caps active working set <= 3 pages (at most ~33-48 tokens)
        l3 = ArkValeLayer(layer_idx=3, page_size=16, active_pages_budget=3, exempt_early_layers=2)
        l3.update(k_pre, v_pre)
        out_k3, _ = l3.update(k_dec, v_dec, query_states=q_dec)
        self.assertLess(out_k3.shape[-2], 81)
        self.assertIn(0, l3.original_positions)
        self.assertGreater(l3.total_bytes_transferred_pcie, 0)

    def test_arkvale_cache_factory_and_pytorch_runner(self):
        technique = get_technique("arkvale", page_size=16, active_pages_budget=3, exempt_early_layers=1)
        cache_factory = technique.pytorch_cache_factory()
        self.assertIsNotNone(cache_factory)

        cache = cache_factory()
        self.assertIsInstance(cache, ArkValeCache)
        self.assertEqual(get_active_arkvale_cache(), cache)

        # Populate layers 0 and 1 with 80 tokens then 1 decode token
        k_pre = torch.randn(1, 4, 80, 64, dtype=torch.float32)
        v_pre = torch.randn(1, 4, 80, 64, dtype=torch.float32)
        k_dec = torch.randn(1, 4, 1, 64, dtype=torch.float32)
        v_dec = torch.randn(1, 4, 1, 64, dtype=torch.float32)
        q_dec = torch.randn(1, 8, 1, 64, dtype=torch.float32)

        for l_idx in range(2):
            cache.update(k_pre, v_pre, layer_idx=l_idx)
            cache.update(k_dec, v_dec, layer_idx=l_idx, query_states=q_dec)

        diag = inspect_sample_diagnostic(
            cache=cache,
            prompt_text="Test ArkVale diagnostic",
            prompt_token_ids=list(range(80)),
            sample_meta={"id": "av_diag_0", "task_type": "unit"},
            is_correct=True,
            baseline_correct=True,
            bits_per_scalar=16.0,
        )
        self.assertGreater(diag.overall_retention_ratio, 0.0)
        self.assertLess(diag.overall_retention_ratio, 1.0)

        runner = PyTorchRunner(
            model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
            revision="cdbee75f17c01a7cc42f958dc650907174af0554",
            cache_factory=cache_factory,
            dry_run=True,
        )
        outputs = runner.generate(["Test ArkVale prompt."], max_tokens=16)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(outputs[0].outputs[0].token_ids), 16)

    def test_query_based_digest_scoring_and_sdpa_hook(self):
        import transformers.modeling_utils as mu
        from src.techniques.arkvale import _uninstall_pytorch_arkvale_hooks

        technique = get_technique("arkvale", page_size=16, active_pages_budget=3, exempt_early_layers=1)
        technique.setup()
        try:
            cache = technique.pytorch_cache_factory()()
            batch, num_kv_heads, num_q_heads, dim = 1, 2, 8, 64
            # 5 pages (80 tokens): page 2 strongly aligned with +e0, page 1 strongly aligned with +e1
            k_pre = torch.zeros(batch, num_kv_heads, 80, dim, dtype=torch.float32)
            k_pre[:, :, 16:32, 1] = 20.0   # page 1 -> dim 1
            k_pre[:, :, 32:48, 0] = 20.0   # page 2 -> dim 0
            v_pre = torch.randn(batch, num_kv_heads, 80, dim, dtype=torch.float32)

            # Decode key points along dim 1 (page 1), while actual query points along dim 0 (page 2)
            k_dec = torch.zeros(batch, num_kv_heads, 1, dim, dtype=torch.float32)
            k_dec[:, :, :, 1] = 20.0
            v_dec = torch.randn(batch, num_kv_heads, 1, dim, dtype=torch.float32)
            q_dec = torch.zeros(batch, num_q_heads, 1, dim, dtype=torch.float32)
            q_dec[:, :, :, 0] = 20.0

            cache.update(k_pre, v_pre, layer_idx=0)
            cache.update(k_pre, v_pre, layer_idx=1)

            # Layer 1 update without query_states defers selection to SDPA hook
            out_k, out_v = cache.update(k_dec, v_dec, layer_idx=1)
            self.assertTrue(cache.layers[1].needs_page_selection)
            self.assertEqual(out_k.shape[-2], 1)

            sdpa_fn = mu.ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
            self.assertIsNotNone(sdpa_fn)
            attn_mask = torch.zeros(batch, 1, 1, 81, dtype=torch.float32)
            attn_out, _ = sdpa_fn(
                SimpleNamespace(layer_idx=1, num_key_value_groups=4, config=SimpleNamespace(is_causal=True)),
                q_dec,
                out_k,
                out_v,
                attention_mask=attn_mask,
                scaling=dim**-0.5,
                is_causal=True,
            )
            self.assertEqual(attn_out.shape, (batch, 1, num_q_heads, dim))
            self.assertFalse(cache.layers[1].needs_page_selection)
            # Page 2 (tokens 32..47) must be selected because query_states aligned with dim 0, not k_dec (dim 1)
            self.assertIn(32, cache.layers[1].original_positions)
            self.assertNotIn(16, cache.layers[1].original_positions)
        finally:
            _uninstall_pytorch_arkvale_hooks()

    def test_telemetry_persists_after_cache_cleanup_and_resets_in_setup(self):
        import gc
        from src.techniques.arkvale import _uninstall_pytorch_arkvale_hooks

        technique = get_technique("arkvale", page_size=16, active_pages_budget=3, exempt_early_layers=1)
        technique.setup()
        try:
            cache = technique.pytorch_cache_factory()()
            k_pre = torch.randn(1, 2, 96, 64, dtype=torch.float32)
            v_pre = torch.randn(1, 2, 96, 64, dtype=torch.float32)
            k_dec = torch.randn(1, 2, 1, 64, dtype=torch.float32)
            v_dec = torch.randn(1, 2, 1, 64, dtype=torch.float32)

            cache.update(k_pre, v_pre, layer_idx=0)
            cache.update(k_pre, v_pre, layer_idx=1)

            # Step 1: query selects page 1; Step 2: query selects page 2 -> triggers page recall from CPU
            q1 = torch.zeros(1, 8, 1, 64, dtype=torch.float32)
            q1[:, :, :, 0] = 10.0
            # Bias page 1 vs page 2 in cpu_k_buf/digests by running two distinct queries across steps
            cache.layers[1].gpu_b_min[:, :, 1, 0] = 50.0
            cache.layers[1].gpu_b_max[:, :, 1, 0] = 50.0
            cache.layers[1].gpu_b_min[:, :, 2, 1] = 50.0
            cache.layers[1].gpu_b_max[:, :, 2, 1] = 50.0

            cache.update(k_dec, v_dec, layer_idx=1, query_states=q1)

            q2 = torch.zeros(1, 8, 1, 64, dtype=torch.float32)
            q2[:, :, :, 1] = 10.0
            cache.update(k_dec, v_dec, layer_idx=1, query_states=q2)

            # Simulate PyTorchRunner.generate() cleanup: del cache
            del cache
            gc.collect()
            self.assertIsNone(get_active_arkvale_cache())

            metrics = technique.extra_metrics(None, None)
            self.assertGreater(metrics["pcie_transfer_bytes"], 0)
            self.assertGreater(metrics["recalled_pages"], 0)

            # Calling setup() must reset run-level counters
            technique.setup()
            reset_metrics = technique.extra_metrics(None, None)
            self.assertEqual(reset_metrics["pcie_transfer_bytes"], 0)
            self.assertEqual(reset_metrics["recalled_pages"], 0)
        finally:
            _uninstall_pytorch_arkvale_hooks()


if __name__ == "__main__":
    unittest.main()
