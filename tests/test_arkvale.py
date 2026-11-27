import inspect
import json
from pathlib import Path
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

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
    set_active_arkvale_cache,
)


class TestArkValeTechnique(unittest.TestCase):
    def test_registration_and_axes(self):
        self.assertIn("arkvale", list_techniques())
        technique = get_technique("arkvale")
        self.assertIsInstance(technique, ArkValeTechnique)
        self.assertEqual(technique.axes, Axis.STRUCTURAL | Axis.SPATIAL)
        self.assertFalse(hasattr(technique, "compression_ratio"))

    def test_defaults_and_custom_parameters(self):
        technique = get_technique("arkvale")
        self.assertEqual(technique.page_size, 16)
        self.assertEqual(technique.active_pages_budget, 16)
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

    def tearDown(self):
        set_active_arkvale_cache(None)
        try:
            from src.techniques.arkvale import _uninstall_pytorch_arkvale_hooks

            _uninstall_pytorch_arkvale_hooks()
        except ImportError:
            pass

    def test_arkvale_layer_early_exempt_vs_deep_eviction(self):
        torch.manual_seed(88)
        batch, heads, dim = 1, 4, 64
        # 80 tokens -> 5 pages of size 16; budget is 3 pages
        k_pre = torch.randn(batch, 80, heads, dim, dtype=torch.bfloat16).transpose(1, 2)
        v_pre = torch.randn(batch, 80, heads, dim, dtype=torch.bfloat16).transpose(1, 2)
        k_dec = torch.randn(batch, heads, 1, dim, dtype=torch.bfloat16)
        v_dec = torch.randn(batch, heads, 1, dim, dtype=torch.bfloat16)
        q_dec = torch.randn(batch, heads, 1, dim, dtype=torch.bfloat16)

        # Layer 0 (exempt): retains all 81 tokens
        l0 = ArkValeLayer(layer_idx=0, page_size=16, active_pages_budget=3, exempt_early_layers=2)
        l0.update(k_pre, v_pre)
        out_k0, _ = l0.update(k_dec, v_dec)
        self.assertEqual(out_k0.shape[-2], 81)

        # Layer 3 (deep): deferred SDPA resolution via resolve_active_kv(query) caps active set <= 3 pages
        l3 = ArkValeLayer(layer_idx=3, page_size=16, active_pages_budget=3, exempt_early_layers=2)
        l3.update(k_pre, v_pre)
        l3.update(k_dec, v_dec)
        out_k3, _ = l3.resolve_active_kv(q_dec)
        self.assertLess(out_k3.shape[-2], 81)
        self.assertIn(0, l3.original_positions)

    def test_arkvale_cache_factory_and_pytorch_runner(self):
        technique = get_technique("arkvale", page_size=16, active_pages_budget=3, exempt_early_layers=1)
        technique.setup()
        cache_factory = technique.pytorch_cache_factory()
        self.assertIsNotNone(cache_factory)

        cache = cache_factory()
        self.assertIsInstance(cache, ArkValeCache)
        self.assertEqual(get_active_arkvale_cache(), cache)

        # Populate layers 0 (exempt) and 1 (deep) with 80 tokens then 1 decode token
        k_pre = torch.randn(1, 4, 80, 64, dtype=torch.float32)
        v_pre = torch.randn(1, 4, 80, 64, dtype=torch.float32)
        k_dec = torch.randn(1, 4, 1, 64, dtype=torch.float32)
        v_dec = torch.randn(1, 4, 1, 64, dtype=torch.float32)
        q_dec = torch.randn(1, 4, 1, 64, dtype=torch.float32)

        for l_idx in range(2):
            cache.update(k_pre, v_pre, layer_idx=l_idx)
            cache.update(k_dec, v_dec, layer_idx=l_idx)

        # Invoke SDPA hook for deep layer (layer_idx=1) to trigger deferred decode pruning
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        sdpa_fn = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
        sdpa_fn(SimpleNamespace(layer_idx=1), q_dec, k_dec, v_dec, attention_mask=None, is_causal=False)

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

    def test_arkvale_layer_reset_releases_buffers_and_persists_telemetry(self):
        technique = ArkValeTechnique(page_size=16, active_pages_budget=2, exempt_early_layers=0)
        technique.setup()
        cache = technique.pytorch_cache_factory()()

        k_pre = torch.randn(1, 2, 48, 32, dtype=torch.float32)
        v_pre = torch.randn(1, 2, 48, 32, dtype=torch.float32)
        cache.update(k_pre, v_pre, layer_idx=0)
        self.assertIsNotNone(cache.layers[0].cpu_k_buf)
        self.assertGreater(cache.total_bytes_transferred_pcie, 0)

        expected_pcie = cache.total_bytes_transferred_pcie
        technique.on_sample_complete(cache)
        cache.reset()

        layer = cache.layers[0]
        self.assertIsNone(layer.cpu_k_buf)
        self.assertIsNone(layer.cpu_v_buf)
        self.assertIsNone(layer.gpu_k_buf)
        self.assertIsNone(layer.gpu_v_buf)
        self.assertIsNone(layer.gpu_slot_k)
        self.assertIsNone(layer.gpu_slot_v)
        self.assertIsNone(layer.gpu_b_min)
        self.assertIsNone(layer.gpu_b_max)
        self.assertEqual(len(layer.slot_to_page), 0)
        self.assertEqual(len(layer.page_to_slot), 0)
        self.assertEqual(len(layer.resident_pages), 0)
        self.assertEqual(layer.total_tokens_seen, 0)
        self.assertEqual(layer.capacity, 0)
        self.assertEqual(layer.page_capacity, 0)

        del cache
        metrics = technique.extra_metrics(None, SimpleNamespace(prompt_tokens=48, generated_tokens=1))
        self.assertEqual(metrics["pcie_transfer_bytes"], expected_pcie)

    def test_query_driven_sdpa_recall_distinct_pages(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        technique = ArkValeTechnique(page_size=16, active_pages_budget=3, exempt_early_layers=2)
        technique.setup()
        cache = technique.pytorch_cache_factory()()

        b, h_q, h_kv, d = 1, 4, 2, 64
        k_pre = torch.zeros(b, h_kv, 80, d, dtype=torch.float32)
        v_pre = torch.zeros(b, h_kv, 80, d, dtype=torch.float32)

        # Page 1 (tokens 16..32): strong signal on dims 0:16, value = +1.0
        k_pre[:, :, 16:32, :16] = 5.0
        v_pre[:, :, 16:32, :] = 1.0
        # Page 2 (tokens 32..48): strong signal on dims 16:32, value = -1.0
        k_pre[:, :, 32:48, 16:32] = 5.0
        v_pre[:, :, 32:48, :] = -1.0

        k_dec = torch.zeros(b, h_kv, 1, d, dtype=torch.float32)
        v_dec = torch.zeros(b, h_kv, 1, d, dtype=torch.float32)

        cache.update(k_pre, v_pre, layer_idx=2)
        cache.update(k_dec, v_dec, layer_idx=2)

        q_a = torch.zeros(b, h_q, 1, d, dtype=torch.float32)
        q_a[:, :, :, :16] = 5.0

        q_b = torch.zeros(b, h_q, 1, d, dtype=torch.float32)
        q_b[:, :, :, 16:32] = 5.0

        sdpa_fn = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
        module = SimpleNamespace(layer_idx=2, num_key_value_groups=h_q // h_kv)
        scale = d**-0.5

        out_a, _ = sdpa_fn(
            module, q_a, k_dec, v_dec, attention_mask=None, scaling=scale, is_causal=False, enable_gqa=True
        )
        layer = cache.layers[2]
        self.assertIn(1, layer.resident_pages)
        self.assertNotIn(2, layer.resident_pages)

        # Reference attention over recalled active pages {0, 1, 5}
        k_act_a, v_act_a = layer.resolve_active_kv(q_a)
        ref_a = F.scaled_dot_product_attention(
            q_a,
            k_act_a.repeat_interleave(h_q // h_kv, dim=1),
            v_act_a.repeat_interleave(h_q // h_kv, dim=1),
            scale=scale,
            is_causal=False,
        ).transpose(1, 2).contiguous()
        self.assertTrue(torch.allclose(out_a, ref_a, atol=1e-5))

        # Switch to q_B with identical dummy k_dec/v_dec -> recalls page 2 instead of page 1
        out_b, _ = sdpa_fn(
            module, q_b, k_dec, v_dec, attention_mask=None, scaling=scale, is_causal=False, enable_gqa=True
        )
        self.assertIn(2, layer.resident_pages)
        self.assertNotIn(1, layer.resident_pages)

        k_act_b, v_act_b = layer.resolve_active_kv(q_b)
        ref_b = F.scaled_dot_product_attention(
            q_b,
            k_act_b.repeat_interleave(h_q // h_kv, dim=1),
            v_act_b.repeat_interleave(h_q // h_kv, dim=1),
            scale=scale,
            is_causal=False,
        ).transpose(1, 2).contiguous()
        self.assertTrue(torch.allclose(out_b, ref_b, atol=1e-5))
        self.assertFalse(torch.allclose(out_a, out_b))

    def test_vectorized_incremental_digests_unaligned_chunked_prefill(self):
        torch.manual_seed(123)
        b, seq_len, h_kv, d = 1, 64, 4, 32
        page_size = 16
        k_full = torch.randn(b, seq_len, h_kv, d, dtype=torch.float32).transpose(1, 2)
        v_full = torch.randn(b, seq_len, h_kv, d, dtype=torch.float32).transpose(1, 2)
        self.assertFalse(k_full.is_contiguous())

        layer = ArkValeLayer(layer_idx=2, page_size=page_size, active_pages_budget=2, exempt_early_layers=2)
        # Unaligned chunks: 13 + 27 + 24 = 64
        for s, e in [(0, 13), (13, 40), (40, 64)]:
            layer.update(k_full[:, :, s:e, :], v_full[:, :, s:e, :])

        ref_min, ref_max, num_pages = compute_page_aabb_digests(k_full, page_size=page_size)
        self.assertEqual(num_pages, 4)
        self.assertTrue(torch.allclose(layer.gpu_b_min[:, :, :num_pages, :], ref_min, atol=1e-5))
        self.assertTrue(torch.allclose(layer.gpu_b_max[:, :, :num_pages, :], ref_max, atol=1e-5))

        src = inspect.getsource(ArkValeLayer._update_incremental_digests)
        self.assertNotIn("for p in range", src)

    def test_resident_gpu_slots_and_delta_only_fetch_accounting(self):
        b, h_q, h_kv, d, page_size = 1, 4, 2, 64, 16
        layer = ArkValeLayer(layer_idx=2, page_size=page_size, active_pages_budget=3, exempt_early_layers=2)

        k_pre = torch.zeros(b, h_kv, 80, d, dtype=torch.float32)
        v_pre = torch.zeros(b, h_kv, 80, d, dtype=torch.float32)
        k_pre[:, :, 16:32, :16] = 5.0
        k_pre[:, :, 32:48, 16:32] = 5.0
        layer.update(k_pre, v_pre)

        k_dec = torch.zeros(b, h_kv, 1, d, dtype=torch.float32)
        v_dec = torch.zeros(b, h_kv, 1, d, dtype=torch.float32)
        q_a = torch.zeros(b, h_q, 1, d, dtype=torch.float32)
        q_a[:, :, :, :16] = 5.0
        q_b = torch.zeros(b, h_q, 1, d, dtype=torch.float32)
        q_b[:, :, :, 16:32] = 5.0

        # Step 1: initial resolve selects {0, 1, 5}
        layer.update(k_dec, v_dec)
        layer.resolve_active_kv(q_a)
        self.assertEqual(layer.resident_pages, {0, 1, 5})

        # Step 2: slot hit (same pages {0, 1, 5}) -> 0 recalled pages and 0 decode HtoD bytes
        layer.update(k_dec, v_dec)
        recalled_before_hit = layer.total_recalled_pages
        bytes_before_hit = layer.total_bytes_transferred_pcie
        layer.resolve_active_kv(q_a)
        self.assertEqual(layer.total_recalled_pages, recalled_before_hit)
        self.assertEqual(layer.total_bytes_transferred_pcie, bytes_before_hit)

        # Step 3: slot miss (q_B recalls page 2, delta = {2} with valid_len=16)
        layer.update(k_dec, v_dec)
        recalled_before_miss = layer.total_recalled_pages
        bytes_before_miss = layer.total_bytes_transferred_pcie
        layer.resolve_active_kv(q_b)
        self.assertEqual(layer.resident_pages, {0, 2, 5})
        self.assertEqual(layer.total_recalled_pages, recalled_before_miss + 1)
        expected_delta_bytes = 2 * b * h_kv * page_size * d * k_dec.element_size()
        self.assertEqual(layer.total_bytes_transferred_pcie - bytes_before_miss, expected_delta_bytes)

    def test_exempt_layers_gpu_only_no_cpu_or_digests(self):
        torch.manual_seed(7)
        layer = ArkValeLayer(layer_idx=0, page_size=16, active_pages_budget=2, exempt_early_layers=2)
        k_pre = torch.randn(1, 2, 48, 32, dtype=torch.float32)
        v_pre = torch.randn(1, 2, 48, 32, dtype=torch.float32)
        k_dec = torch.randn(1, 2, 1, 32, dtype=torch.float32)
        v_dec = torch.randn(1, 2, 1, 32, dtype=torch.float32)

        layer.update(k_pre, v_pre)
        out_k, out_v = layer.update(k_dec, v_dec)

        self.assertIsNone(layer.cpu_k_buf)
        self.assertIsNone(layer.cpu_v_buf)
        self.assertIsNone(layer.gpu_b_min)
        self.assertIsNone(layer.gpu_b_max)
        self.assertIsNotNone(layer.gpu_k_buf)
        self.assertEqual(layer.total_bytes_transferred_pcie, 0)
        self.assertEqual(out_k.shape[-2], 49)
        self.assertEqual(out_v.shape[-2], 49)
        self.assertEqual(layer.keys.shape[-2], 49)
        self.assertEqual(layer.values.shape[-2], 49)

    def test_budget_one_boundary_selects_latest_page_without_overflow(self):
        torch.manual_seed(11)
        layer = ArkValeLayer(layer_idx=2, page_size=16, active_pages_budget=1, exempt_early_layers=2)
        k_pre = torch.randn(1, 2, 48, 32, dtype=torch.float32)
        v_pre = torch.randn(1, 2, 48, 32, dtype=torch.float32)
        k_dec = torch.randn(1, 2, 1, 32, dtype=torch.float32)
        v_dec = torch.randn(1, 2, 1, 32, dtype=torch.float32)
        q_dec = torch.randn(1, 4, 1, 32, dtype=torch.float32)

        layer.update(k_pre, v_pre)
        layer.update(k_dec, v_dec)  # 49 tokens -> total_pages = 4
        k_act, v_act = layer.resolve_active_kv(q_dec)

        self.assertEqual(layer.resident_pages, {3})
        self.assertEqual(k_act.shape[-2], 1)
        self.assertEqual(v_act.shape[-2], 1)

    def test_short_prompt_and_new_page_boundary_digests_accurate(self):
        torch.manual_seed(19)
        page_size = 16
        # Short prompt: start_pos=0, end_pos=7 < page_size
        l_short = ArkValeLayer(layer_idx=2, page_size=page_size, active_pages_budget=2, exempt_early_layers=2)
        k_short = torch.randn(1, 2, 7, 32, dtype=torch.float32) + 1.0
        v_short = torch.randn(1, 2, 7, 32, dtype=torch.float32)
        l_short.update(k_short, v_short)

        c_short = 0.5 * (k_short.amin(dim=-2) + k_short.amax(dim=-2))
        r_short = (k_short - c_short.unsqueeze(-2)).abs().mean(dim=-2)
        self.assertGreater(l_short.gpu_b_min[:, :, 0].abs().sum().item(), 0.0)
        self.assertTrue(torch.allclose(l_short.gpu_b_min[:, :, 0], c_short - r_short, atol=1e-5))
        self.assertTrue(torch.allclose(l_short.gpu_b_max[:, :, 0], c_short + r_short, atol=1e-5))

        # New-page boundary decode: curr == 16 (% page_size == 0)
        l_bound = ArkValeLayer(layer_idx=2, page_size=page_size, active_pages_budget=2, exempt_early_layers=2)
        l_bound.update(torch.randn(1, 2, 16, 32), torch.randn(1, 2, 16, 32))
        k_dec = torch.randn(1, 2, 1, 32) + 2.0
        l_bound.update(k_dec, torch.randn(1, 2, 1, 32))
        self.assertGreater(l_bound.gpu_b_min[:, :, 1].abs().sum().item(), 0.0)
        self.assertTrue(torch.allclose(l_bound.gpu_b_min[:, :, 1], k_dec.squeeze(-2), atol=1e-5))
        self.assertTrue(torch.allclose(l_bound.gpu_b_max[:, :, 1], k_dec.squeeze(-2), atol=1e-5))

    def test_sdpa_hook_passthrough_and_uninstall(self):
        from transformers.cache_utils import DynamicCache
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from src.techniques.arkvale import _install_pytorch_arkvale_hooks, _uninstall_pytorch_arkvale_hooks

        orig_sdpa = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
        _install_pytorch_arkvale_hooks()
        hooked_sdpa = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
        self.assertIsNot(hooked_sdpa, orig_sdpa)

        b, h_q, h_kv, d = 1, 4, 2, 32
        # 1. q_len > 1 pass-through
        cache = ArkValeCache(page_size=16, active_pages_budget=2, exempt_early_layers=1)
        k_pre = torch.randn(b, h_kv, 48, d)
        v_pre = torch.randn(b, h_kv, 48, d)
        cache.update(k_pre, v_pre, layer_idx=0)
        cache.update(k_pre, v_pre, layer_idx=1)

        q_pre = torch.randn(b, h_q, 48, d)
        mod_l1 = SimpleNamespace(layer_idx=1, num_key_value_groups=h_q // h_kv)
        mod_l0 = SimpleNamespace(layer_idx=0, num_key_value_groups=h_q // h_kv)
        out_hook, _ = hooked_sdpa(
            mod_l1, q_pre, k_pre, v_pre, attention_mask=None, is_causal=False, enable_gqa=True
        )
        out_orig, _ = orig_sdpa(
            mod_l1, q_pre, k_pre, v_pre, attention_mask=None, is_causal=False, enable_gqa=True
        )
        self.assertTrue(torch.allclose(out_hook, out_orig, atol=1e-5))

        # 2. Exempt layer pass-through (q_len == 1, layer_idx=0 < exempt_early_layers=1)
        q_dec = torch.randn(b, h_q, 1, d)
        k_dec = torch.randn(b, h_kv, 1, d)
        v_dec = torch.randn(b, h_kv, 1, d)
        k_full_0, v_full_0 = cache.update(k_dec, v_dec, layer_idx=0)
        out_ex_hook, _ = hooked_sdpa(
            mod_l0, q_dec, k_full_0, v_full_0, attention_mask=None, is_causal=False, enable_gqa=True
        )
        out_ex_orig, _ = orig_sdpa(
            mod_l0, q_dec, k_full_0, v_full_0, attention_mask=None, is_causal=False, enable_gqa=True
        )
        self.assertTrue(torch.allclose(out_ex_hook, out_ex_orig, atol=1e-5))

        # 3. Under-budget layer pass-through (total_pages=1 <= active_pages_budget=2)
        cache_small = ArkValeCache(page_size=16, active_pages_budget=2, exempt_early_layers=0)
        k_sm, v_sm = cache_small.update(torch.randn(b, h_kv, 10, d), torch.randn(b, h_kv, 10, d), layer_idx=0)
        out_ub_hook, _ = hooked_sdpa(
            mod_l0, q_dec, k_sm, v_sm, attention_mask=None, is_causal=False, enable_gqa=True
        )
        out_ub_orig, _ = orig_sdpa(
            mod_l0, q_dec, k_sm, v_sm, attention_mask=None, is_causal=False, enable_gqa=True
        )
        self.assertTrue(torch.allclose(out_ub_hook, out_ub_orig, atol=1e-5))

        # 4. Non-ArkVale DynamicCache pass-through
        dyn_cache = DynamicCache()
        set_active_arkvale_cache(dyn_cache)  # type: ignore[arg-type]
        out_dyn_hook, _ = hooked_sdpa(
            mod_l0, q_dec, k_dec, v_dec, attention_mask=None, is_causal=False, enable_gqa=True
        )
        out_dyn_orig, _ = orig_sdpa(
            mod_l0, q_dec, k_dec, v_dec, attention_mask=None, is_causal=False, enable_gqa=True
        )
        self.assertTrue(torch.allclose(out_dyn_hook, out_dyn_orig, atol=1e-5))

        # 5. Uninstall restores original SDPA
        _uninstall_pytorch_arkvale_hooks()
        self.assertIs(ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None), orig_sdpa)

    def test_suites_json_contains_fix_arkvale(self):
        suites_path = Path(__file__).resolve().parents[1] / "src" / "config" / "suites.json"
        suites = json.loads(suites_path.read_text(encoding="utf-8"))
        self.assertIn("fix_arkvale", suites)
        fix_arkvale = suites["fix_arkvale"]
        self.assertEqual(fix_arkvale["techniques"], ["arkvale"])
        self.assertFalse(fix_arkvale["force_longmem_ref"])
        self.assertIn("contexts", fix_arkvale)
        self.assertIn("eval_contexts", fix_arkvale)
        self.assertIn("niah_contexts", fix_arkvale)


if __name__ == "__main__":
    unittest.main()
