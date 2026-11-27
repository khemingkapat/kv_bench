import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.core.axis import Axis
from src.core.diagnostic import inspect_sample_diagnostic
from src.core.pytorch_runner import PyTorchRunner
from src.core.registry import get_technique, list_techniques
from src.techniques.epicache import (
    EpiCacheTechnique,
    compute_gqa_epicache_attention,
    compute_layer_sensitivity_budgets,
    evict_block_prefill,
    route_query_to_episode,
)
from src.techniques.epicache_cache import (
    EpiCacheCache,
    EpiCacheLayer,
    get_active_epicache_cache,
)


class TestEpiCacheTechnique(unittest.TestCase):
    def test_registration_and_axes(self):
        self.assertIn("epicache", list_techniques())
        technique = get_technique("epicache")
        self.assertIsInstance(technique, EpiCacheTechnique)
        self.assertEqual(technique.axes, Axis.STRUCTURAL | Axis.SPATIAL)
        self.assertFalse(hasattr(technique, "compression_ratio"))

    def test_defaults_and_custom_parameters(self):
        technique = get_technique("epicache")
        self.assertEqual(technique.num_episodes, 4)
        self.assertEqual(technique.episode_budget_tokens, 1024)
        self.assertEqual(technique.layer_sensitivity_alpha, 2.0)
        self.assertEqual(technique.block_size, 16)
        self.assertEqual(technique.sink_tokens, 4)
        self.assertEqual(technique.recent_window, 512)

        metrics = technique.extra_metrics(None, None)
        self.assertEqual(metrics["num_episodes"], 4)
        self.assertTrue(metrics["block_wise_prefill_active"])
        self.assertIs(metrics["episodic_routing_active"], False)
        self.assertEqual(metrics["sink_tokens"], 4)
        self.assertEqual(metrics["recent_window"], 512)

        custom = EpiCacheTechnique(sink_tokens=8, recent_window=128)
        self.assertEqual(custom.sink_tokens, 8)
        self.assertEqual(custom.recent_window, 128)
        custom_cache = custom.pytorch_cache_factory()()
        self.assertEqual(custom_cache.sink_tokens, 8)
        self.assertEqual(custom_cache.recent_window, 128)

        kwargs = technique.engine_kwargs(context_length=8192, max_tokens=128)
        self.assertEqual(kwargs["block_size"], 16)
        self.assertEqual(kwargs["swap_space"], 8)

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            EpiCacheTechnique(num_episodes=0)
        with self.assertRaises(ValueError):
            EpiCacheTechnique(episode_budget_tokens=-1)
        with self.assertRaises(ValueError):
            EpiCacheTechnique(layer_sensitivity_alpha=0.0)
        with self.assertRaises(ValueError):
            EpiCacheTechnique(block_size=0)
        with self.assertRaises(ValueError):
            EpiCacheTechnique(sink_tokens=-1)
        with self.assertRaises(ValueError):
            EpiCacheTechnique(recent_window=-1)

        k = torch.randn(1, 2, 8, 16)
        v = torch.randn(1, 2, 8, 16)
        for bad_budget in (0, -4):
            with self.assertRaises(ValueError):
                evict_block_prefill(k, v, budget=bad_budget)
            with self.assertRaises(ValueError):
                EpiCacheCache(total_layers=2, episode_budget_tokens=bad_budget)
            with self.assertRaises(ValueError):
                EpiCacheLayer(layer_idx=0, total_layers=2, episode_budget_tokens=bad_budget)

        for bad_arg in (-1,):
            with self.assertRaises(ValueError):
                evict_block_prefill(k, v, budget=4, sink_tokens=bad_arg)
            with self.assertRaises(ValueError):
                evict_block_prefill(k, v, budget=4, recent_window=bad_arg)
            with self.assertRaises(ValueError):
                EpiCacheCache(total_layers=2, episode_budget_tokens=4, sink_tokens=bad_arg)
            with self.assertRaises(ValueError):
                EpiCacheCache(total_layers=2, episode_budget_tokens=4, recent_window=bad_arg)
            with self.assertRaises(ValueError):
                EpiCacheLayer(layer_idx=0, total_layers=2, episode_budget_tokens=4, sink_tokens=bad_arg)
            with self.assertRaises(ValueError):
                EpiCacheLayer(layer_idx=0, total_layers=2, episode_budget_tokens=4, recent_window=bad_arg)

    def test_layer_sensitivity_budgets(self):
        budgets = compute_layer_sensitivity_budgets(total_layers=16, base_budget_per_layer=512, alpha=2.0)
        self.assertEqual(len(budgets), 16)
        for l, b in budgets.items():
            self.assertGreaterEqual(b, 64)
        self.assertAlmostEqual(sum(budgets.values()) / (16 * 512), 1.0, delta=0.05)

    def test_route_query_to_episode(self):
        dim = 32
        num_episodes = 4

        torch.manual_seed(42)
        centroids = torch.randn(num_episodes, dim, dtype=torch.float32)
        query_emb = centroids[2] + 0.01 * torch.randn(dim)

        best_ep = route_query_to_episode(query_emb, centroids)
        self.assertEqual(best_ep, 2)

    def test_evict_block_prefill(self):
        batch, heads, seq_len, dim = 1, 2, 64, 32
        budget = 16

        torch.manual_seed(100)
        k = torch.randn(batch, heads, seq_len, dim, dtype=torch.float32)
        v = torch.randn(batch, heads, seq_len, dim, dtype=torch.float32)

        k_ret, v_ret, indices = evict_block_prefill(k, v, budget=budget)
        self.assertEqual(k_ret.shape[-2], budget)
        self.assertEqual(v_ret.shape[-2], budget)
        self.assertIn(0, indices.tolist())

    def test_compute_gqa_epicache_attention(self):
        batch, seq_len_q, head_dim = 1, 1, 32
        num_q_heads, num_kv_heads = 8, 2
        seq_len_kv = 64

        torch.manual_seed(200)
        q = torch.randn(batch, num_q_heads, seq_len_q, head_dim, dtype=torch.float32)
        k = torch.randn(batch, num_kv_heads, seq_len_kv, head_dim, dtype=torch.float32)
        v = torch.randn(batch, num_kv_heads, seq_len_kv, head_dim, dtype=torch.float32)

        attn_out, attn_weights, telemetry = compute_gqa_epicache_attention(q, k, v, budget=16)

        self.assertEqual(attn_out.shape, (batch, num_q_heads, seq_len_q, head_dim))
        self.assertTrue(torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1)), atol=1e-4))
        self.assertEqual(telemetry["active_tokens"], 16)
        self.assertTrue(telemetry["block_prefill_capped"])

    def test_epicache_layer_chunk_pruning_and_decode(self):
        torch.manual_seed(123)
        cache = EpiCacheCache(total_layers=4, episode_budget_tokens=16)
        self.assertEqual(get_active_epicache_cache(), cache)

        batch, heads, dim = 1, 4, 64
        k_chunk = torch.randn(batch, 32, heads, dim, dtype=torch.bfloat16).transpose(1, 2)
        v_chunk = torch.randn(batch, 32, heads, dim, dtype=torch.bfloat16).transpose(1, 2)

        cache.update(k_chunk, v_chunk, layer_idx=0)
        cache.prune_after_chunk()
        self.assertEqual(cache.get_seq_length(0), 16)
        self.assertIn(0, cache.original_positions)

        # Decode 1 token beyond budget -> automatically re-prunes to budget 16
        k_dec = torch.randn(batch, heads, 1, dim, dtype=torch.bfloat16)
        v_dec = torch.randn(batch, heads, 1, dim, dtype=torch.bfloat16)
        out_k, _ = cache.update(k_dec, v_dec, layer_idx=0)
        self.assertEqual(out_k.shape[-2], 16)

    def test_epicache_factory_diagnostic_and_pytorch_runner(self):
        technique = get_technique("epicache", episode_budget_tokens=16)
        cache_factory = technique.pytorch_cache_factory()
        self.assertIsNotNone(cache_factory)

        cache = cache_factory()
        self.assertIsInstance(cache, EpiCacheCache)

        k1 = torch.randn(1, 4, 40, 64, dtype=torch.float32)
        v1 = torch.randn(1, 4, 40, 64, dtype=torch.float32)
        cache.update(k1, v1, layer_idx=0)
        cache.prune_after_chunk()

        diag = inspect_sample_diagnostic(
            cache=cache,
            prompt_text="Test EpiCache diagnostic",
            prompt_token_ids=list(range(40)),
            sample_meta={"id": "ec_diag_0", "task_type": "unit"},
            is_correct=True,
            baseline_correct=True,
            bits_per_scalar=16.0,
        )
        self.assertAlmostEqual(diag.overall_retention_ratio, 16 / 40, places=3)

        runner = PyTorchRunner(
            model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
            revision="cdbee75f17c01a7cc42f958dc650907174af0554",
            cache_factory=cache_factory,
            dry_run=True,
        )
        outputs = runner.generate(["Test EpiCache prompt."], max_tokens=16)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(outputs[0].outputs[0].token_ids), 16)

    def test_epicache_chunked_prefill_causal_mask_alignment(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        cfg = Qwen3Config(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
        )
        for dtype in (torch.float32, torch.bfloat16):
            model = Qwen3ForCausalLM(cfg).to(dtype=dtype).eval()
            cache = EpiCacheCache(
                total_layers=6, episode_budget_tokens=128, layer_sensitivity_alpha=2.0
            )
            with torch.no_grad():
                model(
                    input_ids=torch.randint(0, 256, (1, 128)),
                    position_ids=torch.arange(0, 128).unsqueeze(0),
                    past_key_values=cache,
                    use_cache=True,
                )
                cache.prune_after_chunk()
                out2 = model(
                    input_ids=torch.randint(0, 256, (1, 96)),
                    position_ids=torch.arange(128, 224).unsqueeze(0),
                    past_key_values=cache,
                    use_cache=True,
                )
                cache.prune_after_chunk()
                next_tok = out2.logits[:, -1:, :].argmax(dim=-1)
                out3 = model(
                    input_ids=next_tok,
                    position_ids=torch.tensor([[224]]),
                    past_key_values=cache,
                    use_cache=True,
                )
            self.assertEqual(cache.get_seq_length(0), 128)
            self.assertEqual(len(cache.original_positions), 128)
            self.assertEqual(out3.logits.shape, (1, 1, 256))

    def test_adversarial_l2_norm_sink_and_recent_pinning_prefill_and_decode(self):
        batch, heads, dim = 1, 2, 16
        seq_len = 40
        budget = 16
        sink_tokens = 4
        recent_window = 4
        target_middle = [10, 11, 12, 13, 20, 21, 22, 23]

        def make_normed_keys(norms: list[float]) -> torch.Tensor:
            k_tensor = torch.zeros(batch, heads, len(norms), dim, dtype=torch.float32)
            for idx, val in enumerate(norms):
                k_tensor[:, :, idx, 0] = val
            return k_tensor

        norms = [100.0] * seq_len
        for i in range(sink_tokens):
            norms[i] = 0.01
        for i in range(seq_len - recent_window, seq_len):
            norms[i] = 0.01
        for i in target_middle:
            norms[i] = 1000.0

        k = make_normed_keys(norms)
        v = torch.randn(batch, heads, seq_len, dim, dtype=torch.float32)

        k_ret, v_ret, indices = evict_block_prefill(
            k, v, budget=budget, sink_tokens=sink_tokens, recent_window=recent_window
        )
        self.assertEqual(k_ret.shape[-2], budget)
        self.assertEqual(v_ret.shape[-2], budget)
        expected_indices = sorted([0, 1, 2, 3] + target_middle + [36, 37, 38, 39])
        self.assertEqual(indices.tolist(), expected_indices)

        # EpiCacheCache prefill + decode with adversarial L2 norms
        cache = EpiCacheCache(
            total_layers=2,
            episode_budget_tokens=budget,
            sink_tokens=sink_tokens,
            recent_window=recent_window,
        )
        prefill_len = 32
        prefill_norms = [100.0] * prefill_len
        for i in range(4):
            prefill_norms[i] = 0.01
        for i in range(28, 32):
            prefill_norms[i] = 0.01
        cache_targets = list(range(8, 16))
        for i in cache_targets:
            prefill_norms[i] = 1000.0

        k_prefill = make_normed_keys(prefill_norms)
        v_prefill = torch.randn(batch, heads, prefill_len, dim, dtype=torch.float32)
        cache.update(k_prefill, v_prefill, layer_idx=0)
        cache.prune_after_chunk()

        self.assertEqual(cache.get_seq_length(0), budget)
        self.assertEqual(cache.original_positions, sorted([0, 1, 2, 3] + cache_targets + [28, 29, 30, 31]))

        # Decode 2 tiny-norm tokens; both must remain pinned in recent_window along with sinks & target middle
        for _ in range(2):
            k_dec = make_normed_keys([0.01])
            v_dec = torch.randn(batch, heads, 1, dim, dtype=torch.float32)
            out_k, _ = cache.update(k_dec, v_dec, layer_idx=0)
            self.assertEqual(out_k.shape[-2], budget)

        retained_pos = cache.original_positions
        self.assertEqual(len(retained_pos), budget)
        for p in [0, 1, 2, 3] + cache_targets + [32, 33]:
            self.assertIn(p, retained_pos)

    def test_clamping_and_zero_recent_window_edge_cases(self):
        k = torch.zeros(1, 2, 16, 8, dtype=torch.float32)
        v = torch.randn(1, 2, 16, 8, dtype=torch.float32)
        k[:, :, :, 0] = 10.0

        # Small budget < sink_tokens + recent_window clamps to [0, 1]
        k_ret, v_ret, idx = evict_block_prefill(k, v, budget=2, sink_tokens=4, recent_window=512)
        self.assertEqual(k_ret.shape[-2], 2)
        self.assertEqual(v_ret.shape[-2], 2)
        self.assertEqual(idx.tolist(), [0, 1])

        # recent_window=0 must not trigger -0: slice corruption
        norms = [100.0] * 20
        for i in range(4):
            norms[i] = 0.01
        for i in range(16, 20):
            norms[i] = 0.01
        for i in (6, 7, 8, 9):
            norms[i] = 1000.0
        k20 = torch.zeros(1, 2, 20, 8, dtype=torch.float32)
        for i, val in enumerate(norms):
            k20[:, :, i, 0] = val
        v20 = torch.randn(1, 2, 20, 8, dtype=torch.float32)

        k20_ret, v20_ret, idx20 = evict_block_prefill(k20, v20, budget=8, sink_tokens=4, recent_window=0)
        self.assertEqual(k20_ret.shape[-2], 8)
        self.assertEqual(v20_ret.shape[-2], 8)
        self.assertEqual(idx20.tolist(), [0, 1, 2, 3, 6, 7, 8, 9])

    def test_gpu_original_positions_zero_sync_and_reset(self):
        layer = EpiCacheLayer(
            layer_idx=0, total_layers=2, episode_budget_tokens=8, sink_tokens=2, recent_window=2
        )
        k1 = torch.randn(1, 2, 6, 16, dtype=torch.float32)
        v1 = torch.randn(1, 2, 6, 16, dtype=torch.float32)
        k2 = torch.randn(1, 2, 6, 16, dtype=torch.float32)
        v2 = torch.randn(1, 2, 6, 16, dtype=torch.float32)

        with patch.object(torch.Tensor, "tolist", wraps=torch.Tensor.tolist) as mock_tolist:
            layer.update(k1, v1)
            layer.update(k2, v2)
            layer.prune()
            self.assertEqual(mock_tolist.call_count, 0)

        self.assertIsInstance(layer._original_positions, torch.Tensor)
        self.assertEqual(layer._original_positions.dtype, torch.int32)
        self.assertEqual(layer._original_positions.device, k1.device)
        self.assertEqual(layer._original_positions.numel(), 8)

        positions = layer.original_positions
        self.assertIsInstance(positions, list)
        self.assertTrue(all(isinstance(p, int) for p in positions))
        self.assertEqual(len(positions), 8)

        # Verify diagnostic integration matches layer.original_positions retention ratio
        cache = EpiCacheCache(total_layers=1, episode_budget_tokens=8, sink_tokens=2, recent_window=2)
        cache.update(torch.cat([k1, k2], dim=-2), torch.cat([v1, v2], dim=-2), layer_idx=0)
        cache.prune_after_chunk()
        diag = inspect_sample_diagnostic(
            cache=cache,
            prompt_text="Zero sync diag",
            prompt_token_ids=list(range(12)),
            sample_meta={"id": "zero_sync_0", "task_type": "unit"},
            is_correct=True,
            baseline_correct=True,
            bits_per_scalar=16.0,
        )
        self.assertAlmostEqual(diag.overall_retention_ratio, len(positions) / 12, places=3)

        layer.reset()
        self.assertIsNone(layer._original_positions)
        self.assertEqual(layer.original_positions, [])

    def test_fix_epicache_suite_config(self):
        suites_path = Path("src/config/suites.json")
        suites = json.loads(suites_path.read_text(encoding="utf-8"))
        self.assertIn("fix_epicache", suites)
        cfg = suites["fix_epicache"]
        self.assertEqual(cfg["techniques"], ["epicache"])
        self.assertIs(cfg["force_longmem_ref"], False)
        self.assertIs(cfg["skip_existing"], False)


if __name__ == "__main__":
    unittest.main()
