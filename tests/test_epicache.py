import unittest
from types import SimpleNamespace

import torch

from src.core.axis import Axis
from src.core.diagnostic import inspect_sample_diagnostic
from src.core.pytorch_runner import PyTorchRunner
from src.core.registry import get_technique, list_techniques
from src.techniques.epicache import (
    EpiCacheTechnique,
    _uninstall_pytorch_epicache_hooks,
    compute_gqa_epicache_attention,
    compute_layer_sensitivity_budgets,
    evict_block_prefill,
    route_query_to_episode,
)
from src.techniques.epicache_cache import (
    EpiCacheCache,
    EpiCacheLayer,
    get_active_epicache_cache,
    record_epicache_attention_scores,
)


class TestEpiCacheTechnique(unittest.TestCase):
    def test_registration_and_axes(self):
        self.assertIn("epicache", list_techniques())
        technique = get_technique("epicache")
        self.assertIsInstance(technique, EpiCacheTechnique)
        self.assertEqual(technique.axes, Axis.STRUCTURAL | Axis.SPATIAL)
        self.assertEqual(technique.compression_ratio, 3.8)

    def test_defaults_and_custom_parameters(self):
        technique = get_technique("epicache")
        self.assertEqual(technique.num_episodes, 4)
        self.assertEqual(technique.episode_budget_tokens, 1024)
        self.assertEqual(technique.layer_sensitivity_alpha, 2.0)
        self.assertEqual(technique.block_size, 16)

        mock_out = SimpleNamespace(prompt_tokens=4096, generated_tokens=64)
        metrics = technique.extra_metrics(None, mock_out)
        self.assertEqual(metrics["configured_num_episodes"], 4)
        self.assertEqual(metrics["executed_episodes"], 0)
        self.assertFalse(metrics["block_wise_prefill_active"])
        self.assertFalse(metrics["episodic_routing_active"])
        self.assertEqual(metrics["eviction_scoring_mode"], "uninitialized")

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
        self.assertEqual(telemetry["scoring_mode"], "attention_guided")

    def test_attention_guided_eviction_prefers_high_attention_over_key_norm(self):
        batch, heads, seq_len, dim = 1, 2, 16, 8
        budget = 4

        k = torch.ones(batch, heads, seq_len, dim) * 0.1
        v = torch.randn(batch, heads, seq_len, dim)
        # Token 5 has huge L2 key norm, but zero attention score
        k[:, :, 5, :] = 100.0
        # Tokens 2, 7, 11 have small key norms, but high attention scores
        attn_scores = torch.zeros(batch, heads, seq_len)
        attn_scores[:, :, 2] = 0.9
        attn_scores[:, :, 7] = 0.8
        attn_scores[:, :, 11] = 0.7

        _, _, retained_idx = evict_block_prefill(k, v, budget=budget, attn_scores=attn_scores)
        retained = retained_idx.tolist()
        self.assertEqual(retained, [0, 2, 7, 11])
        self.assertNotIn(5, retained)

    def test_record_attention_scores_single_token_kv_len(self):
        layer = EpiCacheLayer(layer_idx=0, total_layers=4, episode_budget_tokens=16)
        k1 = torch.randn(1, 2, 1, 8)
        v1 = torch.randn(1, 2, 1, 8)
        layer.update(k1, v1)
        q1 = torch.randn(1, 4, 1, 8)
        record_epicache_attention_scores(layer, q1, k1)
        self.assertEqual(layer.importance_scores.shape, (1, 2, 1))

        # Subsequent decode token must concatenate cleanly without ndim mismatch
        k2 = torch.randn(1, 2, 1, 8)
        v2 = torch.randn(1, 2, 1, 8)
        layer.update(k2, v2)
        self.assertEqual(layer.importance_scores.shape, (1, 2, 2))

    def test_epicache_layer_chunk_pruning_and_decode(self):
        torch.manual_seed(123)
        cache = EpiCacheCache(total_layers=4, episode_budget_tokens=16)
        self.assertEqual(get_active_epicache_cache(), cache)

        batch, heads, dim = 1, 4, 64
        k_chunk = torch.randn(batch, 32, heads, dim, dtype=torch.bfloat16).transpose(1, 2)
        v_chunk = torch.randn(batch, 32, heads, dim, dtype=torch.bfloat16).transpose(1, 2)

        cache.update(k_chunk, v_chunk, layer_idx=0)
        cache.prune_after_chunk()
        expected_l0_budget = cache.layers[0].layer_budget
        self.assertEqual(expected_l0_budget, cache.layers[0].sensitivity_budget)
        self.assertEqual(cache.get_seq_length(0), expected_l0_budget)
        self.assertIn(0, cache.original_positions)

        # Decode 1 token beyond budget -> automatically re-prunes to layer_budget
        k_dec = torch.randn(batch, heads, 1, dim, dtype=torch.bfloat16)
        v_dec = torch.randn(batch, heads, 1, dim, dtype=torch.bfloat16)
        out_k, _ = cache.update(k_dec, v_dec, layer_idx=0)
        self.assertEqual(out_k.shape[-2], expected_l0_budget)

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
        technique = get_technique("epicache", episode_budget_tokens=64, layer_sensitivity_alpha=2.0)
        technique.setup()
        try:
            for dtype in (torch.float32, torch.bfloat16):
                technique.setup()
                model = Qwen3ForCausalLM(cfg).to(dtype=dtype).eval()
                cache = technique.pytorch_cache_factory()()
                with torch.no_grad():
                    model(
                        input_ids=torch.randint(0, 256, (1, 160)),
                        position_ids=torch.arange(0, 160).unsqueeze(0),
                        past_key_values=cache,
                        use_cache=True,
                    )
                    cache.prune_after_chunk()
                    out2 = model(
                        input_ids=torch.randint(0, 256, (1, 96)),
                        position_ids=torch.arange(160, 256).unsqueeze(0),
                        past_key_values=cache,
                        use_cache=True,
                    )
                    cache.prune_after_chunk()
                    next_tok = out2.logits[:, -1:, :].argmax(dim=-1)
                    out3 = model(
                        input_ids=next_tok,
                        position_ids=torch.tensor([[256]]),
                        past_key_values=cache,
                        use_cache=True,
                    )
                seq_lens = [cache.get_seq_length(i) for i in range(6)]
                # Edge layers (0, 5) have larger sensitivity budgets than middle layers (2, 3)
                self.assertGreater(seq_lens[0], seq_lens[2])
                self.assertGreater(seq_lens[5], seq_lens[3])
                self.assertEqual(len(cache.original_positions), seq_lens[0])
                self.assertEqual(out3.logits.shape, (1, 1, 256))

                # Verify telemetry persistence after cache deletion (simulating PyTorchRunner cleanup)
                del cache
                metrics = technique.extra_metrics(None, None)
                self.assertGreater(metrics["active_retained_tokens"], 0)
                self.assertGreater(metrics["attention_guided_steps"], 0)
                self.assertEqual(metrics["eviction_scoring_mode"], "attention_guided")
                self.assertTrue(metrics["per_layer_sensitivity_active"])
                self.assertTrue(metrics["block_wise_prefill_active"])
                self.assertFalse(metrics["episodic_routing_active"])
                self.assertEqual(metrics["executed_episodes"], 1)

                # Verify setup() resets run-level telemetry
                technique.setup()
                reset_metrics = technique.extra_metrics(None, None)
                self.assertEqual(reset_metrics["active_retained_tokens"], 0)
                self.assertEqual(reset_metrics["attention_guided_steps"], 0)
                self.assertEqual(reset_metrics["eviction_scoring_mode"], "uninitialized")
        finally:
            _uninstall_pytorch_epicache_hooks()


if __name__ == "__main__":
    unittest.main()
