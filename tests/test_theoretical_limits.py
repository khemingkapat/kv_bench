"""Unit tests for theoretical capacity modeling and test matrix generation."""

import unittest
from scripts.calculate_theoretical_limits import (
    calculate_kv_bytes_per_token,
    calculate_memory_breakdown,
    calculate_oom_cliff,
    calculate_roofline,
    format_markdown_report,
    generate_test_matrix,
    get_weights_bytes,
    get_weights_gib,
)
from src.core.config import get_model_spec


class TestTheoreticalLimits(unittest.TestCase):
    def setUp(self):
        self.qwen_spec = get_model_spec("qwen3-4b")
        self.phi_spec = get_model_spec("phi4-mini")
        self.local_spec = get_model_spec("qwen3-1.7b")
        self.hybrid_spec = get_model_spec("qwen3.5-4b")
        self.assertIsNotNone(self.qwen_spec)
        self.assertIsNotNone(self.phi_spec)
        self.assertIsNotNone(self.local_spec)
        self.assertIsNotNone(self.hybrid_spec)

    def test_kv_bytes_per_token(self):
        # Qwen3-4B: 4 * 36 * 8 * 128 = 147,456 bytes = 144 KiB
        self.assertEqual(calculate_kv_bytes_per_token(self.qwen_spec), 147456)
        # Phi-4-mini: 4 * 32 * 8 * 128 = 131,072 bytes = 128 KiB
        self.assertEqual(calculate_kv_bytes_per_token(self.phi_spec), 131072)
        # Qwen3-1.7B: 4 * 28 * 8 * 128 = 114,688 bytes = 112 KiB
        self.assertEqual(calculate_kv_bytes_per_token(self.local_spec), 114688)
        # Qwen3.5-4B (Hybrid): 4 * 8 * 4 * 256 = 32,768 bytes = 32 KiB
        self.assertEqual(calculate_kv_bytes_per_token(self.hybrid_spec), 32768)

    def test_weights_conversion(self):
        # Qwen3-4B: 8.06 GB decimal = 8.06e9 bytes = ~7.506 GiB
        w_bytes = get_weights_bytes(self.qwen_spec)
        w_gib = get_weights_gib(self.qwen_spec)
        self.assertEqual(w_bytes, 8060000000)
        self.assertAlmostEqual(w_gib, 7.506, delta=0.01)

    def test_memory_breakdown_and_two_tier_oom(self):
        # At 512 context, Qwen3-4B total memory should be ~8.38 GiB (Fits both pool and cap)
        mb_512 = calculate_memory_breakdown(self.qwen_spec, context_len=512, num_users=1, vram_cap_gib=16.0, gpu_utilization=0.85)
        self.assertFalse(mb_512.is_oom_cap)
        self.assertFalse(mb_512.is_oom_vllm_pool)
        self.assertAlmostEqual(mb_512.total_memory_gib, 8.38, delta=0.1)

        # At 32768 context, total memory is ~12.81 GiB (Fits inside 13.6 GiB pool, but near edge)
        mb_32k = calculate_memory_breakdown(self.qwen_spec, context_len=32768, num_users=1, vram_cap_gib=16.0, gpu_utilization=0.85)
        self.assertFalse(mb_32k.is_oom_cap)
        self.assertFalse(mb_32k.is_oom_vllm_pool)
        self.assertAlmostEqual(mb_32k.total_memory_gib, 12.81, delta=0.2)

        # At 49152 context, total memory is ~15.06 GiB (Exceeds 13.6 GiB pool, but fits under 16.0 GiB cap)
        mb_49k = calculate_memory_breakdown(self.qwen_spec, context_len=49152, num_users=1, vram_cap_gib=16.0, gpu_utilization=0.85)
        self.assertFalse(mb_49k.is_oom_cap)
        self.assertTrue(mb_49k.is_oom_vllm_pool)

        # At 65536 context, total memory is ~17.31 GiB (Deterministic Hard OOM on 16.0 GiB cap)
        mb_64k = calculate_memory_breakdown(self.qwen_spec, context_len=65536, num_users=1, vram_cap_gib=16.0, gpu_utilization=0.85)
        self.assertTrue(mb_64k.is_oom_cap)
        self.assertTrue(mb_64k.is_oom_vllm_pool)

    def test_oom_cliff_calculation(self):
        # Qwen3-4B: 1 user on 13.6 GiB pool (util=0.85): net headroom 5.294 GiB -> 38,546 tokens
        cliff_pool_1u = calculate_oom_cliff(self.qwen_spec, num_users=1, vram_target_gib=13.6, runtime_overhead_gib=0.8)
        self.assertEqual(cliff_pool_1u, 38546)

        # Qwen3-4B: 12 users on 13.6 GiB pool -> 3,212 tokens per user
        cliff_pool_12u = calculate_oom_cliff(self.qwen_spec, num_users=12, vram_target_gib=13.6, runtime_overhead_gib=0.8)
        self.assertEqual(cliff_pool_12u, 3212)

        # Qwen3-4B: 1 user on 16.0 GiB hard cap -> 56,022 tokens
        cliff_cap_1u = calculate_oom_cliff(self.qwen_spec, num_users=1, vram_target_gib=16.0, runtime_overhead_gib=0.8)
        self.assertEqual(cliff_cap_1u, 56022)

        # Peer models (exact alignment with docs/THEORETICAL_LIMITS.md Section 5)
        self.assertEqual(calculate_oom_cliff(self.phi_spec, num_users=1, vram_target_gib=13.6, runtime_overhead_gib=0.8), 46492)
        self.assertEqual(calculate_oom_cliff(self.phi_spec, num_users=12, vram_target_gib=13.6, runtime_overhead_gib=0.8), 3874)
        self.assertEqual(calculate_oom_cliff(self.local_spec, num_users=1, vram_target_gib=6.8, runtime_overhead_gib=0.6), 27964)
        self.assertEqual(calculate_oom_cliff(self.local_spec, num_users=12, vram_target_gib=6.8, runtime_overhead_gib=0.6), 2330)
        self.assertEqual(calculate_oom_cliff(self.hybrid_spec, num_users=1, vram_target_gib=13.6, runtime_overhead_gib=0.8), 172238)
        self.assertEqual(calculate_oom_cliff(self.hybrid_spec, num_users=12, vram_target_gib=13.6, runtime_overhead_gib=0.8), 14353)

    def test_roofline_bounds(self):
        # RTX 4090 memory bandwidth 1008 GB/s
        rf_512 = calculate_roofline(self.qwen_spec, context_len=512, batch_size=1, memory_bw_gbs=1008.0)
        # Weights 8.06e9 bytes / 1008e9 B/s = ~8.00ms base step latency -> ~125 tok/s
        self.assertGreater(rf_512.decode_throughput_tok_per_sec, 120.0)
        self.assertLess(rf_512.decode_step_latency_ms, 8.5)

        # At 32K context, KV cache adds 4.5GB -> bytes read = 12.89GB -> TPOT rises to ~12.79ms -> ~78 tok/s
        rf_32k = calculate_roofline(self.qwen_spec, context_len=32768, batch_size=1, memory_bw_gbs=1008.0)
        self.assertGreater(rf_32k.decode_step_latency_ms, rf_512.decode_step_latency_ms)
        # Latency increase should be ~+60%
        self.assertAlmostEqual(rf_32k.tpot_latency_increase_pct, 60.0, delta=2.0)
        # Throughput reduction should be ~-37.5%
        self.assertAlmostEqual(rf_32k.throughput_reduction_pct, 37.5, delta=2.0)

        # Prefill min TTFT should scale with linear + causal quadratic terms
        self.assertGreater(rf_32k.prefill_min_ttft_sec, rf_512.prefill_min_ttft_sec * 50)

    def test_generate_test_matrix_and_markdown(self):
        matrix = generate_test_matrix(self.qwen_spec, vram_cap_gib=16.0, gpu_utilization=0.85)
        self.assertIn("single_stream_sweep", matrix)
        self.assertIn("concurrency_grid", matrix)
        self.assertIn("theoretical_oom_cliffs_vllm_pool", matrix)
        self.assertIn("theoretical_oom_cliffs_hard_cap", matrix)

        report = format_markdown_report(matrix, "NVIDIA GeForce RTX 4090")
        self.assertIn("Theoretical Capacity & Roofline Analysis", report)
        self.assertIn("38,546 tokens", report)
        self.assertIn("Hard OOM", report)
        self.assertIn("Pool OOM", report)
        self.assertNotIn("\t", report)
        self.assertIn(r"$8.9\text{--}9.4\text{ GiB}$", report)


if __name__ == "__main__":
    unittest.main()
