"""Unit tests for theoretical capacity modeling and test matrix generation."""

import unittest
from scripts.calculate_theoretical_limits import (
    calculate_kv_bytes_per_token,
    calculate_memory_breakdown,
    calculate_oom_cliff,
    calculate_roofline,
    format_markdown_report,
    generate_test_matrix,
)
from src.core.config import get_model_spec


class TestTheoreticalLimits(unittest.TestCase):
    def setUp(self):
        self.qwen_spec = get_model_spec("qwen3-4b")
        self.phi_spec = get_model_spec("phi4-mini")
        self.local_spec = get_model_spec("qwen3-1.7b")
        self.assertIsNotNone(self.qwen_spec)
        self.assertIsNotNone(self.phi_spec)
        self.assertIsNotNone(self.local_spec)

    def test_kv_bytes_per_token(self):
        # Qwen3-4B: 4 * 36 * 8 * 128 = 147,456 bytes = 144 KiB
        self.assertEqual(calculate_kv_bytes_per_token(self.qwen_spec), 147456)
        # Phi-4-mini: 4 * 32 * 8 * 128 = 131,072 bytes = 128 KiB
        self.assertEqual(calculate_kv_bytes_per_token(self.phi_spec), 131072)
        # Qwen3-1.7B: 4 * 28 * 8 * 128 = 114,688 bytes = 112 KiB
        self.assertEqual(calculate_kv_bytes_per_token(self.local_spec), 114688)

    def test_memory_breakdown_and_oom_flag(self):
        # At 512 context, Qwen3-4B total memory should be ~8.93 GB (Fits in 16GB)
        mb_512 = calculate_memory_breakdown(self.qwen_spec, context_len=512, num_users=1, vram_cap_gb=16.0)
        self.assertFalse(mb_512.is_oom)
        self.assertAlmostEqual(mb_512.total_memory_gb, 8.93, delta=0.1)
        self.assertGreater(mb_512.headroom_remaining_gb, 6.5)

        # At 32768 context, total memory is ~13.36 GB (Fits in 16GB)
        mb_32k = calculate_memory_breakdown(self.qwen_spec, context_len=32768, num_users=1, vram_cap_gb=16.0)
        self.assertFalse(mb_32k.is_oom)
        self.assertAlmostEqual(mb_32k.total_memory_gb, 13.36, delta=0.2)

        # At 65536 context, total memory is ~17.86 GB (Deterministic OOM on 16GB)
        mb_64k = calculate_memory_breakdown(self.qwen_spec, context_len=65536, num_users=1, vram_cap_gb=16.0)
        self.assertTrue(mb_64k.is_oom)
        self.assertLess(mb_64k.headroom_remaining_gb, 0.0)

    def test_oom_cliff_calculation(self):
        # 1 user on 16GB cap: net headroom 7.14 GB -> ~51,991 tokens
        cliff_1u = calculate_oom_cliff(self.qwen_spec, num_users=1, vram_cap_gb=16.0, runtime_overhead_gb=0.8)
        self.assertGreater(cliff_1u, 50000)
        self.assertLess(cliff_1u, 53000)

        # 12 users on 16GB cap: net headroom 7.14 GB -> ~4,332 tokens per user
        cliff_12u = calculate_oom_cliff(self.qwen_spec, num_users=12, vram_cap_gb=16.0, runtime_overhead_gb=0.8)
        self.assertGreater(cliff_12u, 4200)
        self.assertLess(cliff_12u, 4400)

        # 1 user on 8GB cap for Qwen3-1.7B (weights 3.45GB, runtime 0.6GB -> headroom 3.95GB -> ~36K tokens)
        cliff_local = calculate_oom_cliff(self.local_spec, num_users=1, vram_cap_gb=8.0, runtime_overhead_gb=0.6)
        self.assertGreater(cliff_local, 35000)

    def test_roofline_bounds(self):
        # RTX 4090 memory bandwidth 1008 GB/s
        rf_512 = calculate_roofline(self.qwen_spec, context_len=512, batch_size=1, memory_bw_gbs=1008.0)
        # Weights ~8.06 GB / 1008 GB/s = ~8.0ms step latency -> ~115 tok/s
        self.assertGreater(rf_512.decode_throughput_tok_per_sec, 100.0)
        self.assertLess(rf_512.decode_step_latency_ms, 10.0)

        # At 32K context, KV cache adds 4.5GB -> bytes read = 12.56GB -> TPOT rises to ~13.4ms -> ~75 tok/s
        rf_32k = calculate_roofline(self.qwen_spec, context_len=32768, batch_size=1, memory_bw_gbs=1008.0)
        self.assertGreater(rf_32k.decode_step_latency_ms, rf_512.decode_step_latency_ms)
        self.assertGreater(rf_32k.decode_slowdown_vs_zero_ctx_pct, 40.0)

        # Prefill min TTFT should scale quadratically with context length
        self.assertGreater(rf_32k.prefill_min_ttft_sec, rf_512.prefill_min_ttft_sec * 50)

    def test_generate_test_matrix_and_markdown(self):
        matrix = generate_test_matrix(self.qwen_spec, vram_cap_gb=16.0)
        self.assertIn("single_stream_sweep", matrix)
        self.assertIn("concurrency_grid", matrix)
        self.assertIn("theoretical_oom_cliffs", matrix)

        report = format_markdown_report(matrix, "NVIDIA GeForce RTX 4090")
        self.assertIn("Theoretical Capacity & Roofline Analysis", report)
        self.assertIn("51,991 tokens", report)
        self.assertIn("OOM", report)


if __name__ == "__main__":
    unittest.main()
