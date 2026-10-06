import unittest

from src.core.metrics import ResultEntry, calc_percentile
from src.core.quality import (
    compute_kl_divergence_from_probs,
    compute_logit_divergence,
    compute_top1_agreement,
    evaluate_fidelity,
)
from src.core.workload import ConcurrentChatWorkload, SyntheticFiller
from src.workloads.memory_adherence import LongMemEvalWorkload


class TestMetrics(unittest.TestCase):
    def test_calc_percentile(self):
        self.assertEqual(calc_percentile([], 95), 0.0)
        self.assertEqual(calc_percentile([10.0], 95), 10.0)
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        # 95th percentile
        p95 = calc_percentile(vals, 95.0)
        self.assertAlmostEqual(p95, 9.55, places=2)
        p50 = calc_percentile(vals, 50.0)
        self.assertAlmostEqual(p50, 5.5, places=2)

    def test_result_entry_fields(self):
        entry = ResultEntry(
            requested_ctx=1024,
            status="SUCCESS",
            axes=["SPATIAL"],
            tpot_mean_ms=15.4,
            tpot_p95_ms=18.2,
            ttft_p95_sec=0.45,
            memcpy_bytes_per_token=2048.5,
        )
        self.assertEqual(entry.tpot_mean_ms, 15.4)
        self.assertEqual(entry.tpot_p95_ms, 18.2)
        self.assertEqual(entry.ttft_p95_sec, 0.45)
        self.assertEqual(entry.memcpy_bytes_per_token, 2048.5)


class TestQualityMetrics(unittest.TestCase):
    def test_top1_agreement(self):
        self.assertEqual(compute_top1_agreement([], []), 0.0)
        self.assertEqual(compute_top1_agreement([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertEqual(compute_top1_agreement([1, 2, 3, 4], [1, 9, 3, 8]), 0.5)
        self.assertEqual(compute_top1_agreement([1, 2], [3, 4]), 0.0)

    def test_kl_divergence(self):
        # Identical distributions -> KL = 0
        p = {101: -0.1, 102: -2.3}
        mean_kl, p99_kl = compute_logit_divergence([p], [p])
        self.assertAlmostEqual(mean_kl, 0.0, places=4)
        self.assertAlmostEqual(p99_kl, 0.0, places=4)

        # Perturbed distribution
        q = {101: -0.8, 102: -0.6}
        mean_kl_diff, _ = compute_logit_divergence([p], [q])
        self.assertGreater(mean_kl_diff, 0.0)

    def test_evaluate_fidelity(self):
        base_tokens = [10, 20, 30]
        target_tokens = [10, 20, 99]
        p = {10: -0.05, 20: -3.0}
        q = {10: -0.1, 20: -2.5}
        fidelity = evaluate_fidelity(base_tokens, target_tokens, [p], [q])
        self.assertIn("top1_agreement", fidelity)
        self.assertAlmostEqual(fidelity["top1_agreement"], 2 / 3, places=2)
        self.assertIn("kl_div_mean", fidelity)
        self.assertIn("kl_div_p99", fidelity)


class TestWorkloads(unittest.TestCase):
    def test_concurrent_workload_init(self):
        workload = ConcurrentChatWorkload(num_users=12)
        self.assertEqual(workload.num_users, 12)

    def test_longmemeval_dataset(self):
        workload = LongMemEvalWorkload()
        self.assertGreaterEqual(len(workload.samples), 10)
        task_types = {s["task_type"] for s in workload.samples}
        expected_types = {
            "fact_recall",
            "cross_session_reasoning",
            "information_update",
            "relative_timeframe",
            "unanswerable_refusal",
        }
        self.assertTrue(expected_types.issubset(task_types))

    def test_longmemeval_run_mock(self):
        import sys
        from unittest.mock import MagicMock

        mock_vllm = MagicMock()
        class FakeSamplingParams:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
        mock_vllm.SamplingParams = FakeSamplingParams

        mock_llm = MagicMock()
        fake_out = MagicMock()
        fake_out.prompt_token_ids = [1, 2, 3]
        fake_choice = MagicMock()
        fake_choice.text = "Paris"
        fake_choice.token_ids = [4, 5]
        fake_out.outputs = [fake_choice]
        fake_metrics = MagicMock()
        fake_metrics.first_token_time = 0.5
        fake_metrics.arrival_time = 0.0
        fake_metrics.finished_time = 1.0
        fake_out.metrics = fake_metrics

        with unittest.mock.patch.dict(sys.modules, {"vllm": mock_vllm}):
            workload = LongMemEvalWorkload()
            mock_llm.generate.return_value = [fake_out] * len(workload.samples)
            result = workload.run(mock_llm, context_length=512, max_tokens=16)
            self.assertGreater(result.prompt_tokens, 0)
            self.assertIn("longmem_acc_overall", result.extra_summary)


class TestComparison(unittest.TestCase):
    def test_compare_runs(self):
        from src.core.compare import compare_runs, format_markdown_table

        base = {
            "technique": "baseline",
            "results": [
                {
                    "requested_ctx": 1024,
                    "status": "SUCCESS",
                    "peak_alloc_mb": 4000.0,
                    "ttft_sec": 1.0,
                    "throughput_tok_per_sec": 50.0,
                    "tpot_mean_ms": 20.0,
                    "memcpy_bytes_per_token": 100.0,
                }
            ],
        }
        target = {
            "technique": "flexgen_style",
            "results": [
                {
                    "requested_ctx": 1024,
                    "status": "SUCCESS",
                    "peak_alloc_mb": 2000.0,
                    "ttft_sec": 0.8,
                    "throughput_tok_per_sec": 60.0,
                    "tpot_mean_ms": 16.6,
                    "memcpy_bytes_per_token": 25.0,
                    "extra": {
                        "top1_agreement": 0.99,
                        "kl_div_mean": 0.005,
                        "longmem_acc_overall": 0.95,
                    },
                }
            ],
        }
        rows = compare_runs(base, target)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertAlmostEqual(r["vram_reduction_pct"], 50.0)
        self.assertAlmostEqual(r["pcie_reduction_pct"], 75.0)
        self.assertAlmostEqual(r["top1_agreement"], 0.99)
        self.assertAlmostEqual(r["longmem_acc_overall"], 0.95)

        table = format_markdown_table("flexgen_style", "baseline", rows)
        self.assertIn("50.0%", table)
        self.assertIn("75.0%", table)


if __name__ == "__main__":
    unittest.main()

