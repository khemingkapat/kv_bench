import os
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

    def test_validate_metrics(self):
        from src.core.metrics import validate_metrics

        valid_entry = ResultEntry(
            requested_ctx=1024,
            status="SUCCESS",
            prompt_tokens=500,
            generated_tokens=64,
            ttft_sec=0.25,
            decode_time_sec=0.5,
            total_time_sec=0.75,
            tpot_mean_ms=8.0,
            tpot_p95_ms=9.5,
            peak_alloc_mb=6500.0,
        )
        self.assertEqual(validate_metrics(valid_entry, max_vram_mb=8000.0), [])

        # VRAM cap exceeded
        errs = validate_metrics(valid_entry, max_vram_mb=6000.0)
        self.assertEqual(len(errs), 1)
        self.assertIn("exceeded cap", errs[0])

        # Negative timing
        invalid_entry = ResultEntry(
            requested_ctx=1024,
            status="SUCCESS",
            ttft_sec=-0.1,
            peak_alloc_mb=-5.0,
        )
        errs_neg = validate_metrics(invalid_entry)
        self.assertGreaterEqual(len(errs_neg), 2)

    def test_format_metrics_summary(self):
        from src.core.metrics import format_metrics_summary

        entry = ResultEntry(
            requested_ctx=1024,
            status="SUCCESS",
            prompt_tokens=1024,
            generated_tokens=64,
            ttft_sec=0.45,
            decode_time_sec=1.2,
            total_time_sec=1.65,
            throughput_tok_per_sec=52.5,
            tpot_mean_ms=19.0,
            tpot_p95_ms=21.5,
            peak_alloc_mb=5500.0,
            host_rss_delta_mb=250.0,
            cold_ttft_sec=0.85,
            ttft_p95_sec=0.55,
            memcpy_bytes_per_token=120.0,
        )
        summary = format_metrics_summary(entry)
        self.assertIn("peak VRAM 5500.0 MB", summary)
        self.assertIn("TTFT 0.450s", summary)
        self.assertIn("TPOT 19.0 ms (p95: 21.5 ms)", summary)
        self.assertIn("52.50 tok/s", summary)
        self.assertIn("host RSS Δ +250 MB", summary)
        self.assertIn("cold TTFT 0.850s → warm 0.450s", summary)
        self.assertIn("memcpy 120.0 B/tok", summary)

    def test_parse_memcpy_summary(self):
        import json
        import tempfile
        from src.core.harness import _parse_memcpy_summary

        trace_data = {
            "traceEvents": [
                {
                    "name": "Memcpy HtoD (Device)",
                    "cat": "gpu_memcpy",
                    "args": {"bytes": 4096, "memory bandwidth (GB/s)": 12.5},
                },
                {
                    "name": "Memcpy DtoH (Device)",
                    "cat": "gpu_memcpy",
                    "args": {"bytes": 2048, "memory bandwidth (GB/s)": 8.0},
                },
                {
                    "name": "KernelExecution",
                    "cat": "cuda_runtime",
                    "args": {"bytes": 9999},
                },
            ]
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(trace_data, f)
            temp_path = f.name

        summary = _parse_memcpy_summary(temp_path)
        self.assertEqual(summary["memcpy_htod_bytes"], 4096)
        self.assertEqual(summary["memcpy_dtoh_bytes"], 2048)
        self.assertEqual(summary["memcpy_peak_bw_gbs"], 12.5)


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

    def test_extract_step_logprobs(self):
        from unittest.mock import MagicMock
        from src.core.quality import extract_step_logprobs

        mock_out = MagicMock()
        mock_choice = MagicMock()
        mock_lp1 = MagicMock()
        mock_lp1.logprob = -0.25
        mock_lp2 = MagicMock()
        mock_lp2.logprob = -1.75
        mock_choice.logprobs = [{100: mock_lp1, 101: mock_lp2}]
        mock_out.outputs = [mock_choice]

        step_lps = extract_step_logprobs(mock_out)
        self.assertEqual(len(step_lps), 1)
        self.assertAlmostEqual(step_lps[0][100], -0.25)
        self.assertAlmostEqual(step_lps[0][101], -1.75)

    def test_quality_reference_persistence(self):
        import tempfile
        from src.core.quality import (
            evaluate_fidelity_against_reference,
            load_quality_reference,
            save_quality_reference,
        )

        ref_data = {
            "model": "qwen3-4b",
            "contexts": {
                "512": {
                    "token_ids": [10, 20, 30],
                    "step_logprobs": [{10: -0.1}, {20: -0.2}, {30: -0.3}],
                }
            },
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            temp_path = f.name

        save_quality_reference(temp_path, ref_data)
        loaded = load_quality_reference(temp_path)
        self.assertEqual(loaded["model"], "qwen3-4b")

        # In-memory target with int token IDs vs loaded reference (which was serialized to JSON strings)
        target_step_lps = [{10: -0.1}, {20: -0.2}, {30: -0.3}]
        fidelity = evaluate_fidelity_against_reference(
            loaded, 512, [10, 20, 30], target_step_logprobs=target_step_lps
        )
        self.assertAlmostEqual(fidelity["top1_agreement"], 1.0)
        self.assertAlmostEqual(fidelity["kl_div_mean"], 0.0, places=4)
        self.assertAlmostEqual(fidelity["kl_div_p99"], 0.0, places=4)

    def test_score_longmemeval(self):
        from src.core.quality import (
            compute_token_f1,
            extract_candidate_answers,
            score_longmemeval_dataset,
            score_longmemeval_sample,
        )

        self.assertTrue(score_longmemeval_sample("Paris is the capital of France.", "Paris"))
        self.assertFalse(score_longmemeval_sample("London is the capital of England.", "Paris"))
        self.assertTrue(score_longmemeval_sample("I don't know the answer based on the context.", "", is_unanswerable=True))
        self.assertFalse(score_longmemeval_sample("The answer is 42.", "", is_unanswerable=True))

        # Candidate extraction and robust matching
        cands = extract_candidate_answers("25 minutes and 50 seconds (or 25:50)")
        self.assertIn("25:50", cands)
        self.assertIn("25 minutes and 50 seconds", cands)
        self.assertTrue(score_longmemeval_sample("It took 25:50 to finish.", "25 minutes and 50 seconds (or 25:50)"))

        # Number word vs digit
        self.assertTrue(score_longmemeval_sample("I have four cats.", "4"))
        self.assertTrue(score_longmemeval_sample("I visited 3 doctors.", "three"))

        # Token F1
        f1_perfect = compute_token_f1("Paris France", "Paris France")
        self.assertEqual(f1_perfect, 1.0)
        f1_partial = compute_token_f1("The answer is Paris", "Paris")
        self.assertGreater(f1_partial, 0.0)

        samples = [
            {"task_type": "fact_recall", "expected_answer": "Paris", "unanswerable": False},
            {"task_type": "unanswerable_refusal", "expected_answer": "", "unanswerable": True},
        ]
        preds = ["Paris", "I don't know"]
        res = score_longmemeval_dataset(samples, preds)
        self.assertEqual(res["longmem_acc_overall"], 1.0)
        self.assertEqual(res["longmem_acc_fact_recall"], 1.0)
        self.assertEqual(res["longmem_acc_unanswerable_refusal"], 1.0)
        self.assertIn("longmem_f1_overall", res)
        self.assertGreater(res["longmem_f1_overall"], 0.0)


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

    def test_longmemeval_max_samples(self):
        workload = LongMemEvalWorkload(max_samples=5)
        self.assertEqual(len(workload.samples), 5)

    def test_longmemeval_samples_per_task_and_seed(self):
        # 1 sample per task with fixed seed
        w1 = LongMemEvalWorkload(samples_per_task=1, seed=42)
        self.assertEqual(len(w1.samples), 5)
        tasks1 = {s["task_type"] for s in w1.samples}
        expected_tasks = {
            "fact_recall",
            "cross_session_reasoning",
            "information_update",
            "relative_timeframe",
            "unanswerable_refusal",
        }
        self.assertEqual(tasks1, expected_tasks)

        # Exact same seed -> exact same sample IDs
        w2 = LongMemEvalWorkload(samples_per_task=1, seed=42)
        self.assertEqual([s["id"] for s in w1.samples], [s["id"] for s in w2.samples])

        # 2 samples per task
        w3 = LongMemEvalWorkload(samples_per_task=2, seed=42)
        self.assertEqual(len(w3.samples), 10)
        from collections import Counter
        counts = Counter(s["task_type"] for s in w3.samples)
        for t in expected_tasks:
            self.assertEqual(counts[t], 2)


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
            workload = LongMemEvalWorkload(max_samples=4)
            mock_llm.generate.return_value = [fake_out] * len(workload.samples)
            result = workload.run(mock_llm, context_length=512, max_tokens=16)
            self.assertGreater(result.prompt_tokens, 0)
            self.assertIn("longmem_acc_overall", result.extra_summary)
            self.assertGreaterEqual(result.tpot_mean_ms, 0.0)
            self.assertGreaterEqual(result.tpot_p95_ms, 0.0)



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
        self.assertIn("0.0050", table)

    def test_consolidate_concurrency_results(self):
        import json
        import tempfile
        from scripts.consolidate_concurrency_results import consolidate

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "concurrent_1users_16gb.json")
            file2 = os.path.join(tmpdir, "concurrent_4users_16gb.json")
            out_file = os.path.join(tmpdir, "concurrent_baseline_16gb.json")

            with open(file1, "w") as f:
                json.dump({
                    "model": "qwen3-4b",
                    "technique": "baseline",
                    "results": [{"requested_ctx": 1024, "status": "SUCCESS", "extra": {"num_users": 1}}],
                }, f)

            with open(file2, "w") as f:
                json.dump({
                    "model": "qwen3-4b",
                    "technique": "baseline",
                    "results": [{"requested_ctx": 1024, "status": "SUCCESS", "extra": {"num_users": 4}}],
                }, f)

            consolidate([os.path.join(tmpdir, "concurrent_*users_16gb.json")], out_file)

            with open(out_file, "r") as f:
                merged = json.load(f)

            self.assertEqual(merged["model"], "qwen3-4b")
            self.assertEqual(len(merged["concurrency_grid"]), 2)
            self.assertEqual(merged["concurrency_grid"][0]["num_users"], 1)
            self.assertEqual(merged["concurrency_grid"][1]["num_users"], 4)


if __name__ == "__main__":
    unittest.main()


