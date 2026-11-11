"""Unit tests for Phase 2 baseline profiling analysis and report generation."""

import os
import tempfile
import unittest

from src.analysis.report_generator import (
    EMPIRICAL_CLIFF_POOL_TOKENS,
    THEORETICAL_CLIFF_POOL_TOKENS,
    compute_theoretical_sweep,
    generate_baseline_markdown_report,
    generate_publication_figures,
    load_baseline_results,
    reconcile_hardware_gaps,
)


class TestAnalysisReport(unittest.TestCase):
    def setUp(self):
        results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
        hw_file = os.path.join(results_dir, "baseline_hardware_16gb.json")
        if os.path.exists(hw_file):
            self.data = load_baseline_results(results_dir)
        else:
            self.data = self._create_synthetic_fixture()

    @staticmethod
    def _create_synthetic_fixture():
        hw_results = [
            {"requested_ctx": 512, "status": "SUCCESS", "peak_alloc_mb": 7500.0, "peak_reserved_mb": 8000.0, "tpot_mean_ms": 10.5, "throughput_tok_per_sec": 95.0, "ttft_sec": 0.05},
            {"requested_ctx": 1024, "status": "SUCCESS", "peak_alloc_mb": 7600.0, "peak_reserved_mb": 8100.0, "tpot_mean_ms": 10.8, "throughput_tok_per_sec": 92.0, "ttft_sec": 0.06},
            {"requested_ctx": 2048, "status": "SUCCESS", "peak_alloc_mb": 7800.0, "peak_reserved_mb": 8300.0, "tpot_mean_ms": 11.2, "throughput_tok_per_sec": 89.0, "ttft_sec": 0.07},
            {"requested_ctx": 4096, "status": "SUCCESS", "peak_alloc_mb": 8200.0, "peak_reserved_mb": 8700.0, "tpot_mean_ms": 11.8, "throughput_tok_per_sec": 84.0, "ttft_sec": 0.09},
            {"requested_ctx": 8192, "status": "SUCCESS", "peak_alloc_mb": 9000.0, "peak_reserved_mb": 9500.0, "tpot_mean_ms": 12.5, "throughput_tok_per_sec": 80.0, "ttft_sec": 0.12},
            {"requested_ctx": 16384, "status": "SUCCESS", "peak_alloc_mb": 10600.0, "peak_reserved_mb": 11100.0, "tpot_mean_ms": 13.5, "throughput_tok_per_sec": 74.0, "ttft_sec": 0.20},
            {"requested_ctx": 32768, "status": "SUCCESS", "peak_alloc_mb": 13800.0, "peak_reserved_mb": 14300.0, "tpot_mean_ms": 16.0, "throughput_tok_per_sec": 62.0, "ttft_sec": 0.35},
            {"requested_ctx": 49152, "status": "FAILED", "error": "Requested context length is larger than the available KV cache memory pool."},
        ]
        n1_results = [
            {"requested_ctx": 512, "status": "SUCCESS", "tpot_mean_ms": 10.0, "throughput_tok_per_sec": 100.0},
            {"requested_ctx": 1024, "status": "SUCCESS", "tpot_mean_ms": 10.2, "throughput_tok_per_sec": 98.0},
            {"requested_ctx": 2048, "status": "SUCCESS", "tpot_mean_ms": 10.5, "throughput_tok_per_sec": 95.0},
            {"requested_ctx": 4096, "status": "SUCCESS", "tpot_mean_ms": 11.0, "throughput_tok_per_sec": 90.0},
        ]
        concurrency_grid = [
            {"num_users": 1, "results": n1_results},
            {"num_users": 4, "results": []},
            {"num_users": 8, "results": []},
            {"num_users": 12, "results": []},
            {"num_users": 16, "results": []},
        ]
        lm_results = [
            {"requested_ctx": 512, "status": "SUCCESS", "adherence_score": 0.95, "ttft_sec": 0.05, "tpot_mean_ms": 10.5},
            {"requested_ctx": 1024, "status": "SUCCESS", "adherence_score": 0.92, "ttft_sec": 0.08, "tpot_mean_ms": 11.0},
            {"requested_ctx": 2048, "status": "SUCCESS", "adherence_score": 0.90, "ttft_sec": 0.12, "tpot_mean_ms": 11.5},
            {"requested_ctx": 4096, "status": "SUCCESS", "adherence_score": 0.88, "ttft_sec": 0.20, "tpot_mean_ms": 12.0},
        ]
        return {
            "hardware": {"model": "qwen3-4b", "technique": "baseline", "axes": [], "results": hw_results},
            "concurrent": {"model": "qwen3-4b", "technique": "baseline", "concurrency_grid": concurrency_grid},
            "longmem": {"model": "qwen3-4b", "technique": "baseline", "axes": [], "results": lm_results},
            "verify_identity": {},
            "verify_cross_model": {},
        }

    def test_load_baseline_results_structure(self):
        self.assertIn("hardware", self.data)
        self.assertIn("concurrent", self.data)
        self.assertIn("longmem", self.data)

        # Hardware results
        hw_results = self.data["hardware"].get("results", [])
        self.assertGreaterEqual(len(hw_results), 7)
        self.assertEqual(self.data["hardware"]["model"], "qwen3-4b")

        # Concurrency grid
        grid = self.data["concurrent"].get("concurrency_grid", [])
        self.assertEqual(len(grid), 5)
        user_counts = [item["num_users"] for item in grid]
        self.assertEqual(user_counts, [1, 4, 8, 12, 16])

        # LongMem results
        lm_results = self.data["longmem"].get("results", [])
        self.assertEqual(len(lm_results), 4)

    def test_compute_theoretical_sweep(self):
        contexts = [512, 1024, 4096, 32768, 49152, 65536]
        sweep = compute_theoretical_sweep(contexts)
        self.assertEqual(len(sweep), len(contexts))

        # 512 context fits in pool
        row_512 = sweep[0]
        self.assertEqual(row_512["context"], 512)
        self.assertTrue(row_512["fits_pool"])
        self.assertTrue(row_512["fits_cap"])
        self.assertAlmostEqual(row_512["min_tpot_ms"], 8.07, delta=0.1)

        # 32K context fits in pool
        row_32k = next(r for r in sweep if r["context"] == 32768)
        self.assertTrue(row_32k["fits_pool"])
        self.assertTrue(row_32k["fits_cap"])
        self.assertAlmostEqual(row_32k["min_tpot_ms"], 12.79, delta=0.2)

        # 49K context exceeds pool but fits under 16GB process cap
        row_49k = next(r for r in sweep if r["context"] == 49152)
        self.assertFalse(row_49k["fits_pool"])
        self.assertTrue(row_49k["fits_cap"])

        # 65K context exceeds process cap
        row_65k = next(r for r in sweep if r["context"] == 65536)
        self.assertFalse(row_65k["fits_cap"])

    def test_reconcile_hardware_gaps(self):
        reconciled = reconcile_hardware_gaps(
            self.data["hardware"]["results"],
            self.data["concurrent"]["concurrency_grid"],
        )
        self.assertGreaterEqual(len(reconciled), 7)

        # Check successful row (512 ctx)
        r_512 = next(r for r in reconciled if r["context"] == 512)
        self.assertEqual(r_512["status"], "SUCCESS")
        self.assertGreater(r_512["bandwidth_efficiency_pct"], 65.0)
        self.assertLess(r_512["bandwidth_efficiency_pct"], 85.0)
        self.assertGreater(r_512["alloc_overhead_mb"], 0.0)

        # Check pure decode efficiency at 1024 ctx (~82.3%)
        r_1k = next(r for r in reconciled if r["context"] == 1024)
        self.assertIsNotNone(r_1k["pure_bandwidth_efficiency_pct"])
        self.assertGreater(r_1k["pure_bandwidth_efficiency_pct"], 80.0)

        # Check cliff point (32768 ctx)
        r_32k = next(r for r in reconciled if r["context"] == 32768)
        self.assertEqual(r_32k["status"], "SUCCESS")
        self.assertGreater(r_32k["bandwidth_efficiency_pct"], 70.0)

        # Check pool OOM row (49152 ctx)
        r_49k = next(r for r in reconciled if r["context"] == 49152)
        self.assertEqual(r_49k["status"], "FAILED")
        self.assertIn("larger than the available KV cache memory", r_49k["error"])

    def test_generate_figures_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fig_dir = os.path.join(tmp_dir, "figures")
            report_file = os.path.join(tmp_dir, "REPORT.md")
            hub_fig_dir = os.path.join(tmp_dir, "hub_figures")
            hub_report_file = os.path.join(tmp_dir, "HUB_REPORT.md")

            # Generate figures
            saved_figs = generate_publication_figures(self.data, output_dir=fig_dir, hub_figures_dir=hub_fig_dir)
            self.assertEqual(len(saved_figs), 8)  # 4 figures * 2 (png + svg)

            for fpath in saved_figs:
                self.assertTrue(os.path.exists(fpath))
                self.assertGreater(os.path.getsize(fpath), 1000)

            # Reconcile and generate report
            reconciled = reconcile_hardware_gaps(
                self.data["hardware"]["results"],
                self.data["concurrent"]["concurrency_grid"],
            )
            content = generate_baseline_markdown_report(
                self.data,
                reconciled,
                output_path=report_file,
                hub_report_path=hub_report_file,
            )

            self.assertTrue(os.path.exists(report_file))
            self.assertTrue(os.path.exists(hub_report_file))
            self.assertIn(f"{THEORETICAL_CLIFF_POOL_TOKENS:,}", content)
            self.assertIn(f"{EMPIRICAL_CLIFF_POOL_TOKENS:,}", content)
            self.assertIn("Decode Memory Bandwidth Roofline Efficiency", content)
            self.assertIn("Multi-User Concurrency Grid Reconciliation", content)
            self.assertIn("Quality Ground-Truth & Logit Fidelity Calibration", content)


if __name__ == "__main__":
    unittest.main()
