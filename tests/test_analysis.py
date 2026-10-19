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
        self.data = load_baseline_results("results")

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
