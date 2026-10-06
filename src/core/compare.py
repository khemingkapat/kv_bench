"""Comparison and reporting module (Section 3.3.1 and 3.3.2 in proposal.md).

Compares a target technique's run results against the uncompressed baseline:
1. VRAM reduction (% vs baseline)
2. TTFT speedup / delta
3. TPOT (mean & P95 jitter)
4. PCIe transfer volume reduction (% vs baseline)
5. Model quality & LongMemEval accuracy delta
"""

from __future__ import annotations

import json
from typing import Any


def compare_runs(baseline_data: dict[str, Any], target_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare per-context results of target technique against baseline."""
    base_results = {r["requested_ctx"]: r for r in baseline_data.get("results", []) if r.get("status") == "SUCCESS" or r.get("status") == "DRY_RUN"}
    target_results = {r["requested_ctx"]: r for r in target_data.get("results", []) if r.get("status") == "SUCCESS" or r.get("status") == "DRY_RUN"}

    common_contexts = sorted(set(base_results.keys()) & set(target_results.keys()))
    comparison_rows = []

    for ctx in common_contexts:
        b = base_results[ctx]
        t = target_results[ctx]

        # VRAM reduction
        b_vram = b.get("peak_alloc_mb", 0.0)
        t_vram = t.get("peak_alloc_mb", 0.0)
        vram_reduct_pct = round(((b_vram - t_vram) / b_vram * 100.0), 2) if b_vram > 0 else 0.0

        # Throughput
        b_tp = b.get("throughput_tok_per_sec", 0.0)
        t_tp = t.get("throughput_tok_per_sec", 0.0)

        # TTFT
        b_ttft = b.get("ttft_sec", 0.0)
        t_ttft = t.get("ttft_sec", 0.0)

        # TPOT
        t_tpot = t.get("tpot_mean_ms", 0.0)
        t_tpot_p95 = t.get("tpot_p95_ms", 0.0)

        # PCIe Transfer
        b_pcie = b.get("memcpy_bytes_per_token", 0.0)
        t_pcie = t.get("memcpy_bytes_per_token", 0.0)
        pcie_reduct_pct = round(((b_pcie - t_pcie) / b_pcie * 100.0), 2) if b_pcie > 0 else 0.0

        # Quality metrics (if present)
        t_extra = t.get("extra", {})
        top1 = t_extra.get("top1_agreement", None)
        kl_mean = t_extra.get("kl_div_mean", None)
        longmem_acc = t_extra.get("longmem_acc_overall", None)

        comparison_rows.append({
            "ctx": ctx,
            "baseline_vram_mb": b_vram,
            "target_vram_mb": t_vram,
            "vram_reduction_pct": vram_reduct_pct,
            "baseline_ttft_sec": b_ttft,
            "target_ttft_sec": t_ttft,
            "baseline_tp": b_tp,
            "target_tp": t_tp,
            "tpot_mean_ms": t_tpot,
            "tpot_p95_ms": t_tpot_p95,
            "baseline_pcie_bytes_per_tok": b_pcie,
            "target_pcie_bytes_per_tok": t_pcie,
            "pcie_reduction_pct": pcie_reduct_pct,
            "top1_agreement": top1,
            "kl_div_mean": kl_mean,
            "longmem_acc_overall": longmem_acc,
        })

    return comparison_rows


def format_markdown_table(
    target_name: str,
    baseline_name: str,
    rows: list[dict[str, Any]],
) -> str:
    """Format comparison rows into a clean GitHub Flavored Markdown table."""
    lines = [
        f"### Comparative Evaluation: `{target_name}` vs. `{baseline_name}`",
        "",
        "| Context | Base VRAM (MB) | Opt VRAM (MB) | VRAM Reduct (%) | Base TTFT (s) | Opt TTFT (s) | TP (tok/s) | TPOT (ms) | PCIe Reduct (%) | Top-1 Match | LongMem Acc |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in rows:
        vram_red = f"{r['vram_reduction_pct']:+.1f}%" if r['vram_reduction_pct'] != 0 else "-"
        pcie_red = f"{r['pcie_reduction_pct']:+.1f}%" if r['pcie_reduction_pct'] != 0 else "-"
        top1 = f"{r['top1_agreement']*100:.1f}%" if r['top1_agreement'] is not None else "-"
        longmem = f"{r['longmem_acc_overall']*100:.1f}%" if r['longmem_acc_overall'] is not None else "-"

        lines.append(
            f"| {r['ctx']} | {r['baseline_vram_mb']:.1f} | {r['target_vram_mb']:.1f} | {vram_red} | "
            f"{r['baseline_ttft_sec']:.3f} | {r['target_ttft_sec']:.3f} | {r['target_tp']:.1f} | "
            f"{r['tpot_mean_ms']:.1f} | {pcie_red} | {top1} | {longmem} |"
        )

    return "\n".join(lines)
