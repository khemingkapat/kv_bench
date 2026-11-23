#!/usr/bin/env python3
"""Decoupled Offline Grading Utility for LongMemEval Results.

Separates GPU inference (`results/*_longmem_*.json`, kept strictly immutable) from
semantic grading by any evaluator (human colleague, coding agent, or external frontier LLM):

1. `export`: Reads one or more `*_longmem_*.json` files and writes a clean CSV or JSONL
   review sheet with `id`, `task_type`, `prompt_tokens`, `unanswerable`, `question`,
   `expected_answer`, and each technique's `prediction` + `finish_reason`.
2. `apply`: Joins a raw `*_longmem_*.json` with an offline `*_grades.json` sidecar
   (`{sample_id: {"correct": bool, "reason": str, "grader": str}}`) and optionally a
   baseline `*_grades.json` sidecar, writing an immutable `*_judged.json` summary with:
   - overall & per-capability accuracy
   - exact-token-length per-sample records (ready for continuous plotting)
   - baseline-controlled causal verdicts (`PASS_FIDELITY_MAINTAINED`, `QUALITY_REGRESSION`,
     `BASELINE_LIMITATION`, `UNEXPECTED_GAIN`)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any


def _load_sample_records(result_json_path: str, ctx: int | None = None) -> tuple[str, int, str, list[dict[str, Any]]]:
    """Load `(technique, requested_ctx, dataset_sha256, sample_records)` from a result JSON."""
    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    technique = data.get("technique", os.path.basename(result_json_path))
    results = data.get("results", [])
    if not results:
        raise ValueError(f"No results entries found in {result_json_path}")

    entry = None
    if ctx is not None:
        entry = next((r for r in results if r.get("requested_ctx") == ctx), None)
        if entry is None:
            raise ValueError(f"Requested context {ctx} not found in {result_json_path}")
    else:
        entry = results[-1]

    extra = entry.get("extra", {})
    records = extra.get("sample_records")
    if not isinstance(records, list) or not records:
        raise ValueError(
            f"{result_json_path} (ctx={entry.get('requested_ctx')}) does not contain `sample_records`. "
            "Re-run LongMemEval with the updated harness."
        )
    return technique, int(entry.get("requested_ctx", 0)), str(extra.get("dataset_sha256", "")), records


def export_review_sheet(
    input_paths: list[str],
    output_path: str,
    ctx: int | None = None,
    fmt: str = "csv",
) -> int:
    """Export one or more LongMemEval result JSONs into a clean CSV/JSONL review sheet."""
    by_id: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []
    tech_names: list[str] = []

    for path in input_paths:
        tech, _, _, records = _load_sample_records(path, ctx=ctx)
        if tech not in tech_names:
            tech_names.append(tech)
        for rec in records:
            sid = str(rec["id"])
            if sid not in by_id:
                ordered_ids.append(sid)
                by_id[sid] = {
                    "id": sid,
                    "task_type": rec.get("task_type", ""),
                    "unanswerable": rec.get("unanswerable", False),
                    "prompt_tokens": rec.get("prompt_tokens", 0),
                    "question": rec.get("question", ""),
                    "expected_answer": rec.get("expected_answer", ""),
                }
            by_id[sid][f"{tech}_prediction"] = rec.get("prediction", "")
            by_id[sid][f"{tech}_finish_reason"] = rec.get("finish_reason", "")
            by_id[sid][f"{tech}_generated_tokens"] = rec.get("generated_tokens", 0)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    rows = [by_id[sid] for sid in ordered_ids]

    if fmt == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        fieldnames = ["id", "task_type", "unanswerable", "prompt_tokens", "question", "expected_answer"]
        for tech in tech_names:
            fieldnames.extend([f"{tech}_prediction", f"{tech}_finish_reason", f"{tech}_generated_tokens"])
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return len(rows)


def _load_grades_map(grades_path: str) -> dict[str, dict[str, Any]]:
    """Load `{sample_id: {"correct": bool, ...}}` from a JSON object file."""
    with open(grades_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "grades" in raw and isinstance(raw["grades"], dict):
        raw = raw["grades"]
    if not isinstance(raw, dict):
        raise ValueError(f"Grades file {grades_path} must be a JSON object mapping sample_id -> grade dict.")
    return raw


def _causal_verdict(tech_correct: bool, base_correct: bool | None) -> str:
    if base_correct is None:
        return "PASS" if tech_correct else "FAIL"
    if base_correct and tech_correct:
        return "PASS_FIDELITY_MAINTAINED"
    if base_correct and not tech_correct:
        return "QUALITY_REGRESSION"
    if not base_correct and not tech_correct:
        return "BASELINE_LIMITATION"
    return "UNEXPECTED_GAIN"


def apply_grades(
    result_path: str,
    grades_path: str,
    output_path: str,
    baseline_grades_path: str | None = None,
    ctx: int | None = None,
) -> dict[str, Any]:
    """Combine immutable raw predictions with offline semantic grades into a `*_judged.json` sidecar."""
    technique, resolved_ctx, dataset_sha256, records = _load_sample_records(result_path, ctx=ctx)
    grades_map = _load_grades_map(grades_path)
    base_grades_map = _load_grades_map(baseline_grades_path) if baseline_grades_path else None

    per_task_total: dict[str, int] = defaultdict(int)
    per_task_correct: dict[str, int] = defaultdict(int)
    verdict_counts: dict[str, int] = defaultdict(int)
    judged_samples: list[dict[str, Any]] = []

    missing_ids: list[str] = []
    for rec in records:
        sid = str(rec["id"])
        if sid not in grades_map:
            missing_ids.append(sid)
            continue
        g = grades_map[sid]
        is_corr = bool(g["correct"]) if isinstance(g, dict) else bool(g)
        reason = g.get("reason", "") if isinstance(g, dict) else ""
        grader = g.get("grader", "offline") if isinstance(g, dict) else "offline"

        base_corr: bool | None = None
        if base_grades_map is not None and sid in base_grades_map:
            bg = base_grades_map[sid]
            base_corr = bool(bg["correct"]) if isinstance(bg, dict) else bool(bg)

        verdict = _causal_verdict(is_corr, base_corr)
        verdict_counts[verdict] += 1

        task = str(rec.get("task_type", "unknown"))
        per_task_total[task] += 1
        if is_corr:
            per_task_correct[task] += 1

        judged_samples.append({
            "id": sid,
            "task_type": task,
            "unanswerable": rec.get("unanswerable", False),
            "prompt_tokens": rec.get("prompt_tokens", 0),
            "generated_tokens": rec.get("generated_tokens", 0),
            "finish_reason": rec.get("finish_reason", "unknown"),
            "correct": is_corr,
            "baseline_correct": base_corr,
            "causal_verdict": verdict,
            "reason": reason,
            "grader": grader,
            "diagnostic": rec.get("diagnostic"),
        })

    if missing_ids:
        raise ValueError(f"Grades file '{grades_path}' is missing {len(missing_ids)} sample IDs (e.g. {missing_ids[:5]}).")

    total = len(judged_samples)
    correct = sum(1 for s in judged_samples if s["correct"])
    per_task_acc = {
        t: round(per_task_correct[t] / per_task_total[t], 4)
        for t in sorted(per_task_total.keys())
        if per_task_total[t] > 0
    }

    summary = {
        "technique": technique,
        "requested_ctx": resolved_ctx,
        "source_result": os.path.relpath(result_path),
        "source_grades": os.path.relpath(grades_path),
        "dataset_sha256": dataset_sha256,
        "total_samples": total,
        "correct_samples": correct,
        "overall_accuracy": round(correct / total, 4) if total > 0 else 0.0,
        "per_task_accuracy": per_task_acc,
        "per_task_counts": dict(sorted(per_task_total.items())),
        "causal_verdict_counts": dict(sorted(verdict_counts.items())),
        "samples": judged_samples,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline LongMemEval grading export/apply utility.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_exp = sub.add_parser("export", help="Export raw LongMemEval result JSON(s) to CSV/JSONL review sheet.")
    p_exp.add_argument("--inputs", nargs="+", required=True, help="One or more *_longmem_*.json result files.")
    p_exp.add_argument("--output", required=True, help="Output CSV or JSONL path.")
    p_exp.add_argument("--ctx", type=int, default=None, help="Optional context length filter (defaults to last entry).")
    p_exp.add_argument("--format", choices=["csv", "jsonl"], default="csv", help="Output format (default: csv).")

    p_app = sub.add_parser("apply", help="Apply offline grades sidecar to raw result JSON -> *_judged.json.")
    p_app.add_argument("--result", required=True, help="Raw immutable *_longmem_*.json file.")
    p_app.add_argument("--grades", required=True, help="Offline grades JSON mapping sample_id -> {correct, reason, grader}.")
    p_app.add_argument("--baseline-grades", default=None, help="Optional baseline grades JSON for causal attribution.")
    p_app.add_argument("--output", required=True, help="Output *_judged.json sidecar path.")
    p_app.add_argument("--ctx", type=int, default=None, help="Optional context length filter (defaults to last entry).")

    args = parser.parse_args()
    if args.cmd == "export":
        n = export_review_sheet(args.inputs, args.output, ctx=args.ctx, fmt=args.format)
        print(f"Exported {n} LongMemEval samples -> {args.output}")
    elif args.cmd == "apply":
        summary = apply_grades(
            args.result,
            args.grades,
            args.output,
            baseline_grades_path=args.baseline_grades,
            ctx=args.ctx,
        )
        print(
            f"Wrote judged sidecar -> {args.output} "
            f"({summary['correct_samples']}/{summary['total_samples']} = {summary['overall_accuracy']*100:.1f}%)"
        )


if __name__ == "__main__":
    main()
