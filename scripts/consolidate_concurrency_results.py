#!/usr/bin/env python3
"""Consolidate multi-user concurrency benchmark JSONs into a single unified grid file.

Usage:
    uv run python scripts/consolidate_concurrency_results.py \
        --inputs results/concurrent_*users_16gb.json \
        --output results/concurrent_baseline_16gb.json
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path


def consolidate(input_paths: list[str], output_path: str) -> None:
    all_files = []
    for pattern in input_paths:
        expanded = glob.glob(pattern)
        if expanded:
            all_files.extend(expanded)
        elif os.path.isfile(pattern):
            all_files.append(pattern)

    all_files = sorted(set(all_files))
    if not all_files:
        print("No concurrency result files found to consolidate.", file=sys.stderr)
        return

    merged_grid = []
    model = "unknown"
    technique = "unknown"

    for fpath in all_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to read {fpath}: {e}", file=sys.stderr)
            continue

        model = data.get("model", model)
        technique = data.get("technique", technique)
        results = data.get("results", [])

        # Determine num_users from extra or filename
        num_users = None
        for r in results:
            if "extra" in r and "num_users" in r["extra"]:
                num_users = r["extra"]["num_users"]
                break

        if num_users is None:
            # Fallback parse from filename like concurrent_4users_16gb.json
            base = os.path.basename(fpath)
            parts = base.split("users")[0].split("_")
            try:
                num_users = int(parts[-1])
            except ValueError:
                num_users = 1

        merged_grid.append({
            "num_users": num_users,
            "source_file": os.path.basename(fpath),
            "results": results,
        })

    merged_grid = sorted(merged_grid, key=lambda x: x["num_users"])

    unified_output = {
        "model": model,
        "technique": technique,
        "concurrency_grid": merged_grid,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(unified_output, f, indent=2)

    print(f"Consolidated {len(merged_grid)} concurrency result sets into {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate multi-user concurrency result files")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input file paths or globs")
    parser.add_argument("--output", required=True, help="Output consolidated JSON path")
    args = parser.parse_args()

    consolidate(args.inputs, args.output)


if __name__ == "__main__":
    main()
