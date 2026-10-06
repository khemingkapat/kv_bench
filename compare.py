#!/usr/bin/env python3
"""CLI utility to compare benchmark run results against the uncompressed baseline.

Usage:
  python3 compare.py results/baseline.json results/flexgen_style.json
"""

import argparse
import json
import sys

from src.core.compare import compare_runs, format_markdown_table


def main():
    parser = argparse.ArgumentParser(
        description="Compare KV-cache technique results against baseline (Section 3.3 in proposal.md)"
    )
    parser.add_argument("baseline", help="Path to baseline result JSON (e.g. results/baseline_16gb.json)")
    parser.add_argument("target", help="Path to target technique result JSON (e.g. results/flexgen_style_16gb.json)")
    parser.add_argument("--markdown", action="store_true", default=True, help="Print as Markdown table")

    args = parser.parse_args()

    try:
        with open(args.baseline, "r") as f:
            base_data = json.load(f)
    except Exception as e:
        print(f"Error reading baseline file {args.baseline}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.target, "r") as f:
            target_data = json.load(f)
    except Exception as e:
        print(f"Error reading target file {args.target}: {e}", file=sys.stderr)
        sys.exit(1)

    rows = compare_runs(base_data, target_data)
    if not rows:
        print("No matching context lengths found between runs.", file=sys.stderr)
        sys.exit(1)

    base_name = base_data.get("technique", "baseline")
    target_name = target_data.get("technique", "target")

    md_table = format_markdown_table(target_name, base_name, rows)
    print("\n" + md_table + "\n")


if __name__ == "__main__":
    main()
