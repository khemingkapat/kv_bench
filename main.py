import argparse
import json
import os
import sys

from src.benchmark_baseline import run_benchmark as run_baseline
from src.benchmark_lmcache import run_lmcache_benchmark as run_lmcache
from src.benchmark_profiling import run_profiling
from src.benchmark_disaggregation import run_sequential_disaggregation
from src.benchmark_flexgen import run_flexgen_benchmark as run_flexgen

def parse_contexts(ctx_input):
    """
    Robustly parses context length inputs in multiple formats:
    - Comma-separated string: "512,1024,2048"
    - Space-separated items from nargs='+': ["512", "1024", "2048"]
    - Pre-parsed integer list: [512, 1024, 2048]
    """
    if isinstance(ctx_input, list):
        raw = ",".join(str(item) for item in ctx_input)
    else:
        raw = str(ctx_input)
    
    cleaned = raw.replace(",", " ").split()
    return [int(c) for c in cleaned if c.strip()]

DEFAULT_OUTPUTS = {
    "baseline": "results/baseline_phase1.json",
    "lmcache": "results/lmcache_phase2.json",
    "profiling": "results/profiler_trace_phase3/trace.json",
    "disaggregation": "results/disaggregation_phase4.json",
    "flexgen": "results/flexgen.json",
}

def main():
    parser = argparse.ArgumentParser(description="KV Cache Benchmark & Profiling Router")
    parser.add_argument(
        "--mode",
        choices=["baseline", "lmcache", "profiling", "disaggregation", "flexgen"],
        required=True,
        help="Benchmark mode to run"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B",
        help="HuggingFace model identifier or local path"
    )
    parser.add_argument(
        "--contexts",
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192],
        help="Context lengths to test (comma-separated e.g. 512,1024 or space-separated)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (defaults to results/<mode>_phaseX.json)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/lmcache_config.yaml",
        help="LMCache configuration YAML file path"
    )

    args = parser.parse_args()

    contexts = parse_contexts(args.contexts)

    # Determine output path according to interface contracts
    output_file = args.output
    if not output_file:
        output_file = DEFAULT_OUTPUTS.get(args.mode, f"results/{args.mode}_results.json")

    # Ensure parent output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if args.mode == "baseline":
        results = run_baseline(args.model, contexts)
    elif args.mode == "lmcache":
        os.environ["LMCACHE_CONFIG_FILE"] = args.config
        results = run_lmcache(args.model, contexts)
    elif args.mode == "profiling":
        os.environ["LMCACHE_CONFIG_FILE"] = args.config
        results = []
        for ctx in contexts:
            results.extend(run_profiling(args.model, ctx))
    elif args.mode == "disaggregation":
        results = []
        for ctx in contexts:
            results.extend(run_sequential_disaggregation(args.model, ctx))
    elif args.mode == "flexgen":
        results = run_flexgen(args.model, contexts)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    # Standardize top-level JSON structure
    if isinstance(results, dict) and "results" in results:
        output_data = results
    else:
        output_data = {
            "model": args.model,
            "mode": args.mode,
            "results": results
        }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    
    print(f"\nSaved {args.mode} benchmark results to {output_file}")
    
    # Clean exit to prevent vLLM background thread hangs
    os._exit(0)

if __name__ == "__main__":
    main()

