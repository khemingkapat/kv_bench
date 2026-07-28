import argparse
import json
import os
from src.benchmark_baseline import run_benchmark as run_baseline
from src.benchmark_lmcache import run_lmcache_benchmark as run_lmcache

def main():
    parser = argparse.ArgumentParser(description="KV Cache Benchmark Runner")
    parser.add_argument("--mode", choices=["baseline", "lmcache"], required=True, help="Benchmark mode to run")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B", help="Model name on HuggingFace")
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192], help="Context lengths to test")
    parser.add_argument("--output", type=str, help="Output JSON file path (defaults to results/<mode>_results.json)")
    args = parser.parse_args()

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    output_file = args.output
    if not output_file:
        output_file = f"results/{args.mode}_results.json"

    if args.mode == "baseline":
        results = run_baseline(args.model, args.contexts)
    elif args.mode == "lmcache":
        # Required environment setup for LMCache
        os.environ["LMCACHE_CONFIG_FILE"] = "src/config/lmcache_config.yaml"
        results = run_lmcache(args.model, args.contexts)
    
    # Wrap the results with metadata
    output_data = {
        "model": args.model,
        "mode": args.mode,
        "results": results
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved {args.mode} benchmark results to {output_file}")
    
    # vLLM often leaves background threads running that prevent clean exit.
    # We forcefully terminate the process here since we've already saved our data.
    os._exit(0)

if __name__ == "__main__":
    main()
