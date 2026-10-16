import argparse
import json
import os
import sys

from src.core.config import BenchmarkConfig
from src.core.harness import run_benchmark
from src.core.registry import get_technique, list_techniques


def parse_contexts(raw) -> list[int]:
    """Accepts space-separated (argparse nargs='+') or comma-separated input."""
    if isinstance(raw, list):
        joined = ",".join(str(c) for c in raw)
    else:
        joined = str(raw)
    return [int(c) for c in joined.replace(",", " ").split() if c.strip()]


def parse_technique_args(raw: list[str]) -> dict:
    """Parse --technique-args key=val key=val into a dict.

    Values are cast to int or float if possible, otherwise kept as str.
    This lets you tune technique parameters from the CLI without touching
    the technique file -- e.g. --technique-args swap_space_gb=8.

    All Technique.__init__ parameters MUST have defaults (zero-arg convention)
    so the registry can instantiate any technique without required arguments.
    These overrides are then applied on top of the defaults.
    """
    result = {}
    for item in raw or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"--technique-args items must be key=value, got: {item!r}"
            )
        k, v = item.split("=", 1)
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                pass
        result[k] = v
    return result


def main() -> None:
    defaults = BenchmarkConfig()
    available = list_techniques()

    parser = argparse.ArgumentParser(
        description="Unified KV-cache technique benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run main.py --technique baseline --model qwen3-4b --contexts 512 1024 2048
  uv run main.py --technique baseline --model phi4-mini --dry-run
  uv run main.py --technique flexgen_style --contexts 4096 --technique-args swap_space_gb=8
  uv run main.py --technique lmcache --contexts 2048 --profile
""",
    )
    parser.add_argument("--technique", required=True, help=f"one of: {', '.join(available)}")
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--contexts", nargs="+", default=defaults.contexts)
    parser.add_argument("--max-tokens", type=int, default=defaults.max_tokens)
    parser.add_argument("--gpu-utilization", type=float, default=defaults.gpu_utilization)
    parser.add_argument(
        "--technique-args",
        nargs="*",
        metavar="KEY=VAL",
        help="Override technique __init__ kwargs, e.g. swap_space_gb=8 cpu_offload_gb=4",
    )
    parser.add_argument(
        "--workload",
        choices=["default", "filler", "coldwarm", "concurrent", "longmem"],
        default="default",
        help="Workload pattern: default (use technique default), filler (single prompt), coldwarm (prefix reuse), concurrent (10-15 simulated users), longmem (LongMemEval adherence)",
    )
    parser.add_argument(
        "--num-users",
        type=int,
        default=10,
        help="Number of simulated concurrent users for --workload concurrent (default: 10)",
    )
    parser.add_argument(
        "--max-vram-gb",
        type=float,
        default=16.0,
        help="Simulate a maximum VRAM ceiling in GB (default: 16.0 for consumer GPU ceiling on 24GB RTX 4090)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Wrap workload in torch.profiler; saves Chrome trace and extracts memcpy stats into results",
    )
    parser.add_argument(
        "--profile-dir",
        default="results/traces",
        help="Directory for profiler trace files (default: results/traces)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM() construction -- verifies technique registration and routing without a GPU",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Fast verification mode: run a single short prompt (128 ctx, 16 tokens) to verify CUDA memory capping and clean engine initialization",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: results/<technique>.json or results/verify_<technique>.json)",
    )
    args = parser.parse_args()

    technique_kwargs = parse_technique_args(args.technique_args)
    technique_cls_instance = get_technique(args.technique, **technique_kwargs)

    workload_override = None
    if args.workload == "filler" or (args.verify and args.workload == "default"):
        from src.core.workload import SyntheticFiller
        workload_override = SyntheticFiller()
    elif args.workload == "coldwarm":
        from src.core.workload import ColdWarm
        workload_override = ColdWarm()
    elif args.workload == "concurrent":
        from src.core.workload import ConcurrentChatWorkload
        workload_override = ConcurrentChatWorkload(num_users=args.num_users)
    elif args.workload == "longmem":
        from src.workloads.memory_adherence import LongMemEvalWorkload
        workload_override = LongMemEvalWorkload()

    if args.verify:
        contexts = [128]
        max_tokens = 16
        default_output = f"results/verify_{args.technique}.json"
    else:
        contexts = parse_contexts(args.contexts)
        max_tokens = args.max_tokens
        default_output = f"results/{args.technique}.json"

    output_path = args.output or default_output
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    run_result = run_benchmark(
        technique_cls_instance,
        args.model,
        contexts,
        max_tokens=max_tokens,
        gpu_utilization=args.gpu_utilization,
        profile=args.profile,
        profile_dir=args.profile_dir,
        dry_run=args.dry_run,
        workload_override=workload_override,
        max_vram_gb=args.max_vram_gb,
    )

    with open(output_path, "w") as f:
        json.dump(run_result.to_dict(), f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())

    print(f"\nSaved {args.technique} results -> {output_path}")

    exit_code = 0
    if args.verify:
        failed_entries = [r for r in run_result.results if r.status == "FAILED"]
        if failed_entries:
            print(f"[VERIFY FAILED] Engine failed on {len(failed_entries)} context(s): {failed_entries[0].error}", file=sys.stderr)
            exit_code = 1
        elif not args.dry_run:
            entry = run_result.results[0]
            vram_cap_mb = (args.max_vram_gb * 1024.0) if args.max_vram_gb else float("inf")
            if entry.generated_tokens <= 0:
                print("[VERIFY FAILED] Engine generated 0 tokens.", file=sys.stderr)
                exit_code = 1
            elif entry.peak_alloc_mb > vram_cap_mb:
                print(
                    f"[VERIFY FAILED] Peak VRAM {entry.peak_alloc_mb:.1f} MB exceeded cap {vram_cap_mb:.1f} MB.",
                    file=sys.stderr,
                )
                exit_code = 1
            else:
                print(
                    f"[VERIFY OK] Clean engine init | generated {entry.generated_tokens} tokens | "
                    f"peak VRAM {entry.peak_alloc_mb:.1f} MB <= {vram_cap_mb:.1f} MB cap"
                )
        else:
            print(f"[VERIFY OK] Dry-run verification passed (contexts={contexts}, max_tokens={max_tokens}).")
    elif run_result.results and all(r.status == "FAILED" for r in run_result.results):
        print("[BENCHMARK FAILED] All context lengths failed.", file=sys.stderr)
        exit_code = 1

    sys.stdout.flush()
    sys.stderr.flush()

    if args.dry_run:
        # Use normal sys.exit in dry-run so CI gets a clean exit and we don't
        # swallow real failures. os._exit() bypasses Python cleanup.
        sys.exit(exit_code)

    # vLLM spawns background threads that can hang a clean process exit.
    os._exit(exit_code)


if __name__ == "__main__":
    main()
