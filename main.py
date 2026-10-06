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
  uv run main.py --technique baseline --model Qwen/Qwen2.5-1.5B --contexts 512 1024 2048
  uv run main.py --technique flexgen_style --contexts 4096 --technique-args swap_space_gb=8
  uv run main.py --technique lmcache --contexts 2048 --profile
  uv run main.py --technique baseline --dry-run   # verify routing without a GPU
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
        "--output",
        default=None,
        help="Output JSON path (default: results/<technique>.json)",
    )
    args = parser.parse_args()

    technique_kwargs = parse_technique_args(args.technique_args)
    technique_cls_instance = get_technique(args.technique, **technique_kwargs)

    workload_override = None
    if args.workload == "filler":
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

    contexts = parse_contexts(args.contexts)
    output_path = args.output or f"results/{args.technique}.json"
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    run_result = run_benchmark(
        technique_cls_instance,
        args.model,
        contexts,
        max_tokens=args.max_tokens,
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

    if args.dry_run:
        # Use normal sys.exit in dry-run so CI gets a clean 0 and we don't
        # swallow real failures. os._exit(0) bypasses Python cleanup and would
        # mask any exception that happened before this point.
        sys.exit(0)

    # vLLM spawns background threads that can hang a clean process exit.
    os._exit(0)


if __name__ == "__main__":
    main()
