#!/usr/bin/env python3
"""Theoretical Capacity & Roofline Modeling for KV-Cache Optimization.

Analytically models:
1. GQA KV-cache memory footprint per token and total allocation.
2. Usable KV headroom and theoretical Out-Of-Memory (OOM) context cliffs under
   consumer hardware VRAM constraints (e.g. 16.0 GB on RTX 4090, 8.0 GB on RTX 4060).
3. Roofline latency & throughput bounds:
   - Decode phase: Memory bandwidth bound (TPOT min, max tokens/sec).
   - Prefill phase: Compute bound (TTFT min).
4. Analytical empirical test matrix categorizing test points into:
   - Sub-critical (weight-dominated)
   - Inflection (memory bus saturation begins)
   - Cliff edge (approaching VRAM limit)
   - Over-the-cliff (deterministic OOM verification)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import MODEL_REGISTRY, ModelSpec, get_model_spec


# Hardware presets
HARDWARE_PRESETS = {
    "rtx4090_16gb": {
        "name": "NVIDIA GeForce RTX 4090 (16GB Cap)",
        "vram_cap_gb": 16.0,
        "runtime_overhead_gb": 0.80,
        "memory_bw_gbs": 1008.0,
        "tflops_bf16": 82.6,
        "pcie_empirical_bw_gbs": 25.5,
    },
    "rtx4060_8gb": {
        "name": "NVIDIA GeForce RTX 4060 (8GB Local)",
        "vram_cap_gb": 8.0,
        "runtime_overhead_gb": 0.60,
        "memory_bw_gbs": 272.0,
        "tflops_bf16": 18.0,
        "pcie_empirical_bw_gbs": 14.0,
    },
}


@dataclass
class MemoryBreakdown:
    context_len: int
    num_users: int
    weights_gb: float
    runtime_overhead_gb: float
    kv_cache_gb: float
    total_memory_gb: float
    vram_cap_gb: float
    headroom_remaining_gb: float
    is_oom: bool
    kv_fraction_of_total: float


@dataclass
class RooflinePoint:
    context_len: int
    batch_size: int
    kv_cache_bytes: int
    total_bytes_read_per_step: int
    decode_step_latency_ms: float
    decode_throughput_tok_per_sec: float
    prefill_flops: float
    prefill_min_ttft_sec: float
    decode_slowdown_vs_zero_ctx_pct: float


def calculate_kv_bytes_per_token(spec: ModelSpec) -> int:
    """Calculate KV cache size in bytes per token for BF16 precision.

    Equation: 4 * num_full_attn_layers * num_kv_heads * head_dim
    (2 bytes per elem * 2 for Key and Value).
    """
    return spec.kv_bytes_per_token_bf16


def calculate_memory_breakdown(
    spec: ModelSpec,
    context_len: int,
    num_users: int = 1,
    vram_cap_gb: float = 16.0,
    runtime_overhead_gb: float = 0.80,
) -> MemoryBreakdown:
    """Calculate detailed memory footprint for a given model, context length, and concurrency."""
    kv_bytes_per_tok = calculate_kv_bytes_per_token(spec)
    total_kv_bytes = kv_bytes_per_tok * context_len * num_users
    kv_cache_gb = total_kv_bytes / (1024**3)

    weights_gb = spec.weight_size_gb
    total_memory_gb = weights_gb + runtime_overhead_gb + kv_cache_gb
    headroom = vram_cap_gb - total_memory_gb
    is_oom = total_memory_gb > vram_cap_gb
    kv_fraction = kv_cache_gb / total_memory_gb if total_memory_gb > 0 else 0.0

    return MemoryBreakdown(
        context_len=context_len,
        num_users=num_users,
        weights_gb=round(weights_gb, 3),
        runtime_overhead_gb=round(runtime_overhead_gb, 3),
        kv_cache_gb=round(kv_cache_gb, 3),
        total_memory_gb=round(total_memory_gb, 3),
        vram_cap_gb=round(vram_cap_gb, 3),
        headroom_remaining_gb=round(headroom, 3),
        is_oom=is_oom,
        kv_fraction_of_total=round(kv_fraction, 4),
    )


def calculate_oom_cliff(
    spec: ModelSpec,
    num_users: int = 1,
    vram_cap_gb: float = 16.0,
    runtime_overhead_gb: float = 0.80,
) -> int:
    """Calculate theoretical max context length per user before triggering OOM.

    Equation: (VRAM_cap - Weights - Runtime) * 1024^3 / (num_users * KV_bytes_per_token)
    """
    net_headroom_gb = vram_cap_gb - spec.weight_size_gb - runtime_overhead_gb
    if net_headroom_gb <= 0:
        return 0
    net_headroom_bytes = net_headroom_gb * (1024**3)
    kv_bytes_per_tok = calculate_kv_bytes_per_token(spec)
    tokens_total = net_headroom_bytes / (num_users * kv_bytes_per_tok)
    return int(math.floor(tokens_total))


def calculate_roofline(
    spec: ModelSpec,
    context_len: int,
    batch_size: int = 1,
    memory_bw_gbs: float = 1008.0,
    tflops_bf16: float = 82.6,
) -> RooflinePoint:
    """Calculate theoretical memory-bandwidth bound decode throughput and compute-bound prefill TTFT."""
    kv_bytes_per_tok = calculate_kv_bytes_per_token(spec)
    kv_bytes_total = kv_bytes_per_tok * context_len * batch_size
    weights_bytes = int(spec.weight_size_gb * (1024**3))
    bytes_per_decode_step = weights_bytes + kv_bytes_total

    # Memory-bandwidth bound TPOT (ms)
    # decode_step_latency_sec = bytes_per_decode_step / (memory_bw_gbs * 10^9)
    # Multiply by 1000 for ms
    decode_step_latency_ms = (bytes_per_decode_step / (memory_bw_gbs * 1e9)) * 1000.0
    decode_throughput = (batch_size / (decode_step_latency_ms / 1000.0)) if decode_step_latency_ms > 0 else 0.0

    # Base zero-context step latency (only reading weights)
    base_latency_ms = (weights_bytes / (memory_bw_gbs * 1e9)) * 1000.0
    slowdown_pct = ((decode_step_latency_ms - base_latency_ms) / base_latency_ms) * 100.0

    # Compute-bound prefill FLOPs
    # Linear projection FLOPs: ~ 2 * params * context_len
    # Self-attention FLOPs: ~ 4 * num_layers * hidden_size * context_len^2
    hidden_size = spec.hidden_size
    layers = spec.num_full_attn_layers
    params_est = spec.weight_size_gb * (1024**3) / 2  # estimated param count in BF16
    linear_flops = 2.0 * params_est * context_len
    attn_flops = 4.0 * layers * hidden_size * (context_len**2)
    total_prefill_flops = linear_flops + attn_flops

    # Compute-bound min TTFT (sec)
    peak_flops = tflops_bf16 * 1e12
    min_ttft_sec = total_prefill_flops / peak_flops if peak_flops > 0 else 0.0

    return RooflinePoint(
        context_len=context_len,
        batch_size=batch_size,
        kv_cache_bytes=kv_bytes_total,
        total_bytes_read_per_step=bytes_per_decode_step,
        decode_step_latency_ms=round(decode_step_latency_ms, 3),
        decode_throughput_tok_per_sec=round(decode_throughput, 2),
        prefill_flops=round(total_prefill_flops, 2),
        prefill_min_ttft_sec=round(min_ttft_sec, 4),
        decode_slowdown_vs_zero_ctx_pct=round(slowdown_pct, 2),
    )


def generate_test_matrix(
    spec: ModelSpec,
    vram_cap_gb: float = 16.0,
    runtime_overhead_gb: float = 0.80,
    memory_bw_gbs: float = 1008.0,
    tflops_bf16: float = 82.6,
) -> Dict[str, Any]:
    """Generate structured empirical test matrix for a model under given hardware constraints."""
    kv_bytes = calculate_kv_bytes_per_token(spec)
    oom_1u = calculate_oom_cliff(spec, 1, vram_cap_gb, runtime_overhead_gb)
    oom_4u = calculate_oom_cliff(spec, 4, vram_cap_gb, runtime_overhead_gb)
    oom_8u = calculate_oom_cliff(spec, 8, vram_cap_gb, runtime_overhead_gb)
    oom_12u = calculate_oom_cliff(spec, 12, vram_cap_gb, runtime_overhead_gb)
    oom_16u = calculate_oom_cliff(spec, 16, vram_cap_gb, runtime_overhead_gb)

    # Standard contexts to evaluate
    eval_contexts = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    if oom_1u > 32768:
        eval_contexts.extend([49152, 65536])

    single_stream_breakdowns = []
    for ctx in eval_contexts:
        mb = calculate_memory_breakdown(spec, ctx, 1, vram_cap_gb, runtime_overhead_gb)
        rf = calculate_roofline(spec, ctx, 1, memory_bw_gbs, tflops_bf16)
        
        # Categorize regime
        if mb.is_oom:
            regime = "over_the_cliff"
        elif mb.total_memory_gb >= (vram_cap_gb * 0.80):
            regime = "cliff_edge"
        elif mb.kv_cache_gb >= (spec.weight_size_gb * 0.25):
            regime = "inflection"
        else:
            regime = "sub_critical"

        single_stream_breakdowns.append({
            "context_len": ctx,
            "regime": regime,
            "memory": asdict(mb),
            "roofline": asdict(rf),
        })

    # Multi-user concurrency sweep matrix
    concurrency_users = [1, 4, 8, 12, 16]
    concurrency_ctxs = [1024, 2048, 4096, 8192]
    concurrency_grid = []
    for u in concurrency_users:
        user_row = []
        for c in concurrency_ctxs:
            mb = calculate_memory_breakdown(spec, c, u, vram_cap_gb, runtime_overhead_gb)
            user_row.append({
                "users": u,
                "context_len": c,
                "total_tokens": u * c,
                "kv_gb": mb.kv_cache_gb,
                "total_gb": mb.total_memory_gb,
                "is_oom": mb.is_oom,
            })
        concurrency_grid.append({"users": u, "contexts": user_row})

    return {
        "model": spec.name,
        "hf_repo_id": spec.hf_repo_id,
        "vram_cap_gb": vram_cap_gb,
        "runtime_overhead_gb": runtime_overhead_gb,
        "kv_bytes_per_token": kv_bytes,
        "weight_size_gb": spec.weight_size_gb,
        "net_headroom_gb": round(vram_cap_gb - spec.weight_size_gb - runtime_overhead_gb, 3),
        "theoretical_oom_cliffs": {
            "1_user": oom_1u,
            "4_users": oom_4u,
            "8_users": oom_8u,
            "12_users": oom_12u,
            "16_users": oom_16u,
        },
        "single_stream_sweep": single_stream_breakdowns,
        "concurrency_grid": concurrency_grid,
    }


def format_markdown_report(matrix: Dict[str, Any], hw_name: str) -> str:
    """Format test matrix and theoretical calculations as GitHub markdown."""
    m = matrix
    lines = [
        f"# Theoretical Capacity & Roofline Analysis: {m['model'].upper()}",
        "",
        f"> **Target Hardware:** {hw_name}  ",
        f"> **Model Checkpoint:** `{m['hf_repo_id']}`  ",
        f"> **VRAM Ceiling:** `{m['vram_cap_gb']} GB` (Simulated Consumer Cap)  ",
        f"> **KV Bytes / Token (BF16):** `{m['kv_bytes_per_token']} bytes` ({m['kv_bytes_per_token'] // 1024} KiB)",
        "",
        "---",
        "",
        "## 1. Static Budget & Headroom",
        "",
        f"- **Model Weights (BF16):** `{m['weight_size_gb']} GB`",
        f"- **Runtime / CUDA Overhead Buffer:** `{m['runtime_overhead_gb']} GB`",
        f"- **Net Usable KV Headroom:** `{m['net_headroom_gb']} GB`",
        "",
        "### Theoretical OOM Context Thresholds",
        "",
        "| Concurrency ($N$) | Theoretical Max Context per Stream | Total Concurrent Tokens | Failure Mode |",
        "| :---: | :---: | :---: | :--- |",
    ]

    for u_key, oom_val in m["theoretical_oom_cliffs"].items():
        users = int(u_key.split("_")[0])
        total_toks = users * oom_val
        lines.append(f"| **{users} user(s)** | **{oom_val:,} tokens** | {total_toks:,} | OOM if context > {oom_val:,} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Single-Stream Context Sweep & Roofline Bounds ($N=1$)",
        "",
        "| Context ($C$) | Regime | Total VRAM (GB) | Headroom (GB) | Min TTFT (ms) | Min TPOT (ms) | Peak Tok/s | Slowdown vs Base | Status |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for pt in m["single_stream_sweep"]:
        c = pt["context_len"]
        reg = pt["regime"].replace("_", " ").title()
        mem = pt["memory"]
        rf = pt["roofline"]
        status = "⚠️ **OOM**" if mem["is_oom"] else "✅ Fits"
        lines.append(
            f"| `{c:,}` | {reg} | {mem['total_memory_gb']:.2f} | {mem['headroom_remaining_gb']:.2f} | "
            f"{rf['prefill_min_ttft_sec']*1000:.1f} | {rf['decode_step_latency_ms']:.2f} | "
            f"{rf['decode_throughput_tok_per_sec']:.1f} | +{rf['decode_slowdown_vs_zero_ctx_pct']:.1f}% | {status} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Multi-User Concurrency Grid ($N \\times C$)",
        "",
        "| Users ($N$) | 1024 Context | 2048 Context | 4096 Context | 8192 Context |",
        "| :---: | :---: | :---: | :---: | :---: |",
    ])

    for row in m["concurrency_grid"]:
        u = row["users"]
        cols = []
        for c_entry in row["contexts"]:
            vram = c_entry["total_gb"]
            if c_entry["is_oom"]:
                cols.append(f"❌ **{vram:.1f} GB** (OOM)")
            elif vram >= (m["vram_cap_gb"] * 0.90):
                cols.append(f"⚠️ **{vram:.1f} GB** (Edge)")
            else:
                cols.append(f"✅ {vram:.1f} GB")
        lines.append(f"| **{u} Users** | " + " | ".join(cols) + " |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Test Matrix Specification for Empirical Profiling",
        "",
        "Based on these boundaries, the profiling run will execute:",
        "1. **Single-User Depth Sweep:** `512, 1024, 2048, 4096, 8192, 16384, 32768` (Fits), plus `49152, 65536` to verify OOM cliff boundary.",
        "2. **Multi-User Concurrency Sweep:** `1, 4, 8, 12, 16` users across `1024, 2048, 4096, 8192` contexts.",
        "   - **Critical Edge Point:** `N=12, C=4096` ($15.61\\text{ GB}$ total $\\rightarrow$ verifies near-capacity paging).",
        "   - **Deterministic OOM Point:** `N=12, C=8192` ($22.36\\text{ GB}$ total $\\rightarrow$ verifies graceful OOM handling).",
    ])

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate theoretical limits & test matrix for KV-cache benchmark")
    parser.add_argument("--model", default="qwen3-4b", help=f"Model name (default: qwen3-4b, available: {', '.join(MODEL_REGISTRY.keys())})")
    parser.add_argument("--hardware", default="rtx4090_16gb", choices=list(HARDWARE_PRESETS.keys()), help="Hardware preset")
    parser.add_argument("--max-vram-gb", type=float, default=None, help="Override VRAM cap in GB")
    parser.add_argument("--runtime-overhead-gb", type=float, default=None, help="Override runtime overhead in GB")
    parser.add_argument("--memory-bw-gbs", type=float, default=None, help="Override memory bandwidth in GB/s")
    parser.add_argument("--tflops-bf16", type=float, default=None, help="Override BF16 dense TFLOPS")
    parser.add_argument("--json", action="store_true", help="Print raw JSON output")
    parser.add_argument("--markdown", action="store_true", help="Print Markdown report")
    parser.add_argument("--output", default=None, help="Save report to file path")
    args = parser.parse_args()

    hw = HARDWARE_PRESETS[args.hardware]
    vram_cap = args.max_vram_gb if args.max_vram_gb is not None else hw["vram_cap_gb"]
    runtime_overhead = args.runtime_overhead_gb if args.runtime_overhead_gb is not None else hw["runtime_overhead_gb"]
    mem_bw = args.memory_bw_gbs if args.memory_bw_gbs is not None else hw["memory_bw_gbs"]
    tflops = args.tflops_bf16 if args.tflops_bf16 is not None else hw["tflops_bf16"]

    spec = get_model_spec(args.model)
    if spec is None:
        print(f"Error: Model '{args.model}' not found in registry.", file=sys.stderr)
        sys.exit(1)

    matrix = generate_test_matrix(
        spec=spec,
        vram_cap_gb=vram_cap,
        runtime_overhead_gb=runtime_overhead,
        memory_bw_gbs=mem_bw,
        tflops_bf16=tflops,
    )

    if args.json:
        out_str = json.dumps(matrix, indent=2)
    else:
        out_str = format_markdown_report(matrix, hw["name"])

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_str, encoding="utf-8")
        print(f"Wrote theoretical report to {out_path}")
    else:
        print(out_str)


if __name__ == "__main__":
    main()
