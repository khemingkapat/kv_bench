#!/usr/bin/env python3
"""Theoretical Capacity & Roofline Modeling for KV-Cache Optimization.

Analytically models:
1. GQA KV-cache memory footprint per token and total allocation in exact binary GiB (1024^3).
2. Two-tier hardware memory boundaries under consumer GPU constraints:
   - Tier 1: Process allocator ceiling (e.g. 16.0 GiB on RTX 4090, 8.0 GiB on RTX 4060).
   - Tier 2: vLLM PagedAttention KV pool ceiling governed by `gpu_memory_utilization`
     (default 0.85 = 13.60 GiB budget), subtracting static weights, CUDA context,
     and peak activation buffers.
3. Roofline latency & throughput bounds:
   - Decode phase: Memory bandwidth bound (TPOT min ms, peak tokens/sec,
     TPOT latency increase %, and throughput reduction %).
   - Prefill phase: Compute bound with FlashAttention-2 causal masking (TTFT min ms).
4. Analytical empirical test matrix categorizing test points into:
   - Sub-critical (weight-dominated, linear throughput)
   - Inflection (memory bus saturation begins)
   - Cliff edge (approaching vLLM pool limit)
   - Over-the-cliff (deterministic OOM / pool exhaustion verification)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import MODEL_REGISTRY, ModelSpec, get_model_spec


# Hardware presets
HARDWARE_PRESETS = {
    "rtx4090_16gb": {
        "name": "NVIDIA GeForce RTX 4090 (16GB Cap)",
        "vram_cap_gib": 16.0,
        "gpu_utilization": 0.85,
        "runtime_overhead_gib": 0.80,  # non-torch CUDA context (~0.20-0.27 GiB) + activation buffer (~0.55-0.60 GiB)
        "memory_bw_gbs": 1008.0,
        "tflops_bf16": 82.6,
        "pcie_empirical_bw_gbs": 25.5,
    },
    "rtx4060_8gb": {
        "name": "NVIDIA GeForce RTX 4060 (8GB Local)",
        "vram_cap_gib": 8.0,
        "gpu_utilization": 0.85,
        "runtime_overhead_gib": 0.60,
        "memory_bw_gbs": 272.0,
        "tflops_bf16": 18.0,
        "pcie_empirical_bw_gbs": 14.0,
    },
}


@dataclass
class MemoryBreakdown:
    context_len: int
    num_users: int
    weights_gib: float
    runtime_overhead_gib: float
    kv_cache_gib: float
    total_memory_gib: float
    vram_cap_gib: float
    vllm_budget_gib: float
    headroom_cap_gib: float
    headroom_vllm_pool_gib: float
    is_oom_cap: bool
    is_oom_vllm_pool: bool
    kv_fraction_of_total: float


@dataclass
class RooflinePoint:
    context_len: int
    batch_size: int
    kv_cache_bytes: int
    total_bytes_read_per_step: int
    decode_step_latency_ms: float
    decode_throughput_tok_per_sec: float
    tpot_latency_increase_pct: float
    throughput_reduction_pct: float
    prefill_flops: float
    prefill_min_ttft_sec: float


def get_weights_bytes(spec: ModelSpec) -> int:
    """Return exact model weights size in bytes.

    spec.weight_size_gb is stored in decimal GB (10^9 bytes), e.g. 8.06 GB = 8.06e9 bytes.
    """
    return int(round(spec.weight_size_gb * 1e9))


def get_weights_gib(spec: ModelSpec) -> float:
    """Return model weights size in binary GiB (1024^3 bytes)."""
    return get_weights_bytes(spec) / (1024**3)


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
    vram_cap_gib: float = 16.0,
    gpu_utilization: float = 0.85,
    runtime_overhead_gib: float = 0.80,
) -> MemoryBreakdown:
    """Calculate detailed memory footprint against both process cap and vLLM KV pool."""
    kv_bytes_per_tok = calculate_kv_bytes_per_token(spec)
    total_kv_bytes = kv_bytes_per_tok * context_len * num_users
    kv_cache_gib = total_kv_bytes / (1024**3)

    weights_gib = get_weights_gib(spec)
    total_memory_gib = weights_gib + runtime_overhead_gib + kv_cache_gib
    vllm_budget_gib = vram_cap_gib * gpu_utilization

    headroom_cap = vram_cap_gib - total_memory_gib
    headroom_vllm = vllm_budget_gib - total_memory_gib

    is_oom_cap = total_memory_gib > vram_cap_gib
    is_oom_vllm_pool = total_memory_gib > vllm_budget_gib
    kv_fraction = kv_cache_gib / total_memory_gib if total_memory_gib > 0 else 0.0

    return MemoryBreakdown(
        context_len=context_len,
        num_users=num_users,
        weights_gib=round(weights_gib, 3),
        runtime_overhead_gib=round(runtime_overhead_gib, 3),
        kv_cache_gib=round(kv_cache_gib, 3),
        total_memory_gib=round(total_memory_gib, 3),
        vram_cap_gib=round(vram_cap_gib, 3),
        vllm_budget_gib=round(vllm_budget_gib, 3),
        headroom_cap_gib=round(headroom_cap, 3),
        headroom_vllm_pool_gib=round(headroom_vllm, 3),
        is_oom_cap=is_oom_cap,
        is_oom_vllm_pool=is_oom_vllm_pool,
        kv_fraction_of_total=round(kv_fraction, 4),
    )


def calculate_oom_cliff(
    spec: ModelSpec,
    num_users: int = 1,
    vram_target_gib: float = 16.0,
    runtime_overhead_gib: float = 0.80,
) -> int:
    """Calculate theoretical max context length per user before exhausting target VRAM budget.

    Equation: (VRAM_target - Weights - Runtime) * 1024^3 / (num_users * KV_bytes_per_token)
    """
    weights_gib = get_weights_gib(spec)
    net_headroom_gib = vram_target_gib - weights_gib - runtime_overhead_gib
    if net_headroom_gib <= 0:
        return 0
    net_headroom_bytes = net_headroom_gib * (1024**3)
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
    weights_bytes = get_weights_bytes(spec)
    bytes_per_decode_step = weights_bytes + kv_bytes_total

    # Memory-bandwidth bound TPOT (ms)
    # decode_step_latency_sec = bytes_per_decode_step / (memory_bw_gbs * 10^9)
    # Multiply by 1000 for ms
    decode_step_latency_ms = (bytes_per_decode_step / (memory_bw_gbs * 1e9)) * 1000.0
    decode_throughput = (batch_size / (decode_step_latency_ms / 1000.0)) if decode_step_latency_ms > 0 else 0.0

    # Base zero-context step latency (only reading weights)
    base_latency_ms = (weights_bytes / (memory_bw_gbs * 1e9)) * 1000.0
    base_throughput = (batch_size / (base_latency_ms / 1000.0)) if base_latency_ms > 0 else 0.0

    # Percentage changes:
    # 1. Latency increase (%)
    tpot_increase_pct = ((decode_step_latency_ms - base_latency_ms) / base_latency_ms) * 100.0
    # 2. Throughput reduction (%)
    throughput_reduction_pct = ((base_throughput - decode_throughput) / base_throughput) * 100.0

    # Compute-bound prefill FLOPs:
    # 1. Linear projection FLOPs: 2 * param_count * context_len
    # 2. Causal self-attention FLOPs (FlashAttention-2 causal lower-triangle):
    #    2 * num_layers * (num_q_heads * head_dim) * context_len^2
    param_count = weights_bytes / 2  # 2 bytes per param in BF16
    linear_flops = 2.0 * param_count * context_len
    q_dim = spec.num_q_heads * spec.head_dim
    causal_attn_flops = 2.0 * spec.num_full_attn_layers * q_dim * (context_len**2)
    total_prefill_flops = linear_flops + causal_attn_flops

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
        tpot_latency_increase_pct=round(tpot_increase_pct, 2),
        throughput_reduction_pct=round(throughput_reduction_pct, 2),
        prefill_flops=round(total_prefill_flops, 2),
        prefill_min_ttft_sec=round(min_ttft_sec, 4),
    )


def generate_test_matrix(
    spec: ModelSpec,
    vram_cap_gib: float = 16.0,
    gpu_utilization: float = 0.85,
    runtime_overhead_gib: float = 0.80,
    memory_bw_gbs: float = 1008.0,
    tflops_bf16: float = 82.6,
) -> Dict[str, Any]:
    """Generate structured empirical test matrix for a model under given hardware constraints."""
    kv_bytes = calculate_kv_bytes_per_token(spec)
    weights_gib = get_weights_gib(spec)
    vllm_budget_gib = vram_cap_gib * gpu_utilization

    concurrency_users = [1, 4, 8, 12, 16]
    cliffs_pool = {}
    cliffs_cap = {}
    for u in concurrency_users:
        key = f"{u}_user" if u == 1 else f"{u}_users"
        cliffs_pool[key] = calculate_oom_cliff(spec, u, vllm_budget_gib, runtime_overhead_gib)
        cliffs_cap[key] = calculate_oom_cliff(spec, u, vram_cap_gib, runtime_overhead_gib)

    eval_contexts = [512, 1024, 2048, 4096, 8192, 16384, 32768, 49152, 65536]

    single_stream_breakdowns = []
    for ctx in eval_contexts:
        mb = calculate_memory_breakdown(spec, ctx, 1, vram_cap_gib, gpu_utilization, runtime_overhead_gib)
        rf = calculate_roofline(spec, ctx, 1, memory_bw_gbs, tflops_bf16)

        # Categorize regime based on vLLM pool and process cap
        if mb.is_oom_cap:
            regime = "hard_process_oom"
        elif mb.is_oom_vllm_pool:
            regime = "vllm_pool_exhaustion"
        elif mb.total_memory_gib >= (vllm_budget_gib * 0.90):
            regime = "cliff_edge"
        elif mb.kv_cache_gib >= (weights_gib * 0.25):
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
    concurrency_ctxs = [1024, 2048, 4096, 8192]
    concurrency_grid = []
    for u in concurrency_users:
        user_row = []
        for c in concurrency_ctxs:
            mb = calculate_memory_breakdown(spec, c, u, vram_cap_gib, gpu_utilization, runtime_overhead_gib)
            user_row.append({
                "users": u,
                "context_len": c,
                "total_tokens": u * c,
                "kv_gib": mb.kv_cache_gib,
                "total_gib": mb.total_memory_gib,
                "is_oom_cap": mb.is_oom_cap,
                "is_oom_vllm_pool": mb.is_oom_vllm_pool,
            })
        concurrency_grid.append({"users": u, "contexts": user_row})

    return {
        "model": spec.name,
        "hf_repo_id": spec.hf_repo_id,
        "vram_cap_gib": vram_cap_gib,
        "gpu_utilization": gpu_utilization,
        "vllm_budget_gib": round(vllm_budget_gib, 3),
        "runtime_overhead_gib": runtime_overhead_gib,
        "kv_bytes_per_token": kv_bytes,
        "weights_decimal_gb": spec.weight_size_gb,
        "weights_gib": round(weights_gib, 3),
        "net_headroom_cap_gib": round(vram_cap_gib - weights_gib - runtime_overhead_gib, 3),
        "net_headroom_vllm_pool_gib": round(vllm_budget_gib - weights_gib - runtime_overhead_gib, 3),
        "theoretical_oom_cliffs_vllm_pool": cliffs_pool,
        "theoretical_oom_cliffs_hard_cap": cliffs_cap,
        "single_stream_sweep": single_stream_breakdowns,
        "concurrency_grid": concurrency_grid,
    }


def _grid_lookup(grid: list, users: int, ctx: int) -> float:
    """Helper to fetch total_gib from concurrency_grid."""
    for row in grid:
        if row["users"] == users:
            for cell in row["contexts"]:
                if cell["context_len"] == ctx:
                    return cell["total_gib"]
    return 0.0


def format_markdown_report(matrix: Dict[str, Any], hw_name: str) -> str:
    """Format test matrix and theoretical calculations as GitHub markdown."""
    m = matrix
    lines = [
        f"# Theoretical Capacity & Roofline Analysis: {m['model'].upper()}",
        "",
        f"> **Target Hardware:** {hw_name}  ",
        f"> **Model Checkpoint:** `{m['hf_repo_id']}`  ",
        f"> **VRAM Allocator Cap:** `{m['vram_cap_gib']} GiB` (Simulated Consumer Cap)  ",
        f"> **vLLM Target Utilization:** `{m['gpu_utilization']} ({m['vllm_budget_gib']} GiB Budget)`  ",
        f"> **KV Bytes / Token (BF16):** `{m['kv_bytes_per_token']} bytes` ({m['kv_bytes_per_token'] // 1024} KiB)",
        "",
        "---",
        "",
        "## 1. Static Budget & Two-Tier Headroom",
        "",
        f"- **Model Weights:** `{m['weights_decimal_gb']} GB` decimal = `{m['weights_gib']} GiB` binary",
        f"- **Runtime / CUDA Overhead Buffer:** `{m['runtime_overhead_gib']} GiB`",
        f"- **Net KV Headroom (Tier 1 Process Cap {m['vram_cap_gib']} GiB):** `{m['net_headroom_cap_gib']} GiB`",
        f"- **Net KV Headroom (Tier 2 vLLM Pool {m['vllm_budget_gib']} GiB):** `{m['net_headroom_vllm_pool_gib']} GiB`",
        "",
        "### Theoretical OOM Context Thresholds",
        "",
        f"| Concurrency ($N$) | vLLM Pool Limit (`util={m['gpu_utilization']}`) | Process Cap Limit (`util=1.0`) | Failure / Saturation Mode |",
        "| :---: | :---: | :---: | :--- |",
    ]

    for u_key, oom_pool in m["theoretical_oom_cliffs_vllm_pool"].items():
        users = int(u_key.split("_")[0])
        oom_cap = m["theoretical_oom_cliffs_hard_cap"][u_key]
        lines.append(
            f"| **{users} user(s)** | **{oom_pool:,} tokens** | {oom_cap:,} tokens | "
            f"KV pool exhausted if context > {oom_pool:,} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Single-Stream Context Sweep & Roofline Bounds ($N=1$)",
        "",
        r"| Context ($C$) | Regime | Total VRAM (GiB) | Pool Headroom | Min TTFT (ms) | Min TPOT (ms) | Peak Tok/s | Latency $\Delta$ | Throughput $\Delta$ | Status |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for pt in m["single_stream_sweep"]:
        c = pt["context_len"]
        reg = pt["regime"].replace("_", " ").title()
        mem = pt["memory"]
        rf = pt["roofline"]
        if mem["is_oom_cap"]:
            status = "❌ **Hard OOM**"
        elif mem["is_oom_vllm_pool"]:
            status = "⚠️ **Pool OOM**"
        elif mem["total_memory_gib"] >= (m["vllm_budget_gib"] * 0.90):
            status = "⚡ **Near Edge**"
        else:
            status = "✅ Fits"

        lines.append(
            f"| `{c:,}` | {reg} | {mem['total_memory_gib']:.2f} | {mem['headroom_vllm_pool_gib']:.2f} GiB | "
            f"{rf['prefill_min_ttft_sec']*1000:.1f} | {rf['decode_step_latency_ms']:.2f} | "
            f"{rf['decode_throughput_tok_per_sec']:.1f} | +{rf['tpot_latency_increase_pct']:.1f}% | "
            f"-{rf['throughput_reduction_pct']:.1f}% | {status} |"
        )

    lines.extend([
        "",
        "---",
        "",
        r"## 3. Multi-User Concurrency Grid ($N \times C$)",
        "",
        f"Total memory footprint against the **{m['vllm_budget_gib']:.2f} GiB vLLM Pool** (`util={m['gpu_utilization']}`):",
        "",
        "| Users ($N$) | 1024 Context | 2048 Context | 4096 Context | 8192 Context |",
        "| :---: | :---: | :---: | :---: | :---: |",
    ])

    for row in m["concurrency_grid"]:
        u = row["users"]
        cols = []
        for c_entry in row["contexts"]:
            vram = c_entry["total_gib"]
            if c_entry["is_oom_cap"]:
                cols.append(f"❌ **{vram:.1f} GiB** (Hard OOM)")
            elif c_entry["is_oom_vllm_pool"]:
                cols.append(f"⚠️ **{vram:.1f} GiB** (Pool OOM)")
            elif vram >= (m["vllm_budget_gib"] * 0.90):
                cols.append(f"⚡ **{vram:.1f} GiB** (Edge)")
            else:
                cols.append(f"✅ {vram:.1f} GiB")
        lines.append(f"| **{u} Users** | " + " | ".join(cols) + " |")

    g = m["concurrency_grid"]
    v_4_1k = _grid_lookup(g, 4, 1024)
    v_4_2k = _grid_lookup(g, 4, 2048)
    v_8_2k = _grid_lookup(g, 8, 2048)
    v_4_4k = _grid_lookup(g, 4, 4096)
    v_12_2k = _grid_lookup(g, 12, 2048)
    v_8_4k = _grid_lookup(g, 8, 4096)
    v_12_4k = _grid_lookup(g, 12, 4096)
    v_8_8k = _grid_lookup(g, 8, 8192)

    lines.extend([
        "",
        "---",
        "",
        "## 4. Test Matrix Specification for Empirical Profiling",
        "",
        "Based on these boundaries, the profiling run will execute:",
        "1. **Single-User Depth Sweep:** `512, 1024, 2048, 4096, 8192, 16384, 32768` (Fits inside pool), plus `49152, 65536` to verify pool exhaustion and process OOM.",
        "2. **Multi-User Concurrency Sweep:** `1, 4, 8, 12, 16` users across `1024, 2048, 4096, 8192` contexts.",
        f"   - **Sub-Critical Points:** $N=4$ at $1024, 2048$ (${v_4_1k:.1f}\\text{{--}}{v_4_2k:.1f}\\text{{ GiB}}$).",
        f"   - **Inflection Point:** $N=8$ at $2048$ (${v_8_2k:.1f}\\text{{ GiB}}$) and $N=4$ at $4096$ (${v_4_4k:.1f}\\text{{ GiB}}$).",
        f"   - **Critical Edge Point:** $N=12$ at $2048$ (${v_12_2k:.1f}\\text{{ GiB}}$) and $N=8$ at $4096$ (${v_8_4k:.1f}\\text{{ GiB}}$).",
        f"   - **Pool Exhaustion / Hard OOM Points:** $N=12$ at $4096$ (${v_12_4k:.1f}\\text{{ GiB}}$) and $N=8$ at $8192$ (${v_8_8k:.1f}\\text{{ GiB}}$).",
    ])

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate theoretical limits & test matrix for KV-cache benchmark")
    parser.add_argument("--model", default="qwen3-4b", help=f"Model name (default: qwen3-4b, available: {', '.join(MODEL_REGISTRY.keys())})")
    parser.add_argument("--hardware", default="rtx4090_16gb", choices=list(HARDWARE_PRESETS.keys()), help="Hardware preset")
    parser.add_argument("--max-vram-gb", type=float, default=None, help="Override VRAM cap in GiB")
    parser.add_argument("--gpu-utilization", type=float, default=None, help="Override vLLM GPU memory utilization fraction (e.g. 0.85)")
    parser.add_argument("--runtime-overhead-gb", type=float, default=None, help="Override runtime overhead in GiB")
    parser.add_argument("--memory-bw-gbs", type=float, default=None, help="Override memory bandwidth in GB/s")
    parser.add_argument("--tflops-bf16", type=float, default=None, help="Override BF16 dense TFLOPS")
    parser.add_argument("--json", action="store_true", help="Print raw JSON output")
    parser.add_argument("--markdown", action="store_true", help="Print Markdown report")
    parser.add_argument("--output", default=None, help="Save report to file path")
    args = parser.parse_args()

    hw = HARDWARE_PRESETS[args.hardware]
    vram_cap = args.max_vram_gb if args.max_vram_gb is not None else hw["vram_cap_gib"]
    gpu_util = args.gpu_utilization if args.gpu_utilization is not None else hw["gpu_utilization"]
    runtime_overhead = args.runtime_overhead_gb if args.runtime_overhead_gb is not None else hw["runtime_overhead_gib"]
    mem_bw = args.memory_bw_gbs if args.memory_bw_gbs is not None else hw["memory_bw_gbs"]
    tflops = args.tflops_bf16 if args.tflops_bf16 is not None else hw["tflops_bf16"]

    spec = get_model_spec(args.model)
    if spec is None:
        print(f"Error: Model '{args.model}' not found in registry.", file=sys.stderr)
        sys.exit(1)

    matrix = generate_test_matrix(
        spec=spec,
        vram_cap_gib=vram_cap,
        gpu_utilization=gpu_util,
        runtime_overhead_gib=runtime_overhead,
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
