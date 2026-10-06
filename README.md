# KV-cache co-optimization pipeline

Unified benchmark runner for KV-cache optimization techniques on consumer
GPUs (8–16 GB VRAM). Replaces one hand-copied `benchmark_*.py` per technique
with a single harness that every technique plugs into.

## Structure

```
main.py                        CLI entry point
compare.py                     Comparative reporting CLI (Markdown tables vs baseline)
benchmark.sbatch               Slurm batch submission script (CPE cluster / RTX 4090)
docs/
  METRICS_AND_AXES.md          Co-optimization framework, metrics, & quality evaluation
src/core/
  axis.py                      Axis enum: STRUCTURAL / SPATIAL / TEMPORAL
  technique.py                 Technique interface every method implements
  registry.py                  @register("name") decorator + lookup
  workload.py                  Workload hierarchy: SyntheticFiller, ColdWarm, ConcurrentChatWorkload
  quality.py                   Model fidelity metrics (Top-1 agreement, Mean & P99 KL-Divergence)
  compare.py                   Run comparison and Pareto markdown table generation
  metrics.py                   GPU/RAM helpers + ResultEntry schema (TPOT, P95 TTFT, PCIe per token)
  harness.py                   Runner loop: build engine, 16GB VRAM capping, timing, profiling
  config.py                    BenchmarkConfig defaults
src/workloads/
  memory_adherence.py          LongMemEvalWorkload across 5 conversational capabilities
src/data/
  longmemeval_subset.json      Curated LongMemEval evaluation benchmark
src/techniques/
  baseline.py                  No optimization (reference point)
  flexgen_style.py             CPU swap + weight offload + fp8 KV (SPATIAL, STRUCTURAL)
  lmcache_offload.py           CPU offload + prefix reuse via LMCache (SPATIAL, TEMPORAL)
results/                       JSON output lands here
results/traces/                Chrome profiler traces (when --profile is used)
```

## Usage

```bash
# Basic sweep with 16GB consumer VRAM limit enforced
uv run main.py --technique baseline --model Qwen/Qwen2.5-1.5B --contexts 512 1024 2048 4096 --max-vram-gb 16.0

# Tune technique parameters from the CLI (no need to edit the file)
uv run main.py --technique flexgen_style --contexts 2048 --technique-args swap_space_gb=8 cpu_offload_gb=4

# Prefix-reuse technique -- ColdWarm workload, cold vs. warm delta reported
uv run main.py --technique lmcache --contexts 2048

# Multi-user concurrency workload (simulating 10-15 SME staff members)
uv run main.py --technique baseline --workload concurrent --num-users 12 --contexts 2048

# Long-term memory adherence benchmark (LongMemEval across 5 core capabilities)
uv run main.py --technique baseline --workload longmem --contexts 2048

# Profile a run: saves Chrome trace + extracts CPU↔GPU transfer stats into the JSON
uv run main.py --technique flexgen_style --contexts 2048 --profile

# Verify technique routing without a GPU (works on non-GPU dev machines)
uv run main.py --technique baseline --dry-run
```

## Comparative Reporting (`compare.py`)

Compare any optimization technique against the uncompressed baseline:

```bash
python3 compare.py results/baseline_16gb.json results/flexgen_style_16gb.json
```

Outputs a formatted markdown table evaluating:
- **Peak VRAM (MB)** & % reduction vs. baseline
- **TTFT (s)** & speedup vs. baseline
- **TPOT (ms)** & P95 decode jitter
- **PCIe Transfer (bytes/token)** & % reduction vs. baseline
- **Fidelity & Quality:** Top-1 agreement (%) and LongMemEval adherence accuracy (%)

## Slurm Cluster Execution (16GB VRAM Capped)

To run on cluster nodes with 24GB GPUs (e.g. RTX 4090 on `portal.slurm.cpe.kmutt.ac.th`), the benchmark automatically caps the PyTorch allocator and vLLM PagedAttention pool to 16GB via `--max-vram-gb 16.0`.

Submit via Slurm:
```bash
sbatch benchmark.sbatch
```

## Output schema

```json
{
  "model": "Qwen/Qwen2.5-1.5B",
  "technique": "flexgen_style",
  "axes": ["SPATIAL", "STRUCTURAL"],
  "results": [
    {
      "requested_ctx": 2048,
      "status": "SUCCESS",
      "axes": ["SPATIAL", "STRUCTURAL"],
      "ttft_sec": 1.23,
      "cold_ttft_sec": 0.0,
      "cold_total_sec": 0.0,
      "ttft_p95_sec": 1.45,
      "tpot_mean_ms": 20.8,
      "tpot_p95_ms": 24.1,
      "peak_alloc_mb": 4200.0,
      "host_rss_delta_mb": 312.0,
      "throughput_tok_per_sec": 48.2,
      "memcpy_bytes_per_token": 19.3,
      "extra": {
        "memcpy_htod_bytes": 1234567,
        "memcpy_dtoh_bytes": 987654,
        "memcpy_peak_bw_gbs": 7.18,
        "top1_agreement": 0.98,
        "kl_div_mean": 0.012,
        "kl_div_p99": 0.085,
        "longmem_acc_overall": 0.90
      }
    }
  ]
}
```

**Key fields:**
| Field | What it means |
|---|---|
| `axes` | Which of STRUCTURAL/SPATIAL/TEMPORAL this technique touches (per-entry for easy CSV filtering) |
| `ttft_sec` | Time to first token (warm run for ColdWarm workloads, mean across concurrent requests) |
| `cold_ttft_sec` | TTFT of the cold (cache-miss) run — 0 for SyntheticFiller |
| `cold_ttft_delta_sec` | `cold_ttft_sec − ttft_sec`: positive = warm is faster (prefix reuse working) |
| `ttft_p95_sec` | 95th-percentile TTFT under 10–15 concurrent simulated users |
| `tpot_mean_ms` / `tpot_p95_ms` | Mean and P95 Time-Per-Output-Token (detects decode stalls) |
| `host_rss_delta_mb` | CPU RAM added by this technique (delta from before engine init to after workload) |
| `memcpy_bytes_per_token` | Normalized PCIe transfer volume per generated token |
| `extra.top1_agreement` | Fraction of generated tokens identical to FP16 baseline |
| `extra.kl_div_mean` / `kl_div_p99` | Output distribution divergence vs uncompressed baseline |
| `extra.longmem_acc_overall` | Accuracy on LongMemEval across 5 core dialogue capabilities |

> See [docs/METRICS_AND_AXES.md](file:///home/khemi/workspace/kv_bench/docs/METRICS_AND_AXES.md) for full metric rationales, axis mappings, and the structural quality evaluation strategy.

## Adding a new technique

Drop one file in `src/techniques/`. Zero other changes needed.

```python
# src/techniques/h2o_eviction.py
from src.core.axis import Axis
from src.core.registry import register
from src.core.technique import Technique

@register("h2o")
class H2OTechnique(Technique):
    axes = Axis.STRUCTURAL  # eviction changes what's kept, not where it lives

    # RULE: all __init__ params must have defaults (zero-arg convention).
    # This lets the registry instantiate any technique without required args.
    # Override values from the CLI: --technique-args eviction_ratio=0.2
    def __init__(self, eviction_ratio: float = 0.1):
        self.eviction_ratio = eviction_ratio

    def engine_kwargs(self, context_length, max_tokens):
        return {...}
```

**Tips:**
- Put env-var patches or import-order fixes in `setup()`, not at module level (see `lmcache_offload.py`).
- Override `workload()` to return `ColdWarm()` for any prefix-reuse technique.
- Use `extra_metrics()` to return technique-specific numbers that flow into `ResultEntry.extra`.

## Metric roadmap

| Step | Status |
|---|---|
| Peak VRAM & Context Sweep (OOM) | ✅ |
| Host RAM delta (RSS) | ✅ |
| Cold vs. warm TTFT delta | ✅ |
| Inter-Token Latency (TPOT & P95 jitter) | ✅ |
| 10–15 Simulated User Concurrency (P95 TTFT) | ✅ |
| Normalized PCIe transfer bytes & bandwidth | ✅ |
| Logit Divergence (Top-1 agreement & KL Mean/P99) | ✅ |
| Long-Term Memory Adherence (LongMemEval 5 subtasks) | ✅ |
| Verify baseline + flexgen_style on real GPU | ✅ (Verified on CPE Cluster RTX 4090) |
| First STRUCTURAL technique (H2O or StreamingLLM sink+window) | ⬜ |
| Perplexity delta (once STRUCTURAL lands) | ⬜ |
