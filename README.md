# KV-cache co-optimization pipeline

Unified benchmark runner for KV-cache optimization techniques on consumer
GPUs (8–16 GB VRAM). Replaces one hand-copied `benchmark_*.py` per technique
with a single harness that every technique plugs into.

## Structure

```
main.py                        CLI entry point
docs/
  METRICS_AND_AXES.md          Co-optimization framework, metrics, & quality evaluation
src/core/
  axis.py                      Axis enum: STRUCTURAL / SPATIAL / TEMPORAL
  technique.py                 Technique interface every method implements
  registry.py                  @register("name") decorator + lookup
  workload.py                  SyntheticFiller (single shot) · ColdWarm (prefix reuse)
  metrics.py                   GPU/RAM helpers + the ResultEntry schema
  harness.py                   Runner loop: build engine, run, measure, handle failure
  config.py                    BenchmarkConfig defaults
src/techniques/
  baseline.py                  No optimization (reference point)
  flexgen_style.py             CPU swap + weight offload + fp8 KV (SPATIAL, STRUCTURAL)
  lmcache_offload.py           CPU offload + prefix reuse via LMCache (SPATIAL, TEMPORAL)
results/                       JSON output lands here
results/traces/                Chrome profiler traces (when --profile is used)
```

## Usage

```bash
# Basic sweep
uv run main.py --technique baseline --model Qwen/Qwen2.5-1.5B --contexts 512 1024 2048

# Tune technique parameters from the CLI (no need to edit the file)
uv run main.py --technique flexgen_style --contexts 4096 --technique-args swap_space_gb=8

# Prefix-reuse technique -- ColdWarm workload, cold vs. warm delta reported
uv run main.py --technique lmcache --contexts 2048

# Profile a run: saves Chrome trace + extracts CPU↔GPU transfer stats into the JSON
uv run main.py --technique flexgen_style --contexts 2048 --profile

# Verify technique routing without a GPU (useful for CI or new-machine setup)
uv run main.py --technique baseline --dry-run
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
      "peak_alloc_mb": 4200.0,
      "host_rss_delta_mb": 312.0,
      "throughput_tok_per_sec": 48.2,
      "extra": {
        "memcpy_htod_bytes": 1234567,
        "memcpy_dtoh_bytes": 987654,
        "memcpy_peak_bw_gbs": 7.18
      }
    }
  ]
}
```

**Key fields:**
| Field | What it means |
|---|---|
| `axes` | Which of STRUCTURAL/SPATIAL/TEMPORAL this technique touches (per-entry for easy CSV filtering) |
| `ttft_sec` | Time to first token (warm run for ColdWarm workloads) |
| `cold_ttft_sec` | TTFT of the cold (cache-miss) run — 0 for SyntheticFiller |
| `cold_ttft_delta_sec` | `cold_ttft_sec − ttft_sec`: positive = warm is faster (prefix reuse working) |
| `host_rss_delta_mb` | CPU RAM added by this technique (delta from before engine init to after workload) |
| `extra.memcpy_htod_bytes` | Bytes transferred CPU→GPU during the run (requires `--profile`) |
| `extra.memcpy_peak_bw_gbs` | Peak PCIe bandwidth achieved (requires `--profile`) |

> See [docs/METRICS_AND_AXES.md](file:///home/khemi/workspace/llm_bench/docs/METRICS_AND_AXES.md) for full metric rationales, axis mappings, and the structural quality evaluation strategy.

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
| Verify baseline + flexgen_style on real GPU | ⬜ first task on machine |
| Peak VRAM | ✅ |
| Host RAM delta (RSS) | ✅ |
| Cold vs. warm TTFT delta | ✅ |
| CPU↔GPU transfer bytes + peak bandwidth | ✅ (via `--profile`) |
| First STRUCTURAL technique (H2O or StreamingLLM sink+window) | ⬜ |
| Perplexity delta (once STRUCTURAL lands) | ⬜ |
| First cross-axis combination experiment | ⬜ |
