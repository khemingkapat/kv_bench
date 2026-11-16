# WP3-9: InfiniGen

**Primary Axis:** `TEMPORAL` | **All Axes:** `SPATIAL, TEMPORAL, STRUCTURAL` | **Model:** `Qwen3-4B-Instruct-2507` (16.0 GiB Cap, PyTorch Native SDPA)

**Artifact Provenance (SHA-256):** `infinigen_hardware_pytorch_16gb.json`: `VERIFIED` (`51ce698454d1…`) | `infinigen_longmem_pytorch_16gb.json`: `UPDATED (post-manifest re-run)` (`c3d9ece137f4…`) | `infinigen_niah_pytorch_16gb.json`: `UPDATED (post-manifest re-run)` (`3edd0b98a4cc…`)

---

## 1. What It Does & How (Brief)

InfiniGen speculatively prefetches important KV blocks from CPU memory one layer ahead by using low-rank key proxies and threshold-based importance prediction to overlap host-to-device transfers with layer computation.

- **Evaluated Configuration:** `k_proxy_ratio=0.3`, `alpha_threshold=4.0`, `block_size=16`, `max_budget_ratio=0.35`, `swap_space_gb=4`

---

## 2. Empirical Results

### 2.1 Hardware Metric Sweep (512 → 32,768)

| Context | Status | Peak Alloc (MB) | Peak Reserved (MB) | Warm TTFT (s) | Cold TTFT (s) | TPOT Mean (ms) | TPOT P95 (ms) | Throughput (tok/s) | Host RSS Δ (MB) | Memcpy (B/tok) |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | SUCCESS | 7982.1 | 7998.0 | 0.049 | 0.000 | 46.16 | 46.16 | 21.66 | 5816.9 | 29696601.2 |
| 1024 | SUCCESS | 8273.5 | 8400.0 | 0.086 | 0.000 | 55.62 | 55.62 | 17.98 | 10586.8 | 55804006.9 |
| 2048 | SUCCESS | 8720.7 | 8822.0 | 0.193 | 0.000 | 71.88 | 71.88 | 13.91 | 19738.4 | 86615915.8 |
| 4096 | FAILED | 0.0 | 0.0 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 |
| 8192 | FAILED | 0.0 | 0.0 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 |
| 16384 | SUCCESS | 11871.8 | 12444.0 | 3.606 | 0.000 | 352.46 | 352.46 | 2.84 | 13334.8 | 888775168.8 |
| 32768 | SUCCESS | 15158.1 | 15774.0 | 14.158 | 0.000 | 657.32 | 657.32 | 1.52 | 18551.8 | 2085318671.2 |

![InfiniGen Hardware Sweep](../figures/wp3/infinigen/hardware_sweep.png)

### 2.2 KL Divergence & Logit Fidelity

#### PG-19 Hardware Sweep

| Context | KL Div Mean | KL Div P99 | Top-1 Agreement (%) |
|---:|---:|---:|---:|
| 512 | 12.7462 | 15.4200 | 9.38 |
| 1024 | 8.5429 | 15.4243 | 37.50 |
| 2048 | 14.2447 | 15.4249 | 0.00 |
| 16384 | 14.0537 | 15.4249 | 0.00 |
| 32768 | 14.1768 | 15.4249 | 1.56 |

#### LongMemEval Conversational Sweep

| Context Tier | KL Div Mean | KL Div P99 | Top-1 Agreement (%) |
|---:|---:|---:|---:|
| 512 | 6.4060 | 15.4249 | 54.37 |
| 1024 | 6.1251 | 15.4249 | 54.37 |
| 2048 | 6.7887 | 15.4249 | 51.56 |
| 4096 | 7.4502 | 15.4249 | 46.70 |
| 8192 | 7.1101 | 15.4249 | 48.44 |
| 16384 | 4.9029 | 15.4249 | 63.33 |

![InfiniGen KL Fidelity](../figures/wp3/infinigen/kl_fidelity.png)

### 2.3 Needle-In-A-Haystack (NIAH) 2D Sweep

| Depth \ Context | 2048 | 4096 |
|---:|:---:|:---:|
| 0.0 | PASS | PASS |
| 0.1 | PASS | PASS |
| 0.2 | PASS | PASS |
| 0.3 | PASS | PASS |
| 0.4 | PASS | PASS |
| 0.5 | PASS | PASS |
| 0.6 | PASS | PASS |
| 0.7 | PASS | PASS |
| 0.8 | PASS | PASS |
| 0.9 | PASS | PASS |
| **Overall Accuracy** | **100.0%** | **100.0%** |

![InfiniGen NIAH Heatmap](../figures/wp3/infinigen/niah_heatmap.png)

### 2.4 LongMemEval & Post-Hoc Diagnostics

| Context Tier | Acc | F1 | Pass / Total | Quality Regressions | Baseline Limitations | Mean Retention | Eff. Bitwidth |
|---:|---:|---:|:---:|---:|---:|---:|---:|
| 512 | 0.20 | 0.11 | 2/10 | 0 | 8 | 1.00 | 16.00 |
| 1024 | 0.20 | 0.11 | 2/10 | 1 | 7 | 1.00 | 16.00 |
| 2048 | 0.20 | 0.13 | 4/20 | 1 | 15 | 1.00 | 16.00 |
| 4096 | 0.28 | 0.07 | 5/18 | 2 | 11 | 1.00 | 16.00 |
| 8192 | 0.31 | 0.08 | 8/26 | 0 | 18 | 1.00 | 16.00 |
| 16384 | 0.20 | 0.10 | 3/15 | 0 | 12 | 1.00 | 16.00 |

#### 5-Capability Accuracy Breakdown (Sample-Weighted Across Context Tiers)

| Capability | Samples | Technique Acc | Baseline Acc | Δ vs. Baseline |
|---|---:|---:|---:|---:|
| `fact_recall` | 22 | 36.4% | 50.0% | -13.6% |
| `cross_session_reasoning` | 16 | 6.2% | 6.2% | +0.0% |
| `information_update` | 20 | 50.0% | 45.0% | +5.0% |
| `relative_timeframe` | 19 | 26.3% | 21.1% | +5.3% |
| `unanswerable_refusal` | 22 | 0.0% | 0.0% | +0.0% |

![InfiniGen LongMemEval Diagnostics](../figures/wp3/infinigen/longmem_diagnostics.png)

---

## 3. Diagnostic Analysis & Metric-Backed Pros/Cons

### Why It Is Better or Worse Than Baseline

Tested with k_proxy_ratio=0.3, alpha_threshold=4.0, and max_budget_ratio=0.35. Maintaining proxy projections and speculative prefetch buffers alongside CPU swap introduces substantial auxiliary GPU memory and control-plane overhead.

On the 2D NIAH diagnostic sweep, exact-match retrieval across depths 0.0–0.9 is **2048: 100%, 4096: 100%**.

On LongMemEval (99 inspected samples across 512–16,384 tokens), InfiniGen scores **24/99** with **4** quality regressions relative to the uncompressed baseline (25/99) and a mean token retention ratio of **1.00**.

### Pros

- Achieves **100% NIAH exact-match retrieval** across all depths (0.0–0.9) at both 2,048 and 4,096 context lengths.

### Cons

- Increases peak allocated VRAM at 32K by **+14.6%** (15158 MB vs. 13225 MB baseline) due to auxiliary metadata/prefetch buffers.
- Decode latency at 32K is **22.53x slower** than baseline (**657.3 ms/tok** vs. 29.2 ms/tok).
- High host-device transfer volume at 32K (**2085318671 B/tok**, Host RSS Δ = **18552 MB**).
- Elevated PG-19 logit divergence (mean KL = **12.75**, Top-1 = **9.7%**).
- Trails baseline LongMemEval pass count (**24/99** vs. 25/99 baseline) with **4** quality regressions (mean KV retention ratio = **1.00**).
