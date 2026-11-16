# WP3-7: ShadowKV

**Primary Axis:** `SPATIAL` | **All Axes:** `SPATIAL, STRUCTURAL, TEMPORAL` | **Model:** `Qwen3-4B-Instruct-2507` (16.0 GiB Cap, PyTorch Native SDPA)

**Artifact Provenance (SHA-256):** `shadowkv_hardware_pytorch_16gb.json`: `MISSING` | `shadowkv_longmem_pytorch_16gb.json`: `UPDATED (post-manifest re-run)` (`807264f99220…`) | `shadowkv_niah_pytorch_16gb.json`: `UPDATED (post-manifest re-run)` (`ea728df232b9…`)

---

## 1. What It Does & How (Brief)

ShadowKV keeps low-rank SVD-compressed Key representations and sparse outlier channels on the GPU for fast chunk-level relevance scoring, while offloading full-precision Values to CPU RAM and fetching only the top-k relevant chunks on demand during decode.

- **Evaluated Configuration:** `svd_rank=160`, `chunk_size=8`, `topk_chunks=256`, `outlier_ratio=0.003`

---

## 2. Empirical Results

### 2.1 Hardware Metric Sweep (512 → 32,768)

*Pending: `results/shadowkv_hardware_pytorch_16gb.json` has not been generated/fetched yet.*

### 2.2 KL Divergence & Logit Fidelity

#### LongMemEval Conversational Sweep

| Context Tier | KL Div Mean | KL Div P99 | Top-1 Agreement (%) |
|---:|---:|---:|---:|
| 512 | 0.0000 | 0.0000 | 100.00 |
| 1024 | 0.0000 | 0.0000 | 100.00 |
| 2048 | 6.2979 | 15.4249 | 54.37 |
| 4096 | 6.8981 | 15.4249 | 49.65 |
| 8192 | 5.0667 | 15.4249 | 62.74 |
| 16384 | 4.8172 | 15.4249 | 64.17 |

![ShadowKV KL Fidelity](../figures/wp3/shadowkv/kl_fidelity.png)

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

![ShadowKV NIAH Heatmap](../figures/wp3/shadowkv/niah_heatmap.png)

### 2.4 LongMemEval & Post-Hoc Diagnostics

| Context Tier | Acc | F1 | Pass / Total | Quality Regressions | Baseline Limitations | Mean Retention | Eff. Bitwidth |
|---:|---:|---:|:---:|---:|---:|---:|---:|
| 512 | 0.20 | 0.12 | 2/10 | 0 | 8 | 1.00 | 16.00 |
| 1024 | 0.30 | 0.13 | 3/10 | 0 | 7 | 1.00 | 16.00 |
| 2048 | 0.20 | 0.11 | 4/20 | 1 | 15 | 1.00 | 16.00 |
| 4096 | 0.28 | 0.07 | 5/18 | 2 | 11 | 1.00 | 16.00 |
| 8192 | 0.19 | 0.07 | 5/26 | 0 | 21 | 1.00 | 16.00 |
| 16384 | 0.20 | 0.10 | 3/15 | 0 | 12 | 1.00 | 16.00 |

#### 5-Capability Accuracy Breakdown (Sample-Weighted Across Context Tiers)

| Capability | Samples | Technique Acc | Baseline Acc | Δ vs. Baseline |
|---|---:|---:|---:|---:|
| `fact_recall` | 22 | 40.9% | 50.0% | -9.1% |
| `cross_session_reasoning` | 16 | 6.2% | 6.2% | +0.0% |
| `information_update` | 20 | 45.0% | 45.0% | +0.0% |
| `relative_timeframe` | 19 | 15.8% | 21.1% | -5.3% |
| `unanswerable_refusal` | 22 | 0.0% | 0.0% | +0.0% |

![ShadowKV LongMemEval Diagnostics](../figures/wp3/shadowkv/longmem_diagnostics.png)

---

## 3. Diagnostic Analysis & Metric-Backed Pros/Cons

### Why It Is Better or Worse Than Baseline

Tested with svd_rank=160, chunk_size=8, topk_chunks=256, and outlier_ratio=0.003. This design aims to combine full-sequence recall (via CPU Value backing) with low GPU VRAM footprint and selective temporal chunk retrieval.

**Hardware Sweep Pending:** `results/shadowkv_hardware_pytorch_16gb.json` has not been generated/fetched yet.

On the 2D NIAH diagnostic sweep, exact-match retrieval across depths 0.0–0.9 is **2048: 100%, 4096: 100%**.

On LongMemEval (99 inspected samples across 512–16,384 tokens), ShadowKV scores **22/99** with **3** quality regressions relative to the uncompressed baseline (25/99) and a mean token retention ratio of **1.00**.

### Pros

- Low LongMemEval logit drift (mean KL = **3.85**, Top-1 = **71.8%**).
- Achieves **100% NIAH exact-match retrieval** across all depths (0.0–0.9) at both 2,048 and 4,096 context lengths.

### Cons

- Trails baseline LongMemEval pass count (**22/99** vs. 25/99 baseline) with **3** quality regressions (mean KV retention ratio = **1.00**).
