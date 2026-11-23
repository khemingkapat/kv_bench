# WP3-8: CacheGen

**Primary Axis:** `STRUCTURAL` | **All Axes:** `STRUCTURAL, SPATIAL` | **Model:** `Qwen3-4B-Instruct-2507` (16.0 GiB Cap, PyTorch Native SDPA)

**Artifact Provenance (SHA-256):** `cachegen_hardware_pytorch_16gb.json`: `VERIFIED` (`acc433feceac…`) | `cachegen_longmem_pytorch_16gb.json`: `UPDATED (post-manifest re-run)` (`d6420e6a6e3f…`) | `cachegen_niah_pytorch_16gb.json`: `VERIFIED` (`67b12f9ee081…`)

---

## 1. What It Does & How (Brief)

CacheGen encodes KV tensors using anchor-delta compression (storing periodic anchor tokens at higher precision and encoding intermediate tokens as deltas) combined with layer-tiered bit allocation across shallow, middle, and deep transformer layers.

- **Evaluated Configuration:** `anchor_group_size=10`, `shallow_bits=8`, `mid_bits=4`, `deep_bits=2`, `block_size=16`

---

## 2. Empirical Results

### 2.1 Hardware Metric Sweep (512 → 32,768)

| Context | Status | Peak Alloc (MB) | Peak Reserved (MB) | Warm TTFT (s) | Cold TTFT (s) | TPOT Mean (ms) | TPOT P95 (ms) | Throughput (tok/s) | Host RSS Δ (MB) | Memcpy (B/tok) |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | SUCCESS | 7891.8 | 7906.0 | 0.064 | 0.000 | 40.66 | 40.66 | 24.59 | 3127.7 | 253.5 |
| 1024 | SUCCESS | 8101.6 | 8112.0 | 0.095 | 0.000 | 40.99 | 40.99 | 24.40 | 3136.8 | 1888125.5 |
| 2048 | SUCCESS | 8394.0 | 8436.0 | 0.223 | 0.000 | 40.88 | 40.88 | 24.46 | 3198.5 | 5659266.6 |
| 4096 | SUCCESS | 8523.5 | 8726.0 | 0.524 | 0.000 | 40.07 | 40.07 | 24.96 | 3255.3 | 11318405.9 |
| 8192 | SUCCESS | 8753.6 | 9026.0 | 1.338 | 0.000 | 39.97 | 39.97 | 25.02 | 3438.2 | 52794513.2 |
| 16384 | SUCCESS | 9311.6 | 9646.0 | 3.803 | 0.000 | 43.09 | 43.09 | 23.21 | 3826.7 | 226229415.7 |
| 32768 | SUCCESS | 10600.2 | 10966.0 | 12.555 | 0.000 | 82.87 | 82.87 | 12.07 | 4535.8 | 935016150.9 |

![CacheGen Hardware Sweep](../figures/wp3/cachegen/hardware_sweep.png)

### 2.2 KL Divergence & Logit Fidelity

#### PG-19 Hardware Sweep

| Context | KL Div Mean | KL Div P99 | Top-1 Agreement (%) |
|---:|---:|---:|---:|
| 512 | 12.6342 | 15.4199 | 6.25 |
| 1024 | 13.0502 | 15.4249 | 6.25 |
| 2048 | 11.8169 | 15.4248 | 9.38 |
| 4096 | 13.4983 | 15.2393 | 3.12 |
| 8192 | 12.2941 | 15.4249 | 9.38 |
| 16384 | 13.6631 | 15.4249 | 1.56 |
| 32768 | 13.1353 | 15.4249 | 7.81 |

#### LongMemEval Conversational Sweep

| Context Tier | KL Div Mean | KL Div P99 | Top-1 Agreement (%) |
|---:|---:|---:|---:|
| 32768 | 12.7129 | 15.4249 | 11.74 |

![CacheGen KL Fidelity](../figures/wp3/cachegen/kl_fidelity.png)

### 2.3 Needle-In-A-Haystack (NIAH) 2D Sweep

| Depth \ Context | 2048 | 4096 |
|---:|:---:|:---:|
| 0.0 | FAIL | PASS |
| 0.1 | FAIL | PASS |
| 0.2 | PASS | FAIL |
| 0.3 | FAIL | PASS |
| 0.4 | PASS | PASS |
| 0.5 | PASS | PASS |
| 0.6 | FAIL | PASS |
| 0.7 | FAIL | PASS |
| 0.8 | FAIL | PASS |
| 0.9 | PASS | FAIL |
| **Overall Accuracy** | **40.0%** | **80.0%** |

### 2.4 LongMemEval & Post-Hoc Diagnostics

**LongMemEval (150 Quantile-Sampled Official Questions, Single-Pass Native Length @ 32K, Chat Template, `max_tokens=128`, Semantic Graded):**

| Context Ceiling | Semantic Acc | Pass / Total | Quality Regressions | Baseline Limitations | Unexpected Gains | EOS Clean Stops | Mean Retention | Eff. Bitwidth |
|---:|---:|:---:|---:|---:|---:|:---:|---:|---:|
| 32768 | **44.0%** | **66/150** | 36 | 48 | 3 | 45/150 | 1.00 | 6.39 |

#### 5-Capability Semantic Accuracy Breakdown (30 Official Questions / Capability)

| Capability | Samples | Technique Acc | Baseline Acc | Δ vs. Baseline |
|---|---:|---:|---:|---:|
| `fact_recall` | 30 | 46.7% | 90.0% | -43.3% |
| `cross_session_reasoning` | 30 | 20.0% | 43.3% | -23.3% |
| `information_update` | 30 | 50.0% | 66.7% | -16.7% |
| `relative_timeframe` | 30 | 16.7% | 30.0% | -13.3% |
| `unanswerable_refusal` | 30 | 86.7% | 100.0% | -13.3% |

![CacheGen LongMemEval Diagnostics](../figures/wp3/cachegen/longmem_diagnostics.png)

---

## 3. Diagnostic Analysis & Metric-Backed Pros/Cons

### Why It Is Better or Worse Than Baseline

Tested with anchor_group_size=10, shallow_bits=8, mid_bits=4, and deep_bits=2. Aggressive 2-bit quantization in deep layers reduces memory footprint but can degrade fine-grained multi-hop and exact-match retrieval.

On the 2D NIAH diagnostic sweep, exact-match retrieval across depths 0.0–0.9 is **2048: 40%, 4096: 80%**.

On LongMemEval (150 quantile-sampled official questions at native length up to 32K, chat-templated, `max_tokens=128`, semantically graded), CacheGen achieves **66/150 (44.0%)** vs. baseline **99/150 (66.0%)**, with **36** quality regressions and mean KV retention ratio **1.00**.

### Pros

- Reduces peak allocated VRAM at 32K by **19.8%** (10600 MB vs. 13225 MB baseline).

### Cons

- Decode latency at 32K is **2.84x slower** than baseline (**82.9 ms/tok** vs. 29.2 ms/tok).
- High host-device transfer volume at 32K (**935016151 B/tok**, Host RSS Δ = **4536 MB**).
- Elevated PG-19 logit divergence (mean KL = **12.87**, Top-1 = **6.2%**).
- NIAH retrieval drops to **80%** at 4,096 tokens (failing at depths `[0.2, 0.9]`), confirming loss of early/non-local needle tokens.
- Degrades LongMemEval semantic accuracy to **66/150 (44.0%)** vs. 99/150 (66.0%) baseline (**36** quality regressions).
