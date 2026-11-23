# WP3-3: TurboQuant

**Primary Axis:** `STRUCTURAL` | **All Axes:** `STRUCTURAL` | **Model:** `Qwen3-4B-Instruct-2507` (16.0 GiB Cap, PyTorch Native SDPA)

**Artifact Provenance (SHA-256):** `turboquant_hardware_pytorch_16gb.json`: `VERIFIED` (`6dfdb5bbd319…`) | `turboquant_longmem_pytorch_16gb.json`: `UPDATED (post-manifest re-run)` (`bbfd97bf6053…`) | `turboquant_niah_pytorch_16gb.json`: `VERIFIED` (`08562dd7526b…`)

---

## 1. What It Does & How (Brief)

TurboQuant applies asymmetric low-bit quantization to Keys and Values, using a Walsh-Hadamard rotation to smooth outlier channels before quantization and a Quantized Johnson-Lindenstrauss (QJL) sketch on Values to preserve inner-product fidelity.

- **Evaluated Configuration:** `k_bits=4`, `v_bits=3`, `qjl_v=True`, `qjl_k=False`, `block_size=16`, `compression_ratio=3.657`, `theoretical_bytes_per_token_head=140`

---

## 2. Empirical Results

### 2.1 Hardware Metric Sweep (512 → 32,768)

| Context | Status | Peak Alloc (MB) | Peak Reserved (MB) | Warm TTFT (s) | Cold TTFT (s) | TPOT Mean (ms) | TPOT P95 (ms) | Throughput (tok/s) | Host RSS Δ (MB) | Memcpy (B/tok) |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | SUCCESS | 7869.6 | 7892.0 | 0.148 | 0.000 | 82.25 | 82.25 | 12.16 | 4816.1 | 4922.2 |
| 1024 | SUCCESS | 8053.1 | 8074.0 | 0.177 | 0.000 | 81.73 | 81.73 | 12.24 | 4816.2 | 5050.2 |
| 2048 | SUCCESS | 8358.0 | 8388.0 | 0.327 | 0.000 | 79.53 | 79.53 | 12.57 | 4678.9 | 5315.2 |
| 4096 | SUCCESS | 8436.8 | 8688.0 | 0.677 | 0.000 | 81.24 | 81.24 | 12.31 | 4969.8 | 5845.2 |
| 8192 | SUCCESS | 8594.3 | 8888.0 | 1.619 | 0.000 | 128.84 | 128.84 | 7.76 | 5159.3 | 6905.2 |
| 16384 | SUCCESS | 9152.2 | 9408.0 | 5.436 | 0.000 | 420.20 | 420.20 | 2.38 | 6146.4 | 9025.2 |
| 32768 | SUCCESS | 10294.4 | 10732.0 | 20.627 | 0.000 | 864.12 | 864.12 | 1.16 | 7528.9 | 13265.2 |

![TurboQuant Hardware Sweep](../figures/wp3/turboquant/hardware_sweep.png)

### 2.2 KL Divergence & Logit Fidelity

#### PG-19 Hardware Sweep

| Context | KL Div Mean | KL Div P99 | Top-1 Agreement (%) |
|---:|---:|---:|---:|
| 512 | 12.7666 | 15.4200 | 9.38 |
| 1024 | 11.9099 | 15.4249 | 12.50 |
| 2048 | 10.9512 | 15.4249 | 21.88 |
| 4096 | 13.3091 | 15.2225 | 0.00 |
| 8192 | 10.5429 | 15.4249 | 23.44 |
| 16384 | 11.5250 | 15.4249 | 18.75 |
| 32768 | 12.2347 | 15.4249 | 15.62 |

#### LongMemEval Conversational Sweep

| Context Tier | KL Div Mean | KL Div P99 | Top-1 Agreement (%) |
|---:|---:|---:|---:|
| 32768 | 11.2884 | 15.4249 | 22.60 |

![TurboQuant KL Fidelity](../figures/wp3/turboquant/kl_fidelity.png)

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

![TurboQuant NIAH Heatmap](../figures/wp3/turboquant/niah_heatmap.png)

### 2.4 LongMemEval & Post-Hoc Diagnostics

**LongMemEval v2 (150 Quantile-Sampled Official Questions, Single-Pass Native Length @ 32K, Chat Template, `max_tokens=128`, Semantic Graded):**

| Context Ceiling | Semantic Acc | Pass / Total | Quality Regressions | Baseline Limitations | Unexpected Gains | EOS Clean Stops | Mean Retention | Eff. Bitwidth |
|---:|---:|:---:|---:|---:|---:|:---:|---:|---:|
| 32768 | **53.3%** | **80/150** | 20 | 50 | 1 | 142/150 | 1.00 | 4.38 |

#### 5-Capability Semantic Accuracy Breakdown (30 Official Questions / Capability)

| Capability | Samples | Technique Acc | Baseline Acc | Δ vs. Baseline |
|---|---:|---:|---:|---:|
| `fact_recall` | 30 | 73.3% | 90.0% | -16.7% |
| `cross_session_reasoning` | 30 | 23.3% | 43.3% | -20.0% |
| `information_update` | 30 | 53.3% | 66.7% | -13.3% |
| `relative_timeframe` | 30 | 20.0% | 30.0% | -10.0% |
| `unanswerable_refusal` | 30 | 96.7% | 100.0% | -3.3% |

![TurboQuant LongMemEval Diagnostics](../figures/wp3/turboquant/longmem_diagnostics.png)

---

## 3. Diagnostic Analysis & Metric-Backed Pros/Cons

### Why It Is Better or Worse Than Baseline

Tested at 4-bit Keys and 3-bit Values (qjl_v=True, qjl_k=False, ~3.66x theoretical compression, 140 B/token-head). All tokens remain resident in the cache rather than being evicted, trading Python-level dequantization/rotation latency for full-context retention.

On the 2D NIAH diagnostic sweep, exact-match retrieval across depths 0.0–0.9 is **2048: 100%, 4096: 100%**.

On LongMemEval v2 (150 quantile-sampled official questions at native length up to 32K, chat-templated, `max_tokens=128`, semantically graded), TurboQuant achieves **80/150 (53.3%)** vs. baseline **99/150 (66.0%)**, with **20** quality regressions and mean KV retention ratio **1.00**.

### Pros

- Reduces peak allocated VRAM at 32K by **22.2%** (10294 MB vs. 13225 MB baseline).
- Achieves **100% NIAH exact-match retrieval** across all depths (0.0–0.9) at both 2,048 and 4,096 context lengths.

### Cons

- Decode latency at 32K is **29.62x slower** than baseline (**864.1 ms/tok** vs. 29.2 ms/tok).
- High host-device transfer volume at 32K (**13265 B/tok**, Host RSS Δ = **7529 MB**).
- Elevated PG-19 logit divergence (mean KL = **11.89**, Top-1 = **14.5%**).
- Degrades LongMemEval v2 semantic accuracy to **80/150 (53.3%)** vs. 99/150 (66.0%) baseline (**20** quality regressions).
