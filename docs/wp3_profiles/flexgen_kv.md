# WP3-6: FlexGen-KV

**Primary Axis:** `SPATIAL` | **All Axes:** `SPATIAL, STRUCTURAL` | **Model:** `Qwen3-4B-Instruct-2507` (16.0 GiB Cap, PyTorch Native SDPA)

**Artifact Provenance (SHA-256):** `flexgen_kv_hardware_pytorch_16gb.json`: `VERIFIED` (`bd79f1dd9965…`) | `flexgen_kv_longmem_pytorch_16gb.json`: `VERIFIED` (`6861240031c7…`) | `flexgen_kv_niah_pytorch_16gb.json`: `VERIFIED` (`6ff0ae38d44b…`)

---

## 1. What It Does & How (Brief)

FlexGen-KV tiers the KV cache across GPU VRAM and host CPU RAM, keeping a bounded active budget on the GPU while offloading older KV blocks to CPU memory with group-wise quantization to reduce host-device PCIe transfer volume.

- **Evaluated Configuration:** `gpu_cache_tokens=1024`, `cpu_cache_tokens=32768`, `block_size=16`, `quantize_kv=True`, `group_size=64`, `cpu_delegation=False`, `swap_space_gb=4`

---

## 2. Empirical Results

### 2.1 Hardware Metric Sweep (512 → 32,768)

| Context | Status | Peak Alloc (MB) | Peak Reserved (MB) | Warm TTFT (s) | Cold TTFT (s) | TPOT Mean (ms) | TPOT P95 (ms) | Throughput (tok/s) | Host RSS Δ (MB) | Memcpy (B/tok) |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | SUCCESS | 7911.7 | 7998.0 | 0.039 | 0.000 | 20.33 | 20.33 | 49.18 | 1981.8 | 201.8 |
| 1024 | SUCCESS | 8126.5 | 8164.0 | 0.070 | 0.000 | 44.94 | 44.94 | 22.25 | 3447.6 | 1347521.8 |
| 2048 | SUCCESS | 8422.9 | 8464.0 | 0.188 | 0.000 | 54.37 | 54.37 | 18.39 | 3500.6 | 44478657.8 |
| 4096 | SUCCESS | 8422.9 | 8766.0 | 0.440 | 0.000 | 68.92 | 68.92 | 14.51 | 3709.9 | 132731585.8 |
| 8192 | SUCCESS | 8423.0 | 8766.0 | 1.099 | 0.000 | 71.55 | 71.55 | 13.98 | 4123.0 | 317200065.8 |
| 16384 | SUCCESS | 8520.8 | 8786.0 | 3.126 | 0.000 | 111.51 | 111.51 | 8.97 | 4925.4 | 717987521.8 |
| 32768 | SUCCESS | 8888.9 | 8986.0 | 10.205 | 0.000 | 229.57 | 229.57 | 4.36 | 6564.6 | 1646964417.8 |

![FlexGen-KV Hardware Sweep](../figures/wp3/flexgen_kv/hardware_sweep.png)

### 2.2 KL Divergence & Logit Fidelity

#### PG-19 Hardware Sweep

| Context | KL Div Mean | KL Div P99 | Top-1 Agreement (%) |
|---:|---:|---:|---:|
| 512 | 0.0000 | 0.0000 | 100.00 |
| 1024 | 10.8596 | 15.4247 | 23.44 |
| 2048 | 12.2849 | 15.4235 | 7.81 |
| 4096 | 13.2523 | 15.2318 | 3.12 |
| 8192 | 14.1239 | 15.4249 | 0.00 |
| 16384 | 13.1506 | 15.4249 | 6.25 |
| 32768 | 11.9269 | 15.4249 | 10.94 |

#### LongMemEval Conversational Sweep

| Context Tier | KL Div Mean | KL Div P99 | Top-1 Agreement (%) |
|---:|---:|---:|---:|
| 512 | 0.0000 | 0.0000 | 100.00 |
| 1024 | 0.0000 | 0.0000 | 100.00 |
| 2048 | 7.4357 | 15.4249 | 46.25 |
| 4096 | 10.8082 | 15.4249 | 23.44 |
| 8192 | 11.7038 | 15.4249 | 18.15 |
| 16384 | 14.1135 | 15.4249 | 3.12 |

![FlexGen-KV KL Fidelity](../figures/wp3/flexgen_kv/kl_fidelity.png)

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

![FlexGen-KV NIAH Heatmap](../figures/wp3/flexgen_kv/niah_heatmap.png)

### 2.4 LongMemEval & Post-Hoc Diagnostics

| Context Tier | Acc | F1 | Pass / Total | Quality Regressions | Baseline Limitations | Mean Retention | Eff. Bitwidth |
|---:|---:|---:|:---:|---:|---:|---:|---:|
| 512 | 0.20 | 0.12 | 2/10 | 0 | 8 | 1.00 | 16.00 |
| 1024 | 0.30 | 0.13 | 3/10 | 0 | 7 | 1.00 | 16.00 |
| 2048 | 0.25 | 0.11 | 5/20 | 1 | 14 | 1.00 | 16.00 |
| 4096 | 0.39 | 0.09 | 7/18 | 1 | 10 | 1.00 | 16.00 |
| 8192 | 0.23 | 0.08 | 6/26 | 1 | 19 | 1.00 | 16.00 |
| 16384 | 0.07 | 0.06 | 1/15 | 3 | 11 | 1.00 | 16.00 |

#### 5-Capability Accuracy Breakdown (Sample-Weighted Across Context Tiers)

| Capability | Samples | Technique Acc | Baseline Acc | Δ vs. Baseline |
|---|---:|---:|---:|---:|
| `fact_recall` | 22 | 50.0% | 50.0% | +0.0% |
| `cross_session_reasoning` | 16 | 0.0% | 6.2% | -6.2% |
| `information_update` | 20 | 40.0% | 45.0% | -5.0% |
| `relative_timeframe` | 19 | 26.3% | 21.1% | +5.3% |
| `unanswerable_refusal` | 22 | 0.0% | 0.0% | +0.0% |

![FlexGen-KV LongMemEval Diagnostics](../figures/wp3/flexgen_kv/longmem_diagnostics.png)

---

## 3. Diagnostic Analysis & Metric-Backed Pros/Cons

### Why It Is Better or Worse Than Baseline

Tested with gpu_cache_tokens=1,024, cpu_cache_tokens=32,768, group_size=64, and quantize_kv=True. All context tokens remain accessible via CPU tiering, making PCIe bandwidth and host-to-device staging the primary latency bottleneck at long contexts.

On the 2D NIAH diagnostic sweep, exact-match retrieval across depths 0.0–0.9 is **2048: 100%, 4096: 100%**.

On LongMemEval (99 inspected samples across 512–16,384 tokens), FlexGen-KV scores **24/99** with **6** quality regressions relative to the uncompressed baseline (25/99) and a mean token retention ratio of **1.00**.

### Pros

- Reduces peak allocated VRAM at 32K by **32.8%** (8889 MB vs. 13225 MB baseline).
- Achieves **100% NIAH exact-match retrieval** across all depths (0.0–0.9) at both 2,048 and 4,096 context lengths.

### Cons

- Decode latency at 32K is **7.87x slower** than baseline (**229.6 ms/tok** vs. 29.2 ms/tok).
- High host-device transfer volume at 32K (**1646964418 B/tok**, Host RSS Δ = **6565 MB**).
- Elevated PG-19 logit divergence (mean KL = **10.80**, Top-1 = **21.7%**).
- Trails baseline LongMemEval pass count (**24/99** vs. 25/99 baseline) with **6** quality regressions (mean KV retention ratio = **1.00**).
