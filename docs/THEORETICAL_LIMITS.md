# Theoretical Capacity Modeling & Roofline Analysis (WP2-1)

> **Work Package:** `[WP2-1] Theoretical Capacity Modeling & Test Matrix Specification`  
> **Parent Issue:** [`khemingkapat/kv_co-op_hub#2`](https://github.com/khemingkapat/kv_co-op_hub/issues/2)  
> **Child Issue:** [`khemingkapat/kv_bench#11`](https://github.com/khemingkapat/kv_bench/issues/11)  
> **Target Hardware:**  
> - **Primary Cluster:** NVIDIA GeForce RTX 4090 ($1,008\text{ GB/s}$, $82.6\text{ TFLOPS}$ BF16) capped at **$16.0\text{ GiB}$ VRAM**  
> - **Local Testbed:** NVIDIA GeForce RTX 4060 ($272\text{ GB/s}$, $18.0\text{ TFLOPS}$ BF16) capped at **$8.0\text{ GiB}$ VRAM**

---

## 1. Executive Summary & Purpose

Before conducting empirical hardware sweeps on the cluster, we establish an **analytical theoretical baseline**. This ensures benchmark configurations are grounded in first-principles physics rather than arbitrary context powers of two.

By modeling the memory allocation equations, paging allocator constraints, and memory bandwidth roofline curves ahead of time:
1. We identify the exact mathematical **Out-Of-Memory (OOM) cliff edges** across two distinct tiers:
   - **Tier 1 (Hard Process Allocator Cap):** The $16.0\text{ GiB}$ simulated ceiling enforced via `torch.cuda.set_per_process_memory_fraction`.
   - **Tier 2 (vLLM PagedAttention KV Pool):** The internal budget allocated to vLLM via `gpu_memory_utilization = 0.85` ($13.60\text{ GiB}$ effective budget), which determines whether sequences can be admitted simultaneously.
2. We design an empirical test matrix that intentionally probes **sub-critical**, **inflection**, **cliff-edge**, and **over-the-cliff** regimes.
3. We establish the reference ceiling against which empirical allocator overhead and memory bus contention will be quantified in **WP2-4**.

---

## 2. Mathematical Formulations

### A. KV-Cache Memory per Token (GQA in BF16)
In modern Grouped-Query Attention (GQA) architectures, attention keys and values are stored at native 16-bit precision ($2\text{ bytes}$ per element):

$$\text{KV Bytes per Token} = 2 \text{ (Key \& Value)} \times 2 \text{ bytes (BF16)} \times L_{\text{full\_attn}} \times H_{KV} \times D_{\text{head}} = 4 \cdot L_{\text{full\_attn}} \cdot H_{KV} \cdot D_{\text{head}}$$

Where:
* $L_{\text{full\_attn}}$: Number of full attention layers storing KV states.
* $H_{KV}$: Number of key/value heads.
* $D_{\text{head}}$: Head dimension.

$$\text{Total KV Cache (GiB)} = \frac{\text{KV Bytes per Token} \times C \times N}{1024^3}$$

Where $C$ is context sequence length and $N$ is concurrent session concurrency.

### B. Static Budget & Two-Tier Headroom
The total VRAM footprint at any step is:

$$M_{\text{total}} = M_{\text{weights}} + M_{\text{runtime}} + M_{\text{KV}}(C, N)$$

Under a simulated hardware ceiling $M_{\text{cap}} = 16.0\text{ GiB}$:
1. **Tier 1: Hard Process Allocator Cap ($M_{\text{cap}} = 16.0\text{ GiB}$):**
   $$\text{Net Headroom}_{\text{cap}} = M_{\text{cap}} - M_{\text{weights}} - M_{\text{runtime}}$$
2. **Tier 2: vLLM PagedAttention KV Pool ($M_{\text{vLLM}} = M_{\text{cap}} \times \text{util} = 13.60\text{ GiB}$ for $\text{util}=0.85$):**
   $$\text{Net Headroom}_{\text{pool}} = M_{\text{vLLM}} - M_{\text{weights}} - M_{\text{runtime}}$$

Where:
* $M_{\text{weights}}$: Base model weights loaded in BF16 ($P \times 2\text{ bytes} / 1024^3 \text{ GiB}$; $\approx 7.506\text{ GiB}$ for Qwen3-4B).
* $M_{\text{runtime}}$: Non-torch CUDA context, PyTorch library overhead, and peak activation workspace buffer ($\approx 0.80\text{ GiB}$ for RTX 4090, $0.60\text{ GiB}$ for RTX 4060).

### C. Theoretical OOM Cliff Threshold ($C_{\text{cliff}}$)
The maximum sustainable context length per user before exhausting the respective tier budget:

$$C_{\text{cliff}}(N) = \left\lfloor \frac{\text{Net Headroom} \times 1024^3}{N \times \text{KV Bytes per Token}} \right\rfloor$$

* If single-stream context exceeds the **Tier 2 KV Pool Limit**, vLLM fails at engine initialization (`max_seq_len > KV cache capacity`).
* If multi-user aggregate context exceeds the **Tier 2 KV Pool Limit**, vLLM enters the **queueing / preemption cliff** (excess requests wait or are recomputed).
* If total process memory exceeds the **Tier 1 Hard Cap**, PyTorch triggers a hard `CUDA out of memory`.

### D. Roofline Memory-Bandwidth Bound (Decode Phase)
Autoregressive token generation is strictly memory-bandwidth bound at small batch sizes. In each decode step, the GPU memory controller must stream all model weights plus all accumulated KV cache tensors across the memory bus:

$$\text{Bytes Read per Step} = M_{\text{weights\_bytes}} + M_{\text{KV\_bytes}}(C, B)$$

$$\text{Minimum TPOT (ms)} = \frac{\text{Bytes Read per Step}}{\text{Peak Memory Bandwidth (GB/s)} \times 10^6}$$

$$\text{Maximum Decode Throughput (tokens/sec)} = \frac{B}{\text{TPOT} / 1000}$$

$$\text{TPOT Latency Increase (\%)} = \frac{\text{TPOT}(C) - \text{TPOT}(0)}{\text{TPOT}(0)} \times 100\% = \frac{M_{\text{KV\_bytes}}}{M_{\text{weights\_bytes}}} \times 100\%$$

$$\text{Throughput Reduction (\%)} = \frac{\text{Throughput}(0) - \text{Throughput}(C)}{\text{Throughput}(0)} \times 100\%$$

### E. Prefill Phase Compute Bound (TTFT)
Prompt prefill FLOPs incorporate linear matrix projections and causal self-attention (FlashAttention-2 computing the causal lower triangle $C^2/2$):

$$\text{Linear FLOPs} = 2 \cdot P \cdot C$$

$$\text{Causal Attention FLOPs} = 2 \cdot L_{\text{full\_attn}} \cdot (H_Q \cdot D_{\text{head}}) \cdot C^2$$

$$\text{Total Prefill FLOPs} = \text{Linear FLOPs} + \text{Causal Attention FLOPs}$$

$$\text{Minimum TTFT (sec)} = \frac{\text{Total Prefill FLOPs}}{\text{Peak Tensor Core TFLOPS} \times 10^{12}}$$

---

## 3. Theoretical Analysis: Primary Model (`qwen3-4b`)

Target Checkpoint: `Qwen/Qwen3-4B-Instruct-2507`  
Hardware: RTX 4090 capped at **$16.0\text{ GiB}$ VRAM** ($1,008\text{ GB/s}$, $82.6\text{ TFLOPS}$)

* **Architecture:** $L=36$, $H_{KV}=8$, $H_Q=32$, $D_{\text{head}}=128$, Params $\approx 4.03\text{B}$  
* **KV Bytes per Token:** $4 \times 36 \times 8 \times 128 = 147,456\text{ bytes} = \mathbf{144\text{ KiB/token}}$  
* **Model Weights:** $8.06\text{ GB}$ decimal = $\mathbf{7.506\text{ GiB}}$ binary  
* **Runtime Overhead Buffer:** $\mathbf{0.80\text{ GiB}}$  
* **Net Usable Headroom (Tier 1 Process Cap $16.0\text{ GiB}$):** $16.0 - 7.506 - 0.80 = \mathbf{7.694\text{ GiB}}$  
* **Net Usable Headroom (Tier 2 vLLM Pool $13.6\text{ GiB}$ at `util=0.85`):** $13.6 - 7.506 - 0.80 = \mathbf{5.294\text{ GiB}}$  

### Theoretical OOM Context Thresholds

| Concurrency ($N$) | vLLM Pool Limit (`util=0.85`) | Process Cap Limit (`util=1.0`) | Failure / Saturation Mode |
| :---: | :---: | :---: | :--- |
| **1 user** | **$38,546$ tokens** | $56,022$ tokens | Single-stream fits 32K safely; pool exhausted if $>38.5\text{K}$. |
| **4 users** | **$9,636$ tokens** | $14,005$ tokens | 4K context fits easily ($4 \times 4096 = 16.3\text{K}$ tokens); 16K triggers OOM. |
| **8 users** | **$4,818$ tokens** | $7,002$ tokens | 4K context fits near edge; 8K triggers pool exhaustion. |
| **12 users** | **$3,212$ tokens** | $4,668$ tokens | **Critical inflection:** 2K context fits ($24.5\text{K}$ tokens); 4K context demands $49.1\text{K}$ tokens $\rightarrow$ triggers pool queueing/preemption. |
| **16 users** | **$2,409$ tokens** | $3,501$ tokens | 2K context fits near edge; 4K context triggers pool exhaustion. |

### Single-Stream Context Sweep & Roofline Bounds ($N=1$)

| Context ($C$) | Regime | Total VRAM (GiB) | Pool Headroom | Min TTFT (ms) | Min TPOT (ms) | Peak Tok/s | Latency $\Delta$ | Throughput $\Delta$ | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `512` | Sub-Critical | $8.38$ | $5.22\text{ GiB}$ | $50.9$ | $8.07$ | $123.9$ | $+0.9\%$ | $-0.9\%$ | ✅ Fits |
| `1,024` | Sub-Critical | $8.45$ | $5.15\text{ GiB}$ | $103.7$ | $8.15$ | $122.8$ | $+1.9\%$ | $-1.8\%$ | ✅ Fits |
| `2,048` | Sub-Critical | $8.59$ | $5.01\text{ GiB}$ | $214.8$ | $8.30$ | $120.5$ | $+3.8\%$ | $-3.6\%$ | ✅ Fits |
| `4,096` | Sub-Critical | $8.87$ | $4.73\text{ GiB}$ | $459.6$ | $8.60$ | $116.3$ | $+7.5\%$ | $-7.0\%$ | ✅ Fits |
| `8,192` | Sub-Critical | $9.43$ | $4.17\text{ GiB}$ | $1039.0$ | $9.19$ | $108.8$ | $+15.0\%$ | $-13.0\%$ | ✅ Fits |
| `16,384` | Inflection | $10.56$ | $3.04\text{ GiB}$ | $2557.1$ | $10.39$ | $96.2$ | $+30.0\%$ | $-23.1\%$ | ✅ Fits |
| `32,768` | Cliff Edge | $12.81$ | $0.79\text{ GiB}$ | $7031.1$ | $12.79$ | $78.2$ | $+60.0\%$ | $-37.5\%$ | ⚡ **Near Edge** |
| `49,152` | Pool Exhaustion | $15.06$ | $-1.46\text{ GiB}$ | $13421.9$ | $15.19$ | $65.8$ | $+89.9\%$ | $-47.4\%$ | ⚠️ **Pool OOM** |
| `65,536` | Hard Process OOM | $17.31$ | $-3.71\text{ GiB}$ | $21729.5$ | $17.58$ | $56.9$ | $+119.9\%$ | $-54.5\%$ | ❌ **Hard OOM** |

---

## 4. Multi-User Concurrency Grid ($N \times C$)

Total memory footprint against the **$13.60\text{ GiB}$ vLLM Pool** (`util=0.85`):

| Concurrent Users ($N$) | $C=1,024$ | $C=2,048$ | $C=4,096$ | $C=8,192$ |
| :---: | :---: | :---: | :---: | :---: |
| **$N = 1$** | ✅ $8.4\text{ GiB}$ | ✅ $8.6\text{ GiB}$ | ✅ $8.9\text{ GiB}$ | ✅ $9.4\text{ GiB}$ |
| **$N = 4$** | ✅ $8.9\text{ GiB}$ | ✅ $9.4\text{ GiB}$ | ✅ $10.6\text{ GiB}$ | ⚡ **$12.8\text{ GiB}$ (Edge)** |
| **$N = 8$** | ✅ $9.4\text{ GiB}$ | ✅ $10.6\text{ GiB}$ | ⚡ **$12.8\text{ GiB}$ (Edge)** | ❌ **$17.3\text{ GiB}$ (Hard OOM)** |
| **$N = 12$** | ✅ $10.0\text{ GiB}$ | ✅ $11.7\text{ GiB}$ | ⚠️ **$15.1\text{ GiB}$ (Pool OOM)** | ❌ **$21.8\text{ GiB}$ (Hard OOM)** |
| **$N = 16$** | ✅ $10.6\text{ GiB}$ | ⚡ **$12.8\text{ GiB}$ (Edge)** | ❌ **$17.3\text{ GiB}$ (Hard OOM)** | ❌ **$26.3\text{ GiB}$ (Hard OOM)** |

---

## 5. Peer Model Theoretical Comparison

| Model | Hardware Budget | Weights (Binary) | KV Bytes/Token | Pool Headroom (`0.85`) | Single-User Pool Cliff | 12-User Pool Cliff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qwen3-4b`** | 16 GiB | $7.506\text{ GiB}$ | $144\text{ KiB}$ | $5.294\text{ GiB}$ | $38,546$ tokens | $3,212$ tokens |
| **`phi4-mini`** | 16 GiB | $7.125\text{ GiB}$ | $128\text{ KiB}$ | $5.675\text{ GiB}$ | $46,492$ tokens | $3,874$ tokens |
| **`qwen3-1.7b`** | 8 GiB | $3.213\text{ GiB}$ | $112\text{ KiB}$ | $2.987\text{ GiB}$ | $27,964$ tokens | $2,330$ tokens |
| **`qwen3.5-4b` (Hybrid)** | 16 GiB | $7.544\text{ GiB}$ | $32\text{ KiB}$ (+GDN) | $5.256\text{ GiB}$ | $172,238$ tokens | $14,353$ tokens |

---

## 6. The Empirical Test Matrix Specification

To ensure rigorous validation of both scaling performance and failure boundaries, the profiling run in **WP2-2** will execute:

### Stream A: Single-Stream Depth Sweep
* **Context Array:** `512, 1024, 2048, 4096, 8192, 16384, 32768, 49152, 65536`
* **Test Categorization:**
  * **Sub-Critical (`512` to `8192`):** Validates raw throughput, baseline TTFT scaling, and verifies linear memory growth without paging pressure.
  * **Inflection (`16384`):** Verifies memory bandwidth degradation (predicted $-23.1\%$ throughput reduction / $+30.0\%$ TPOT latency).
  * **Cliff-Edge (`32768`):** Tests allocator behavior at $>90\%$ vLLM pool utilization (predicted $-37.5\%$ throughput reduction).
  * **Pool Exhaustion (`49152`):** Verifies vLLM pool exhaustion behavior ($15.06\text{ GiB} > 13.60\text{ GiB}$ pool).
  * **Over-the-Cliff (`65536`):** Verifies graceful out-of-memory handling and vLLM abort behavior under process cap ($17.31\text{ GiB} > 16.0\text{ GiB}$).

### Stream B: Multi-User Concurrency Sweep
* **User Array:** `1, 4, 8, 12, 16` concurrent sessions.
* **Context Depths:** `1024, 2048, 4096, 8192` tokens.
* **Critical Probe Points:**
  * **Normal Operation:** $N=4$ at $1024, 2048$ ($8.9\text{--}9.4\text{ GiB}$, ample pool headroom).
  * **Borderline Edge Point:** $N=8$ at $4096$ and $N=12$ at $2048$ ($11.7\text{--}12.8\text{ GiB}$, near pool capacity).
  * **Pool Exhaustion / Preemption Point:** $N=12$ at $4096$ ($15.1\text{ GiB}$). Verifies whether vLLM preempts/queues excess requests or degrades P95 latency.
  * **Guaranteed Hard OOM Point:** $N=8$ and $N=12$ at $8192$ ($>17.3\text{ GiB}$). Verifies cluster scheduler cancellation and memory reclamation.

### Stream C: Quality Ground-Truth Reference (LongMemEval)
* **Context Depths:** `512, 1024, 2048, 4096` tokens.
* **Sampling:** 10 samples per task across all 5 task capabilities (50 prompts total), deterministic random seed (`seed=42`).
* **Artifact Output:** Persist golden reference logprobs and token IDs (`results/references/baseline_qwen3_4b_ref.json`).
