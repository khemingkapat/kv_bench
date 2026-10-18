# Theoretical Capacity Modeling & Roofline Analysis (WP2-1)

> **Work Package:** `[WP2-1] Theoretical Capacity Modeling & Test Matrix Specification`  
> **Parent Issue:** [`khemingkapat/kv_co-op_hub#2`](https://github.com/khemingkapat/kv_co-op_hub/issues/2)  
> **Child Issue:** [`khemingkapat/kv_bench#11`](https://github.com/khemingkapat/kv_bench/issues/11)  
> **Target Hardware:**  
> - **Primary Cluster:** NVIDIA GeForce RTX 4090 ($1,008\text{ GB/s}$, $82.6\text{ TFLOPS}$ BF16) capped at **$16.0\text{ GB}$ VRAM**  
> - **Local Testbed:** NVIDIA GeForce RTX 4060 ($272\text{ GB/s}$, $18.0\text{ TFLOPS}$ BF16) capped at **$8.0\text{ GB}$ VRAM**

---

## 1. Executive Summary & Purpose

Before conducting empirical hardware sweeps on the cluster, we establish an **analytical theoretical baseline**. This ensures benchmark configurations are grounded in first-principles physics rather than arbitrary context powers of two.

By modeling the memory allocation equations, paging allocator constraints, and memory bandwidth roofline curves ahead of time:
1. We identify the exact mathematical **Out-Of-Memory (OOM) cliff edges**.
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

$$\text{Total KV Cache (GB)} = \frac{\text{KV Bytes per Token} \times C \times N}{1024^3}$$

Where $C$ is context sequence length and $N$ is concurrent session concurrency.

### B. Static Budget & Net Usable Headroom
The total VRAM footprint at any step is:

$$M_{\text{total}} = M_{\text{weights}} + M_{\text{runtime}} + M_{\text{KV}}(C, N)$$

Under a simulated hardware ceiling $M_{\text{cap}}$ (e.g. $16.0\text{ GB}$):

$$\text{Net KV Headroom} = M_{\text{cap}} - M_{\text{weights}} - M_{\text{runtime}}$$

Where:
* $M_{\text{weights}}$: Unmodified base model weights loaded in BF16 ($P \times 2\text{ bytes}$).
* $M_{\text{runtime}}$: CUDA runtime context, PyTorch library overhead, and minimum activation workspace buffer ($\approx 0.80\text{ GB}$ for RTX 4090, $0.60\text{ GB}$ for RTX 4060).

### C. Theoretical OOM Cliff Threshold ($C_{\text{cliff}}$)
The maximum sustainable context length per user before deterministic VRAM exhaustion:

$$C_{\text{cliff}}(N) = \left\lfloor \frac{\text{Net KV Headroom} \times 1024^3}{N \times \text{KV Bytes per Token}} \right\rfloor$$

### D. Roofline Memory-Bandwidth Bound (Decode Phase)
Autoregressive token generation is strictly memory-bandwidth bound at small batch sizes. In each decode step, the GPU memory controller must stream all model weights plus all accumulated KV cache tensors across the memory bus:

$$\text{Bytes Read per Step} = M_{\text{weights}} + M_{\text{KV}}(C, B)$$

$$\text{Minimum TPOT (ms)} = \frac{\text{Bytes Read per Step}}{\text{Peak Memory Bandwidth (GB/s)} \times 10^6}$$

$$\text{Maximum Decode Throughput (tokens/sec)} = \frac{B}{\text{TPOT} / 1000}$$

$$\text{Slowdown vs. Zero-Context Base (\%)} = \frac{M_{\text{KV}}(C, B)}{M_{\text{weights}}} \times 100\%$$

### E. Prefill Phase Compute Bound (TTFT)
During prompt prefill, computation scales with both linear projections and quadratic attention FLOPs:

$$\text{Prefill FLOPs} \approx 2 \cdot P \cdot C + 4 \cdot L \cdot D_{\text{model}} \cdot C^2$$

$$\text{Minimum TTFT (sec)} = \frac{\text{Prefill FLOPs}}{\text{Peak Tensor Core TFLOPS} \times 10^{12}}$$

---

## 3. Theoretical Analysis: Primary Model (`qwen3-4b`)

Target Checkpoint: `Qwen/Qwen3-4B-Instruct-2507`  
Hardware: RTX 4090 capped at **$16.0\text{ GB}$ VRAM** ($1,008\text{ GB/s}$, $82.6\text{ TFLOPS}$)

* **Architecture:** $L=36$, $H_{KV}=8$, $D_{\text{head}}=128$, $D_{\text{model}}=2560$  
* **KV Bytes per Token:** $4 \times 36 \times 8 \times 128 = 147,456\text{ bytes} = \mathbf{144\text{ KiB/token}}$  
* **Model Weights:** $\mathbf{8.06\text{ GB}}$  
* **Runtime Overhead Buffer:** $\mathbf{0.80\text{ GB}}$  
* **Net Usable Headroom:** $16.0 - 8.06 - 0.80 = \mathbf{7.14\text{ GB}}$ ($\approx 51,991\text{ total tokens}$)

### Theoretical OOM Context Thresholds

| Concurrency ($N$) | Max Context per Stream | Total Active Tokens | Predicted Failure Mode |
| :---: | :---: | :---: | :--- |
| **1 user** | **$51,991$ tokens** | $51,991$ | Single-stream long context safely spans 32K; OOMs past 52K. |
| **4 users** | **$12,997$ tokens** | $51,988$ | Multi-user 4K context fits; 16K context triggers OOM. |
| **8 users** | **$6,498$ tokens** | $51,984$ | Multi-user 4K context fits; 8K context triggers OOM. |
| **12 users** | **$4,332$ tokens** | $51,984$ | **Critical edge:** 4K context uses $15.61\text{ GB}$ ($97.6\%$ of cap); 8K context triggers hard OOM ($22.36\text{ GB}$). |
| **16 users** | **$3,249$ tokens** | $51,984$ | 4K context triggers deterministic OOM ($17.86\text{ GB}$). |

### Single-Stream Context Sweep & Roofline Bounds ($N=1$)

| Context ($C$) | Regime | Total VRAM (GB) | Headroom (GB) | Min TTFT (ms) | Min TPOT (ms) | Peak Tok/s | Slowdown vs Base | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `512` | Sub-Critical | $8.93$ | $7.07$ | $54.8$ | $8.66$ | $115.5$ | $+0.9\%$ | ✅ Fits |
| `1,024` | Sub-Critical | $9.00$ | $7.00$ | $112.0$ | $8.73$ | $114.5$ | $+1.7\%$ | ✅ Fits |
| `2,048` | Sub-Critical | $9.14$ | $6.86$ | $233.3$ | $8.88$ | $112.5$ | $+3.5\%$ | ✅ Fits |
| `4,096` | Sub-Critical | $9.42$ | $6.58$ | $504.0$ | $9.19$ | $108.9$ | $+7.0\%$ | ✅ Fits |
| `8,192` | Sub-Critical | $9.98$ | $6.01$ | $1157.8$ | $9.78$ | $102.2$ | $+14.0\%$ | ✅ Fits |
| `16,384` | Inflection | $11.11$ | $4.89$ | $2914.6$ | $10.98$ | $91.0$ | $+27.9\%$ | ✅ Fits |
| `32,768` | Cliff Edge | $13.36$ | $2.64$ | $8225.3$ | $13.38$ | $74.7$ | $+55.8\%$ | ✅ Fits |
| `49,152` | Cliff Edge | $15.61$ | $0.39$ | $15932.0$ | $15.78$ | $63.4$ | $+83.8\%$ | ✅ Near Edge |
| `65,536` | Over-the-Cliff | $17.86$ | $-1.86$ | $26034.7$ | $18.17$ | $55.0$ | $+111.7\%$ | ⚠️ **OOM** |

---

## 4. Multi-User Concurrency Grid ($N \times C$)

Total VRAM allocation ($M_{\text{weights}} + M_{\text{runtime}} + M_{\text{KV}}$) for `qwen3-4b` on 16GB VRAM cap:

| Concurrent Users ($N$) | $C=1,024$ | $C=2,048$ | $C=4,096$ | $C=8,192$ |
| :---: | :---: | :---: | :---: | :---: |
| **$N = 1$** | ✅ $9.00\text{ GB}$ | ✅ $9.14\text{ GB}$ | ✅ $9.42\text{ GB}$ | ✅ $9.98\text{ GB}$ |
| **$N = 4$** | ✅ $9.42\text{ GB}$ | ✅ $9.98\text{ GB}$ | ✅ $11.11\text{ GB}$ | ✅ $13.36\text{ GB}$ |
| **$N = 8$** | ✅ $9.98\text{ GB}$ | ✅ $11.11\text{ GB}$ | ✅ $13.36\text{ GB}$ | ❌ **$17.86\text{ GB}$ (OOM)** |
| **$N = 12$** | ✅ $10.54\text{ GB}$ | ✅ $12.23\text{ GB}$ | ⚠️ **$15.61\text{ GB}$ (Edge)** | ❌ **$22.36\text{ GB}$ (OOM)** |
| **$N = 16$** | ✅ $11.11\text{ GB}$ | ✅ $13.36\text{ GB}$ | ❌ **$17.86\text{ GB}$ (OOM)** | ❌ **$26.86\text{ GB}$ (OOM)** |

---

## 5. Peer Model Theoretical Comparison

| Model | Hardware Budget | Weights (BF16) | KV Bytes/Token | Net Headroom | Single-User OOM Cliff | 12-User OOM Cliff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qwen3-4b`** | 16 GB | $8.06\text{ GB}$ | $144\text{ KiB}$ | $7.14\text{ GB}$ | $51,991$ tokens | $4,332$ tokens |
| **`phi4-mini`** | 16 GB | $7.65\text{ GB}$ | $128\text{ KiB}$ | $7.55\text{ GB}$ | $63,334$ tokens | $5,277$ tokens |
| **`qwen3-1.7b`** | 8 GB | $3.45\text{ GB}$ | $112\text{ KiB}$ | $3.95\text{ GB}$ | $37,921$ tokens | $3,160$ tokens |
| **`qwen3.5-4b` (Hybrid)** | 16 GB | $8.10\text{ GB}$ | $32\text{ KiB}$ (+GDN) | $7.10\text{ GB}$ | $238,000$ tokens | $19,833$ tokens |

---

## 6. The Empirical Test Matrix Specification

To ensure rigorous validation of both scaling performance and failure boundaries, the profiling run in **WP2-2** will execute:

### Stream A: Single-Stream Depth Sweep
* **Context Array:** `512, 1024, 2048, 4096, 8192, 16384, 32768, 49152, 65536`
* **Test Categorization:**
  * **Sub-Critical (`512` to `8192`):** Validates raw throughput, baseline TTFT scaling, and verifies linear memory growth without paging pressure.
  * **Inflection (`16384`):** Verifies memory bandwidth degradation (predicted $-27.9\%$ throughput reduction).
  * **Cliff-Edge (`32768`, `49152`):** Tests allocator behavior at $>80\%$ and $>95\%$ capacity utilization.
  * **Over-the-Cliff (`65536`):** Verifies graceful out-of-memory handling and vLLM abort behavior.

### Stream B: Multi-User Concurrency Sweep
* **User Array:** `1, 4, 8, 12, 16` concurrent sessions.
* **Context Depths:** `1024, 2048, 4096, 8192` tokens.
* **Critical Probe Points:**
  * **Normal Operation:** $N=4$ at $4096$ ctx ($11.11\text{ GB}$, ample headroom).
  * **Borderline Edge Point:** $N=12$ at $4096$ ctx ($15.61\text{ GB}$, $97.6\%$ capacity). Verifies if block fragmentation triggers premature OOM.
  * **Guaranteed OOM Point:** $N=12$ at $8192$ ctx ($22.36\text{ GB}$). Verifies cluster scheduler cancellation and memory reclamation.

### Stream C: Quality Ground-Truth Reference (LongMemEval)
* **Context Depths:** `512, 1024, 2048, 4096` tokens.
* **Sampling:** 10 samples per task across all 5 task capabilities (50 prompts total), deterministic random seed (`seed=42`).
* **Artifact Output:** Persist golden reference logprobs and token IDs (`results/references/baseline_qwen3_4b_ref.json`).
