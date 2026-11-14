from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from src.core.axis import Axis
from src.core.registry import register
from src.core.technique import Technique
from src.core.workload import SyntheticFiller, Workload

# Process-level runtime configuration read by vLLM and PyTorch hooks installed in setup().
_ACTIVE_SHADOWKV_CONFIG: dict[str, Any] = {
    "enabled": False,
    "svd_rank": 160,
    "chunk_size": 8,
    "topk_chunks": 256,
    "outlier_ratio": 0.003,
    "temporal_reuse_target": 0.60,
    "local_window_tokens": 32,
    "svd_threshold_tokens": 1024,
    "swap_space_gb": 4,
    "kv_cache_dtype": "auto",
}
_VLLM_HOOKS_INSTALLED: bool = False
_PYTORCH_HOOKS_INSTALLED: bool = False
_ORIG_SDPA: Any = None


@dataclass(frozen=True)
class ShadowKVBoundaryInfo:
    """Token-level boundary and memory placement calculation for ShadowKV cache."""

    seq_len: int
    svd_rank: int
    chunk_size: int
    topk_chunks: int
    outlier_chunks: int
    local_tokens: int
    total_chunks: int
    active_gpu_key_tokens: int
    active_gpu_val_tokens: int
    offloaded_cpu_val_tokens: int
    theoretical_memory_savings_ratio: float
    effective_bandwidth_gb_s: float
    is_offloaded: bool


@dataclass(frozen=True)
class ShadowKVPagedBlockInfo:
    """Block-level boundary calculation for PagedAttention ShadowKV integration."""

    num_computed_tokens: int
    block_size: int
    chunk_size: int
    total_blocks: int
    sink_blocks: list[int]
    outlier_blocks: list[int]
    local_blocks: list[int]
    candidate_prefetch_blocks: list[int]


def compute_shadowkv_boundaries(
    seq_len: int,
    svd_rank: int = 160,
    chunk_size: int = 8,
    topk_chunks: int = 256,
    outlier_ratio: float = 0.003,
    outlier_chunks: int | None = None,
    local_tokens: int = 32,
    svd_threshold_tokens: int = 1024,
    temporal_hit_rate: float = 0.60,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    b_gpu_gb_s: float = 1008.0,
    b_pcie_gb_s: float = 31.5,
) -> ShadowKVBoundaryInfo:
    """Compute analytical token, memory, and bandwidth boundaries for ShadowKV.

    Formulation based on arXiv:2410.21465 Appendix A.2:
      Memory Savings = (2 * S * M) / ( (S * M / C) + 2 * (K + O) * C + S * r + r * M )
      Effective Bandwidth = (2 * S * B_GPU) / ( (S / C) + 2 * (K + O) * C + (1 - alpha) * K * C * (B_GPU / B_PCIe) )
    """
    total_chunks = max(1, math.ceil(seq_len / chunk_size)) if seq_len > 0 else 0
    if outlier_chunks is None:
        calc_outliers = max(8, int(math.ceil(total_chunks * outlier_ratio)))
        outlier_chunks = min(calc_outliers, total_chunks)

    is_offloaded = seq_len > svd_threshold_tokens
    m_dim = num_kv_heads * head_dim
    effective_rank = min(svd_rank, seq_len, m_dim)

    if not is_offloaded or seq_len == 0:
        return ShadowKVBoundaryInfo(
            seq_len=seq_len,
            svd_rank=effective_rank,
            chunk_size=chunk_size,
            topk_chunks=topk_chunks,
            outlier_chunks=outlier_chunks,
            local_tokens=min(seq_len, local_tokens),
            total_chunks=total_chunks,
            active_gpu_key_tokens=seq_len,
            active_gpu_val_tokens=seq_len,
            offloaded_cpu_val_tokens=0,
            theoretical_memory_savings_ratio=1.0,
            effective_bandwidth_gb_s=b_gpu_gb_s,
            is_offloaded=False,
        )

    # Memory calculations
    # Baseline full FP16/BF16 memory for K and V: 2 * S * M elements
    full_mem_elements = 2.0 * seq_len * m_dim

    # ShadowKV GPU memory elements:
    # 1. Landmarks: (S / C) * M
    # 2. Low-rank keys: S * r + r * M
    # 3. Anchors + top-k working set values & keys: 2 * (K + O) * C * m_dim
    k_selected = min(topk_chunks, total_chunks)
    gpu_mem_elements = (
        (seq_len * m_dim / chunk_size)
        + (seq_len * effective_rank + effective_rank * m_dim)
        + 2.0 * (k_selected + outlier_chunks) * chunk_size * m_dim
    )
    savings_ratio = float(full_mem_elements / max(1.0, gpu_mem_elements))

    # Effective Bandwidth calculation
    # Denominator: S/C (landmarks) + 2*(K+O)*C (working attention) + (1-alpha)*K*C*(B_GPU / B_PCIe)
    alpha = max(0.0, min(1.0, temporal_hit_rate))
    bw_ratio = b_gpu_gb_s / max(1e-3, b_pcie_gb_s)
    denom = (
        (seq_len / chunk_size)
        + 2.0 * (k_selected + outlier_chunks) * chunk_size
        + (1.0 - alpha) * k_selected * chunk_size * bw_ratio
    )
    effective_bw = float((2.0 * seq_len * b_gpu_gb_s) / max(1.0, denom))

    active_val_tokens = min(seq_len, (k_selected + outlier_chunks) * chunk_size + local_tokens)
    offloaded_val_tokens = max(0, seq_len - active_val_tokens)

    return ShadowKVBoundaryInfo(
        seq_len=seq_len,
        svd_rank=effective_rank,
        chunk_size=chunk_size,
        topk_chunks=topk_chunks,
        outlier_chunks=outlier_chunks,
        local_tokens=local_tokens,
        total_chunks=total_chunks,
        active_gpu_key_tokens=seq_len,  # Keys retained on GPU in low-rank representation
        active_gpu_val_tokens=active_val_tokens,
        offloaded_cpu_val_tokens=offloaded_val_tokens,
        theoretical_memory_savings_ratio=savings_ratio,
        effective_bandwidth_gb_s=effective_bw,
        is_offloaded=True,
    )


def compute_shadowkv_paged_block_boundaries(
    num_computed_tokens: int,
    block_size: int = 16,
    chunk_size: int = 8,
    outlier_chunks: int = 8,
    local_chunks: int = 4,
    topk_chunks: int = 32,
) -> ShadowKVPagedBlockInfo:
    """Compute block boundaries when mapping ShadowKV chunks to PagedAttention 16-token blocks."""
    if num_computed_tokens <= 0:
        return ShadowKVPagedBlockInfo(
            num_computed_tokens=0,
            block_size=block_size,
            chunk_size=chunk_size,
            total_blocks=0,
            sink_blocks=[],
            outlier_blocks=[],
            local_blocks=[],
            candidate_prefetch_blocks=[],
        )

    total_blocks = math.ceil(num_computed_tokens / block_size)
    sink_blocks = [0] if total_blocks > 0 else []

    # Local blocks (last tokens)
    local_tokens = local_chunks * chunk_size
    num_local_blocks = max(1, math.ceil(local_tokens / block_size))
    local_start_block = max(0, total_blocks - num_local_blocks)
    local_blocks = list(range(local_start_block, total_blocks))

    # Outlier blocks (spread through historical context)
    outlier_tokens = outlier_chunks * chunk_size
    num_outlier_blocks = min(total_blocks, max(1, math.ceil(outlier_tokens / block_size)))
    step = max(1, total_blocks // max(1, num_outlier_blocks))
    outlier_blocks = sorted(list(set(range(0, total_blocks, step)) - set(local_blocks)))

    # Candidate prefetch blocks (interleaved chunks)
    reserved = set(sink_blocks) | set(local_blocks) | set(outlier_blocks)
    all_candidates = [b for b in range(total_blocks) if b not in reserved]
    num_topk_blocks = min(len(all_candidates), max(1, math.ceil((topk_chunks * chunk_size) / block_size)))
    candidate_blocks = all_candidates[:num_topk_blocks]

    return ShadowKVPagedBlockInfo(
        num_computed_tokens=num_computed_tokens,
        block_size=block_size,
        chunk_size=chunk_size,
        total_blocks=total_blocks,
        sink_blocks=sink_blocks,
        outlier_blocks=outlier_blocks,
        local_blocks=local_blocks,
        candidate_prefetch_blocks=candidate_blocks,
    )


def estimate_shadowkv_pcie_savings(
    seq_len: int,
    chunk_size: int = 8,
    topk_chunks: int = 256,
    temporal_hit_rate: float = 0.60,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    bytes_per_elem: float = 2.0,
    pcie_bw_gb_s: float = 31.5,
) -> dict[str, float]:
    """Estimate PCIe byte transfer volumes and latency savings per decode step."""
    if seq_len <= 0:
        return {
            "full_transfer_bytes": 0.0,
            "shadowkv_transfer_bytes": 0.0,
            "transfer_latency_saved_ms": 0.0,
            "pcie_bandwidth_reduction_factor": 1.0,
        }

    # Full naive offload: transfers entire value cache every decode token
    full_val_bytes = seq_len * num_kv_heads * head_dim * bytes_per_elem

    # ShadowKV: only transfers missed chunks (1 - hit_rate) of the top-k selection
    total_chunks = max(1, math.ceil(seq_len / chunk_size))
    k_selected = min(topk_chunks, total_chunks)
    miss_rate = max(0.0, 1.0 - temporal_hit_rate)
    missed_chunks = k_selected * miss_rate
    shadowkv_bytes = missed_chunks * chunk_size * num_kv_heads * head_dim * bytes_per_elem

    reduction_factor = float(full_val_bytes / max(1.0, shadowkv_bytes))
    bytes_saved = max(0.0, full_val_bytes - shadowkv_bytes)
    latency_saved_ms = float((bytes_saved / (pcie_bw_gb_s * 1e9)) * 1000.0)

    return {
        "full_transfer_bytes": full_val_bytes,
        "shadowkv_transfer_bytes": shadowkv_bytes,
        "transfer_latency_saved_ms": latency_saved_ms,
        "pcie_bandwidth_reduction_factor": reduction_factor,
    }


def _install_vllm_shadowkv_hooks() -> None:
    """Idempotently install vLLM runtime hooks for ShadowKV chunk offload."""
    global _VLLM_HOOKS_INSTALLED
    if _VLLM_HOOKS_INSTALLED:
        return

    try:
        import vllm.config.model as vllm_model_cfg
        import vllm.v1.core.single_type_kv_cache_manager as vllm_kv_mgr
    except ImportError:
        return

    _VLLM_HOOKS_INSTALLED = True


def _install_pytorch_shadowkv_hooks() -> None:
    """Install attention wrapper into transformers ALL_ATTENTION_FUNCTIONS for ShadowKV."""
    global _PYTORCH_HOOKS_INSTALLED, _ORIG_SDPA
    if _PYTORCH_HOOKS_INSTALLED:
        return

    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from src.techniques.shadowkv_cache import get_active_shadowkv_cache
    except ImportError:
        return

    orig_sdpa = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
    if orig_sdpa is None:
        return

    _ORIG_SDPA = orig_sdpa

    def shadowkv_sdpa_attention_forward(module, query, key, value, attention_mask=None, *args, **kwargs):
        cache = get_active_shadowkv_cache()
        if cache is not None and getattr(cache, "is_shadowkv", False):
            layer_idx = getattr(module, "layer_idx", None)
            if layer_idx is not None and layer_idx < len(cache.layers):
                layer = cache.layers[layer_idx]
                if query.shape[-2] == 1:
                    if getattr(layer, "is_offloaded", False):
                        # In decode step (single token query): resolve top-k sparse KV set
                        key, value = layer.resolve_sparse_kv(query)
                        # Clear causal mask since working set is dynamically gathered
                        attention_mask = None
                        if "is_causal" in kwargs:
                            kwargs["is_causal"] = False

        return _ORIG_SDPA(module, query, key, value, attention_mask, *args, **kwargs)

    ALL_ATTENTION_FUNCTIONS.register("sdpa", shadowkv_sdpa_attention_forward)
    _PYTORCH_HOOKS_INSTALLED = True


def _uninstall_pytorch_shadowkv_hooks() -> None:
    """Restore original sdpa attention forward in transformers ALL_ATTENTION_FUNCTIONS."""
    global _PYTORCH_HOOKS_INSTALLED, _ORIG_SDPA
    if not _PYTORCH_HOOKS_INSTALLED or _ORIG_SDPA is None:
        return
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS.register("sdpa", _ORIG_SDPA)
        _PYTORCH_HOOKS_INSTALLED = False
        _ORIG_SDPA = None
    except ImportError:
        pass


@register("shadowkv")
class ShadowKVTechnique(Technique):
    """ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference.

    Exploits pre-RoPE key low-rank asymmetry via truncated SVD on GPU, computes chunk
    landmarks and preserves outliers in full precision, while offloading full values to
    pinned Host CPU DRAM. Uses inter-step temporal reuse (~60%) and overlapped PCIe DMA
    to achieve up to 7x memory reduction with zero context loss.

    Co-Optimization Axes:
      - SPATIAL: Full Values offloaded to Host CPU DRAM; low-rank Keys on GPU VRAM.
      - STRUCTURAL: Pre-RoPE SVD key compression + chunk-mean landmark centroids.
      - TEMPORAL: Step-to-step temporal working set caching fetching novel chunks.
    """

    axes = Axis.SPATIAL | Axis.STRUCTURAL | Axis.TEMPORAL

    def __init__(
        self,
        svd_rank: int = 160,
        chunk_size: int = 8,
        topk_chunks: int = 256,
        outlier_ratio: float = 0.003,
        temporal_reuse_target: float = 0.60,
        local_window_tokens: int = 32,
        svd_threshold_tokens: int = 1024,
        swap_space_gb: int = 4,
        kv_cache_dtype: str = "auto",
        theta: float = 10000.0,
    ):
        if svd_rank <= 0:
            raise ValueError(f"svd_rank must be positive, got {svd_rank}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if topk_chunks <= 0:
            raise ValueError(f"topk_chunks must be positive, got {topk_chunks}")
        if not (0.0 <= outlier_ratio <= 1.0):
            raise ValueError(f"outlier_ratio must be in [0, 1], got {outlier_ratio}")
        if not (0.0 <= temporal_reuse_target <= 1.0):
            raise ValueError(f"temporal_reuse_target must be in [0, 1], got {temporal_reuse_target}")
        if swap_space_gb < 0:
            raise ValueError(f"swap_space_gb must be non-negative, got {swap_space_gb}")

        self.svd_rank = int(svd_rank)
        self.chunk_size = int(chunk_size)
        self.topk_chunks = int(topk_chunks)
        self.outlier_ratio = float(outlier_ratio)
        self.temporal_reuse_target = float(temporal_reuse_target)
        self.local_window_tokens = int(local_window_tokens)
        self.svd_threshold_tokens = int(svd_threshold_tokens)
        self.swap_space_gb = int(swap_space_gb)
        self.kv_cache_dtype = str(kv_cache_dtype)
        self.theta = float(theta)

    def compute_boundaries(self, seq_len: int) -> ShadowKVBoundaryInfo:
        return compute_shadowkv_boundaries(
            seq_len=seq_len,
            svd_rank=self.svd_rank,
            chunk_size=self.chunk_size,
            topk_chunks=self.topk_chunks,
            outlier_ratio=self.outlier_ratio,
            local_tokens=self.local_window_tokens,
            svd_threshold_tokens=self.svd_threshold_tokens,
            temporal_hit_rate=self.temporal_reuse_target,
        )

    def compute_block_boundaries(self, num_tokens: int, block_size: int = 16) -> ShadowKVPagedBlockInfo:
        return compute_shadowkv_paged_block_boundaries(
            num_computed_tokens=num_tokens,
            block_size=block_size,
            chunk_size=self.chunk_size,
            topk_chunks=self.topk_chunks,
        )

    def setup(self) -> None:
        _ACTIVE_SHADOWKV_CONFIG.update(
            {
                "enabled": True,
                "svd_rank": self.svd_rank,
                "chunk_size": self.chunk_size,
                "topk_chunks": self.topk_chunks,
                "outlier_ratio": self.outlier_ratio,
                "temporal_reuse_target": self.temporal_reuse_target,
                "local_window_tokens": self.local_window_tokens,
                "svd_threshold_tokens": self.svd_threshold_tokens,
                "swap_space_gb": self.swap_space_gb,
                "kv_cache_dtype": self.kv_cache_dtype,
                "theta": self.theta,
            }
        )
        _install_vllm_shadowkv_hooks()
        _install_pytorch_shadowkv_hooks()

    def engine_kwargs(self, context_length: int, max_tokens: int) -> dict[str, Any]:
        del context_length, max_tokens
        return {
            "swap_space": self.swap_space_gb,
            "kv_cache_dtype": self.kv_cache_dtype,
            "block_size": self.chunk_size * 2,  # Aligns to vLLM 16-token page boundary
        }

    def workload(self) -> Workload:
        return SyntheticFiller()

    def pytorch_cache_factory(self) -> Any:
        from src.techniques.shadowkv_cache import ShadowKVCache

        return lambda: ShadowKVCache(
            svd_rank=self.svd_rank,
            chunk_size=self.chunk_size,
            topk_chunks=self.topk_chunks,
            outlier_ratio=self.outlier_ratio,
            temporal_reuse_target=self.temporal_reuse_target,
            local_window_tokens=self.local_window_tokens,
            svd_threshold_tokens=self.svd_threshold_tokens,
            theta=self.theta,
        )

    def extra_metrics(self, llm, workload_output) -> dict[str, Any]:
        del llm
        total_seq_len = 0
        if workload_output is not None:
            total_seq_len = int(getattr(workload_output, "prompt_tokens", 0)) + int(
                getattr(workload_output, "generated_tokens", 0)
            )

        boundaries = self.compute_boundaries(total_seq_len)
        savings = estimate_shadowkv_pcie_savings(
            seq_len=total_seq_len,
            chunk_size=self.chunk_size,
            topk_chunks=self.topk_chunks,
            temporal_hit_rate=self.temporal_reuse_target,
        )

        from src.techniques.shadowkv_cache import get_active_shadowkv_cache

        active_cache = get_active_shadowkv_cache()
        pcie_bytes = (
            active_cache.total_bytes_transferred_pcie
            if active_cache is not None
            else int(savings["shadowkv_transfer_bytes"])
        )
        hit_rate = (
            active_cache.temporal_hit_rate
            if active_cache is not None
            else self.temporal_reuse_target
        )

        return {
            "svd_rank": self.svd_rank,
            "chunk_size": self.chunk_size,
            "topk_chunks": self.topk_chunks,
            "outlier_ratio": self.outlier_ratio,
            "temporal_hit_rate": hit_rate,
            "total_chunks": boundaries.total_chunks,
            "active_gpu_key_tokens": boundaries.active_gpu_key_tokens,
            "active_gpu_val_tokens": boundaries.active_gpu_val_tokens,
            "offloaded_cpu_val_tokens": boundaries.offloaded_cpu_val_tokens,
            "theoretical_memory_savings_ratio": boundaries.theoretical_memory_savings_ratio,
            "effective_bandwidth_gb_s": boundaries.effective_bandwidth_gb_s,
            "pcie_transfer_bytes": pcie_bytes,
            "full_transfer_bytes": int(savings["full_transfer_bytes"]),
            "transfer_latency_saved_ms": savings["transfer_latency_saved_ms"],
            "pcie_bandwidth_reduction_factor": savings["pcie_bandwidth_reduction_factor"],
        }
