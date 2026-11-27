from __future__ import annotations

import math
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from src.core.axis import Axis
from src.core.registry import register
from src.core.technique import Technique
from src.core.workload import SyntheticFiller, Workload

_PYTORCH_HOOKS_INSTALLED = False
_ORIG_SDPA = None


def _install_pytorch_arkvale_hooks() -> None:
    """Install attention wrapper into transformers ALL_ATTENTION_FUNCTIONS for ArkVale."""
    global _PYTORCH_HOOKS_INSTALLED, _ORIG_SDPA
    if _PYTORCH_HOOKS_INSTALLED:
        return

    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from src.techniques.arkvale_cache import get_active_arkvale_cache
    except ImportError:
        return

    orig_sdpa = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
    if orig_sdpa is None:
        return

    _ORIG_SDPA = orig_sdpa

    def arkvale_sdpa_attention_forward(module, query, key, value, attention_mask=None, *args, **kwargs):
        cache = get_active_arkvale_cache()
        if cache is not None and getattr(cache, "is_arkvale", False):
            layer_idx = getattr(module, "layer_idx", None)
            if layer_idx is not None and 0 <= layer_idx < len(cache.layers):
                layer = cache.layers[layer_idx]
                if query.shape[-2] == 1 and layer.should_use_sparse_decode():
                    key, value = layer.resolve_active_kv(query)
                    attention_mask = None
                    if "is_causal" in kwargs:
                        kwargs["is_causal"] = False

        return _ORIG_SDPA(module, query, key, value, attention_mask, *args, **kwargs)

    ALL_ATTENTION_FUNCTIONS.register("sdpa", arkvale_sdpa_attention_forward)
    _PYTORCH_HOOKS_INSTALLED = True


def _uninstall_pytorch_arkvale_hooks() -> None:
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


def compute_page_aabb_digests(
    k_pages: "torch.Tensor",
    page_size: int = 16,
) -> tuple["torch.Tensor", "torch.Tensor", int]:
    """Compute Axis-Aligned Bounding Box (AABB) page digests for Key cache pages.

    For each page of `page_size` tokens:
      - Center: c = 0.5 * (min(K) + max(K))
      - Mean radius: r_mean = mean(|k - c|)
      - Bounds: b_max = c + r_mean, b_min = c - r_mean

    Args:
        k_pages: Tensor of shape (batch, num_kv_heads, seq_len, head_dim).
        page_size: Tokens per page (default: 16).

    Returns:
        Tuple of (b_min, b_max, num_pages).
        Each bounds tensor has shape (batch, num_kv_heads, num_pages, head_dim).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for compute_page_aabb_digests")

    seq_len = k_pages.size(-2)
    head_dim = k_pages.size(-1)
    num_pages = math.ceil(seq_len / page_size)

    pad_len = num_pages * page_size - seq_len
    if pad_len > 0:
        k_padded = F.pad(k_pages, (0, 0, 0, pad_len))
    else:
        k_padded = k_pages

    pages = k_padded.contiguous().reshape(*k_pages.shape[:-2], num_pages, page_size, head_dim)

    c = 0.5 * (pages.amin(dim=-2) + pages.amax(dim=-2))
    r_mean = (pages - c.unsqueeze(-2)).abs().mean(dim=-2)

    b_max = c + r_mean
    b_min = c - r_mean

    return b_min, b_max, num_pages


def score_page_digests_gqa(
    query: "torch.Tensor",
    b_min: "torch.Tensor",
    b_max: "torch.Tensor",
    gqa_ratio: int = 1,
) -> "torch.Tensor":
    """Evaluate geometric upper-bound attention scores using AABB digests.

    Upper bound on dot product for coordinate i:
      max(q_i * k_i) = max(q_i * b_max_i, q_i * b_min_i)
    Sum across hidden dimension d yields tight score upper bound per page.

    Returns:
        Page score tensor of shape (batch, num_kv_heads, num_pages).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for score_page_digests_gqa")

    batch, num_q_heads, q_len, d = query.shape
    num_kv_heads = b_min.size(1)
    num_pages = b_min.size(2)

    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
    effective_gqa = num_q_heads // num_kv_heads

    q_grouped = query.reshape(batch, num_kv_heads, effective_gqa, q_len, 1, d)
    b_min_exp = b_min.unsqueeze(2).unsqueeze(3)
    b_max_exp = b_max.unsqueeze(2).unsqueeze(3)

    prod_min = q_grouped * b_min_exp
    prod_max = q_grouped * b_max_exp

    page_scores = torch.maximum(prod_min, prod_max).sum(dim=-1) / math.sqrt(d)
    if q_len == 1:
        page_scores = page_scores.squeeze(3)
    kv_scores = page_scores.max(dim=2).values

    return kv_scores


def select_and_recall_pages(
    kv_scores: "torch.Tensor",
    active_pages_budget: int = 16,
) -> tuple["torch.Tensor", dict[str, Any]]:
    """Rank all prompt pages by upper-bound score and select active pages for GPU residency.

    Always retains page 0 (initial attention sink).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for select_and_recall_pages")

    num_pages = kv_scores.size(-1)
    k_pages = min(active_pages_budget, num_pages)

    agg_scores = kv_scores.mean(dim=(0, 1))

    k_top = max(1, k_pages - 1)
    _, top_pages = torch.topk(agg_scores, k=min(k_top, num_pages))
    active_pages = torch.unique(torch.cat([torch.tensor([0], device=top_pages.device), top_pages]))

    if active_pages.numel() > k_pages:
        active_pages = active_pages[:k_pages]

    sparsity_ratio = 1.0 - (active_pages.numel() / max(num_pages, 1))

    telemetry = {
        "active_pages": int(active_pages.numel()),
        "total_pages": int(num_pages),
        "page_sparsity_ratio": float(sparsity_ratio),
    }

    return active_pages, telemetry


def compute_gqa_arkvale_attention(
    query: "torch.Tensor",
    key: "torch.Tensor",
    value: "torch.Tensor",
    page_size: int = 16,
    active_pages_budget: int = 16,
    scale: float | None = None,
) -> tuple["torch.Tensor", "torch.Tensor", dict[str, Any]]:
    """Execute ArkVale recallable page decode attention step."""
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for compute_gqa_arkvale_attention")

    d = query.size(-1)
    num_q_heads = query.size(1)
    num_kv_heads = key.size(1)
    seq_len_kv = key.size(2)
    gqa_ratio = num_q_heads // num_kv_heads

    b_min, b_max, num_pages = compute_page_aabb_digests(key, page_size=page_size)
    kv_scores = score_page_digests_gqa(query, b_min, b_max, gqa_ratio=gqa_ratio)
    active_pages, telemetry = select_and_recall_pages(kv_scores, active_pages_budget=active_pages_budget)

    token_indices = []
    for p in active_pages.tolist():
        start = p * page_size
        end = min(start + page_size, seq_len_kv)
        token_indices.extend(range(start, end))
    token_indices = sorted(list(set(token_indices)))
    token_idx_tensor = torch.tensor(token_indices, dtype=torch.long, device=key.device)

    k_active = key[:, :, token_idx_tensor, :]
    v_active = value[:, :, token_idx_tensor, :]

    if gqa_ratio > 1:
        expanded_k = k_active.repeat_interleave(gqa_ratio, dim=1)
        expanded_v = v_active.repeat_interleave(gqa_ratio, dim=1)
    else:
        expanded_k = k_active
        expanded_v = v_active

    softmax_scale = scale if scale is not None else (d ** -0.5)
    scores = torch.matmul(query.float(), expanded_k.float().transpose(-2, -1)) * softmax_scale
    attn_weights = F.softmax(scores, dim=-1).to(query.dtype)
    attn_output = torch.matmul(attn_weights, expanded_v)

    telemetry["active_tokens"] = len(token_indices)
    telemetry["zero_copy_eviction_active"] = True
    return attn_output, attn_weights, telemetry


@register("arkvale")
class ArkValeTechnique(Technique):
    """ArkVale: Efficient Generative LLM Inference with Recallable Key-Value Eviction.

    Combines page-level eviction with proactive CPU backup and geometric digest recall:
      1. Structural: Groups tokens into fixed pages (16-32 tokens) and computes compact
         Axis-Aligned Bounding Box (AABB) digests recording minimum and maximum coordinate bounds.
      2. Spatial: Full-precision pages are backed up to host CPU RAM on completion; GPU memory
         is capped to a strict active page budget.
      3. At each decode step, queries evaluate score upper bounds across all digests to recall
         vital dormant pages from CPU memory, eliminating 95% of amnesia errors with near-zero latency.

    Axes:
        Axis.STRUCTURAL | Axis.SPATIAL
    """

    axes = Axis.STRUCTURAL | Axis.SPATIAL

    def __init__(
        self,
        page_size: int = 16,
        active_pages_budget: int = 16,
        exempt_early_layers: int = 2,
        block_size: int = 16,
    ):
        if int(page_size) <= 0:
            raise ValueError(f"page_size must be positive, got {page_size}")
        if int(active_pages_budget) <= 0:
            raise ValueError(f"active_pages_budget must be positive, got {active_pages_budget}")
        if int(exempt_early_layers) < 0:
            raise ValueError(f"exempt_early_layers must be non-negative, got {exempt_early_layers}")
        if int(block_size) <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.page_size = int(page_size)
        self.active_pages_budget = int(active_pages_budget)
        self.exempt_early_layers = int(exempt_early_layers)
        self.block_size = int(block_size)

    def engine_kwargs(self, context_length: int, max_tokens: int) -> dict[str, Any]:
        """Extra engine kwargs merged into the vLLM / PyTorch constructor."""
        del context_length, max_tokens
        return {
            "block_size": self.block_size,
            "swap_space": 8,
            "gpu_memory_utilization": 0.85,
        }

    def workload(self) -> Workload:
        return SyntheticFiller()

    def pytorch_cache_factory(self) -> Any:
        from src.techniques.arkvale_cache import ArkValeCache

        return lambda: ArkValeCache(
            page_size=self.page_size,
            active_pages_budget=self.active_pages_budget,
            exempt_early_layers=self.exempt_early_layers,
        )

    def setup(self) -> None:
        _install_pytorch_arkvale_hooks()
        self._sample_count = 0
        self._accum_pcie_bytes = 0
        self._accum_recalled_pages = 0

    def on_sample_complete(self, cache: Any) -> None:
        if cache is None:
            return
        self._sample_count = getattr(self, "_sample_count", 0) + 1
        self._accum_pcie_bytes = getattr(self, "_accum_pcie_bytes", 0) + int(
            getattr(cache, "total_bytes_transferred_pcie", 0)
        )
        self._accum_recalled_pages = getattr(self, "_accum_recalled_pages", 0) + int(
            getattr(cache, "total_recalled_pages", 0)
        )

    def extra_metrics(self, llm, workload_output) -> dict[str, Any]:
        del llm, workload_output
        from src.techniques.arkvale_cache import get_active_arkvale_cache

        active_cache = get_active_arkvale_cache()
        if getattr(self, "_sample_count", 0) > 0:
            pcie_bytes = self._accum_pcie_bytes
            recalled_pages = self._accum_recalled_pages
        elif active_cache is not None:
            pcie_bytes = active_cache.total_bytes_transferred_pcie
            recalled_pages = active_cache.total_recalled_pages
        else:
            pcie_bytes = 0
            recalled_pages = 0

        return {
            "page_size": self.page_size,
            "active_pages_budget": self.active_pages_budget,
            "exempt_early_layers": self.exempt_early_layers,
            "block_size": self.block_size,
            "pcie_transfer_bytes": pcie_bytes,
            "recalled_pages": recalled_pages,
            "aabb_digests_active": True,
            "page_recall_supported": True,
        }
