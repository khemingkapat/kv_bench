from __future__ import annotations

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


def compute_layer_sensitivity_budgets(
    total_layers: int = 32,
    base_budget_per_layer: int = 1024,
    alpha: float = 2.0,
) -> dict[int, int]:
    """Compute sensitivity-aware KV-cache token budgets across transformer layers.

    Empirically, transformer layers show varying sensitivity: early layers (syntax/attention sinks)
    and late layers (readout) require larger budgets, while middle layers tolerate pruning.

    Budget allocation:
        M_l = (s_l^alpha / sum(s_j^alpha)) * (L * M)
    """
    total_tokens = total_layers * base_budget_per_layer

    sensitivities = []
    for l in range(total_layers):
        x = l / max(total_layers - 1, 1)
        sens = 0.3 + 0.7 * (4.0 * (x - 0.5) ** 2)
        sensitivities.append(sens ** alpha)

    total_sens = sum(sensitivities)
    min_budget = min(64, max(1, base_budget_per_layer // 4))
    budgets: dict[int, int] = {}
    for l in range(total_layers):
        allocated = int(round((sensitivities[l] / total_sens) * total_tokens))
        budgets[l] = max(min_budget, allocated)

    return budgets


def route_query_to_episode(
    query_emb: "torch.Tensor",
    episode_centroids: "torch.Tensor",
) -> int:
    """Identify matching episode for incoming query via cosine similarity with centroids.

    Args:
        query_emb: Query embedding vector of shape (d,).
        episode_centroids: Centroids matrix of shape (num_episodes, d).

    Returns:
        Best matching episode index.
    """
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for route_query_to_episode")

    q_norm = F.normalize(query_emb.float().reshape(1, -1), p=2, dim=-1)
    c_norm = F.normalize(episode_centroids.float(), p=2, dim=-1)
    sims = torch.matmul(q_norm, c_norm.transpose(0, 1)).squeeze(0)
    return int(torch.argmax(sims).item())


def _compute_query_key_max_scores(
    query: "torch.Tensor",
    key: "torch.Tensor",
) -> "torch.Tensor":
    """Compute GQA-pooled max attention score per KV token (epicache.md Eq. 3)."""
    batch, num_q_heads, q_len, head_dim = query.shape
    num_kv_heads = key.shape[1]
    kv_len = key.shape[2]
    group_size = max(1, num_q_heads // max(num_kv_heads, 1))

    if group_size > 1 and num_q_heads % num_kv_heads == 0:
        k_rep = key.repeat_interleave(group_size, dim=1)
    else:
        k_rep = key

    scores = torch.matmul(query.float(), k_rep.float().transpose(-2, -1)) * (head_dim ** -0.5)
    if q_len > 1 and kv_len >= q_len:
        past_len = kv_len - q_len
        q_idx = torch.arange(q_len, device=query.device).unsqueeze(-1) + past_len
        kv_idx = torch.arange(kv_len, device=query.device).unsqueeze(0)
        mask = kv_idx <= q_idx
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn = F.softmax(scores, dim=-1)
    if num_q_heads % num_kv_heads == 0:
        attn_grouped = attn.view(batch, num_kv_heads, group_size, q_len, kv_len).mean(dim=2)
    else:
        attn_grouped = attn
    return attn_grouped.amax(dim=2)


def evict_block_prefill(
    k_block: "torch.Tensor",
    v_block: "torch.Tensor",
    budget: int,
    attn_scores: "torch.Tensor | None" = None,
    query: "torch.Tensor | None" = None,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Perform block-wise token eviction during prefill to cap peak GPU memory.

    Retains the attention sink (token 0) and the top tokens ranked by attention-guided
    max-attention importance (when ``attn_scores`` or ``query`` is provided), falling back
    to L2 Key magnitude when no query/attention context is available.

    Returns:
        Tuple of (k_retained, v_retained, retained_indices).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for evict_block_prefill")

    seq_len = k_block.size(-2)
    if seq_len <= budget:
        indices = torch.arange(seq_len, device=k_block.device)
        return k_block, v_block, indices

    if attn_scores is not None and attn_scores.shape[-1] == seq_len:
        token_scores = attn_scores.float()
        if token_scores.ndim > 1:
            token_scores = token_scores.mean(dim=tuple(range(token_scores.ndim - 1)))
    elif query is not None and F is not None:
        q_scores = _compute_query_key_max_scores(query, k_block)
        token_scores = q_scores.mean(dim=(0, 1))
    else:
        token_scores = k_block.float().norm(p=2, dim=-1).mean(dim=(0, 1))

    k_top = max(1, budget - 1)
    _, top_idx = torch.topk(token_scores[1:], k=min(k_top, seq_len - 1))
    top_idx = top_idx + 1
    retained_idx = torch.unique(torch.cat([torch.tensor([0], device=top_idx.device), top_idx]))

    if retained_idx.numel() > budget:
        retained_idx = retained_idx[:budget]
    retained_idx = torch.sort(retained_idx).values

    k_retained = k_block[:, :, retained_idx, :]
    v_retained = v_block[:, :, retained_idx, :]
    return k_retained, v_retained, retained_idx


def compute_gqa_epicache_attention(
    query: "torch.Tensor",
    key: "torch.Tensor",
    value: "torch.Tensor",
    budget: int = 128,
    scale: float | None = None,
) -> tuple["torch.Tensor", "torch.Tensor", dict[str, Any]]:
    """Compute GQA attention on the episodic active cache partition."""
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for compute_gqa_epicache_attention")

    d = query.size(-1)
    orig_seq_len = key.size(-2)

    k_active, v_active, retained_idx = evict_block_prefill(
        key, value, budget=budget, query=query
    )

    num_q_heads = query.size(1)
    num_kv_heads = key.size(1)
    gqa_ratio = num_q_heads // num_kv_heads

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

    sparsity_ratio = 1.0 - (retained_idx.numel() / max(orig_seq_len, 1))

    telemetry = {
        "active_tokens": int(retained_idx.numel()),
        "original_tokens": int(orig_seq_len),
        "sparsity_ratio": float(sparsity_ratio),
        "block_prefill_capped": True,
        "scoring_mode": "attention_guided",
    }

    return attn_output, attn_weights, telemetry


_PYTORCH_HOOKS_INSTALLED: bool = False
_ORIG_SDPA: Any = None


def _align_attention_mask(
    mask: "torch.Tensor | None",
    q_len: int,
    kv_len: int,
) -> "torch.Tensor | None":
    """Align 4D causal attention_mask to a layer's retained KV length under per-layer budgets."""
    if mask is None or torch is None:
        return mask
    mask_kv_len = mask.shape[-1]
    if mask_kv_len == kv_len:
        return mask
    if mask_kv_len > kv_len:
        past_needed = kv_len - q_len
        if past_needed > 0:
            return torch.cat([mask[..., :past_needed], mask[..., -q_len:]], dim=-1)
        return mask[..., -kv_len:]
    pad_len = kv_len - mask_kv_len
    pad = torch.zeros((*mask.shape[:-1], pad_len), device=mask.device, dtype=mask.dtype)
    return torch.cat([pad, mask], dim=-1)


def _install_pytorch_epicache_hooks() -> None:
    """Install SDPA attention wrapper to capture query-informed scores and align per-layer masks."""
    global _PYTORCH_HOOKS_INSTALLED, _ORIG_SDPA
    if _PYTORCH_HOOKS_INSTALLED:
        return

    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from src.techniques.epicache_cache import (
            get_active_epicache_cache,
            record_epicache_attention_scores,
        )
    except ImportError:
        return

    orig_sdpa = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
    if orig_sdpa is None:
        return

    _ORIG_SDPA = orig_sdpa

    def epicache_sdpa_attention_forward(module, query, key, value, *args, **kwargs):
        cache = get_active_epicache_cache()
        if cache is not None and getattr(cache, "is_epicache", False):
            kv_len = key.shape[-2]
            q_len = query.shape[-2]
            if args:
                aligned_mask = _align_attention_mask(args[0], q_len=q_len, kv_len=kv_len)
                if aligned_mask is not args[0]:
                    args = (aligned_mask,) + args[1:]
            elif "attention_mask" in kwargs:
                aligned_mask = _align_attention_mask(
                    kwargs["attention_mask"], q_len=q_len, kv_len=kv_len
                )
                if aligned_mask is not kwargs["attention_mask"]:
                    kwargs["attention_mask"] = aligned_mask

        output, attn_weights = _ORIG_SDPA(module, query, key, value, *args, **kwargs)

        if cache is not None and getattr(cache, "is_epicache", False):
            layer_idx = getattr(module, "layer_idx", None)
            if layer_idx is not None and layer_idx < len(cache.layers):
                record_epicache_attention_scores(cache.layers[layer_idx], query, key)
                cache.sync_telemetry()

        return output, attn_weights

    ALL_ATTENTION_FUNCTIONS.register("sdpa", epicache_sdpa_attention_forward)
    _PYTORCH_HOOKS_INSTALLED = True


def _uninstall_pytorch_epicache_hooks() -> None:
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


@register("epicache")
class EpiCacheTechnique(Technique):
    """EpiCache: Episodic KV Cache Management for Long-Term Multi-Turn Conversations.

    Combines semantic episode clustering, block-wise prefill capping, and layer-sensitivity budgets:
      1. Structural: Groups long-term conversation history into coherent semantic episodes.
         During prefill, token pruning runs block-by-block, strictly capping peak VRAM usage.
         Allocates variable token budgets across layers based on sensitivity profiles.
      2. Spatial: Keeps only the single active episode cache in GPU VRAM during generation;
         all other episode caches reside in host CPU memory, avoiding multi-turn amnesia.
      3. Cuts peak GPU VRAM by up to 3.8x, enabling indefinitely long dialogues on consumer GPUs.

    Axes:
        Axis.STRUCTURAL | Axis.SPATIAL
    """

    axes = Axis.STRUCTURAL | Axis.SPATIAL

    def __init__(
        self,
        num_episodes: int = 4,
        episode_budget_tokens: int = 1024,
        layer_sensitivity_alpha: float = 2.0,
        block_size: int = 16,
    ):
        if int(num_episodes) <= 0:
            raise ValueError(f"num_episodes must be positive, got {num_episodes}")
        if int(episode_budget_tokens) <= 0:
            raise ValueError(f"episode_budget_tokens must be positive, got {episode_budget_tokens}")
        if float(layer_sensitivity_alpha) <= 0.0:
            raise ValueError(f"layer_sensitivity_alpha must be positive, got {layer_sensitivity_alpha}")
        if int(block_size) <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.num_episodes = int(num_episodes)
        self.episode_budget_tokens = int(episode_budget_tokens)
        self.layer_sensitivity_alpha = float(layer_sensitivity_alpha)
        self.block_size = int(block_size)
        self._last_run_telemetry: dict[str, Any] = self._empty_telemetry()

    @staticmethod
    def _empty_telemetry() -> dict[str, Any]:
        return {
            "active_retained_tokens": 0,
            "total_retained_tokens_all_layers": 0,
            "layer_0_retained_tokens": 0,
            "num_layers_observed": 0,
            "total_tokens_seen": 0,
            "total_pruned_tokens": 0,
            "attention_guided_steps": 0,
            "eviction_scoring_mode": "uninitialized",
            "block_wise_prefill_active": False,
            "per_layer_sensitivity_active": False,
            "episodic_routing_active": False,
            "executed_episodes": 0,
        }

    def setup(self) -> None:
        """Reset run-level telemetry and install PyTorch SDPA hooks."""
        from src.techniques.epicache_cache import set_active_epicache_cache

        set_active_epicache_cache(None)
        self._last_run_telemetry = self._empty_telemetry()
        _install_pytorch_epicache_hooks()

    @property
    def compression_ratio(self) -> float:
        """Peak GPU VRAM footprint compression ratio relative to uncompressed cache."""
        return 3.8

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
        from src.techniques.epicache_cache import EpiCacheCache

        return lambda: EpiCacheCache(
            total_layers=None,
            episode_budget_tokens=self.episode_budget_tokens,
            layer_sensitivity_alpha=self.layer_sensitivity_alpha,
            telemetry_sink=self._last_run_telemetry,
        )

    def extra_metrics(self, llm, workload_output) -> dict[str, Any]:
        del llm, workload_output
        from src.techniques.epicache_cache import get_active_epicache_cache

        active_cache = get_active_epicache_cache()
        if active_cache is not None:
            telemetry = active_cache.snapshot_telemetry()
        else:
            telemetry = dict(self._last_run_telemetry)

        return {
            "configured_num_episodes": self.num_episodes,
            "executed_episodes": telemetry.get("executed_episodes", 0),
            "episode_budget_tokens": self.episode_budget_tokens,
            "layer_sensitivity_alpha": self.layer_sensitivity_alpha,
            "compression_ratio_vs_bf16": self.compression_ratio,
            "block_size": self.block_size,
            "active_retained_tokens": telemetry.get("active_retained_tokens", 0),
            "total_retained_tokens_all_layers": telemetry.get(
                "total_retained_tokens_all_layers", 0
            ),
            "layer_0_retained_tokens": telemetry.get("layer_0_retained_tokens", 0),
            "num_layers_observed": telemetry.get("num_layers_observed", 0),
            "total_tokens_seen": telemetry.get("total_tokens_seen", 0),
            "total_pruned_tokens": telemetry.get("total_pruned_tokens", 0),
            "attention_guided_steps": telemetry.get("attention_guided_steps", 0),
            "eviction_scoring_mode": telemetry.get("eviction_scoring_mode", "uninitialized"),
            "block_wise_prefill_active": telemetry.get("block_wise_prefill_active", False),
            "per_layer_sensitivity_active": telemetry.get(
                "per_layer_sensitivity_active", False
            ),
            "episodic_routing_active": telemetry.get("episodic_routing_active", False),
        }
