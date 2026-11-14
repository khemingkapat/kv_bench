from __future__ import annotations

import weakref
from collections.abc import Iterator
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from transformers.cache_utils import Cache, DynamicLayer
from src.techniques.epicache import (
    _compute_query_key_max_scores,
    compute_layer_sensitivity_budgets,
    evict_block_prefill,
)

_ACTIVE_EPICACHE_CACHE_REF: weakref.ReferenceType[EpiCacheCache] | None = None


def get_active_epicache_cache() -> EpiCacheCache | None:
    """Return currently active EpiCacheCache instance if alive, else None."""
    global _ACTIVE_EPICACHE_CACHE_REF
    if _ACTIVE_EPICACHE_CACHE_REF is not None:
        return _ACTIVE_EPICACHE_CACHE_REF()
    return None


def set_active_epicache_cache(cache: EpiCacheCache | None) -> None:
    """Register weak reference to active EpiCacheCache ensuring zero tensor retention leaks."""
    global _ACTIVE_EPICACHE_CACHE_REF
    if cache is not None:
        _ACTIVE_EPICACHE_CACHE_REF = weakref.ref(cache)
    else:
        _ACTIVE_EPICACHE_CACHE_REF = None


def record_epicache_attention_scores(
    layer: "EpiCacheLayer",
    query: "torch.Tensor",
    key: "torch.Tensor",
) -> None:
    """Compute GQA-pooled max attention scores and record onto EpiCacheLayer."""
    if torch is None or F is None:
        return

    batch, num_q_heads, q_len, head_dim = query.shape
    num_kv_heads = key.shape[1]
    kv_len = key.shape[2]
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        return

    if q_len == 1:
        group_size = num_q_heads // num_kv_heads
        q_grouped = query.view(batch, num_kv_heads, group_size, 1, head_dim)
        k_expanded = key.unsqueeze(2)
        scores = (q_grouped.float() * k_expanded.float()).sum(dim=-1) * (head_dim ** -0.5)
        attn = F.softmax(scores, dim=-1)
        step_scores = attn.mean(dim=2).detach()
    else:
        step_scores = _compute_query_key_max_scores(query, key).detach()

    if (
        layer.importance_scores is None
        or layer.importance_scores.shape[-1] != kv_len
        or layer.importance_scores.shape[:2] != step_scores.shape[:2]
    ):
        layer.importance_scores = step_scores
    else:
        if getattr(layer, "_has_unscored_decode_token", False):
            layer.importance_scores[:, :, -1] = step_scores[:, :, -1]
            layer._has_unscored_decode_token = False
        layer.importance_scores = torch.maximum(layer.importance_scores, step_scores)

    layer.attention_guided_steps += 1


class EpiCacheLayer(DynamicLayer):
    """Cache layer implementing EpiCache block-prefill capping and layer-sensitivity token budgets."""

    is_sliding = False

    def __init__(
        self,
        layer_idx: int = 0,
        total_layers: int = 36,
        episode_budget_tokens: int = 1024,
        layer_sensitivity_alpha: float = 2.0,
    ):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.total_layers = max(1, int(total_layers))
        self.episode_budget_tokens = int(episode_budget_tokens)
        self.layer_sensitivity_alpha = float(layer_sensitivity_alpha)

        self.sensitivity_budget: int = self.episode_budget_tokens
        self.layer_budget: int = self.episode_budget_tokens
        self.set_total_layers(self.total_layers)

        self.original_positions: list[int] = []
        self.total_tokens_seen: int = 0
        self.importance_scores: torch.Tensor | None = None
        self.attention_guided_steps: int = 0
        self.pruned_tokens: int = 0
        self._has_unscored_decode_token: bool = False

    def set_total_layers(self, total_layers: int) -> None:
        """Recompute sensitivity-aware per-layer budget for the given model depth."""
        self.total_layers = max(1, int(total_layers))
        budgets = compute_layer_sensitivity_budgets(
            total_layers=self.total_layers,
            base_budget_per_layer=self.episode_budget_tokens,
            alpha=self.layer_sensitivity_alpha,
        )
        self.sensitivity_budget = budgets.get(self.layer_idx, self.episode_budget_tokens)
        self.layer_budget = self.sensitivity_budget

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append incoming states and evict excess tokens on decode if exceeding layer budget."""
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)

        new_tokens = key_states.shape[-2]
        self.original_positions.extend(
            range(self.total_tokens_seen, self.total_tokens_seen + new_tokens)
        )
        self.total_tokens_seen += new_tokens

        batch, num_kv_heads = key_states.shape[0], key_states.shape[1]
        if self.importance_scores is None:
            self.importance_scores = torch.zeros(
                batch, num_kv_heads, new_tokens, device=key_states.device, dtype=torch.float32
            )
        else:
            if new_tokens == 1:
                fill_val = float(self.importance_scores.max().item()) + 1.0 if self.importance_scores.numel() > 0 else 1.0
                new_scores = torch.full(
                    (batch, num_kv_heads, 1),
                    fill_val,
                    device=key_states.device,
                    dtype=torch.float32,
                )
                self._has_unscored_decode_token = True
            else:
                new_scores = torch.zeros(
                    batch, num_kv_heads, new_tokens, device=key_states.device, dtype=torch.float32
                )
                self._has_unscored_decode_token = False
            self.importance_scores = torch.cat([self.importance_scores, new_scores], dim=-1)

        if new_tokens == 1 and self.keys.shape[-2] > self.layer_budget:
            self.prune()

        return self.keys, self.values

    def prune(self) -> None:
        """Prune retained block buffer to budget, preserving token 0 sink & salient keys."""
        if self.is_initialized and self.keys is not None and self.values is not None:
            seq_len = self.keys.shape[-2]
            if seq_len > self.layer_budget:
                scores_for_eviction = (
                    self.importance_scores
                    if self.attention_guided_steps > 0
                    and self.importance_scores is not None
                    and self.importance_scores.shape[-1] == seq_len
                    else None
                )
                k_ret, v_ret, retained_idx = evict_block_prefill(
                    self.keys,
                    self.values,
                    budget=self.layer_budget,
                    attn_scores=scores_for_eviction,
                )
                self.keys = k_ret
                self.values = v_ret
                self.pruned_tokens += max(0, seq_len - int(retained_idx.numel()))
                if (
                    self.importance_scores is not None
                    and self.importance_scores.shape[-1] == seq_len
                ):
                    self.importance_scores = self.importance_scores[:, :, retained_idx]
                if self.original_positions:
                    idx_list = retained_idx.tolist()
                    self.original_positions = [
                        self.original_positions[i]
                        for i in idx_list
                        if i < len(self.original_positions)
                    ]

    def get_seq_length(self) -> int:
        return self.keys.shape[-2] if (self.is_initialized and self.keys is not None) else 0

    def get_max_length(self) -> int:
        return self.layer_budget

    def reset(self) -> None:
        self.total_tokens_seen = 0
        self.original_positions = []
        self.importance_scores = None
        self.attention_guided_steps = 0
        self.pruned_tokens = 0
        super().reset()


class EpiCacheCache(Cache):
    """Subclass of transformers.cache_utils.Cache implementing EpiCache layer-budgeted storage."""

    is_epicache = True

    def __init__(
        self,
        total_layers: int | None = 36,
        episode_budget_tokens: int = 1024,
        layer_sensitivity_alpha: float = 2.0,
        telemetry_sink: dict[str, Any] | None = None,
    ):
        self.total_layers: int | None = int(total_layers) if total_layers is not None else None
        self.episode_budget_tokens = int(episode_budget_tokens)
        self.layer_sensitivity_alpha = float(layer_sensitivity_alpha)
        self.telemetry_sink = telemetry_sink
        self._block_wise_prefill_executed = False

        super().__init__(
            layer_class_to_replicate=lambda: EpiCacheLayer(
                layer_idx=len(self.layers),
                total_layers=self.total_layers or max(1, len(self.layers) + 1),
                episode_budget_tokens=self.episode_budget_tokens,
                layer_sensitivity_alpha=self.layer_sensitivity_alpha,
            )
        )
        set_active_epicache_cache(self)

    def _sync_layer_budgets(self) -> None:
        """Align per-layer sensitivity budgets to the observed model depth when auto-detected."""
        target_layers = self.total_layers if self.total_layers is not None else len(self.layers)
        if target_layers <= 0:
            return
        for layer in self.layers:
            if hasattr(layer, "set_total_layers") and layer.total_layers != target_layers:
                layer.set_total_layers(target_layers)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = super().update(key_states, value_states, layer_idx, cache_kwargs)
        self.sync_telemetry()
        return result

    @property
    def original_positions(self) -> list[int]:
        if not self.layers:
            return []
        return self.layers[0].original_positions

    def prune_after_chunk(self) -> None:
        """Execute block-wise prefill pruning across all layers immediately after chunk ingestion."""
        self._sync_layer_budgets()
        for layer in self.layers:
            if hasattr(layer, "prune"):
                layer.prune()
        self._block_wise_prefill_executed = True
        self.sync_telemetry()

    def snapshot_telemetry(self) -> dict[str, Any]:
        """Return plain-Python run telemetry without retaining tensor references."""
        seq_lens = [layer.get_seq_length() for layer in self.layers]
        budgets = [getattr(layer, "layer_budget", self.episode_budget_tokens) for layer in self.layers]
        total_retained = sum(seq_lens)
        num_layers = len(seq_lens)
        mean_retained = int(round(total_retained / num_layers)) if num_layers > 0 else 0
        attn_steps = sum(getattr(layer, "attention_guided_steps", 0) for layer in self.layers)
        total_pruned = sum(getattr(layer, "pruned_tokens", 0) for layer in self.layers)
        total_seen = max((getattr(layer, "total_tokens_seen", 0) for layer in self.layers), default=0)

        return {
            "active_retained_tokens": mean_retained,
            "total_retained_tokens_all_layers": total_retained,
            "layer_0_retained_tokens": seq_lens[0] if seq_lens else 0,
            "num_layers_observed": num_layers,
            "total_tokens_seen": total_seen,
            "total_pruned_tokens": total_pruned,
            "attention_guided_steps": attn_steps,
            "eviction_scoring_mode": "attention_guided" if attn_steps > 0 else "key_norm_fallback",
            "block_wise_prefill_active": self._block_wise_prefill_executed or num_layers > 0,
            "per_layer_sensitivity_active": len(set(budgets)) > 1 or num_layers > 1,
            "episodic_routing_active": False,
            "executed_episodes": 1 if num_layers > 0 else 0,
        }

    def sync_telemetry(self) -> None:
        """Persist plain-Python telemetry into the owning technique's sink if registered."""
        if self.telemetry_sink is not None:
            self.telemetry_sink.update(self.snapshot_telemetry())

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_max_length(self) -> int:
        if not self.layers:
            return self.episode_budget_tokens
        return self.layers[0].get_max_length()

    def __iter__(self) -> Iterator[tuple[Any, Any, Any]]:
        for layer in self.layers:
            yield getattr(layer, "keys", None), getattr(layer, "values", None), None
