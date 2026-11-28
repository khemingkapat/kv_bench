from __future__ import annotations

import math
import weakref
from collections.abc import Iterator
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from transformers.cache_utils import Cache, DynamicLayer

_ACTIVE_ARKVALE_CACHE_REF: weakref.ReferenceType[ArkValeCache] | None = None


def get_active_arkvale_cache() -> ArkValeCache | None:
    """Return currently active ArkValeCache instance if alive, else None."""
    global _ACTIVE_ARKVALE_CACHE_REF
    if _ACTIVE_ARKVALE_CACHE_REF is not None:
        return _ACTIVE_ARKVALE_CACHE_REF()
    return None


def set_active_arkvale_cache(cache: ArkValeCache | None) -> None:
    """Register weak reference to active ArkValeCache ensuring zero tensor retention leaks."""
    global _ACTIVE_ARKVALE_CACHE_REF
    if cache is not None:
        _ACTIVE_ARKVALE_CACHE_REF = weakref.ref(cache)
    else:
        _ACTIVE_ARKVALE_CACHE_REF = None


class ArkValeLayer(DynamicLayer):
    """Cache layer implementing ArkVale page-level backup, cached GPU AABB digests, and page recall."""

    is_sliding = False

    def __init__(
        self,
        layer_idx: int = 0,
        page_size: int = 16,
        active_pages_budget: int = 16,
        exempt_early_layers: int = 2,
    ):
        self._keys_fallback: torch.Tensor | None = None
        self._values_fallback: torch.Tensor | None = None

        self.layer_idx = int(layer_idx)
        self.page_size = int(page_size)
        self.active_pages_budget = int(active_pages_budget)
        self.exempt_early_layers = int(exempt_early_layers)

        self.capacity: int = 0
        self.page_capacity: int = 0
        self.total_tokens_seen: int = 0
        self.head_dim: int = 0
        self.target_dtype: Any = None
        self.target_device: Any = None

        # Host pinned CPU full backup pool for deep layers: [capacity, B, H_kv, D]
        self.cpu_k_buf: torch.Tensor | None = None
        self.cpu_v_buf: torch.Tensor | None = None

        # GPU-only buffers for exempt early layers: [B, H_kv, capacity, D]
        self.gpu_k_buf: torch.Tensor | None = None
        self.gpu_v_buf: torch.Tensor | None = None

        # Resident GPU page slots for deep layers: [active_pages_budget, B, H_kv, page_size, D]
        self.gpu_slot_k: torch.Tensor | None = None
        self.gpu_slot_v: torch.Tensor | None = None
        self.slot_to_page: dict[int, int] = {}
        self.page_to_slot: dict[int, int] = {}

        # Pre-allocated GPU AABB page digests: [B, H_kv, max_pages, D]
        self.gpu_b_min: torch.Tensor | None = None
        self.gpu_b_max: torch.Tensor | None = None

        self.copy_stream: Any = None
        self.active_positions: list[int] = []
        self.total_bytes_transferred_pcie: int = 0
        self.total_recalled_pages: int = 0

        super().__init__()

    @property
    def resident_pages(self) -> set[int]:
        if self.layer_idx < self.exempt_early_layers:
            if self.total_tokens_seen == 0:
                return set()
            return set(range(math.ceil(self.total_tokens_seen / self.page_size)))
        return set(self.page_to_slot.keys())

    @property
    def original_positions(self) -> list[int]:
        if not self.active_positions:
            return list(range(self.total_tokens_seen))
        return list(self.active_positions)

    @property
    def keys(self) -> torch.Tensor | None:
        if not self.is_initialized or self.total_tokens_seen == 0:
            return self._keys_fallback
        if self.layer_idx < self.exempt_early_layers:
            assert self.gpu_k_buf is not None
            return self.gpu_k_buf[:, :, : self.total_tokens_seen]
        assert self.cpu_k_buf is not None
        return (
            self.cpu_k_buf[: self.total_tokens_seen]
            .to(self.target_device, non_blocking=True)
            .permute(1, 2, 0, 3)
        )

    @keys.setter
    def keys(self, value: torch.Tensor | None) -> None:
        self._keys_fallback = value

    @property
    def values(self) -> torch.Tensor | None:
        if not self.is_initialized or self.total_tokens_seen == 0:
            return self._values_fallback
        if self.layer_idx < self.exempt_early_layers:
            assert self.gpu_v_buf is not None
            return self.gpu_v_buf[:, :, : self.total_tokens_seen]
        assert self.cpu_v_buf is not None
        return (
            self.cpu_v_buf[: self.total_tokens_seen]
            .to(self.target_device, non_blocking=True)
            .permute(1, 2, 0, 3)
        )

    @values.setter
    def values(self, value: torch.Tensor | None) -> None:
        self._values_fallback = value

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.target_dtype = key_states.dtype
        self.target_device = key_states.device
        self.head_dim = key_states.shape[-1]
        self.is_initialized = True

    def _compute_new_capacity(self, required_capacity: int) -> int:
        if required_capacity <= 2048:
            return max(required_capacity, self.capacity * 2, 1024)
        return math.ceil((required_capacity + 512) / self.page_size) * self.page_size

    def _ensure_exempt_capacity(
        self,
        batch_size: int,
        num_kv: int,
        head_dim: int,
        target_device: Any,
        target_dtype: Any,
        required_capacity: int,
    ) -> None:
        if self.capacity >= required_capacity:
            return

        new_capacity = self._compute_new_capacity(required_capacity)
        new_gpu_k = torch.empty(
            (batch_size, num_kv, new_capacity, head_dim), dtype=target_dtype, device=target_device
        )
        new_gpu_v = torch.empty_like(new_gpu_k)

        if self.total_tokens_seen > 0 and self.gpu_k_buf is not None and self.gpu_v_buf is not None:
            t = self.total_tokens_seen
            new_gpu_k[:, :, :t].copy_(self.gpu_k_buf[:, :, :t])
            new_gpu_v[:, :, :t].copy_(self.gpu_v_buf[:, :, :t])

        self.gpu_k_buf = new_gpu_k
        self.gpu_v_buf = new_gpu_v
        self.capacity = new_capacity

    def _ensure_capacity(
        self,
        batch_size: int,
        num_kv: int,
        head_dim: int,
        target_device: Any,
        target_dtype: Any,
        required_capacity: int,
    ) -> None:
        if self.capacity >= required_capacity:
            return

        new_capacity = self._compute_new_capacity(required_capacity)
        new_page_cap = math.ceil(new_capacity / self.page_size)
        pin = torch.cuda.is_available()

        new_cpu_k = torch.empty(
            (new_capacity, batch_size, num_kv, head_dim), dtype=target_dtype, device="cpu", pin_memory=pin
        )
        new_cpu_v = torch.empty(
            (new_capacity, batch_size, num_kv, head_dim), dtype=target_dtype, device="cpu", pin_memory=pin
        )
        new_b_min = torch.zeros(
            (batch_size, num_kv, new_page_cap, head_dim), dtype=target_dtype, device=target_device
        )
        new_b_max = torch.zeros_like(new_b_min)

        if self.total_tokens_seen > 0 and self.cpu_k_buf is not None and self.cpu_v_buf is not None:
            if torch.cuda.is_available() and getattr(target_device, "type", None) == "cuda":
                torch.cuda.synchronize(target_device)
            t = self.total_tokens_seen
            p_used = math.ceil(t / self.page_size)
            new_cpu_k[:t].copy_(self.cpu_k_buf[:t])
            new_cpu_v[:t].copy_(self.cpu_v_buf[:t])
            if self.gpu_b_min is not None and self.gpu_b_max is not None:
                new_b_min[:, :, :p_used].copy_(self.gpu_b_min[:, :, :p_used])
                new_b_max[:, :, :p_used].copy_(self.gpu_b_max[:, :, :p_used])

        self.cpu_k_buf = new_cpu_k
        self.cpu_v_buf = new_cpu_v
        self.gpu_b_min = new_b_min
        self.gpu_b_max = new_b_max
        self.capacity = new_capacity
        self.page_capacity = new_page_cap

    def _set_page_digest(self, page_idx: int, page_k: torch.Tensor) -> None:
        """Compute and store AABB digest for a single page tensor [B, H_kv, len, D]."""
        assert self.gpu_b_min is not None and self.gpu_b_max is not None
        c = 0.5 * (page_k.amin(dim=-2) + page_k.amax(dim=-2))
        r_mean = (page_k - c.unsqueeze(-2)).abs().mean(dim=-2)
        self.gpu_b_max[:, :, page_idx] = c + r_mean
        self.gpu_b_min[:, :, page_idx] = c - r_mean

    def _update_incremental_digests(
        self, key_states: torch.Tensor, start_pos: int, end_pos: int
    ) -> None:
        """Vectorized incremental computation of GPU AABB page digests for touched pages."""
        assert self.cpu_k_buf is not None and self.gpu_b_min is not None and self.gpu_b_max is not None
        batch_size, num_kv, _, head_dim = key_states.shape
        p_first = start_pos // self.page_size
        p_last = (end_pos - 1) // self.page_size

        # 1. Leading partial page (when start_pos is not page-aligned)
        if start_pos % self.page_size != 0:
            first_start = p_first * self.page_size
            first_end = min(end_pos, (p_first + 1) * self.page_size)
            valid_len = first_end - first_start
            if p_first in self.page_to_slot and self.gpu_slot_k is not None:
                slot = self.page_to_slot[p_first]
                page_k = self.gpu_slot_k[slot, :, :, :valid_len, :]
            else:
                page_k = (
                    self.cpu_k_buf[first_start:first_end]
                    .to(self.target_device)
                    .permute(1, 2, 0, 3)
                )
            self._set_page_digest(p_first, page_k)
            full_start = (p_first + 1) * self.page_size
        else:
            full_start = start_pos

        # 2. Vectorized middle full pages
        full_end = (end_pos // self.page_size) * self.page_size
        if full_end > full_start:
            num_full = (full_end - full_start) // self.page_size
            p_full_start = full_start // self.page_size
            p_full_end = p_full_start + num_full
            k_full = key_states[:, :, full_start - start_pos : full_end - start_pos, :]
            pages = k_full.contiguous().reshape(batch_size, num_kv, num_full, self.page_size, head_dim)
            c = 0.5 * (pages.amin(dim=-2) + pages.amax(dim=-2))
            r_mean = (pages - c.unsqueeze(-2)).abs().mean(dim=-2)
            self.gpu_b_max[:, :, p_full_start:p_full_end] = c + r_mean
            self.gpu_b_min[:, :, p_full_start:p_full_end] = c - r_mean

        # 3. Trailing partial page (including short prompts and curr % page_size == 0 decode)
        if full_end < end_pos and (start_pos % self.page_size == 0 or p_last != p_first):
            k_tail = key_states[:, :, full_end - start_pos : end_pos - start_pos, :]
            self._set_page_digest(p_last, k_tail)

    def should_use_sparse_decode(self) -> bool:
        """Return True when this layer should resolve active KV pages during 1-token decode."""
        if self.layer_idx < self.exempt_early_layers or self.total_tokens_seen == 0:
            return False
        total_pages = math.ceil(self.total_tokens_seen / self.page_size)
        return total_pages > self.active_pages_budget

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update ArkVale cache; defer sparse page selection/recall to SDPA hook on decode."""
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        batch_size = key_states.shape[0]
        num_kv = key_states.shape[1]
        new_tokens = key_states.shape[-2]
        head_dim = key_states.shape[-1]

        self.target_dtype = key_states.dtype
        self.target_device = key_states.device
        self.head_dim = head_dim

        curr = self.total_tokens_seen
        new_total = curr + new_tokens
        is_exempt = self.layer_idx < self.exempt_early_layers

        if is_exempt:
            self._ensure_exempt_capacity(
                batch_size, num_kv, head_dim, key_states.device, key_states.dtype, new_total
            )
            assert self.gpu_k_buf is not None and self.gpu_v_buf is not None
            self.gpu_k_buf[:, :, curr:new_total].copy_(key_states)
            self.gpu_v_buf[:, :, curr:new_total].copy_(value_states)
            self.total_tokens_seen = new_total
            self.active_positions = list(range(new_total))
            return self.gpu_k_buf[:, :, :new_total], self.gpu_v_buf[:, :, :new_total]

        self._ensure_capacity(
            batch_size, num_kv, head_dim, key_states.device, key_states.dtype, new_total
        )
        assert self.cpu_k_buf is not None and self.cpu_v_buf is not None
        self.cpu_k_buf[curr:new_total].copy_(
            key_states.permute(2, 0, 1, 3).contiguous(), non_blocking=True
        )
        self.cpu_v_buf[curr:new_total].copy_(
            value_states.permute(2, 0, 1, 3).contiguous(), non_blocking=True
        )
        self.total_bytes_transferred_pcie += (
            (key_states.nelement() + value_states.nelement()) * key_states.element_size()
        )

        if new_tokens > 1:
            self.page_to_slot.clear()
            self.slot_to_page.clear()
        else:
            tail_page = curr // self.page_size
            if tail_page in self.page_to_slot and self.gpu_slot_k is not None and self.gpu_slot_v is not None:
                slot = self.page_to_slot[tail_page]
                offset = curr % self.page_size
                self.gpu_slot_k[slot, :, :, offset : offset + 1, :].copy_(key_states)
                self.gpu_slot_v[slot, :, :, offset : offset + 1, :].copy_(value_states)

        self.total_tokens_seen = new_total
        self._update_incremental_digests(key_states, curr, new_total)
        self.active_positions = list(range(new_total))

        total_pages = math.ceil(new_total / self.page_size)
        if new_tokens == 1 and total_pages > self.active_pages_budget:
            # Deferred to SDPA hook via resolve_active_kv(query)
            return key_states, value_states

        if curr == 0:
            return key_states, value_states

        # Single-layer prefill/under-budget materialization (contiguous HtoD before permute)
        prev_k_gpu = self.cpu_k_buf[:curr].to(key_states.device, non_blocking=True).permute(1, 2, 0, 3)
        prev_v_gpu = self.cpu_v_buf[:curr].to(value_states.device, non_blocking=True).permute(1, 2, 0, 3)
        k_out = torch.empty(
            (batch_size, num_kv, new_total, head_dim), dtype=key_states.dtype, device=key_states.device
        )
        v_out = torch.empty_like(k_out)
        k_out[:, :, :curr].copy_(prev_k_gpu)
        v_out[:, :, :curr].copy_(prev_v_gpu)
        k_out[:, :, curr:new_total].copy_(key_states)
        v_out[:, :, curr:new_total].copy_(value_states)
        self.total_bytes_transferred_pcie += (
            2 * batch_size * num_kv * curr * head_dim * key_states.element_size()
        )
        return k_out, v_out

    def resolve_active_kv(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Score AABB digests with post-RoPE query, fetch delta pages into GPU slots, and return active KV."""
        from src.techniques.arkvale import score_page_digests_gqa

        assert self.cpu_k_buf is not None and self.cpu_v_buf is not None
        total_tokens = self.total_tokens_seen
        total_pages = math.ceil(total_tokens / self.page_size)
        batch_size, _, _, head_dim = query.shape
        num_kv = self.cpu_k_buf.shape[2]
        target_device = query.device
        target_dtype = self.target_dtype or query.dtype

        if self.active_pages_budget == 1:
            selected_pages = {total_pages - 1}
        elif total_pages <= self.active_pages_budget:
            selected_pages = set(range(total_pages))
        else:
            assert self.gpu_b_min is not None and self.gpu_b_max is not None
            b_min = self.gpu_b_min[:, :, :total_pages, :]
            b_max = self.gpu_b_max[:, :, :total_pages, :]
            kv_scores = score_page_digests_gqa(query, b_min, b_max)
            agg_scores = kv_scores.amax(dim=1).mean(dim=0)

            k_budget = max(0, min(self.active_pages_budget, total_pages) - 2)
            if k_budget > 0 and total_pages > 2:
                masked_scores = agg_scores.clone()
                masked_scores[0] = float("-inf")
                masked_scores[total_pages - 1] = float("-inf")
                _, top_pages = torch.topk(masked_scores, k=min(k_budget, total_pages - 2))
                selected_pages = set(top_pages.tolist()) | {0, total_pages - 1}
            else:
                selected_pages = {0, total_pages - 1}

        if self.gpu_slot_k is None or self.gpu_slot_v is None:
            self.gpu_slot_k = torch.empty(
                (self.active_pages_budget, batch_size, num_kv, self.page_size, head_dim),
                dtype=target_dtype,
                device=target_device,
            )
            self.gpu_slot_v = torch.empty_like(self.gpu_slot_k)

        current_resident = set(self.page_to_slot.keys())
        evict_pages = current_resident - selected_pages
        delta_pages = sorted(selected_pages - current_resident)

        for p in evict_pages:
            slot = self.page_to_slot.pop(p)
            self.slot_to_page.pop(slot, None)

        used_slots = set(self.slot_to_page.keys())
        available_slots = [s for s in range(self.active_pages_budget) if s not in used_slots]

        use_cuda_stream = query.is_cuda and torch.cuda.is_available()
        if use_cuda_stream and self.copy_stream is None:
            self.copy_stream = torch.cuda.Stream(device=target_device)

        elem_size = self.gpu_slot_k.element_size()
        if delta_pages:
            if use_cuda_stream:
                assert self.copy_stream is not None
                compute_stream = torch.cuda.current_stream(device=target_device)
                self.copy_stream.wait_stream(compute_stream)
            stream_ctx = (
                torch.cuda.stream(self.copy_stream)
                if use_cuda_stream
                else _null_context()
            )
            with stream_ctx:
                for p in delta_pages:
                    slot = available_slots.pop(0)
                    self.page_to_slot[p] = slot
                    self.slot_to_page[slot] = p
                    s_tok = p * self.page_size
                    e_tok = min(total_tokens, (p + 1) * self.page_size)
                    valid_len = e_tok - s_tok

                    k_gpu = self.cpu_k_buf[s_tok:e_tok].to(target_device, non_blocking=True).permute(1, 2, 0, 3)
                    v_gpu = self.cpu_v_buf[s_tok:e_tok].to(target_device, non_blocking=True).permute(1, 2, 0, 3)
                    self.gpu_slot_k[slot, :, :, :valid_len, :].copy_(k_gpu)
                    self.gpu_slot_v[slot, :, :, :valid_len, :].copy_(v_gpu)

                    self.total_recalled_pages += 1
                    self.total_bytes_transferred_pcie += (
                        2 * batch_size * num_kv * valid_len * head_dim * elem_size
                    )

            if use_cuda_stream:
                torch.cuda.current_stream(device=target_device).wait_stream(self.copy_stream)

        k_chunks: list[torch.Tensor] = []
        v_chunks: list[torch.Tensor] = []
        active_positions: list[int] = []
        for p in sorted(selected_pages):
            slot = self.page_to_slot[p]
            s_tok = p * self.page_size
            e_tok = min(total_tokens, (p + 1) * self.page_size)
            valid_len = e_tok - s_tok
            k_chunks.append(self.gpu_slot_k[slot, :, :, :valid_len, :])
            v_chunks.append(self.gpu_slot_v[slot, :, :, :valid_len, :])
            active_positions.extend(range(s_tok, e_tok))

        self.active_positions = active_positions
        return torch.cat(k_chunks, dim=-2), torch.cat(v_chunks, dim=-2)

    def get_seq_length(self) -> int:
        return self.total_tokens_seen

    def get_max_length(self) -> int:
        return max(self.total_tokens_seen, 32768)

    def reset(self) -> None:
        self.total_tokens_seen = 0
        self.active_positions = []
        self.slot_to_page.clear()
        self.page_to_slot.clear()
        self.cpu_k_buf = None
        self.cpu_v_buf = None
        self.gpu_k_buf = None
        self.gpu_v_buf = None
        self.gpu_slot_k = None
        self.gpu_slot_v = None
        self.gpu_b_min = None
        self.gpu_b_max = None
        self.copy_stream = None
        self.capacity = 0
        self.page_capacity = 0
        self.total_bytes_transferred_pcie = 0
        self.total_recalled_pages = 0
        self._keys_fallback = None
        self._values_fallback = None
        super().reset()


class _null_context:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:
        return None


class ArkValeCache(Cache):
    """Subclass of transformers.cache_utils.Cache implementing ArkVale page recall."""

    is_arkvale = True

    def __init__(
        self,
        page_size: int = 16,
        active_pages_budget: int = 16,
        exempt_early_layers: int = 2,
    ):
        self.page_size = int(page_size)
        self.active_pages_budget = int(active_pages_budget)
        self.exempt_early_layers = int(exempt_early_layers)

        super().__init__(
            layer_class_to_replicate=lambda: ArkValeLayer(
                layer_idx=len(self.layers),
                page_size=self.page_size,
                active_pages_budget=self.active_pages_budget,
                exempt_early_layers=self.exempt_early_layers,
            )
        )
        set_active_arkvale_cache(self)

    @property
    def original_positions(self) -> list[int]:
        if not self.layers:
            return []
        return self.layers[-1].original_positions

    @property
    def total_bytes_transferred_pcie(self) -> int:
        return sum(getattr(layer, "total_bytes_transferred_pcie", 0) for layer in self.layers)

    @property
    def total_recalled_pages(self) -> int:
        return sum(getattr(layer, "total_recalled_pages", 0) for layer in self.layers)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_max_length(self) -> int:
        if not self.layers:
            return 32768
        return self.layers[0].get_max_length()

    def __iter__(self) -> Iterator[tuple[Any, Any, Any]]:
        for layer in self.layers:
            yield getattr(layer, "keys", None), getattr(layer, "values", None), None
