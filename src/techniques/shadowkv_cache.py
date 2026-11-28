from __future__ import annotations

import math
import weakref
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from transformers.cache_utils import Cache, DynamicLayer


_ACTIVE_SHADOWKV_CACHE_REF: weakref.ReferenceType[ShadowKVCache] | None = None


def get_active_shadowkv_cache() -> ShadowKVCache | None:
    """Return currently active ShadowKVCache instance if alive, else None."""
    global _ACTIVE_SHADOWKV_CACHE_REF
    if _ACTIVE_SHADOWKV_CACHE_REF is not None:
        return _ACTIVE_SHADOWKV_CACHE_REF()
    return None


def set_active_shadowkv_cache(cache: ShadowKVCache | None) -> None:
    """Register weak reference to active ShadowKVCache ensuring zero tensor retention leaks."""
    global _ACTIVE_SHADOWKV_CACHE_REF
    if cache is not None:
        _ACTIVE_SHADOWKV_CACHE_REF = weakref.ref(cache)
    else:
        _ACTIVE_SHADOWKV_CACHE_REF = None


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def compute_rope_cos_sin(
    seq_len: int,
    head_dim: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    theta: float = 10000.0,
    start_pos: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cosine and sine rotary embeddings for a contiguous range of token positions."""
    half_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim))
    positions = torch.arange(start_pos, start_pos + seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)  # [seq_len, half_dim]
    emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, head_dim]
    cos = emb.cos().to(dtype=dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = emb.sin().to(dtype=dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    return cos, sin


def compute_rope_for_positions(
    positions: torch.Tensor,
    head_dim: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cosine and sine rotary embeddings for arbitrary (potentially non-contiguous) positions."""
    half_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim))
    freqs = torch.outer(positions.to(device=device).float(), inv_freq)  # [num_positions, half_dim]
    emb = torch.cat((freqs, freqs), dim=-1)  # [num_positions, head_dim]
    cos = emb.cos().to(dtype=dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, num_positions, head_dim]
    sin = emb.sin().to(dtype=dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, num_positions, head_dim]
    return cos, sin


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply forward Rotary Position Embedding (RoPE)."""
    return (x * cos) + (rotate_half(x) * sin)


def invert_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply exact inverse Rotary Position Embedding (RoPE) via negative rotation."""
    return (x * cos) + (rotate_half(x) * (-sin))


def compute_chunk_landmarks(
    keys: torch.Tensor,
    chunk_size: int = 8,
) -> tuple[torch.Tensor, int, int]:
    """Compute mean landmark centroids for post-RoPE keys."""
    b, h_kv, s, d = keys.shape
    pad_len = (chunk_size - (s % chunk_size)) % chunk_size
    if pad_len > 0:
        keys_padded = F.pad(keys, (0, 0, 0, pad_len), value=0.0)
    else:
        keys_padded = keys
    padded_s = keys_padded.shape[2]
    num_chunks = padded_s // chunk_size

    chunks = keys_padded.view(b, h_kv, num_chunks, chunk_size, d)
    if pad_len > 0:
        weights = torch.ones(b, h_kv, num_chunks, chunk_size, 1, device=keys.device, dtype=keys.dtype)
        weights[..., -1, chunk_size - pad_len :, :] = 0.0
        landmarks = (chunks * weights).sum(dim=-2) / weights.sum(dim=-2).clamp(min=1.0)
    else:
        landmarks = chunks.mean(dim=-2)

    return landmarks, num_chunks, padded_s


def detect_outlier_chunks(
    keys: torch.Tensor,
    landmarks: torch.Tensor,
    chunk_size: int = 8,
    outlier_ratio: float = 0.003,
    min_outliers: int = 8,
) -> list[int]:
    """Identify outlier chunks having the lowest minimum cosine similarity to their centroid."""
    b, h_kv, s, d = keys.shape
    num_chunks = landmarks.shape[2]

    pad_len = (chunk_size - (s % chunk_size)) % chunk_size
    if pad_len > 0:
        keys_padded = F.pad(keys, (0, 0, 0, pad_len), value=0.0)
    else:
        keys_padded = keys

    chunks = keys_padded.view(b, h_kv, num_chunks, chunk_size, d)
    landmarks_exp = landmarks.unsqueeze(-2)  # [B, H_kv, num_chunks, 1, D]
    norm_chunks = chunks.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    norm_landmarks = landmarks_exp.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    cos_sim = (chunks * landmarks_exp).sum(dim=-1, keepdim=True) / (norm_chunks * norm_landmarks)
    min_sim = cos_sim.squeeze(-1).amin(dim=(0, 1, -1))  # [num_chunks]

    num_outliers = max(min_outliers, int(math.ceil(num_chunks * outlier_ratio)))
    num_outliers = min(num_outliers, num_chunks)

    candidate_indices = torch.topk(-min_sim, k=min(num_outliers, num_chunks)).indices.tolist()
    outlier_set = {0}
    for idx in candidate_indices:
        outlier_set.add(idx)
        if len(outlier_set) >= num_outliers:
            break

    return sorted(outlier_set)


def _validate_theta(theta: float | None, default: float = 10000.0) -> float:
    """Validate RoPE base theta, rejecting bools, non-positive, and non-finite values."""
    if theta is None:
        return float(default)
    if isinstance(theta, bool) or not isinstance(theta, (int, float)):
        raise ValueError(f"theta must be a positive finite number, got {theta!r}")
    val = float(theta)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"theta must be positive and finite, got {theta!r}")
    return val


def extract_rope_theta(config: Any, default: float = 10000.0) -> float:
    """Safely extract rope_theta from a Hugging Face model config."""
    if config is None:
        return float(default)

    rope_params = getattr(config, "rope_parameters", None)
    if isinstance(rope_params, dict):
        val = rope_params.get("rope_theta")
        if val is not None and not isinstance(val, bool) and math.isfinite(float(val)) and float(val) > 0.0:
            return float(val)

    val = getattr(config, "rope_theta", None)
    if val is not None and not isinstance(val, bool) and math.isfinite(float(val)) and float(val) > 0.0:
        return float(val)

    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        val = rope_scaling.get("rope_theta")
        if val is not None and not isinstance(val, bool) and math.isfinite(float(val)) and float(val) > 0.0:
            return float(val)

    return float(default)


def perform_svd_key_decomposition(
    keys_pre_rope: torch.Tensor,
    svd_rank: int = 160,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cross-head truncated SVD on pre-RoPE keys of shape (B, H_kv, S, D)."""
    b, h_kv, s, d = keys_pre_rope.shape
    m_dim = h_kv * d
    r = min(svd_rank, s, m_dim)

    flat_k = keys_pre_rope.permute(0, 2, 1, 3).reshape(b, s, m_dim).float()
    u, s_vals, vh = torch.linalg.svd(flat_k, full_matrices=False)

    u_r = u[:, :, :r]  # [B, S, r]
    s_r = s_vals[:, :r]  # [B, r]
    vh_r = vh[:, :r, :]  # [B, r, H_kv*D]

    sqrt_s = torch.sqrt(s_r).unsqueeze(1)  # [B, 1, r]
    a_proj = (u_r * sqrt_s).to(keys_pre_rope.dtype)  # [B, S, r]
    b_basis = (sqrt_s.transpose(1, 2) * vh_r).to(keys_pre_rope.dtype)  # [B, r, H_kv*D]

    del flat_k, u, s_vals, vh, u_r, s_r, vh_r, sqrt_s
    return a_proj, b_basis


class ShadowKVLayer(DynamicLayer):
    """Cache layer implementing ShadowKV (GPU Low-Rank Keys + Pinned Host CPU Values)."""

    is_sliding = False

    def __init__(
        self,
        svd_rank: int = 160,
        chunk_size: int = 8,
        topk_chunks: int = 256,
        outlier_ratio: float = 0.003,
        temporal_reuse_target: float = 0.60,
        local_window_tokens: int = 32,
        svd_threshold_tokens: int = 1024,
        theta: float | None = None,
        _cache_ref: weakref.ReferenceType[ShadowKVCache] | None = None,
    ):
        super().__init__()
        self._theta_explicit: bool = theta is not None
        self._theta_synced: bool = False
        self.theta: float = _validate_theta(theta)
        self._cache_ref = _cache_ref

        self.svd_rank = int(svd_rank)
        self.chunk_size = int(chunk_size)
        self.topk_chunks = int(topk_chunks)
        self.outlier_ratio = float(outlier_ratio)
        self.temporal_reuse_target = float(temporal_reuse_target)
        self.local_window_tokens = int(local_window_tokens)
        self.svd_threshold_tokens = int(svd_threshold_tokens)

        # Validation
        if self.svd_rank <= 0:
            raise ValueError(f"svd_rank must be positive, got {self.svd_rank}")
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.topk_chunks <= 0:
            raise ValueError(f"topk_chunks must be positive, got {self.topk_chunks}")
        if not (0.0 <= self.outlier_ratio <= 1.0):
            raise ValueError(f"outlier_ratio must be in [0, 1], got {self.outlier_ratio}")

        # GPU-resident low-rank keys and metadata
        self.is_offloaded: bool = False
        self.total_tokens_seen: int = 0
        self.prefill_seq_len: int = 0
        self.head_dim: int = 0
        self._device: torch.device | None = None
        self._dtype: torch.dtype | None = None

        # Legacy dense prefill fields (kept None during CPU-staged prefill)
        self.dense_keys: torch.Tensor | None = None
        self.dense_values: torch.Tensor | None = None

        # Pinned CPU staging buffers for prefill & dense fallback
        self._cpu_k_buf: torch.Tensor | None = None
        self._cpu_v_buf: torch.Tensor | None = None

        # ShadowKV prefill data structures
        self.key_a: torch.Tensor | None = None  # [B, S, r]
        self.key_b: torch.Tensor | None = None  # [B, r, H_kv*D]
        self.landmarks: torch.Tensor | None = None  # [B, H_kv, num_chunks, D]
        self.num_chunks: int = 0
        self.outlier_chunk_indices: list[int] = []
        self._anchor_token_set: set[int] = set()
        self._candidate_tokens: dict[int, list[int]] | None = None
        self._ineligible_chunk_mask: torch.Tensor | None = None

        # GPU working cache: Outliers + Local window
        self.gpu_anchor_keys: torch.Tensor | None = None
        self.gpu_anchor_values: torch.Tensor | None = None
        self.gpu_anchor_pos: torch.Tensor | None = None

        # Host CPU DRAM pinned buffer for values (zero-copy view of _cpu_v_buf after prune)
        self.cpu_values: torch.Tensor | None = None  # [B, H_kv, S, D]

        # Bounded LRU temporal cache for autoregressive decoding
        self._lru_chunks: OrderedDict[int, int] = OrderedDict()
        self._cached_chunk_layout: list[tuple[int, int]] = []
        self.cached_topk_indices: set[int] = set()
        self.cached_sparse_keys: torch.Tensor | None = None
        self.cached_sparse_values: torch.Tensor | None = None
        self.cached_sparse_pos: torch.Tensor | None = None

        # Telemetry
        self.total_bytes_transferred_pcie: int = 0
        self.temporal_hits: int = 0
        self.temporal_misses: int = 0

        # Lazy fallback CUDA copy stream if layer is used standalone without ShadowKVCache
        self._copy_stream: torch.cuda.Stream | None = None

    def _get_copy_stream(self) -> torch.cuda.Stream | None:
        """Return shared CUDA copy stream from parent cache, or lazy standalone stream."""
        if self._cache_ref is not None:
            owner = self._cache_ref()
            if owner is not None:
                return owner.get_copy_stream()
        if self._copy_stream is None and torch is not None and torch.cuda.is_available():
            try:
                self._copy_stream = torch.cuda.Stream()
            except Exception:
                self._copy_stream = None
        return self._copy_stream

    @staticmethod
    def _alloc_cpu_staging(shape: tuple[int, int, int, int], dtype: torch.dtype, pin: bool) -> torch.Tensor:
        """Allocate CPU staging buffer, pinned when CUDA is active."""
        if pin:
            try:
                return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
            except Exception:
                pass
        return torch.empty(shape, dtype=dtype, device="cpu")

    def _append_to_cpu_staging(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append incoming K/V states to CPU staging buffers and return full K/V on input device."""
        b, h_kv, seq_len, d = key_states.shape
        cur_len = self.total_tokens_seen
        needed = cur_len + seq_len
        pin = key_states.is_cuda

        if self._cpu_k_buf is None or self._cpu_v_buf is None:
            init_cap = max(needed, 256) if needed <= 32768 else needed
            self._cpu_k_buf = self._alloc_cpu_staging((b, h_kv, init_cap, d), key_states.dtype, pin)
            self._cpu_v_buf = self._alloc_cpu_staging((b, h_kv, init_cap, d), value_states.dtype, pin)
        elif needed > self._cpu_k_buf.shape[2]:
            cur_cap = self._cpu_k_buf.shape[2]
            if needed <= 32768:
                new_cap = min(32768, max(needed, cur_cap * 2))
            else:
                new_cap = max(needed, cur_cap + 4096)
            new_k = self._alloc_cpu_staging((b, h_kv, new_cap, d), key_states.dtype, pin)
            new_v = self._alloc_cpu_staging((b, h_kv, new_cap, d), value_states.dtype, pin)
            if cur_len > 0:
                new_k[:, :, :cur_len, :].copy_(self._cpu_k_buf[:, :, :cur_len, :])
                new_v[:, :, :cur_len, :].copy_(self._cpu_v_buf[:, :, :cur_len, :])
            self._cpu_k_buf = new_k
            self._cpu_v_buf = new_v

        self._cpu_k_buf[:, :, cur_len:needed, :].copy_(key_states, non_blocking=False)
        self._cpu_v_buf[:, :, cur_len:needed, :].copy_(value_states, non_blocking=False)
        self.total_tokens_seen = needed

        k_slice = self._cpu_k_buf[:, :, :needed, :]
        v_slice = self._cpu_v_buf[:, :, :needed, :]
        if key_states.is_cuda:
            return k_slice.to(device=key_states.device), v_slice.to(device=value_states.device)
        return k_slice.clone(), v_slice.clone()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update layer cache with incoming key and value states."""
        _, _, seq_len, d = key_states.shape
        self.head_dim = d
        self._device = key_states.device
        self._dtype = key_states.dtype

        # Phase 1: Prefill (seq_len > 1) -> stage directly on CPU
        if seq_len > 1:
            return self._append_to_cpu_staging(key_states, value_states)

        # Phase 2: Autoregressive Decode (seq_len == 1)
        # If prefill/dense decode exceeded threshold and prune was not yet called, trigger prune
        if not self.is_offloaded and self.total_tokens_seen > self.svd_threshold_tokens:
            self.prune()

        if not self.is_offloaded:
            # Dense fallback path (S <= svd_threshold_tokens)
            return self._append_to_cpu_staging(key_states, value_states)

        self.total_tokens_seen += 1
        curr_pos = self.total_tokens_seen - 1

        # Append new token to the GPU anchor / local working window
        assert self.gpu_anchor_keys is not None and self.gpu_anchor_values is not None
        assert self.gpu_anchor_pos is not None

        new_pos = torch.tensor([curr_pos], device=key_states.device, dtype=torch.long)
        self.gpu_anchor_keys = torch.cat([self.gpu_anchor_keys, key_states], dim=2)
        self.gpu_anchor_values = torch.cat([self.gpu_anchor_values, value_states], dim=2)
        self.gpu_anchor_pos = torch.cat([self.gpu_anchor_pos, new_pos], dim=0)

        # Return the anchor cache; full sparse decode attention will be assembled in SDPA hook
        return self.gpu_anchor_keys, self.gpu_anchor_values

    def prune(self) -> None:
        """Trigger SVD key decomposition and host value offload across all prompt tokens."""
        if self.is_offloaded or self.total_tokens_seen <= self.svd_threshold_tokens:
            return
        if self._cpu_k_buf is None or self._cpu_v_buf is None:
            return

        self.is_offloaded = True
        s = self.total_tokens_seen
        self.prefill_seq_len = s
        target_device = self._device or torch.device("cpu")

        key_states = self._cpu_k_buf[:, :, :s, :].to(device=target_device)
        _, _, _, d = key_states.shape

        # 1. Compute chunk landmarks and detect outliers while key_states is on GPU
        self.landmarks, self.num_chunks, _ = compute_chunk_landmarks(
            keys=key_states,
            chunk_size=self.chunk_size,
        )
        self.outlier_chunk_indices = detect_outlier_chunks(
            keys=key_states,
            landmarks=self.landmarks,
            chunk_size=self.chunk_size,
            outlier_ratio=self.outlier_ratio,
        )

        # 2. Extract Anchor tokens (Outliers + Local Window) to reside on GPU; slice V on CPU
        outlier_token_indices: list[int] = []
        for c_idx in self.outlier_chunk_indices:
            start = c_idx * self.chunk_size
            end = min(s, start + self.chunk_size)
            outlier_token_indices.extend(range(start, end))

        local_start = max(0, s - self.local_window_tokens)
        local_token_indices = list(range(local_start, s))

        anchor_indices = sorted(set(outlier_token_indices + local_token_indices))
        self._anchor_token_set = set(anchor_indices)
        anchor_idx_gpu = torch.tensor(anchor_indices, device=target_device, dtype=torch.long)
        anchor_idx_cpu = anchor_idx_gpu.cpu()

        self.gpu_anchor_keys = torch.index_select(key_states, dim=2, index=anchor_idx_gpu)
        self.gpu_anchor_values = torch.index_select(
            self._cpu_v_buf[:, :, :s, :], dim=2, index=anchor_idx_cpu
        ).to(device=target_device)
        self.gpu_anchor_pos = anchor_idx_gpu

        # 3. Recover pre-RoPE keys via exact inverse RoPE, then free dense K before SVD
        cos, sin = compute_rope_cos_sin(
            seq_len=s,
            head_dim=d,
            device=target_device,
            dtype=key_states.dtype,
            theta=self.theta,
            start_pos=0,
        )
        keys_pre = invert_rope(key_states, cos, sin)
        del key_states, cos, sin
        self._cpu_k_buf = None

        # 4. Decompose pre-RoPE keys into cross-head low-rank factors
        self.key_a, self.key_b = perform_svd_key_decomposition(
            keys_pre_rope=keys_pre,
            svd_rank=self.svd_rank,
        )
        del keys_pre

        # 5. Keep cpu_values as a zero-copy view of pinned _cpu_v_buf
        self.cpu_values = self._cpu_v_buf[:, :, :s, :]
        self.dense_keys = None
        self.dense_values = None
        self._build_candidate_index(target_device)

    def _build_candidate_index(self, device: torch.device) -> tuple[dict[int, list[int]], torch.Tensor]:
        """Precompute eligible non-anchor chunk tokens and boolean mask once per sample."""
        outlier_set = set(self.outlier_chunk_indices)
        candidate_tokens: dict[int, list[int]] = {}
        ineligible = torch.ones(self.num_chunks, dtype=torch.bool, device="cpu")
        for c_idx in range(self.num_chunks):
            if c_idx in outlier_set:
                continue
            start = c_idx * self.chunk_size
            end = min(self.prefill_seq_len, start + self.chunk_size)
            if start >= self.prefill_seq_len:
                continue
            non_anchor = [pos for pos in range(start, end) if pos not in self._anchor_token_set]
            if non_anchor:
                candidate_tokens[c_idx] = non_anchor
                ineligible[c_idx] = False
        mask = ineligible.to(device=device)
        self._candidate_tokens = candidate_tokens
        self._ineligible_chunk_mask = mask
        return candidate_tokens, mask

    def _reconstruct_keys(
        self,
        missed_idx_tensor: torch.Tensor,
        b: int,
        h_kv: int,
        d: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Reconstruct post-RoPE keys from cross-head SVD factors for missed token positions."""
        assert self.key_a is not None and self.key_b is not None
        a_slice = torch.index_select(self.key_a, dim=1, index=missed_idx_tensor)  # [B, S_miss, r]
        recon_flat = torch.matmul(a_slice, self.key_b)  # [B, S_miss, H_kv*D]
        recon_keys_pre = (
            recon_flat.reshape(b, missed_idx_tensor.shape[0], h_kv, d).permute(0, 2, 1, 3).contiguous()
        )
        cos, sin = compute_rope_for_positions(
            positions=missed_idx_tensor,
            head_dim=d,
            device=missed_idx_tensor.device,
            dtype=dtype,
            theta=self.theta,
        )
        recon_keys = apply_rope(recon_keys_pre, cos, sin)
        del a_slice, recon_flat, recon_keys_pre, cos, sin
        return recon_keys

    def resolve_sparse_kv(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute landmark attention, select top-k chunks, and assemble working KV set."""
        if not self.is_offloaded or self.landmarks is None or self.key_a is None or self.key_b is None:
            if self._cpu_k_buf is not None and self._cpu_v_buf is not None and self.total_tokens_seen > 0:
                return (
                    self._cpu_k_buf[:, :, : self.total_tokens_seen, :].to(device=query.device),
                    self._cpu_v_buf[:, :, : self.total_tokens_seen, :].to(device=query.device),
                )
            if self.dense_keys is not None and self.dense_values is not None:
                return self.dense_keys, self.dense_values
            assert self.gpu_anchor_keys is not None and self.gpu_anchor_values is not None
            return self.gpu_anchor_keys, self.gpu_anchor_values

        b, h_q, q_len, d = query.shape
        h_kv = self.landmarks.shape[1]
        gqa_group_size = h_q // h_kv

        # 1. Approximate attention against landmarks: P = Q @ L.T / sqrt(D)
        landmarks_t = self.landmarks.transpose(-1, -2)  # [B, H_kv, D, num_chunks]
        if gqa_group_size > 1:
            q_grouped = query.view(b, h_kv, gqa_group_size, q_len, d)
            attn_scores = torch.matmul(q_grouped, landmarks_t.unsqueeze(2)) / math.sqrt(d)
            chunk_scores = attn_scores.amax(dim=(2, 3))  # [B, H_kv, num_chunks]
        else:
            attn_scores = torch.matmul(query, landmarks_t) / math.sqrt(d)
            chunk_scores = attn_scores.amax(dim=2)  # [B, H_kv, num_chunks]

        # Center per KV head before cross-head max pooling so high-baseline heads don't starve others
        chunk_scores = chunk_scores - chunk_scores.amax(dim=-1, keepdim=True)
        pooled_scores = chunk_scores.amax(dim=(0, 1))  # [num_chunks]

        # 2. Global top-k chunk selection across heads (masking already-resident anchor chunks)
        if (
            self._candidate_tokens is None
            or self._ineligible_chunk_mask is None
            or self._ineligible_chunk_mask.device != query.device
        ):
            candidate_tokens, ineligible_mask = self._build_candidate_index(query.device)
        else:
            candidate_tokens = self._candidate_tokens
            ineligible_mask = self._ineligible_chunk_mask

        masked_scores = pooled_scores.masked_fill(ineligible_mask, float("-inf"))

        k_val = min(self.topk_chunks, len(candidate_tokens))
        chunk_tokens: dict[int, list[int]] = {}
        selected_chunks: list[int] = []
        if k_val > 0:
            topk_indices = torch.topk(masked_scores, k=k_val, dim=-1).indices.tolist()
            for c_idx in topk_indices:
                if c_idx in candidate_tokens:
                    chunk_tokens[c_idx] = candidate_tokens[c_idx]
                    selected_chunks.append(c_idx)

        # 3. Inter-step temporal cache diff & LRU recency update
        hits: list[int] = []
        misses: list[int] = []
        for c_idx in selected_chunks:
            if c_idx in self._lru_chunks:
                hits.append(c_idx)
                self._lru_chunks.move_to_end(c_idx)
            else:
                misses.append(c_idx)

        self.temporal_hits += len(hits)
        self.temporal_misses += len(misses)

        # 4. On miss: fetch missed value chunks, reconstruct missed keys, evict LRU, rebuild sparse cache
        if misses and self.cpu_values is not None:
            missed_token_indices: list[int] = []
            for c_idx in misses:
                missed_token_indices.extend(chunk_tokens[c_idx])

            missed_idx_tensor = torch.tensor(
                missed_token_indices,
                device=query.device,
                dtype=torch.long,
            )

            # Fetch Values from Host CPU DRAM over PCIe
            cpu_indices = missed_idx_tensor.cpu()
            val_slice = torch.index_select(self.cpu_values, dim=2, index=cpu_indices)
            self.total_bytes_transferred_pcie += val_slice.nelement() * val_slice.element_size()

            copy_stream = self._get_copy_stream()
            if copy_stream is not None and query.is_cuda:
                curr_stream = torch.cuda.current_stream()
                copy_stream.wait_stream(curr_stream)
                with torch.cuda.stream(copy_stream):
                    fetched_values = val_slice.to(device=query.device, non_blocking=True)
                fetched_values.record_stream(curr_stream)
                recon_keys = self._reconstruct_keys(missed_idx_tensor, b, h_kv, d, query.dtype)
                curr_stream.wait_stream(copy_stream)
            else:
                fetched_values = val_slice.to(device=query.device)
                recon_keys = self._reconstruct_keys(missed_idx_tensor, b, h_kv, d, query.dtype)

            # Insert missed chunks and evict least-recently-used chunks beyond topk_chunks
            for c_idx in misses:
                self._lru_chunks[c_idx] = len(chunk_tokens[c_idx])
            while len(self._lru_chunks) > self.topk_chunks:
                self._lru_chunks.popitem(last=False)

            # Rebuild cached_sparse_{k,v,pos} from surviving chunks + newly fetched chunks (zero duplication)
            surviving_k: list[torch.Tensor] = []
            surviving_v: list[torch.Tensor] = []
            surviving_pos: list[torch.Tensor] = []
            new_layout: list[tuple[int, int]] = []

            if (
                self.cached_sparse_keys is not None
                and self.cached_sparse_values is not None
                and self.cached_sparse_pos is not None
            ):
                offset = 0
                for c_idx, tok_len in self._cached_chunk_layout:
                    end_off = offset + tok_len
                    if c_idx in self._lru_chunks:
                        surviving_k.append(self.cached_sparse_keys[:, :, offset:end_off, :])
                        surviving_v.append(self.cached_sparse_values[:, :, offset:end_off, :])
                        surviving_pos.append(self.cached_sparse_pos[offset:end_off])
                        new_layout.append((c_idx, tok_len))
                    offset = end_off

            surviving_k.append(recon_keys)
            surviving_v.append(fetched_values)
            surviving_pos.append(missed_idx_tensor)
            for c_idx in misses:
                new_layout.append((c_idx, len(chunk_tokens[c_idx])))

            if len(surviving_k) == 1:
                self.cached_sparse_keys = surviving_k[0]
                self.cached_sparse_values = surviving_v[0]
                self.cached_sparse_pos = surviving_pos[0]
            else:
                self.cached_sparse_keys = torch.cat(surviving_k, dim=2)
                self.cached_sparse_values = torch.cat(surviving_v, dim=2)
                self.cached_sparse_pos = torch.cat(surviving_pos, dim=0)

            self._cached_chunk_layout = new_layout
            self.cached_topk_indices = set(self._lru_chunks.keys())

        # 5. Assemble Working KV Set: Anchors (Sink + Outliers + Local) + Cached Sparse Chunks
        assert self.gpu_anchor_keys is not None and self.gpu_anchor_values is not None
        if self.cached_sparse_keys is not None and self.cached_sparse_values is not None:
            working_k = torch.cat([self.gpu_anchor_keys, self.cached_sparse_keys], dim=2)
            working_v = torch.cat([self.gpu_anchor_values, self.cached_sparse_values], dim=2)
        else:
            working_k = self.gpu_anchor_keys
            working_v = self.gpu_anchor_values

        return working_k, working_v

    def get_seq_length(self) -> int:
        return self.total_tokens_seen

    def get_max_length(self) -> int:
        return max(self.total_tokens_seen, 32768)

    def reset(self) -> None:
        self.is_offloaded = False
        self.total_tokens_seen = 0
        self.prefill_seq_len = 0
        self.head_dim = 0
        self._device = None
        self._dtype = None
        if not self._theta_explicit:
            self.theta = 10000.0
            self._theta_synced = False
        self.dense_keys = None
        self.dense_values = None
        self._cpu_k_buf = None
        self._cpu_v_buf = None
        self.key_a = None
        self.key_b = None
        self.landmarks = None
        self.num_chunks = 0
        self.outlier_chunk_indices = []
        self._anchor_token_set.clear()
        self._candidate_tokens = None
        self._ineligible_chunk_mask = None
        self.gpu_anchor_keys = None
        self.gpu_anchor_values = None
        self.gpu_anchor_pos = None
        self.cpu_values = None
        self._lru_chunks.clear()
        self._cached_chunk_layout.clear()
        self.cached_topk_indices = set()
        self.cached_sparse_keys = None
        self.cached_sparse_values = None
        self.cached_sparse_pos = None
        self.total_bytes_transferred_pcie = 0
        self.temporal_hits = 0
        self.temporal_misses = 0
        super().reset()


class ShadowKVCache(Cache):
    """Cache implementation for ShadowKV (GPU Low-Rank Keys + Host CPU Values)."""

    is_shadowkv = True

    def __init__(
        self,
        svd_rank: int = 160,
        chunk_size: int = 8,
        topk_chunks: int = 256,
        outlier_ratio: float = 0.003,
        temporal_reuse_target: float = 0.60,
        local_window_tokens: int = 32,
        svd_threshold_tokens: int = 1024,
        theta: float | None = None,
    ):
        self.svd_rank = int(svd_rank)
        self.chunk_size = int(chunk_size)
        self.topk_chunks = int(topk_chunks)
        self.outlier_ratio = float(outlier_ratio)
        self.temporal_reuse_target = float(temporal_reuse_target)
        self.local_window_tokens = int(local_window_tokens)
        self.svd_threshold_tokens = int(svd_threshold_tokens)
        self.theta: float | None = _validate_theta(theta) if theta is not None else None
        self._copy_stream: torch.cuda.Stream | None = None

        cache_ref = weakref.ref(self)
        super().__init__(
            layer_class_to_replicate=lambda: ShadowKVLayer(
                svd_rank=self.svd_rank,
                chunk_size=self.chunk_size,
                topk_chunks=self.topk_chunks,
                outlier_ratio=self.outlier_ratio,
                temporal_reuse_target=self.temporal_reuse_target,
                local_window_tokens=self.local_window_tokens,
                svd_threshold_tokens=self.svd_threshold_tokens,
                theta=self.theta,
                _cache_ref=cache_ref,
            )
        )
        if not self.layers:
            self.layers.append(self.layer_class_to_replicate())
        set_active_shadowkv_cache(self)

    def get_copy_stream(self) -> torch.cuda.Stream | None:
        """Lazily create and return a single shared CUDA copy stream for all layers."""
        if self._copy_stream is None and torch is not None and torch.cuda.is_available():
            try:
                self._copy_stream = torch.cuda.Stream()
            except Exception:
                self._copy_stream = None
        return self._copy_stream

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_max_length(self) -> int:
        return 32768

    @property
    def total_bytes_transferred_pcie(self) -> int:
        return sum(layer.total_bytes_transferred_pcie for layer in self.layers)

    @property
    def temporal_hits(self) -> int:
        return sum(layer.temporal_hits for layer in self.layers)

    @property
    def temporal_misses(self) -> int:
        return sum(layer.temporal_misses for layer in self.layers)

    @property
    def temporal_hit_rate(self) -> float:
        total_hits = self.temporal_hits
        total_misses = self.temporal_misses
        denom = total_hits + total_misses
        return float(total_hits / denom) if denom > 0 else self.temporal_reuse_target
