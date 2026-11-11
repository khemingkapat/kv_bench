from __future__ import annotations

import math
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


def perform_svd_key_decomposition(
    keys_pre_rope: torch.Tensor,
    svd_rank: int = 160,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute truncated SVD on pre-RoPE keys."""
    b, h_kv, s, d = keys_pre_rope.shape
    r = min(svd_rank, s, d)

    flat_k = keys_pre_rope.reshape(b * h_kv, s, d).float()
    u, s_vals, vh = torch.linalg.svd(flat_k, full_matrices=False)

    u_r = u[:, :, :r]  # [B*H, S, r]
    s_r = s_vals[:, :r]  # [B*H, r]
    vh_r = vh[:, :r, :]  # [B*H, r, D]

    sqrt_s = torch.sqrt(s_r).unsqueeze(1)  # [B*H, 1, r]
    a_proj = (u_r * sqrt_s).reshape(b, h_kv, s, r).to(keys_pre_rope.dtype)
    b_basis = (sqrt_s.transpose(1, 2) * vh_r).reshape(b, h_kv, r, d).to(keys_pre_rope.dtype)

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
        theta: float = 10000.0,
    ):
        super().__init__()
        self.svd_rank = int(svd_rank)
        self.chunk_size = int(chunk_size)
        self.topk_chunks = int(topk_chunks)
        self.outlier_ratio = float(outlier_ratio)
        self.temporal_reuse_target = float(temporal_reuse_target)
        self.local_window_tokens = int(local_window_tokens)
        self.svd_threshold_tokens = int(svd_threshold_tokens)
        self.theta = float(theta)

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

        # Dense prefill buffers
        self.dense_keys: torch.Tensor | None = None
        self.dense_values: torch.Tensor | None = None

        # ShadowKV prefill data structures
        self.key_a: torch.Tensor | None = None  # [B, H_kv, S, r]
        self.key_b: torch.Tensor | None = None  # [B, H_kv, r, D]
        self.landmarks: torch.Tensor | None = None  # [B, H_kv, num_chunks, D]
        self.num_chunks: int = 0
        self.outlier_chunk_indices: list[int] = []

        # GPU working cache: Outliers + Local window
        self.gpu_anchor_keys: torch.Tensor | None = None
        self.gpu_anchor_values: torch.Tensor | None = None
        self.gpu_anchor_pos: torch.Tensor | None = None

        # Host CPU DRAM pinned buffer for values
        self.cpu_values: torch.Tensor | None = None  # [B, H_kv, S, D] in pinned memory

        # Temporal cache for autoregressive decoding
        self.cached_topk_indices: set[int] = set()
        self.cached_sparse_keys: torch.Tensor | None = None
        self.cached_sparse_values: torch.Tensor | None = None
        self.cached_sparse_pos: torch.Tensor | None = None

        # Telemetry
        self.total_bytes_transferred_pcie: int = 0
        self.temporal_hits: int = 0
        self.temporal_misses: int = 0

        # CUDA copy stream for overlapped DMA
        self.copy_stream: torch.cuda.Stream | None = None
        if torch is not None and torch.cuda.is_available():
            try:
                self.copy_stream = torch.cuda.Stream()
            except Exception:
                self.copy_stream = None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update layer cache with incoming key and value states."""
        b, h_kv, seq_len, d = key_states.shape
        self.head_dim = d

        # Phase 1: Prefill (seq_len > 1)
        if seq_len > 1:
            if self.dense_keys is None:
                self.dense_keys = key_states
                self.dense_values = value_states
            else:
                self.dense_keys = torch.cat([self.dense_keys, key_states], dim=2)
                self.dense_values = torch.cat([self.dense_values, value_states], dim=2)
            self.total_tokens_seen = self.dense_keys.shape[2]
            return self.dense_keys, self.dense_values

        # Phase 2: Autoregressive Decode (seq_len == 1)
        # If prefill exceeded threshold and prune was not yet called, trigger prune on step 0
        if not self.is_offloaded and self.total_tokens_seen > self.svd_threshold_tokens:
            self.prune()

        self.total_tokens_seen += 1
        curr_pos = self.total_tokens_seen - 1

        if not self.is_offloaded:
            # Dense fallback path (S <= svd_threshold_tokens)
            if self.dense_keys is None:
                self.dense_keys = key_states
                self.dense_values = value_states
            else:
                self.dense_keys = torch.cat([self.dense_keys, key_states], dim=2)
                self.dense_values = torch.cat([self.dense_values, value_states], dim=2)
            return self.dense_keys, self.dense_values

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
        if self.is_offloaded or self.dense_keys is None or self.dense_values is None:
            return

        seq_len = self.dense_keys.shape[2]
        if seq_len <= self.svd_threshold_tokens:
            return

        self.is_offloaded = True
        self.prefill_seq_len = seq_len
        key_states = self.dense_keys
        value_states = self.dense_values
        b, h_kv, s, d = key_states.shape

        # 1. Recover pre-RoPE keys via exact inverse RoPE
        cos, sin = compute_rope_cos_sin(
            seq_len=s,
            head_dim=d,
            device=key_states.device,
            dtype=key_states.dtype,
            theta=self.theta,
            start_pos=0,
        )
        keys_pre = invert_rope(key_states, cos, sin)

        # 2. Decompose pre-RoPE keys into low-rank factors
        self.key_a, self.key_b = perform_svd_key_decomposition(
            keys_pre_rope=keys_pre,
            svd_rank=self.svd_rank,
        )

        # 3. Compute chunk landmarks and detect outliers
        self.landmarks, self.num_chunks, padded_s = compute_chunk_landmarks(
            keys=key_states,
            chunk_size=self.chunk_size,
        )
        self.outlier_chunk_indices = detect_outlier_chunks(
            keys=key_states,
            landmarks=self.landmarks,
            chunk_size=self.chunk_size,
            outlier_ratio=self.outlier_ratio,
        )

        # 4. Offload full values to pinned Host CPU DRAM
        if key_states.is_cuda:
            try:
                self.cpu_values = torch.empty(
                    (b, h_kv, s, d),
                    dtype=value_states.dtype,
                    device="cpu",
                    pin_memory=True,
                )
            except Exception:
                self.cpu_values = torch.empty(
                    (b, h_kv, s, d),
                    dtype=value_states.dtype,
                    device="cpu",
                )
            self.cpu_values.copy_(value_states, non_blocking=True)
        else:
            self.cpu_values = value_states.detach().cpu()

        # 5. Extract Anchor tokens (Outliers + Local Window) to reside on GPU
        outlier_token_indices: list[int] = []
        for c_idx in self.outlier_chunk_indices:
            start = c_idx * self.chunk_size
            end = min(s, start + self.chunk_size)
            outlier_token_indices.extend(range(start, end))

        local_start = max(0, s - self.local_window_tokens)
        local_token_indices = list(range(local_start, s))

        anchor_indices = sorted(set(outlier_token_indices + local_token_indices))
        anchor_idx_tensor = torch.tensor(anchor_indices, device=key_states.device, dtype=torch.long)

        self.gpu_anchor_keys = torch.index_select(key_states, dim=2, index=anchor_idx_tensor)
        self.gpu_anchor_values = torch.index_select(value_states, dim=2, index=anchor_idx_tensor)
        self.gpu_anchor_pos = anchor_idx_tensor

        # Release dense uncompressed GPU buffers to free VRAM
        self.dense_keys = None
        self.dense_values = None

    def resolve_sparse_kv(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute landmark attention, select top-k chunks, and assemble working KV set."""
        if not self.is_offloaded or self.landmarks is None or self.key_a is None or self.key_b is None:
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
            # Max-pool across grouped query heads: [B, H_kv, num_chunks]
            chunk_scores = attn_scores.amax(dim=(2, 3))
        else:
            attn_scores = torch.matmul(query, landmarks_t) / math.sqrt(d)
            chunk_scores = attn_scores.squeeze(2)

        # 2. Top-k chunk selection per KV head
        k_val = min(self.topk_chunks, self.num_chunks)
        topk_indices = torch.topk(chunk_scores, k=k_val, dim=-1).indices  # [B, H_kv, k]

        # Extract unique chunk indices across heads for memory gather
        selected_chunks = set(topk_indices.flatten().tolist())
        selected_chunks.difference_update(self.outlier_chunk_indices)

        # 3. Inter-step temporal cache diff
        missed_chunks = selected_chunks.difference(self.cached_topk_indices)
        hit_count = len(selected_chunks) - len(missed_chunks)
        self.temporal_hits += hit_count
        self.temporal_misses += len(missed_chunks)

        # 4. Fetch missed value chunks and reconstruct missed key chunks
        if missed_chunks and self.cpu_values is not None:
            missed_token_indices: list[int] = []
            for c_idx in sorted(missed_chunks):
                start = c_idx * self.chunk_size
                end = min(self.prefill_seq_len, start + self.chunk_size)
                if start < self.prefill_seq_len:
                    missed_token_indices.extend(range(start, end))

            if missed_token_indices:
                missed_idx_tensor = torch.tensor(
                    missed_token_indices,
                    device=query.device,
                    dtype=torch.long,
                )

                # Fetch Values from Host CPU DRAM over PCIe
                cpu_indices = missed_idx_tensor.cpu()
                val_slice = torch.index_select(self.cpu_values, dim=2, index=cpu_indices)
                bytes_transferred = val_slice.nelement() * val_slice.element_size()
                self.total_bytes_transferred_pcie += bytes_transferred

                # Asynchronous DMA if copy_stream is available
                if self.copy_stream is not None and query.is_cuda:
                    curr_stream = torch.cuda.current_stream()
                    self.copy_stream.wait_stream(curr_stream)
                    with torch.cuda.stream(self.copy_stream):
                        fetched_values = val_slice.to(device=query.device, non_blocking=True)

                    # Concurrently on compute stream: Reconstruct Keys via A @ B and position-exact RoPE
                    a_slice = torch.index_select(self.key_a, dim=2, index=missed_idx_tensor)
                    recon_keys_pre = torch.matmul(a_slice, self.key_b)  # [B, H_kv, S_miss, D]
                    cos, sin = compute_rope_for_positions(
                        positions=missed_idx_tensor,
                        head_dim=d,
                        device=query.device,
                        dtype=query.dtype,
                        theta=self.theta,
                    )
                    recon_keys = apply_rope(recon_keys_pre, cos, sin)

                    # Synchronize compute stream before attention kernel
                    curr_stream.wait_stream(self.copy_stream)
                else:
                    fetched_values = val_slice.to(device=query.device)
                    a_slice = torch.index_select(self.key_a, dim=2, index=missed_idx_tensor)
                    recon_keys_pre = torch.matmul(a_slice, self.key_b)
                    cos, sin = compute_rope_for_positions(
                        positions=missed_idx_tensor,
                        head_dim=d,
                        device=query.device,
                        dtype=query.dtype,
                        theta=self.theta,
                    )
                    recon_keys = apply_rope(recon_keys_pre, cos, sin)

                # Update cached sparse buffer
                if self.cached_sparse_keys is None:
                    self.cached_sparse_keys = recon_keys
                    self.cached_sparse_values = fetched_values
                    self.cached_sparse_pos = missed_idx_tensor
                else:
                    self.cached_sparse_keys = torch.cat([self.cached_sparse_keys, recon_keys], dim=2)
                    self.cached_sparse_values = torch.cat([self.cached_sparse_values, fetched_values], dim=2)
                    self.cached_sparse_pos = torch.cat([self.cached_sparse_pos, missed_idx_tensor], dim=0)

                self.cached_topk_indices.update(missed_chunks)

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
        theta: float = 10000.0,
    ):
        self.svd_rank = int(svd_rank)
        self.chunk_size = int(chunk_size)
        self.topk_chunks = int(topk_chunks)
        self.outlier_ratio = float(outlier_ratio)
        self.temporal_reuse_target = float(temporal_reuse_target)
        self.local_window_tokens = int(local_window_tokens)
        self.svd_threshold_tokens = int(svd_threshold_tokens)
        self.theta = float(theta)

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
            )
        )
        set_active_shadowkv_cache(self)

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
    def temporal_hit_rate(self) -> float:
        total_hits = sum(layer.temporal_hits for layer in self.layers)
        total_misses = sum(layer.temporal_misses for layer in self.layers)
        denom = total_hits + total_misses
        return float(total_hits / denom) if denom > 0 else self.temporal_reuse_target
