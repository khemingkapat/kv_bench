from __future__ import annotations

import weakref
from collections.abc import Iterator
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from transformers.cache_utils import Cache, DynamicLayer
from src.techniques.headq import (
    compute_key_sidecode,
    compute_per_head_query_pca_basis,
    dequantize_key_2bit,
    dequantize_key_sidecode,
    dequantize_value_4bit,
    quantize_key_2bit,
    quantize_value_4bit,
)

_ACTIVE_HEADQ_CACHE_REF: weakref.ReferenceType[HeadQCache] | None = None


def get_active_headq_cache() -> HeadQCache | None:
    """Return currently active HeadQCache instance if alive, else None."""
    global _ACTIVE_HEADQ_CACHE_REF
    if _ACTIVE_HEADQ_CACHE_REF is not None:
        return _ACTIVE_HEADQ_CACHE_REF()
    return None


def set_active_headq_cache(cache: HeadQCache | None) -> None:
    """Register weak reference to active HeadQCache ensuring zero tensor retention leaks."""
    global _ACTIVE_HEADQ_CACHE_REF
    if cache is not None:
        _ACTIVE_HEADQ_CACHE_REF = weakref.ref(cache)
    else:
        _ACTIVE_HEADQ_CACHE_REF = None


class HeadQLayer(DynamicLayer):
    """Cache layer implementing HeadQ 2-bit base Key + rank-r residual side-code + 4-bit Value storage."""

    is_sliding = False

    def __init__(self, rank: int = 4, sidecode_bits: int = 4):
        self._keys_fallback: torch.Tensor | None = None
        self._values_fallback: torch.Tensor | None = None

        self.rank = int(rank)
        self.sidecode_bits = int(sidecode_bits)
        self.capacity: int = 0
        self.total_tokens_seen: int = 0
        self.num_kv: int = 0
        self.head_dim: int = 0
        self.target_dtype: Any = None
        self.target_device: Any = None

        self.basis: torch.Tensor | None = None
        self.k_mean: torch.Tensor | None = None
        self.k_std: torch.Tensor | None = None
        self._pending_chunk0_residual: torch.Tensor | None = None
        self.k_packed_buf: torch.Tensor | None = None
        self.k_scales_buf: torch.Tensor | None = None
        self.z_packed_buf: torch.Tensor | None = None
        self.z_scales_buf: torch.Tensor | None = None
        self.v_packed_buf: torch.Tensor | None = None
        self.v_scales_buf: torch.Tensor | None = None

        super().__init__()

        if self.rank <= 0 or (self.rank % 2 != 0):
            raise ValueError(f"rank must be a positive even integer for sidecode packing, got {rank}")

    @property
    def original_positions(self) -> list[int]:
        return list(range(self.total_tokens_seen))

    @property
    def keys(self) -> torch.Tensor | None:
        if not self.is_initialized or self.total_tokens_seen == 0:
            return self._keys_fallback
        assert self.k_packed_buf is not None and self.k_scales_buf is not None
        t = self.total_tokens_seen
        k_base = dequantize_key_2bit(
            self.k_packed_buf[:, :, :t, :],
            self.k_scales_buf[:, :, :t, :],
            original_dim=self.head_dim,
            dtype=self.target_dtype,
        )
        if self.k_std is not None and self.k_mean is not None:
            k_base.mul_(self.k_std).add_(self.k_mean)

        if self.basis is None:
            return k_base

        assert self.z_packed_buf is not None and self.z_scales_buf is not None
        z_recon = dequantize_key_sidecode(
            self.z_packed_buf[:, :, :t, :],
            self.z_scales_buf[:, :, :t, :],
            rank=self.rank,
            dtype=self.target_dtype,
        )
        k_corr = torch.matmul(z_recon, self.basis.transpose(-2, -1))
        return k_base.add_(k_corr)

    @keys.setter
    def keys(self, value: torch.Tensor | None) -> None:
        self._keys_fallback = value

    @property
    def values(self) -> torch.Tensor | None:
        if not self.is_initialized or self.total_tokens_seen == 0:
            return self._values_fallback
        assert self.v_packed_buf is not None and self.v_scales_buf is not None
        t = self.total_tokens_seen
        return dequantize_value_4bit(
            self.v_packed_buf[:, :, :t, :],
            self.v_scales_buf[:, :, :t, :],
            original_dim=self.head_dim,
            dtype=self.target_dtype,
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

    def _ensure_capacity(
        self,
        batch_size: int,
        num_kv: int,
        head_dim: int,
        target_device: Any,
        target_dtype: Any,
        required_capacity: int,
    ) -> None:
        if self.k_packed_buf is not None:
            existing_batch, existing_kv = self.k_packed_buf.shape[0], self.k_packed_buf.shape[1]
            if (
                batch_size != existing_batch
                or num_kv != existing_kv
                or head_dim != self.head_dim
                or target_device != self.target_device
                or target_dtype != self.target_dtype
            ):
                raise ValueError(
                    "HeadQLayer shape/device/dtype mismatch without reset(): "
                    f"expected ({existing_batch}, {existing_kv}, {self.head_dim}, "
                    f"{self.target_device}, {self.target_dtype}), "
                    f"got ({batch_size}, {num_kv}, {head_dim}, {target_device}, {target_dtype})"
                )
        else:
            self.num_kv = num_kv
            self.head_dim = head_dim
            self.target_device = target_device
            self.target_dtype = target_dtype

        if self.capacity >= required_capacity:
            return

        if self.capacity >= 16384 or required_capacity > 16384:
            new_capacity = max(required_capacity, self.capacity + 2048)
        else:
            new_capacity = max(required_capacity, self.capacity * 2, 1024)

        new_kp = torch.empty((batch_size, num_kv, new_capacity, head_dim // 4), dtype=torch.uint8, device=target_device)
        new_ks = torch.empty((batch_size, num_kv, new_capacity, 1), dtype=target_dtype, device=target_device)
        new_zp = torch.empty((batch_size, num_kv, new_capacity, self.rank // 2), dtype=torch.uint8, device=target_device)
        new_zs = torch.empty((batch_size, num_kv, new_capacity, 1), dtype=target_dtype, device=target_device)
        new_vp = torch.empty((batch_size, num_kv, new_capacity, head_dim // 2), dtype=torch.uint8, device=target_device)
        new_vs = torch.empty((batch_size, num_kv, new_capacity, 1), dtype=target_dtype, device=target_device)

        if self.total_tokens_seen > 0 and self.k_packed_buf is not None:
            t = self.total_tokens_seen
            new_kp[:, :, :t].copy_(self.k_packed_buf[:, :, :t])
            new_ks[:, :, :t].copy_(self.k_scales_buf[:, :, :t])
            new_zp[:, :, :t].copy_(self.z_packed_buf[:, :, :t])
            new_zs[:, :, :t].copy_(self.z_scales_buf[:, :, :t])
            new_vp[:, :, :t].copy_(self.v_packed_buf[:, :, :t])
            new_vs[:, :, :t].copy_(self.v_scales_buf[:, :, :t])

        self.k_packed_buf = new_kp
        self.k_scales_buf = new_ks
        self.z_packed_buf = new_zp
        self.z_scales_buf = new_zs
        self.v_packed_buf = new_vp
        self.v_scales_buf = new_vs
        self.capacity = new_capacity

    def finalize_chunk0_from_queries(self, query_states: torch.Tensor) -> None:
        """Fit per-KV-head PCA basis from post-RoPE queries and materialize Chunk-0 sidecodes."""
        if self._pending_chunk0_residual is None:
            return
        num_kv = self._pending_chunk0_residual.shape[1]
        t0 = self._pending_chunk0_residual.shape[-2]
        self.basis = compute_per_head_query_pca_basis(
            query_states, num_kv_heads=num_kv, rank=self.rank
        ).to(self.target_dtype)
        z0_packed, z0_scales, _ = compute_key_sidecode(
            self._pending_chunk0_residual, self.basis, sidecode_bits=self.sidecode_bits
        )
        assert self.z_packed_buf is not None and self.z_scales_buf is not None
        self.z_packed_buf[:, :, :t0].copy_(z0_packed)
        self.z_scales_buf[:, :, :t0].copy_(z0_scales.to(self.target_dtype))
        self._pending_chunk0_residual = None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        batch_size = key_states.shape[0]
        num_kv = key_states.shape[1]
        new_tokens = key_states.shape[-2]
        head_dim = key_states.shape[-1]

        curr = self.total_tokens_seen
        self._ensure_capacity(
            batch_size, num_kv, head_dim, key_states.device, key_states.dtype, curr + new_tokens
        )

        if self.k_mean is None or self.k_std is None:
            k_f = key_states.float()
            mean_f = k_f.mean(dim=(0, 2), keepdim=True)
            if batch_size * new_tokens == 1:
                std_f = torch.ones_like(mean_f)
            else:
                std_f = k_f.std(dim=(0, 2), keepdim=True, unbiased=False).clamp(min=1e-3)
            self.k_mean = mean_f.to(self.target_dtype)
            self.k_std = std_f.to(self.target_dtype)

        k_norm = (key_states - self.k_mean) / self.k_std
        k_packed, k_scales, k_norm_dequant = quantize_key_2bit(k_norm)
        k_base = k_norm_dequant * self.k_std + self.k_mean
        k_residual = key_states - k_base

        v_packed, v_scales, _ = quantize_value_4bit(value_states)

        self.k_packed_buf[:, :, curr : curr + new_tokens].copy_(k_packed)
        self.k_scales_buf[:, :, curr : curr + new_tokens].copy_(k_scales.to(key_states.dtype))
        self.v_packed_buf[:, :, curr : curr + new_tokens].copy_(v_packed)
        self.v_scales_buf[:, :, curr : curr + new_tokens].copy_(v_scales.to(value_states.dtype))

        if curr == 0 and self.basis is None:
            self._pending_chunk0_residual = k_residual
        else:
            if self.basis is None and self._pending_chunk0_residual is not None:
                # Standalone fallback when no SDPA query hook fired on Chunk 0.
                self.finalize_chunk0_from_queries(self._pending_chunk0_residual)
            z_packed, z_scales, _ = compute_key_sidecode(
                k_residual, self.basis, sidecode_bits=self.sidecode_bits
            )
            self.z_packed_buf[:, :, curr : curr + new_tokens].copy_(z_packed)
            self.z_scales_buf[:, :, curr : curr + new_tokens].copy_(z_scales.to(key_states.dtype))

        self.total_tokens_seen += new_tokens

        recon_k = self.keys
        recon_v = self.values
        assert recon_k is not None and recon_v is not None
        return recon_k, recon_v

    @property
    def quantized_bytes(self) -> int:
        if self.total_tokens_seen == 0 or self.k_packed_buf is None:
            return 0
        t = self.total_tokens_seen
        tensors = [
            self.k_packed_buf[:, :, :t],
            self.k_scales_buf[:, :, :t],
            self.z_packed_buf[:, :, :t],
            self.z_scales_buf[:, :, :t],
            self.v_packed_buf[:, :, :t],
            self.v_scales_buf[:, :, :t],
        ]
        extra_bytes = sum(
            x.nelement() * x.element_size()
            for x in (self.basis, self.k_mean, self.k_std)
            if x is not None
        )
        return extra_bytes + sum(x.nelement() * x.element_size() for x in tensors)

    def get_seq_length(self) -> int:
        return self.total_tokens_seen

    def get_max_length(self) -> int:
        return max(self.total_tokens_seen, 32768)

    def reset(self) -> None:
        self.total_tokens_seen = 0
        self.num_kv = 0
        self.head_dim = 0
        self.target_dtype = None
        self.target_device = None
        self.basis = None
        self.k_mean = None
        self.k_std = None
        self._pending_chunk0_residual = None
        self.k_packed_buf = None
        self.k_scales_buf = None
        self.z_packed_buf = None
        self.z_scales_buf = None
        self.v_packed_buf = None
        self.v_scales_buf = None
        self.capacity = 0
        self._keys_fallback = None
        self._values_fallback = None
        super().reset()


class HeadQCache(Cache):
    """Subclass of transformers.cache_utils.Cache implementing HeadQ 2-bit Key + PCA side-code storage."""

    is_headq = True

    def __init__(self, rank: int = 4, sidecode_bits: int = 4):
        self.rank = int(rank)
        self.sidecode_bits = int(sidecode_bits)
        super().__init__(
            layer_class_to_replicate=lambda: HeadQLayer(
                rank=self.rank,
                sidecode_bits=self.sidecode_bits,
            )
        )
        set_active_headq_cache(self)

    @property
    def original_positions(self) -> list[int]:
        if not self.layers:
            return []
        return self.layers[0].original_positions

    @property
    def total_quantized_bytes(self) -> int:
        return sum(getattr(layer, "quantized_bytes", 0) for layer in self.layers)

    @property
    def effective_bits_per_scalar(self) -> float:
        """Physical bits per KV scalar (2-bit K + rank-4 sidecode + 4-bit V + FP16 scales)."""
        return 3.25

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
