from __future__ import annotations

from collections.abc import Iterator
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from transformers.cache_utils import Cache, DynamicLayer
from src.techniques.turboquant_ops import (
    compute_shannon_phi_metrics,
    generate_rademacher_signs,
    get_codebook_tensors,
    turboquant_decode_key,
    turboquant_decode_value,
    turboquant_encode_key,
    turboquant_encode_value,
)


INITIAL_CAPACITY = 1024
LINEAR_GROWTH_THRESHOLD = 16384
LINEAR_GROWTH_STEP = 2048


def _next_capacity(current: int, required: int) -> int:
    """Compute next pre-allocated capacity: double up to 16384, then +2048 linearly."""
    if required <= current:
        return current
    cap = INITIAL_CAPACITY if current <= 0 else current
    while cap < required:
        if cap < LINEAR_GROWTH_THRESHOLD:
            cap *= 2
        else:
            cap += LINEAR_GROWTH_STEP
    return cap


class TurboQuantLayer(DynamicLayer):
    """Cache layer storing asymmetrically quantized Key-Value representations.

    Keys: 4-bit scalar Lloyd-Max + FP32 norm (68 bytes / token / head).
    Values: 3-bit scalar Lloyd-Max + 1-bit QJL residual sketch + FP32 norms (72 bytes / token / head).
    Total: 140 bytes / token / head (3.657x compression vs BF16).
    """

    is_croppable = False

    def __init__(
        self,
        seed: int = 42,
    ):
        super().__init__()
        self.seed = seed
        self.total_tokens_seen: int = 0
        self.capacity: int = 0
        self.dtype: torch.dtype | None = None
        self.device: torch.device | None = None

        # Compressed storage tensors
        self.key_packed: torch.Tensor | None = None
        self.key_norms: torch.Tensor | None = None
        self.value_packed: torch.Tensor | None = None
        self.value_norms: torch.Tensor | None = None
        self.value_res_norms: torch.Tensor | None = None

        # Sign vectors & codebooks
        self._s: torch.Tensor | None = None
        self._s_prime: torch.Tensor | None = None
        self._c16: torch.Tensor | None = None
        self._t16: torch.Tensor | None = None
        self._c8: torch.Tensor | None = None
        self._t8: torch.Tensor | None = None

        # Transient Shannon-phi tracking accumulators
        self.accum_k_eps_snr: float = 0.0
        self.accum_k_eps_dir: float = 0.0
        self.accum_v_eps_snr: float = 0.0
        self.accum_v_eps_dir: float = 0.0
        self.num_profile_chunks: int = 0

    def lazy_initialization(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor | None = None,
    ) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.is_initialized = True

    def _init_constants(self, device: torch.device, num_kv_heads: int, head_dim: int) -> None:
        if self._c16 is None or self._c16.device != device:
            c16, t16, c8, t8 = get_codebook_tensors(device=device)
            self._c16 = c16
            self._t16 = t16
            self._c8 = c8
            self._t8 = t8

        if (
            self._s is None
            or self._s.device != device
            or self._s.shape[1] != num_kv_heads
            or self._s.shape[-1] != head_dim
        ):
            # Shape (1, num_kv_heads, 1, head_dim) to broadcast seamlessly with (batch, num_kv_heads, seq_len, head_dim)
            self._s = generate_rademacher_signs((1, num_kv_heads, 1, head_dim), device=device, seed=self.seed)
            self._s_prime = generate_rademacher_signs(
                (1, num_kv_heads, 1, head_dim), device=device, seed=self.seed + 10007
            )

    @staticmethod
    def _grow_buffer(
        buf: torch.Tensor | None,
        shape: tuple[int, int, int, int],
        dtype: torch.dtype,
        device: torch.device,
        used_tokens: int,
    ) -> torch.Tensor:
        new_buf = torch.empty(shape, dtype=dtype, device=device)
        if buf is not None and used_tokens > 0:
            new_buf[:, :, :used_tokens, :] = buf[:, :, :used_tokens, :]
        return new_buf

    def _ensure_capacity(
        self,
        batch: int,
        num_kv_heads: int,
        required: int,
        head_dim: int,
        device: torch.device,
    ) -> None:
        if self.key_packed is not None and required <= self.capacity:
            return

        new_cap = _next_capacity(self.capacity, required)
        used = self.total_tokens_seen
        packed_dim = head_dim // 2

        # Grow-copy-replace each buffer individually to avoid 2x peak allocation
        self.key_packed = self._grow_buffer(
            self.key_packed, (batch, num_kv_heads, new_cap, packed_dim), torch.uint8, device, used
        )
        self.key_norms = self._grow_buffer(
            self.key_norms, (batch, num_kv_heads, new_cap, 1), torch.float32, device, used
        )
        self.value_packed = self._grow_buffer(
            self.value_packed, (batch, num_kv_heads, new_cap, packed_dim), torch.uint8, device, used
        )
        self.value_norms = self._grow_buffer(
            self.value_norms, (batch, num_kv_heads, new_cap, 1), torch.float32, device, used
        )
        self.value_res_norms = self._grow_buffer(
            self.value_res_norms, (batch, num_kv_heads, new_cap, 1), torch.float32, device, used
        )
        self.capacity = new_cap

    def _decode_active_keys(self, target_dtype: torch.dtype) -> torch.Tensor:
        assert self.key_packed is not None and self.key_norms is not None
        assert self._s is not None and self._c16 is not None
        seq_len = self.total_tokens_seen
        return turboquant_decode_key(
            self.key_packed[:, :, :seq_len, :],
            self.key_norms[:, :, :seq_len, :],
            self._s,
            self._c16,
            target_dtype=target_dtype,
        )

    def _decode_active_values(self, target_dtype: torch.dtype) -> torch.Tensor:
        assert self.value_packed is not None and self.value_norms is not None and self.value_res_norms is not None
        assert self._s is not None and self._s_prime is not None and self._c8 is not None
        seq_len = self.total_tokens_seen
        return turboquant_decode_value(
            self.value_packed[:, :, :seq_len, :],
            self.value_norms[:, :, :seq_len, :],
            self.value_res_norms[:, :, :seq_len, :],
            self._s,
            self._s_prime,
            self._c8,
            target_dtype=target_dtype,
        )

    @property
    def keys(self) -> torch.Tensor | None:
        if not self.is_initialized or self.key_packed is None or self.total_tokens_seen == 0:
            return None
        return self._decode_active_keys(self.dtype or torch.bfloat16)

    @keys.setter
    def keys(self, value: Any) -> None:
        # DynamicLayer.reset() assigns self.keys = None
        pass

    @property
    def values(self) -> torch.Tensor | None:
        if not self.is_initialized or self.value_packed is None or self.total_tokens_seen == 0:
            return None
        return self._decode_active_values(self.dtype or torch.bfloat16)

    @values.setter
    def values(self, value: Any) -> None:
        # DynamicLayer.reset() assigns self.values = None
        pass

    @property
    def original_positions(self) -> list[int]:
        return list(range(self.total_tokens_seen))

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize and append new tokens, then dequantize accumulated cache for SDPA."""
        if key_states.ndim != 4 or value_states.ndim != 4:
            raise ValueError(
                f"Expected 4D key/value states, got {key_states.ndim}D and {value_states.ndim}D"
            )
        if key_states.shape != value_states.shape:
            raise ValueError(
                f"Key and value shapes must match, got {tuple(key_states.shape)} vs {tuple(value_states.shape)}"
            )
        if key_states.device != value_states.device:
            raise ValueError(
                f"Key and value devices must match, got {key_states.device} vs {value_states.device}"
            )

        batch_size, num_kv_heads, new_tokens, head_dim = key_states.shape
        device = key_states.device
        dtype = key_states.dtype

        if self.key_packed is not None:
            exp_batch, exp_heads, _, exp_packed_dim = self.key_packed.shape
            exp_dim = exp_packed_dim * 2
            if (
                batch_size != exp_batch
                or num_kv_heads != exp_heads
                or head_dim != exp_dim
                or device != self.key_packed.device
            ):
                raise ValueError(
                    "Input shape/device mismatch with initialized TurboQuantLayer: "
                    f"expected (batch={exp_batch}, heads={exp_heads}, *, head_dim={exp_dim}, device={self.key_packed.device}), "
                    f"got (batch={batch_size}, heads={num_kv_heads}, *, head_dim={head_dim}, device={device})"
                )

        if new_tokens == 0:
            if self.key_packed is None or self.total_tokens_seen == 0:
                return torch.empty_like(key_states), torch.empty_like(value_states)
            return self._decode_active_keys(dtype), self._decode_active_values(dtype)

        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        else:
            self.dtype = dtype
            self.device = device

        self._init_constants(device, num_kv_heads, head_dim)
        assert self._s is not None and self._s_prime is not None
        assert self._t16 is not None and self._c16 is not None
        assert self._t8 is not None and self._c8 is not None

        # 1. Encode incoming key and value states
        k_packed_new, k_norm_new = turboquant_encode_key(key_states, self._s, self._t16)
        v_packed_new, v_norm_new, v_res_norm_new = turboquant_encode_value(
            value_states, self._s, self._s_prime, self._t8, self._c8
        )

        # 2. Transient Shannon-phi profiling on prefill (tokens > 1) without memory leaks
        if new_tokens > 1:
            k_rec_new = turboquant_decode_key(k_packed_new, k_norm_new, self._s, self._c16, target_dtype=dtype)
            v_rec_new = turboquant_decode_value(
                v_packed_new, v_norm_new, v_res_norm_new, self._s, self._s_prime, self._c8, target_dtype=dtype
            )
            m_k = compute_shannon_phi_metrics(key_states, k_rec_new)
            m_v = compute_shannon_phi_metrics(value_states, v_rec_new)
            self.accum_k_eps_snr += m_k["eps_snr"]
            self.accum_k_eps_dir += m_k["eps_dir"]
            self.accum_v_eps_snr += m_v["eps_snr"]
            self.accum_v_eps_dir += m_v["eps_dir"]
            self.num_profile_chunks += 1

        # 3. Ensure pre-allocated capacity and write in-place
        curr = self.total_tokens_seen
        end = curr + new_tokens
        self._ensure_capacity(batch_size, num_kv_heads, end, head_dim, device)
        assert self.key_packed is not None and self.key_norms is not None
        assert self.value_packed is not None and self.value_norms is not None and self.value_res_norms is not None

        self.key_packed[:, :, curr:end, :] = k_packed_new
        self.key_norms[:, :, curr:end, :] = k_norm_new
        self.value_packed[:, :, curr:end, :] = v_packed_new
        self.value_norms[:, :, curr:end, :] = v_norm_new
        self.value_res_norms[:, :, curr:end, :] = v_res_norm_new
        self.total_tokens_seen = end

        # 4. Dequantize active slice back to model dtype for Hugging Face SDPA
        return self._decode_active_keys(dtype), self._decode_active_values(dtype)

    def get_seq_length(self) -> int:
        return self.total_tokens_seen

    def get_max_length(self) -> int | None:
        return None

    def get_cache_size_bytes(self) -> tuple[int, int]:
        """Return (physical_stored_bytes, theoretical_compressed_bytes) for active tokens."""
        if not self.is_initialized or self.key_packed is None or self.total_tokens_seen == 0:
            return 0, 0
        seq_len = self.total_tokens_seen
        batch = self.key_packed.shape[0]
        num_heads = self.key_packed.shape[1]
        head_dim = self.key_packed.shape[-1] * 2

        # Physical stored bytes for active slice
        phys = sum(
            t[:, :, :seq_len].numel() * t.element_size()
            for t in (self.key_packed, self.key_norms, self.value_packed, self.value_norms, self.value_res_norms)
            if t is not None
        )

        # Theoretical TurboQuant budget per token per head:
        # Key: 4-bit * head_dim / 8 = 64 bytes + 4 bytes FP32 norm = 68 bytes
        # Value: 3-bit * head_dim / 8 = 48 bytes + 1-bit * head_dim / 8 = 16 bytes + 4B norm + 4B res_norm = 72 bytes
        # Total = 140 bytes
        theo_per_token_head = (head_dim // 2) + 4 + (head_dim * 3 // 8) + (head_dim // 8) + 8
        theo = batch * num_heads * seq_len * theo_per_token_head
        return phys, theo

    def reset(self) -> None:
        self.key_packed = None
        self.key_norms = None
        self.value_packed = None
        self.value_norms = None
        self.value_res_norms = None
        self.capacity = 0
        self.total_tokens_seen = 0
        self.accum_k_eps_snr = 0.0
        self.accum_k_eps_dir = 0.0
        self.accum_v_eps_snr = 0.0
        self.accum_v_eps_dir = 0.0
        self.num_profile_chunks = 0
        self._s = None
        self._s_prime = None
        self.is_initialized = False
        super().reset()


class TurboQuantCache(Cache):
    """Subclass of transformers.cache_utils.Cache implementing TurboQuant asymmetric quantization."""

    is_turboquant = True

    def __init__(self, seed: int = 42):
        self.seed = seed
        super().__init__(
            layer_class_to_replicate=lambda: TurboQuantLayer(seed=self.seed)
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys_out, values_out = super().update(key_states, value_states, layer_idx, cache_kwargs)
        from src.techniques.turboquant import record_turboquant_telemetry

        phys, theo = self.get_cache_size_bytes()
        shannon = self.get_shannon_phi_metrics()
        record_turboquant_telemetry(shannon, phys, theo)
        return keys_out, values_out

    def prune_after_chunk(self) -> None:
        """No-op for TurboQuant (preserves full compressed context)."""
        pass

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_max_length(self) -> int | None:
        return None

    def get_cache_size_bytes(self) -> tuple[int, int]:
        """Aggregate (physical_bytes, theoretical_bytes) across all layers."""
        total_phys = 0
        total_theo = 0
        for layer in self.layers:
            if hasattr(layer, "get_cache_size_bytes"):
                p, t = layer.get_cache_size_bytes()
                total_phys += p
                total_theo += t
        return total_phys, total_theo

    def get_shannon_phi_metrics(self) -> dict[str, float]:
        """Aggregate average Shannon-phi distortion metrics across all initialized layers."""
        k_snr_sum = 0.0
        k_dir_sum = 0.0
        v_snr_sum = 0.0
        v_dir_sum = 0.0
        count = 0
        for layer in self.layers:
            if getattr(layer, "num_profile_chunks", 0) > 0:
                n = layer.num_profile_chunks
                k_snr_sum += layer.accum_k_eps_snr / n
                k_dir_sum += layer.accum_k_eps_dir / n
                v_snr_sum += layer.accum_v_eps_snr / n
                v_dir_sum += layer.accum_v_eps_dir / n
                count += 1
        if count == 0:
            return {
                "k_eps_snr": 0.0,
                "k_eps_dir": 0.0,
                "v_eps_snr": 0.0,
                "v_eps_dir": 0.0,
            }
        return {
            "k_eps_snr": round(k_snr_sum / count, 6),
            "k_eps_dir": round(k_dir_sum / count, 6),
            "v_eps_snr": round(v_snr_sum / count, 6),
            "v_eps_dir": round(v_dir_sum / count, 6),
        }

    @property
    def effective_bits_per_scalar(self) -> float:
        """Physical bits per KV scalar (140B / 512B BF16 * 16, incl. FP32 norms)."""
        return 4.375

    def __iter__(self) -> Iterator[tuple[Any, Any, Any]]:
        for layer in self.layers:
            yield layer.keys, layer.values, None
