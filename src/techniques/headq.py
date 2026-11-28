from __future__ import annotations

import math
from dataclasses import dataclass
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


def _batched_pca_basis(
    x_batched: "torch.Tensor",
    rank: int,
) -> "torch.Tensor":
    """Compute top-r orthonormal PCA bases for a batch of sample matrices.

    Args:
        x_batched: Tensor of shape (num_groups, M, d).
        rank: Number of principal components (1 <= rank <= d).

    Returns:
        Orthonormal basis tensor of shape (num_groups, d, rank) in float32.
    """
    num_groups, m, d = x_batched.shape
    x_f = x_batched.float()
    # Skip mean-centering when M == 1 to avoid zeroing out the single sample.
    if m > 1:
        x_centered = x_f - x_f.mean(dim=1, keepdim=True)
    else:
        x_centered = x_f

    try:
        _, _, vh = torch.linalg.svd(x_centered, full_matrices=(m < rank))
        basis = vh[:, :rank, :].transpose(-2, -1).contiguous()
    except Exception:
        q_rand = torch.eye(d, rank, dtype=torch.float32, device=x_batched.device)
        q_orth, _ = torch.linalg.qr(q_rand)
        basis = q_orth.unsqueeze(0).expand(num_groups, d, rank).contiguous()

    return basis


def compute_query_pca_basis(
    queries: "torch.Tensor",
    rank: int = 4,
) -> "torch.Tensor":
    """Compute top-r orthonormal PCA basis matrix U in R^(d x r) from calibration queries.

    Args:
        queries: Tensor of shape (..., d) or (num_samples, d).
        rank: Number of principal components (default: 4).

    Returns:
        Orthonormal basis tensor of shape (d, rank).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for compute_query_pca_basis")

    d = queries.size(-1)
    if rank <= 0 or rank > d:
        raise ValueError(f"rank ({rank}) must be in [1, {d}]")

    q_flat = queries.contiguous().reshape(1, -1, d)
    basis = _batched_pca_basis(q_flat, rank=rank).squeeze(0)
    return basis.to(queries.dtype)


def compute_per_head_query_pca_basis(
    queries: "torch.Tensor",
    num_kv_heads: int,
    rank: int = 4,
) -> "torch.Tensor":
    """Compute per-KV-head top-r orthonormal PCA basis from post-RoPE 4D queries.

    Args:
        queries: Tensor of shape (B, num_q_heads, T, d).
        num_kv_heads: Number of key/value heads (must divide num_q_heads).
        rank: Number of principal components (default: 4).

    Returns:
        Orthonormal basis tensor of shape (1, num_kv_heads, d, rank).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for compute_per_head_query_pca_basis")

    if queries.ndim != 4:
        raise ValueError(f"queries must be 4D (B, num_q_heads, T, d), got shape {tuple(queries.shape)}")
    batch, num_q_heads, seq_len, d = queries.shape
    if num_kv_heads <= 0:
        raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
    if rank <= 0 or rank > d:
        raise ValueError(f"rank ({rank}) must be in [1, {d}]")

    gqa_ratio = num_q_heads // num_kv_heads
    q_grouped = (
        queries.contiguous()
        .reshape(batch, num_kv_heads, gqa_ratio, seq_len, d)
        .permute(1, 0, 2, 3, 4)
        .reshape(num_kv_heads, batch * gqa_ratio * seq_len, d)
    )
    basis = _batched_pca_basis(q_grouped, rank=rank)
    return basis.unsqueeze(0).to(queries.dtype)


def quantize_key_2bit(
    k: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Quantize Key vector to symmetric 2-bit representation ({-1.5, -0.5, +0.5, +1.5} * scale).

    Uses Gaussian-optimal per-token RMS step size (rms * 0.9957) computed in FP32.
    Packs 4 2-bit values into one uint8 byte.

    Returns:
        Tuple of (packed_uint8, scale, dequantized_tensor).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for quantize_key_2bit")

    dim = k.size(-1)
    if dim % 4 != 0:
        raise ValueError(f"Head dimension must be divisible by 4 for 2-bit packing, got {dim}")

    k_f = k.float()
    rms = torch.sqrt(k_f.pow(2).mean(dim=-1, keepdim=True))
    scale_f = (rms * 0.9957).clamp(min=1e-6)

    # Map {-1.5, -0.5, +0.5, +1.5} <-> {0, 1, 2, 3}
    u2 = torch.round(k_f / scale_f + 1.5).clamp(0, 3).to(torch.uint8)
    scale = scale_f.to(k.dtype)
    levels = u2.to(k.dtype) - 1.5
    dequant = (levels * scale).to(k.dtype)

    # Pack 4 consecutive 2-bit values into one byte
    packed = (
        (u2[..., 0::4])
        | (u2[..., 1::4] << 2)
        | (u2[..., 2::4] << 4)
        | (u2[..., 3::4] << 6)
    )

    return packed, scale, dequant


def dequantize_key_2bit(
    packed: "torch.Tensor",
    scale: "torch.Tensor",
    original_dim: int,
    dtype: "torch.dtype" = torch.bfloat16,
) -> "torch.Tensor":
    """Dequantize packed symmetric 2-bit Key tensor."""
    if torch is None:
        raise RuntimeError("PyTorch is required for dequantize_key_2bit")

    b0 = (packed & 0x03).to(dtype) - 1.5
    b1 = ((packed >> 2) & 0x03).to(dtype) - 1.5
    b2 = ((packed >> 4) & 0x03).to(dtype) - 1.5
    b3 = ((packed >> 6) & 0x03).to(dtype) - 1.5

    shape = list(packed.shape)
    shape[-1] = original_dim
    unpacked = torch.empty(shape, dtype=dtype, device=packed.device)
    unpacked[..., 0::4] = b0
    unpacked[..., 1::4] = b1
    unpacked[..., 2::4] = b2
    unpacked[..., 3::4] = b3

    return (unpacked * scale.to(dtype)).to(dtype)


def compute_key_sidecode(
    k_residual: "torch.Tensor",
    basis: "torch.Tensor",
    sidecode_bits: int = 4,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Project key reconstruction residual onto top-r query PCA basis and quantize to 4 bits.

    Args:
        k_residual: Tensor of shape (..., d) representing e = k - k_base.
        basis: Orthonormal basis matrix of shape (d, r).
        sidecode_bits: Bitwidth for residual code coordinates (default: 4).

    Returns:
        Tuple of (packed_code, scale, dequantized_projection).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for compute_key_sidecode")

    # Project residual: z = e @ basis, shape (..., r)
    z_proj = torch.matmul(k_residual.float(), basis.float()).to(k_residual.dtype)

    # 4-bit uniform quantization in [-8, 7]
    scale = z_proj.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 7.0
    q = torch.round(z_proj / scale).clamp(-8, 7).to(torch.int8)
    z_dequant = q.to(k_residual.dtype) * scale.to(k_residual.dtype)

    # Pack 2 4-bit nibbles into uint8 byte
    u4 = (q + 8).to(torch.uint8)
    packed = u4[..., 0::2] | (u4[..., 1::2] << 4)

    return packed, scale, z_dequant


def dequantize_key_sidecode(
    packed: "torch.Tensor",
    scale: "torch.Tensor",
    rank: int,
    dtype: "torch.dtype" = torch.bfloat16,
) -> "torch.Tensor":
    """Dequantize 4-bit side-code into continuous r-dimensional projection vector."""
    if torch is None:
        raise RuntimeError("PyTorch is required for dequantize_key_sidecode")

    low = (packed & 0x0F).to(torch.int8) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int8) - 8

    shape = list(packed.shape)
    shape[-1] = rank
    unpacked = torch.empty(shape, dtype=torch.int8, device=packed.device)
    unpacked[..., 0::2] = low
    unpacked[..., 1::2] = high

    return (unpacked.to(dtype) * scale.to(dtype)).to(dtype)


def quantize_value_4bit(
    v: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Quantize Value vector to 4-bit uniform representation (16 levels in [-8, 7])."""
    if torch is None:
        raise RuntimeError("PyTorch is required for quantize_value_4bit")

    scale = v.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 7.0
    q = torch.round(v / scale).clamp(-8, 7).to(torch.int8)
    dequant = q.to(v.dtype) * scale.to(v.dtype)

    u4 = (q + 8).to(torch.uint8)
    packed = u4[..., 0::2] | (u4[..., 1::2] << 4)
    return packed, scale, dequant


def dequantize_value_4bit(
    packed: "torch.Tensor",
    scale: "torch.Tensor",
    original_dim: int,
    dtype: "torch.dtype" = torch.bfloat16,
) -> "torch.Tensor":
    """Dequantize packed 4-bit Value tensor."""
    if torch is None:
        raise RuntimeError("PyTorch is required for dequantize_value_4bit")

    low = (packed & 0x0F).to(torch.int8) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int8) - 8

    shape = list(packed.shape)
    shape[-1] = original_dim
    unpacked = torch.empty(shape, dtype=torch.int8, device=packed.device)
    unpacked[..., 0::2] = low
    unpacked[..., 1::2] = high

    return (unpacked.to(dtype) * scale.to(dtype)).to(dtype)


def compute_gqa_headq_attention(
    query: "torch.Tensor",
    key: "torch.Tensor",
    value: "torch.Tensor",
    basis: "torch.Tensor | None" = None,
    rank: int = 4,
    scale: float | None = None,
) -> tuple["torch.Tensor", "torch.Tensor", dict[str, float]]:
    """Compute GQA attention using HeadQ 2-bit Key + rank-4 Additive Logit Correction.

    Follows HeadQ's model-visible distortion architecture:
      1. Coarse logit: s_base = (q @ k_hat_base^T) / sqrt(d)
      2. Query projection: q_proj = q @ U in R^r
      3. Residual logit correction: delta_s = (q_proj @ z_tilde^T) / sqrt(d)
      4. Corrected logit: s_hat = s_base + delta_s
      5. Softmax & Value readout with 4-bit quantized Values.

    Returns:
        Tuple of (attn_output, attn_weights, distortion_metrics).
    """
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for compute_gqa_headq_attention")

    batch, num_q_heads, seq_len_q, d = query.shape
    _, num_kv_heads, seq_len_kv, _ = key.shape
    gqa_ratio = num_q_heads // num_kv_heads

    # Derive or use provided orthonormal query PCA basis
    if basis is None:
        basis = compute_query_pca_basis(query, rank=rank)

    # 1. Base 2-bit Key Quantization
    _, k_scale, k_dequant = quantize_key_2bit(key)
    k_residual = key - k_dequant

    # 2. Key Side-code projection and quantization
    _, z_scale, z_dequant = compute_key_sidecode(k_residual, basis, sidecode_bits=4)

    # 3. Value 4-bit Quantization
    _, v_scale, v_dequant = quantize_value_4bit(value)

    # 4. GQA Expansion
    if gqa_ratio > 1:
        expanded_k_base = k_dequant.repeat_interleave(gqa_ratio, dim=1)
        expanded_z = z_dequant.repeat_interleave(gqa_ratio, dim=1)
        expanded_v = v_dequant.repeat_interleave(gqa_ratio, dim=1)
        expanded_k_exact = key.repeat_interleave(gqa_ratio, dim=1)
    else:
        expanded_k_base = k_dequant
        expanded_z = z_dequant
        expanded_v = v_dequant
        expanded_k_exact = key

    softmax_scale = scale if scale is not None else (d ** -0.5)

    # Coarse logits from 2-bit Key
    coarse_logits = torch.matmul(query.float(), expanded_k_base.float().transpose(-2, -1)) * softmax_scale

    # Additive logit correction via query PCA projection
    q_basis = basis
    if q_basis.ndim == 4 and gqa_ratio > 1:
        q_basis = q_basis.repeat_interleave(gqa_ratio, dim=1)
    q_proj = torch.matmul(query.float(), q_basis.float()).to(query.dtype)
    delta_s = torch.matmul(q_proj.float(), expanded_z.float().transpose(-2, -1)) * softmax_scale

    # Corrected logits
    corrected_logits = coarse_logits + delta_s
    attn_weights = F.softmax(corrected_logits, dim=-1).to(query.dtype)
    attn_output = torch.matmul(attn_weights, expanded_v)

    # Compute Exact Logits for Distortion Profiling
    exact_logits = torch.matmul(query.float(), expanded_k_exact.float().transpose(-2, -1)) * softmax_scale
    exact_weights = F.softmax(exact_logits, dim=-1).to(query.dtype)

    kl_div = F.kl_div(
        F.log_softmax(corrected_logits, dim=-1),
        exact_weights,
        reduction="batchmean",
    ).item()

    kl_coarse = F.kl_div(
        F.log_softmax(coarse_logits, dim=-1),
        exact_weights,
        reduction="batchmean",
    ).item()

    distortion_metrics = {
        "attention_kl_corrected": max(0.0, float(kl_div)),
        "attention_kl_coarse": max(0.0, float(kl_coarse)),
        "kl_reduction_ratio": float(1.0 - (kl_div / max(kl_coarse, 1e-9))),
    }

    return attn_output, attn_weights, distortion_metrics


_PYTORCH_HOOKS_INSTALLED: bool = False
_ORIG_SDPA: Any = None


def _install_pytorch_headq_hooks() -> None:
    """Install idempotent SDPA query hook for deferred Chunk-0 PCA fitting and key correction."""
    global _PYTORCH_HOOKS_INSTALLED, _ORIG_SDPA

    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from src.techniques.headq_cache import get_active_headq_cache
    except ImportError:
        return

    orig_sdpa = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
    if orig_sdpa is None:
        return

    if getattr(orig_sdpa, "_is_headq_sdpa_wrapper", False):
        _PYTORCH_HOOKS_INSTALLED = True
        return

    _ORIG_SDPA = orig_sdpa

    def headq_sdpa_attention_forward(module, query, key, value, *args, **kwargs):
        cache = get_active_headq_cache()
        if cache is not None and getattr(cache, "is_headq", False):
            layer_idx = getattr(module, "layer_idx", None)
            if layer_idx is not None and 0 <= layer_idx < len(cache.layers):
                layer = cache.layers[layer_idx]
                if (
                    getattr(layer, "is_initialized", False)
                    and layer.total_tokens_seen > 0
                    and key.shape[-2] == layer.total_tokens_seen
                    and key.device == layer.target_device
                ):
                    if getattr(layer, "_pending_chunk0_residual", None) is not None:
                        layer.finalize_chunk0_from_queries(query)
                        if layer.basis is not None:
                            corrected_k = layer.keys
                            if corrected_k is not None and corrected_k.shape == key.shape:
                                key = corrected_k.to(dtype=query.dtype, device=query.device)

        return orig_sdpa(module, query, key, value, *args, **kwargs)

    headq_sdpa_attention_forward._is_headq_sdpa_wrapper = True  # type: ignore[attr-defined]
    ALL_ATTENTION_FUNCTIONS.register("sdpa", headq_sdpa_attention_forward)
    _PYTORCH_HOOKS_INSTALLED = True


def _uninstall_pytorch_headq_hooks() -> None:
    """Restore original SDPA attention forward in transformers ALL_ATTENTION_FUNCTIONS."""
    global _PYTORCH_HOOKS_INSTALLED, _ORIG_SDPA
    if not _PYTORCH_HOOKS_INSTALLED or _ORIG_SDPA is None:
        return

    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        cur_sdpa = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
        if getattr(cur_sdpa, "_is_headq_sdpa_wrapper", False):
            ALL_ATTENTION_FUNCTIONS.register("sdpa", _ORIG_SDPA)
    except ImportError:
        pass
    finally:
        _PYTORCH_HOOKS_INSTALLED = False
        _ORIG_SDPA = None


@register("headq")
class HeadQTechnique(Technique):
    """HeadQ: Head-Wise Quantization for KV Cache with Attention Logit Preservation.

    Preserves model-visible attention logits through an asymmetric 2-bit quantizer-corrector:
      1. Key vectors are quantized to an aggressive 2-bit representation (4 levels).
      2. Key reconstruction residuals are projected onto an offline-learned rank-4 query PCA
         orthonormal subspace, stored as a compact 4-bit side-code (16 bits/token per head).
      3. At decode time, queries are projected into the r-dimensional subspace and an additive
         dot-product correction is fused directly into attention logits, eliminating 85-94% of
         2-bit attention divergence with minimal computational overhead.
      4. Value vectors are quantized at 4-bit precision according to readout geometry.

    Axes:
        Axis.STRUCTURAL
    """

    axes = Axis.STRUCTURAL

    def __init__(
        self,
        key_bits: int = 2,
        val_bits: int = 4,
        rank: int = 4,
        sidecode_bits: int = 4,
        block_size: int = 16,
    ):
        if int(key_bits) != 2:
            raise ValueError(f"HeadQ primary contract expects key_bits=2, got {key_bits}")
        if int(val_bits) <= 0:
            raise ValueError(f"val_bits must be positive, got {val_bits}")
        if int(rank) <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if int(sidecode_bits) <= 0:
            raise ValueError(f"sidecode_bits must be positive, got {sidecode_bits}")
        if int(block_size) <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.key_bits = int(key_bits)
        self.val_bits = int(val_bits)
        self.rank = int(rank)
        self.sidecode_bits = int(sidecode_bits)
        self.block_size = int(block_size)

    def engine_kwargs(self, context_length: int, max_tokens: int) -> dict[str, Any]:
        """Extra engine kwargs merged into the vLLM / PyTorch constructor."""
        del context_length, max_tokens
        return {
            "block_size": self.block_size,
            "kv_cache_dtype": "fp8",  # Native quantization knob
        }

    def workload(self) -> Workload:
        return SyntheticFiller()

    def pytorch_cache_factory(self) -> Any:
        from src.techniques.headq_cache import HeadQCache

        return lambda: HeadQCache(
            rank=self.rank,
            sidecode_bits=self.sidecode_bits,
        )

    def setup(self) -> None:
        self._sample_count = 0
        self._accum_quantized_bytes = 0
        from src.techniques.headq_cache import set_active_headq_cache

        set_active_headq_cache(None)
        _install_pytorch_headq_hooks()

    def teardown(self) -> None:
        from src.techniques.headq_cache import set_active_headq_cache

        set_active_headq_cache(None)
        _uninstall_pytorch_headq_hooks()

    def on_sample_complete(self, cache: Any) -> None:
        if cache is None:
            return
        self._sample_count = getattr(self, "_sample_count", 0) + 1
        self._accum_quantized_bytes = getattr(self, "_accum_quantized_bytes", 0) + int(
            getattr(cache, "total_quantized_bytes", 0)
        )

    def extra_metrics(self, llm, workload_output) -> dict[str, Any]:
        del llm, workload_output
        from src.techniques.headq_cache import get_active_headq_cache

        active_cache = get_active_headq_cache()
        if getattr(self, "_sample_count", 0) > 0:
            quantized_bytes = self._accum_quantized_bytes
        elif active_cache is not None:
            quantized_bytes = active_cache.total_quantized_bytes
        else:
            quantized_bytes = 0
        effective_key_bits = self.key_bits + (self.rank * self.sidecode_bits) / 128.0
        return {
            "key_bits": self.key_bits,
            "val_bits": self.val_bits,
            "pca_rank": self.rank,
            "sidecode_bits": self.sidecode_bits,
            "effective_key_bits": effective_key_bits,
            "block_size": self.block_size,
            "model_visible_distortion_aware": True,
            "quantized_cache_bytes": quantized_bytes,
        }
