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


_CALIBRATION_TEXT = (
    "In transformer attention, keys are observed by the model exclusively through "
    "dot-product attention logits followed by softmax normalization, while values "
    "are observed through attention-weighted linear readouts. Standard cache "
    "quantizers minimize raw Euclidean reconstruction error in storage coordinates, "
    "allocating bit precision to directions that queries rarely probe or to common "
    "constant key shifts that cancel out identically under softmax invariance. "
    "HeadQ measures quantization distortion in model-visible coordinates: score-space "
    "Kullback-Leibler divergence for keys and squared-attention weighted error for "
    "values. During offline calibration, post-RoPE query vectors within each attention "
    "head concentrate strongly into a low-dimensional principal subspace. By "
    "projecting the 2-bit key quantization residual onto this top-r orthonormal "
    "query basis and storing a compact 4-bit side-code, decode attention recovers "
    "most of the score-space fidelity with minimal memory overhead. "
) * 6


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
    if rank > d:
        raise ValueError(f"rank ({rank}) cannot exceed head dimension ({d})")

    q_flat = queries.contiguous().reshape(-1, d).float()
    q_centered = q_flat - q_flat.mean(dim=0, keepdim=True)

    try:
        _, _, Vh = torch.linalg.svd(q_centered, full_matrices=False)
        basis = Vh[:rank, :].transpose(0, 1).contiguous()
    except Exception:
        q_rand = torch.eye(d, rank, dtype=torch.float32, device=queries.device)
        basis, _ = torch.linalg.qr(q_rand)

    return basis.to(queries.dtype)


def compute_gqa_query_pca_bases(
    queries: "torch.Tensor",
    num_kv_heads: int,
    rank: int = 4,
) -> "torch.Tensor":
    """Compute per-KV-head orthonormal Query-PCA bases U_h in R^(H_kv x d x r).

    Pools query heads belonging to each GQA group before SVD so every KV head
    gets its own query-aligned subspace.

    Args:
        queries: Post-RoPE query tensor of shape (batch, num_q_heads, seq_len, d).
        num_kv_heads: Number of key/value heads.
        rank: Number of principal components per KV head.

    Returns:
        Orthonormal bases tensor of shape (num_kv_heads, d, rank).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for compute_gqa_query_pca_bases")
    if queries.ndim != 4:
        raise ValueError(f"Expected 4D query tensor (B, H_q, T, d), got shape {tuple(queries.shape)}")

    batch, num_q_heads, seq_len, d = queries.shape
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )

    gqa_ratio = num_q_heads // num_kv_heads
    q_grouped = (
        queries.reshape(batch, num_kv_heads, gqa_ratio, seq_len, d)
        .permute(1, 0, 2, 3, 4)
        .reshape(num_kv_heads, batch * gqa_ratio * seq_len, d)
    )
    bases = [compute_query_pca_basis(q_grouped[h], rank=rank) for h in range(num_kv_heads)]
    return torch.stack(bases, dim=0)


def quantize_key_2bit(
    k: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Quantize Key vector to 2-bit per-token asymmetric min-max representation (levels 0..3).

    Packs 4 2-bit values into one uint8 byte.

    Returns:
        Tuple of (packed_uint8, scale, zero_point, dequantized_tensor).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for quantize_key_2bit")

    dim = k.size(-1)
    if dim % 4 != 0:
        raise ValueError(f"Head dimension must be divisible by 4 for 2-bit packing, got {dim}")

    k_min = k.amin(dim=-1, keepdim=True)
    k_max = k.amax(dim=-1, keepdim=True)
    scale = (k_max - k_min).clamp(min=1e-6) / 3.0

    u2 = torch.round((k - k_min) / scale).clamp(0, 3).to(torch.uint8)
    dequant = (k_min + u2.to(k.dtype) * scale.to(k.dtype)).to(k.dtype)

    packed = (
        (u2[..., 0::4])
        | (u2[..., 1::4] << 2)
        | (u2[..., 2::4] << 4)
        | (u2[..., 3::4] << 6)
    )

    return packed, scale, k_min, dequant


def dequantize_key_2bit(
    packed: "torch.Tensor",
    scale: "torch.Tensor",
    zero_point: "torch.Tensor",
    original_dim: int,
    dtype: "torch.dtype" = torch.bfloat16,
) -> "torch.Tensor":
    """Dequantize packed asymmetric 2-bit Key tensor."""
    if torch is None:
        raise RuntimeError("PyTorch is required for dequantize_key_2bit")

    b0 = packed & 0x03
    b1 = (packed >> 2) & 0x03
    b2 = (packed >> 4) & 0x03
    b3 = (packed >> 6) & 0x03

    shape = list(packed.shape)
    shape[-1] = original_dim
    unpacked = torch.empty(shape, dtype=torch.uint8, device=packed.device)
    unpacked[..., 0::4] = b0
    unpacked[..., 1::4] = b1
    unpacked[..., 2::4] = b2
    unpacked[..., 3::4] = b3

    return (zero_point.to(dtype) + unpacked.to(dtype) * scale.to(dtype)).to(dtype)


def compute_key_sidecode(
    k_residual: "torch.Tensor",
    basis: "torch.Tensor",
    sidecode_bits: int = 4,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Project key reconstruction residual onto top-r query PCA basis and quantize to 4 bits.

    Args:
        k_residual: Tensor of shape (..., d) representing e = k - k_base.
        basis: Orthonormal basis matrix of shape (d, r) or per-KV-head (H_kv, d, r).
        sidecode_bits: Bitwidth for residual code coordinates (must be 4).

    Returns:
        Tuple of (packed_code, scale, dequantized_projection).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for compute_key_sidecode")
    if int(sidecode_bits) != 4:
        raise ValueError(f"Only sidecode_bits=4 is supported, got {sidecode_bits}")

    if basis.ndim == 3:
        z_proj = torch.einsum("bhtd,hdr->bhtr", k_residual.float(), basis.float()).to(k_residual.dtype)
    elif basis.ndim == 2:
        z_proj = torch.matmul(k_residual.float(), basis.float()).to(k_residual.dtype)
    else:
        raise ValueError(f"Expected 2D (d, r) or 3D (H_kv, d, r) basis, got shape {tuple(basis.shape)}")

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
    """Compute GQA attention using HeadQ 2-bit Key + rank-r Additive Logit Correction.

    Follows HeadQ's model-visible distortion architecture:
      1. Coarse logit: s_base = (q @ k_hat_base^T) / sqrt(d)
      2. Query projection: q_proj = q @ U_h in R^r
      3. Residual logit correction: delta_s = (q_proj @ z_tilde^T) / sqrt(d)
      4. Corrected logit: s_hat = s_base + delta_s
      5. Softmax & Value readout with 4-bit quantized Values, plus A^2 distortion profiling.

    Returns:
        Tuple of (attn_output, attn_weights, distortion_metrics).
    """
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for compute_gqa_headq_attention")

    batch, num_q_heads, seq_len_q, d = query.shape
    _, num_kv_heads, seq_len_kv, _ = key.shape
    gqa_ratio = num_q_heads // num_kv_heads

    # Derive or use provided per-KV-head orthonormal query PCA basis
    if basis is None:
        basis = compute_gqa_query_pca_bases(query, num_kv_heads=num_kv_heads, rank=rank)

    # 1. Base asymmetric 2-bit Key Quantization
    _, k_scale, k_zero, k_dequant = quantize_key_2bit(key)
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
        expanded_v_exact = value.repeat_interleave(gqa_ratio, dim=1)
        expanded_k_exact = key.repeat_interleave(gqa_ratio, dim=1)
    else:
        expanded_k_base = k_dequant
        expanded_z = z_dequant
        expanded_v = v_dequant
        expanded_v_exact = value
        expanded_k_exact = key

    softmax_scale = scale if scale is not None else (d ** -0.5)

    # Coarse logits from 2-bit Key
    coarse_logits = torch.matmul(query.float(), expanded_k_base.float().transpose(-2, -1)) * softmax_scale

    # Additive logit correction via query PCA projection
    if basis.ndim == 3:
        expanded_basis = basis.repeat_interleave(gqa_ratio, dim=0) if gqa_ratio > 1 else basis
        q_proj = torch.einsum("bhtd,hdr->bhtr", query.float(), expanded_basis.float()).to(query.dtype)
    else:
        q_proj = torch.matmul(query.float(), basis.float()).to(query.dtype)
    delta_s = torch.matmul(q_proj.float(), expanded_z.float().transpose(-2, -1)) * softmax_scale

    # Corrected logits
    corrected_logits = coarse_logits + delta_s
    attn_weights = F.softmax(corrected_logits, dim=-1).to(query.dtype)
    attn_output = torch.matmul(attn_weights, expanded_v)

    # Compute Exact Logits for Model-Visible Distortion Profiling
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

    v_err_sq = (expanded_v_exact.float() - expanded_v.float()).pow(2).sum(dim=-1, keepdim=True)
    value_a2_distortion = torch.matmul(exact_weights.float().pow(2), v_err_sq).mean().item()

    distortion_metrics = {
        "attention_kl_corrected": max(0.0, float(kl_div)),
        "attention_kl_coarse": max(0.0, float(kl_coarse)),
        "kl_reduction_ratio": float(1.0 - (kl_div / max(kl_coarse, 1e-9))),
        "value_a2_distortion": max(0.0, float(value_a2_distortion)),
    }

    return attn_output, attn_weights, distortion_metrics


@register("headq")
class HeadQTechnique(Technique):
    """HeadQ: Head-Wise Quantization for KV Cache with Attention Logit Preservation.

    Preserves model-visible attention logits through an asymmetric 2-bit quantizer-corrector:
      1. Key vectors are quantized to an asymmetric 2-bit representation (4 levels [0, 3]).
      2. Key reconstruction residuals are projected onto an offline-learned per-KV-head rank-r
         post-RoPE query PCA orthonormal subspace, stored as a compact 4-bit side-code.
      3. At decode time, queries are projected into the r-dimensional subspace and an additive
         dot-product correction is fused into attention logits, eliminating 85-94% of
         2-bit attention divergence with minimal computational overhead.
      4. Value vectors are quantized at 4-bit precision with A^2 readout distortion profiling.

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
        if int(val_bits) != 4:
            raise ValueError(f"HeadQ currently supports val_bits=4 only, got {val_bits}")
        if int(rank) <= 0 or (int(rank) % 2 != 0):
            raise ValueError(f"rank must be a positive even integer for sidecode packing, got {rank}")
        if int(sidecode_bits) != 4:
            raise ValueError(f"HeadQ currently supports sidecode_bits=4 only, got {sidecode_bits}")
        if int(block_size) <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.key_bits = int(key_bits)
        self.val_bits = int(val_bits)
        self.rank = int(rank)
        self.sidecode_bits = int(sidecode_bits)
        self.block_size = int(block_size)
        self._calibrated_bases: dict[int, Any] = {}
        self._calibrated_model_id: str | None = None
        self._calibration_metrics: dict[str, float] = {}

    def setup(self) -> None:
        """Reset active cache references, scalar telemetry, and calibration state."""
        from src.techniques.headq_cache import reset_headq_telemetry, set_active_headq_cache

        set_active_headq_cache(None)
        reset_headq_telemetry()
        self._calibrated_bases = {}
        self._calibrated_model_id = None
        self._calibration_metrics = {}

    def prepare_runner(self, runner: Any) -> None:
        """Run offline post-RoPE Query-PCA calibration on the loaded PyTorchRunner model."""
        if torch is None or F is None:
            return
        if runner is None or getattr(runner, "dry_run", False):
            return
        model = getattr(runner, "model", None)
        tokenizer = getattr(runner, "tokenizer", None)
        if model is None or tokenizer is None:
            return

        model_id = getattr(runner, "model_name_or_path", None)
        if self._calibrated_bases and self._calibrated_model_id == model_id:
            return

        attn_modules = [
            mod for name, mod in model.named_modules() if name.endswith(".self_attn") or name == "self_attn"
        ]
        if not attn_modules:
            return

        enc = tokenizer(
            _CALIBRATION_TEXT,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        device = getattr(runner, "device", "cpu")
        input_ids = enc["input_ids"].to(device)

        captured_q: dict[int, torch.Tensor] = {}
        captured_kv_heads: dict[int, int] = {}
        sample_k_v: dict[str, torch.Tensor] = {}
        active_layer: dict[str, int | None] = {"idx": None}
        hooks = []

        for layer_idx, mod in enumerate(attn_modules):
            def _make_pre_hook(idx: int):
                def _pre_hook(_module, _args, _kwargs=None):
                    active_layer["idx"] = idx
                return _pre_hook
            hooks.append(mod.register_forward_pre_hook(_make_pre_hook(layer_idx)))

        orig_sdpa = F.scaled_dot_product_attention
        config_kv_heads = getattr(getattr(model, "config", None), "num_key_value_heads", None)

        def _wrapped_sdpa(query, key, value, *args, **kwargs):
            idx = active_layer["idx"]
            if idx is not None and idx not in captured_q:
                captured_q[idx] = query.detach().cpu()
                captured_kv_heads[idx] = int(config_kv_heads) if config_kv_heads else int(key.shape[1])
                if idx == 0:
                    sample_k_v["q"] = query[:, :, : min(128, query.shape[2]), :].detach().cpu()
                    sample_k_v["k"] = key[:, :, : min(128, key.shape[2]), :].detach().cpu()
                    sample_k_v["v"] = value[:, :, : min(128, value.shape[2]), :].detach().cpu()
            return orig_sdpa(query, key, value, *args, **kwargs)

        try:
            F.scaled_dot_product_attention = _wrapped_sdpa
            with torch.no_grad():
                model(input_ids, use_cache=False)
        finally:
            F.scaled_dot_product_attention = orig_sdpa
            for h in hooks:
                h.remove()

        bases: dict[int, torch.Tensor] = {}
        for idx, q_tensor in captured_q.items():
            num_kv = captured_kv_heads[idx]
            bases[idx] = compute_gqa_query_pca_bases(q_tensor, num_kv_heads=num_kv, rank=self.rank)

        self._calibrated_bases = bases
        self._calibrated_model_id = model_id
        self._calibration_metrics = {}
        if 0 in bases and "q" in sample_k_v:
            _, _, dist = compute_gqa_headq_attention(
                sample_k_v["q"],
                sample_k_v["k"],
                sample_k_v["v"],
                basis=bases[0],
                rank=self.rank,
            )
            self._calibration_metrics = {
                "calibration_kl_reduction_ratio": dist["kl_reduction_ratio"],
                "calibration_value_a2_distortion": dist["value_a2_distortion"],
            }

    @property
    def compression_ratio(self) -> float:
        """Compression ratio relative to standard 16-bit (FP16/BF16) cache.

        For standard head_dim=128:
          - Standard KV cache: 16 bits (Key) + 16 bits (Value) = 32 bits per element.
          - HeadQ KV cache: 2 bits (Key) + (4 * 4 / 128 = 0.125) sidecode bits + 4 bits (Value) = 6.125 bits.
          - Compression ratio: 32 / 6.125 = ~5.22x.
        """
        effective_key_bits = self.key_bits + (self.rank * self.sidecode_bits) / 128.0
        effective_kv_bits = effective_key_bits + self.val_bits
        return round(32.0 / effective_kv_bits, 2)

    def engine_kwargs(self, context_length: int, max_tokens: int) -> dict[str, Any]:
        """Extra engine kwargs merged into the vLLM / PyTorch constructor."""
        del context_length, max_tokens
        return {
            "block_size": self.block_size,
            "kv_cache_dtype": "auto",
        }

    def workload(self) -> Workload:
        return SyntheticFiller()

    def pytorch_cache_factory(self) -> Any:
        from src.techniques.headq_cache import HeadQCache

        return lambda: HeadQCache(
            rank=self.rank,
            sidecode_bits=self.sidecode_bits,
            val_bits=self.val_bits,
            bases=self._calibrated_bases,
        )

    def extra_metrics(self, llm, workload_output) -> dict[str, Any]:
        del llm, workload_output
        from src.techniques.headq_cache import get_active_headq_cache, get_last_headq_telemetry

        active_cache = get_active_headq_cache()
        telemetry = get_last_headq_telemetry()
        if active_cache is not None:
            quantized_bytes = active_cache.total_quantized_bytes
            head_dim = active_cache.head_dim or telemetry.get("head_dim", 0)
        else:
            quantized_bytes = int(telemetry.get("quantized_cache_bytes", 0))
            head_dim = int(telemetry.get("head_dim", 0))

        effective_head_dim = float(head_dim) if head_dim > 0 else 128.0
        effective_key_bits = self.key_bits + (self.rank * self.sidecode_bits) / effective_head_dim
        effective_kv_bits = effective_key_bits + self.val_bits
        compression_ratio = round(32.0 / effective_kv_bits, 2)

        metrics: dict[str, Any] = {
            "key_bits": self.key_bits,
            "val_bits": self.val_bits,
            "pca_rank": self.rank,
            "sidecode_bits": self.sidecode_bits,
            "head_dim": int(effective_head_dim),
            "effective_key_bits": effective_key_bits,
            "compression_ratio_vs_bf16": compression_ratio,
            "block_size": self.block_size,
            "model_visible_distortion_aware": True,
            "is_query_calibrated": bool(self._calibrated_bases),
            "calibrated_layers": len(self._calibrated_bases),
            "quantized_cache_bytes": quantized_bytes,
        }
        metrics.update(self._calibration_metrics)
        return metrics
