from __future__ import annotations

import math
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


# Beta(63.5, 63.5) Lloyd-Max codebooks for d=128 on unit sphere S^127.
# 4-bit (16 levels) for Keys
CENTROIDS_4BIT: list[float] = [
    -0.23706912, -0.17997439, -0.14069195, -0.10906106,
    -0.08173541, -0.05698214, -0.03362839, -0.01112486,
    0.01112486, 0.03362839, 0.05698214, 0.08173541,
    0.10906106, 0.14069195, 0.17997439, 0.23706912,
]

THRESHOLDS_4BIT: list[float] = [
    -0.20852176, -0.16033317, -0.12487651, -0.09539823,
    -0.06935877, -0.04530526, -0.02237662, 0.0,
    0.02237662, 0.04530526, 0.06935877, 0.09539823,
    0.12487651, 0.16033317, 0.20852176,
]

# 3-bit (8 levels) for Values
CENTROIDS_3BIT: list[float] = [
    -0.18863435, -0.11830974, -0.06671308, -0.02164567,
    0.02164567, 0.06671308, 0.11830974, 0.18863435,
]

THRESHOLDS_3BIT: list[float] = [
    -0.15347205, -0.09251141, -0.04417938, 0.0,
    0.04417938, 0.09251141, 0.15347205,
]


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform along the last dimension (must be power of 2).

    Executes in O(d log2(d)) time using radix-2 butterfly operations in FP32
    to prevent mantissa precision loss.
    """
    orig_shape = x.shape
    d = orig_shape[-1]
    if (d & (d - 1)) != 0:
        raise ValueError(f"Last dimension must be a power of 2, got {d}")

    # Compute in float32 for numerical stability across butterfly stages
    x_flat = x.reshape(-1, d).to(dtype=torch.float32)
    h = 1
    while h < d:
        x_flat = x_flat.view(-1, d // (2 * h), 2, h)
        x1 = x_flat[:, :, 0, :]
        x2 = x_flat[:, :, 1, :]
        x_flat = torch.cat([x1 + x2, x1 - x2], dim=2)
        h *= 2

    # Orthonormal scaling factor 1 / sqrt(d)
    x_out = x_flat.view(orig_shape) * (1.0 / math.sqrt(d))
    return x_out


def ifwht(x: torch.Tensor) -> torch.Tensor:
    """Inverse Fast Walsh-Hadamard Transform.

    Since H = H^T and H^2 = d * I, the normalized transform (1/sqrt(d)) * H
    is strictly self-inverse: ifwht(x) == fwht(x).
    """
    return fwht(x)


def get_codebook_tensors(
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return precomputed centroids and thresholds as float32 tensors on target device."""
    c16 = torch.tensor(CENTROIDS_4BIT, dtype=torch.float32, device=device)
    t16 = torch.tensor(THRESHOLDS_4BIT, dtype=torch.float32, device=device)
    c8 = torch.tensor(CENTROIDS_3BIT, dtype=torch.float32, device=device)
    t8 = torch.tensor(THRESHOLDS_3BIT, dtype=torch.float32, device=device)
    return c16, t16, c8, t8


def generate_rademacher_signs(
    shape: tuple[int, ...],
    device: torch.device | str = "cpu",
    seed: int = 42,
) -> torch.Tensor:
    """Generate deterministic Rademacher random sign vectors in {-1.0, +1.0}."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    bits = torch.randint(0, 2, shape, generator=gen, dtype=torch.float32)
    signs = bits * 2.0 - 1.0
    return signs.to(device=device)


def quantize_lloyd_max(
    x_rot: torch.Tensor,
    thresholds: torch.Tensor,
) -> torch.Tensor:
    """Map continuous rotated values to integer centroid indices via bucketize."""
    x_f32 = x_rot.to(dtype=torch.float32)
    indices = torch.bucketize(x_f32, thresholds).to(dtype=torch.uint8)
    return indices


def dequantize_lloyd_max(
    indices: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """Map integer centroid indices back to continuous centroids."""
    return codebook[indices.to(dtype=torch.long)]


def pack_4bit_nibbles(indices: torch.Tensor) -> torch.Tensor:
    """Pack pairs of 4-bit indices along the last dimension into uint8 bytes."""
    d = indices.shape[-1]
    if d % 2 != 0:
        raise ValueError(f"Last dimension must be even for nibble packing, got {d}")
    x0 = indices[..., 0::2]
    x1 = indices[..., 1::2]
    packed = (x0 & 0x0F) | ((x1 & 0x0F) << 4)
    return packed.to(dtype=torch.uint8)


def unpack_4bit_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 bytes into pairs of 4-bit indices along the last dimension."""
    u0 = packed & 0x0F
    u1 = (packed >> 4) & 0x0F
    out_shape = list(packed.shape)
    out_shape[-1] = out_shape[-1] * 2
    unpacked = torch.empty(out_shape, dtype=torch.uint8, device=packed.device)
    unpacked[..., 0::2] = u0
    unpacked[..., 1::2] = u1
    return unpacked


def pack_value_nibbles(
    lloyd_indices: torch.Tensor,
    qjl_signs: torch.Tensor,
) -> torch.Tensor:
    """Pack 3-bit Lloyd-Max index + 1-bit QJL sign into 4-bit nibbles, then into uint8 bytes.

    qjl_signs can be in {-1.0, 1.0} or {0, 1}.
    """
    d = lloyd_indices.shape[-1]
    if d % 2 != 0:
        raise ValueError(f"Last dimension must be even, got {d}")

    sign_bits = torch.where(qjl_signs > 0, 1, 0).to(dtype=torch.uint8)
    lloyd_3bit = (lloyd_indices & 0x07).to(dtype=torch.uint8)
    # Fuse into 4-bit nibble: (sign_bit << 3) | lloyd_3bit
    nibbles = lloyd_3bit | (sign_bits << 3)

    # Pack 2 nibbles per byte
    n0 = nibbles[..., 0::2]
    n1 = nibbles[..., 1::2]
    packed = (n0 & 0x0F) | ((n1 & 0x0F) << 4)
    return packed.to(dtype=torch.uint8)


def unpack_value_nibbles(
    packed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack uint8 bytes into 3-bit Lloyd indices and {-1.0, +1.0} QJL signs."""
    n0 = packed & 0x0F
    n1 = (packed >> 4) & 0x0F
    out_shape = list(packed.shape)
    out_shape[-1] = out_shape[-1] * 2
    nibbles = torch.empty(out_shape, dtype=torch.uint8, device=packed.device)
    nibbles[..., 0::2] = n0
    nibbles[..., 1::2] = n1

    lloyd_indices = (nibbles & 0x07).to(dtype=torch.uint8)
    sign_bits = (nibbles >> 3) & 0x01
    qjl_signs = torch.where(sign_bits == 1, 1.0, -1.0).to(dtype=torch.float32)
    return lloyd_indices, qjl_signs


def qjl_sketch(
    residual: torch.Tensor,
    s_prime: torch.Tensor,
) -> torch.Tensor:
    """Generate 1-bit Quantized Johnson-Lindenstrauss sketch q in {-1.0, +1.0}."""
    res_f32 = residual.to(dtype=torch.float32)
    proj = fwht(s_prime * res_f32)
    q = torch.sign(proj)
    q = torch.where(q == 0.0, 1.0, q)
    return q


def qjl_reconstruct(
    sketch: torch.Tensor,
    s_prime: torch.Tensor,
    res_norm: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Compute unbiased residual estimator r_hat = sqrt(pi/2) * (||r|| / sqrt(d)) * (s' * fwht(q))."""
    sketch_f32 = sketch.to(dtype=torch.float32)
    bias_correction = math.sqrt(math.pi / 2.0)
    scale = bias_correction * (res_norm.to(dtype=torch.float32) / math.sqrt(head_dim))
    r_hat = scale * (s_prime * fwht(sketch_f32))
    return r_hat


def turboquant_encode_key(
    key: torch.Tensor,
    s: torch.Tensor,
    thresholds_4bit: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize Key tensor with 4-bit scalar Lloyd-Max + FWHT rotation.

    Returns:
      packed_indices: uint8 tensor of shape (..., head_dim // 2)
      norm: float32 tensor of shape (..., 1)
    """
    k_f32 = key.to(dtype=torch.float32)
    norm = torch.norm(k_f32, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    k_unit = k_f32 / norm

    k_rot = fwht(s * k_unit)
    idx = quantize_lloyd_max(k_rot, thresholds_4bit)
    packed = pack_4bit_nibbles(idx)
    return packed, norm


def turboquant_decode_key(
    packed: torch.Tensor,
    norm: torch.Tensor,
    s: torch.Tensor,
    codebook_4bit: torch.Tensor,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize Key tensor back to target model dtype."""
    idx = unpack_4bit_nibbles(packed)
    k_rot_hat = dequantize_lloyd_max(idx, codebook_4bit)
    k_unit_hat = s * ifwht(k_rot_hat)
    k_hat = k_unit_hat * norm.to(dtype=torch.float32)
    return k_hat.to(dtype=target_dtype)


def turboquant_encode_value(
    value: torch.Tensor,
    s: torch.Tensor,
    s_prime: torch.Tensor,
    thresholds_3bit: torch.Tensor,
    codebook_3bit: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize Value tensor with 3-bit scalar Lloyd-Max + 1-bit QJL residual sketch.

    Returns:
      packed_bytes: uint8 tensor of shape (..., head_dim // 2)
      norm: float32 tensor of shape (..., 1)
      res_norm: float32 tensor of shape (..., 1)
    """
    v_f32 = value.to(dtype=torch.float32)
    norm = torch.norm(v_f32, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    v_unit = v_f32 / norm

    v_rot = fwht(s * v_unit)
    lloyd_idx = quantize_lloyd_max(v_rot, thresholds_3bit)
    v_rot_scalar = dequantize_lloyd_max(lloyd_idx, codebook_3bit)

    res = v_rot - v_rot_scalar
    res_norm = torch.norm(res, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    q = qjl_sketch(res, s_prime)

    packed = pack_value_nibbles(lloyd_idx, q)
    return packed, norm, res_norm


def turboquant_decode_value(
    packed: torch.Tensor,
    norm: torch.Tensor,
    res_norm: torch.Tensor,
    s: torch.Tensor,
    s_prime: torch.Tensor,
    codebook_3bit: torch.Tensor,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize Value tensor back to target model dtype."""
    d = s.shape[-1]
    lloyd_idx, q = unpack_value_nibbles(packed)
    v_rot_scalar = dequantize_lloyd_max(lloyd_idx, codebook_3bit)
    res_hat = qjl_reconstruct(q, s_prime, res_norm, head_dim=d)

    v_rot_hat = v_rot_scalar + res_hat
    v_unit_hat = s * ifwht(v_rot_hat)
    v_hat = v_unit_hat * norm.to(dtype=torch.float32)
    return v_hat.to(dtype=target_dtype)


def compute_shannon_phi_metrics(
    x_true: torch.Tensor,
    x_rec: torch.Tensor,
) -> dict[str, float]:
    """Compute 2D Shannon-phi distortion metrics: eps_snr (scale) and eps_dir (direction).

    eps_snr = ||x - x_rec||^2 / (||x||^2 + eps)
    eps_dir = 1 - <x, x_rec> / (||x|| * ||x_rec|| + eps)
    """
    x1 = x_true.to(dtype=torch.float32)
    x2 = x_rec.to(dtype=torch.float32)

    diff = x1 - x2
    num_diff = torch.sum(diff * diff, dim=-1)
    norm_x1_sq = torch.sum(x1 * x1, dim=-1).clamp(min=1e-8)
    eps_snr = float(torch.mean(num_diff / norm_x1_sq).item())

    dot = torch.sum(x1 * x2, dim=-1)
    norm_x1 = torch.norm(x1, p=2, dim=-1).clamp(min=1e-8)
    norm_x2 = torch.norm(x2, p=2, dim=-1).clamp(min=1e-8)
    cos_sim = dot / (norm_x1 * norm_x2)
    eps_dir = float(torch.mean(1.0 - cos_sim).clamp(min=0.0).item())

    return {
        "eps_snr": round(eps_snr, 6),
        "eps_dir": round(eps_dir, 6),
    }
