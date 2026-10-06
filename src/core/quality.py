"""Model quality and fidelity evaluation (Section 3.3.2 in proposal.md).

Implements:
1. Logit Divergence:
   - Mean and P99 KL-Divergence across output token probability distributions.
   - Top-1 Token Agreement between compressed/offloaded technique and FP16 baseline.
"""

from __future__ import annotations

import math
from typing import Any

from .metrics import calc_percentile


def compute_top1_agreement(base_token_ids: list[int], target_token_ids: list[int]) -> float:
    """Calculate the fraction of positions where both models generated the exact same top-1 token."""
    if not base_token_ids or not target_token_ids:
        return 0.0
    compare_len = min(len(base_token_ids), len(target_token_ids))
    if compare_len == 0:
        return 0.0
    matches = sum(1 for i in range(compare_len) if base_token_ids[i] == target_token_ids[i])
    return round(matches / compare_len, 4)


def _normalize_logprobs(logprob_dict: dict[int, float]) -> dict[int, float]:
    """Convert logprobs to a normalized probability distribution over known tokens."""
    if not logprob_dict:
        return {}
    # Use log-sum-exp trick for numerical stability
    max_lp = max(logprob_dict.values())
    exps = {tok: math.exp(lp - max_lp) for tok, lp in logprob_dict.items()}
    sum_exps = sum(exps.values())
    if sum_exps <= 0:
        return {}
    return {tok: prob / sum_exps for tok, prob in exps.items()}


def compute_kl_divergence_from_probs(
    base_probs: dict[int, float],
    target_probs: dict[int, float],
    eps: float = 1e-7,
) -> float:
    """Compute KL(P_base || Q_target) over the common distribution support."""
    if not base_probs:
        return 0.0

    all_tokens = set(base_probs.keys()) | set(target_probs.keys())
    kl = 0.0
    for tok in all_tokens:
        p = base_probs.get(tok, eps)
        q = target_probs.get(tok, eps)
        if p > 0:
            kl += p * math.log((p + eps) / (q + eps))
    return max(0.0, kl)


def compute_logit_divergence(
    base_step_logprobs: list[dict[int, float]],
    target_step_logprobs: list[dict[int, float]],
) -> tuple[float, float]:
    """Compute (mean_kl, p99_kl) across all aligned token generation steps."""
    if not base_step_logprobs or not target_step_logprobs:
        return 0.0, 0.0

    steps = min(len(base_step_logprobs), len(target_step_logprobs))
    kl_vals: list[float] = []

    for i in range(steps):
        p_dist = _normalize_logprobs(base_step_logprobs[i])
        q_dist = _normalize_logprobs(target_step_logprobs[i])
        kl = compute_kl_divergence_from_probs(p_dist, q_dist)
        kl_vals.append(kl)

    if not kl_vals:
        return 0.0, 0.0

    mean_kl = round(sum(kl_vals) / len(kl_vals), 5)
    p99_kl = calc_percentile(kl_vals, 99.0)
    return mean_kl, p99_kl


def evaluate_fidelity(
    base_token_ids: list[int],
    target_token_ids: list[int],
    base_step_logprobs: list[dict[int, float]] | None = None,
    target_step_logprobs: list[dict[int, float]] | None = None,
) -> dict[str, Any]:
    """Unified helper returning Logit Divergence and Top-1 metrics for ResultEntry.extra."""
    top1 = compute_top1_agreement(base_token_ids, target_token_ids)
    res: dict[str, Any] = {
        "top1_agreement": top1,
    }

    if base_step_logprobs and target_step_logprobs:
        mean_kl, p99_kl = compute_logit_divergence(base_step_logprobs, target_step_logprobs)
        res["kl_div_mean"] = mean_kl
        res["kl_div_p99"] = p99_kl

    return res
