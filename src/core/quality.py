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


def _normalize_logprobs(logprob_dict: dict[Any, float]) -> dict[int, float]:
    """Convert logprobs to a normalized probability distribution over known tokens."""
    if not logprob_dict:
        return {}
    # Explicitly normalize keys to integer token IDs to prevent str vs int key mismatches
    # when loading reference files serialized to JSON.
    int_dict: dict[int, float] = {}
    for tok, lp in logprob_dict.items():
        try:
            int_dict[int(tok)] = float(lp)
        except (ValueError, TypeError):
            continue
    if not int_dict:
        return {}
    # Use log-sum-exp trick for numerical stability
    max_lp = max(int_dict.values())
    exps = {tok: math.exp(lp - max_lp) for tok, lp in int_dict.items()}
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


# ---------------------------------------------------------------------------
# Reference Extraction & Persistence Helpers
# ---------------------------------------------------------------------------

def extract_step_logprobs(output: Any) -> list[dict[int, float]]:
    """Extract step logprobs from vLLM output (RequestOutput or CompletionOutput).

    Supports vLLM Logprob objects, raw dicts, or nested outputs.
    Returns a list (one entry per generation step) of {token_id: float_logprob}.
    """
    raw_logprobs = None
    if hasattr(output, "outputs") and len(output.outputs) > 0:
        raw_logprobs = getattr(output.outputs[0], "logprobs", None)
    elif hasattr(output, "logprobs"):
        raw_logprobs = getattr(output, "logprobs", None)
    elif isinstance(output, dict):
        raw_logprobs = output.get("logprobs")

    if not raw_logprobs or not isinstance(raw_logprobs, list):
        return []

    step_logprobs: list[dict[int, float]] = []
    for step in raw_logprobs:
        if not step:
            continue
        step_dict: dict[int, float] = {}
        if isinstance(step, dict):
            for tok_id, val in step.items():
                lp = getattr(val, "logprob", val)
                try:
                    step_dict[int(tok_id)] = float(lp)
                except (ValueError, TypeError):
                    continue
        step_logprobs.append(step_dict)

    return step_logprobs


def save_quality_reference(path: str, reference_data: dict[str, Any]) -> None:
    """Save reference token and logit artifacts to JSON for fidelity comparisons."""
    import json
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reference_data, f, indent=2)


def load_quality_reference(path: str) -> dict[str, Any]:
    """Load reference token and logit artifacts from JSON."""
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def _normalize_steps(steps: Any) -> Any:
        if not isinstance(steps, list):
            return steps
        normalized = []
        for step in steps:
            if isinstance(step, dict):
                normalized.append({int(k): float(v) for k, v in step.items()})
            else:
                normalized.append(step)
        return normalized

    if "contexts" in data and isinstance(data["contexts"], dict):
        for ctx_entry in data["contexts"].values():
            if isinstance(ctx_entry, dict) and "step_logprobs" in ctx_entry:
                ctx_entry["step_logprobs"] = _normalize_steps(ctx_entry["step_logprobs"])

    if "step_logprobs" in data:
        data["step_logprobs"] = _normalize_steps(data["step_logprobs"])

    return data


def evaluate_fidelity_against_reference(
    reference_data: dict[str, Any],
    target_ctx: int,
    target_token_ids: list[int],
    target_step_logprobs: list[dict[int, float]] | None = None,
) -> dict[str, Any]:
    """Compare a target run against saved reference data for a matching context length."""
    ctx_key = str(target_ctx)
    contexts_map = reference_data.get("contexts", {})
    ref_entry = contexts_map.get(ctx_key) or reference_data.get(ctx_key)

    if not ref_entry:
        return {}

    base_tokens = ref_entry.get("token_ids", [])
    base_lps = ref_entry.get("step_logprobs", None)
    return evaluate_fidelity(base_tokens, target_token_ids, base_lps, target_step_logprobs)


# ---------------------------------------------------------------------------
# LongMemEval Memory Adherence Scoring
# ---------------------------------------------------------------------------

DEFAULT_REFUSAL_TRIGGERS: list[str] = [
    "not mentioned",
    "don't know",
    "do not know",
    "cannot answer",
    "can't answer",
    "no information",
    "unknown",
    "not provided",
    "unable to find",
]

NUMBER_WORDS: dict[str, str] = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten"
}
WORD_TO_NUM: dict[str, str] = {v: k for k, v in NUMBER_WORDS.items()}


def extract_candidate_answers(expected: str) -> list[str]:
    """Extract candidate acceptable answer variants from expected answer string."""
    import re

    raw = expected.strip()
    candidates: set[str] = set()

    # 1. Clean primary
    primary = re.sub(r"[\.\,\!\?]+$", "", raw).strip().lower()
    if primary:
        candidates.add(primary)

    # 2. Extract '(or ...)' or alternative parentheticals
    match_or = re.search(r"\((?:or\s+)?([^\)]+)\)", raw, re.IGNORECASE)
    if match_or:
        inside = match_or.group(1).strip().lower()
        if inside:
            candidates.add(inside)
        outside = re.sub(r"\([^\)]+\)", "", raw).strip()
        outside = re.sub(r"[\.\,\!\?]+$", "", outside).strip().lower()
        if outside:
            candidates.add(outside)

    # 3. Extract '... is also acceptable'
    if "is also acceptable" in raw.lower():
        parts = re.split(r"[\.\;]\s*", raw)
        for p in parts:
            clean_p = re.sub(r"\(.*?\)", "", p)
            clean_p = re.sub(r"is also acceptable.*", "", clean_p, flags=re.IGNORECASE)
            clean_p = re.sub(r"[\.\,\!\?]+$", "", clean_p).strip().lower()
            if clean_p:
                candidates.add(clean_p)

    # 4. Number word vs digit equivalence
    for c in list(candidates):
        words = c.split()
        if len(words) == 1:
            if words[0] in NUMBER_WORDS:
                candidates.add(NUMBER_WORDS[words[0]])
            elif words[0] in WORD_TO_NUM:
                candidates.add(WORD_TO_NUM[words[0]])

    return [c for c in candidates if c]


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth strings."""
    import re
    from collections import Counter

    pred_tokens = re.findall(r"\w+", prediction.lower())
    gt_tokens = re.findall(r"\w+", ground_truth.lower())
    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return round(f1, 4)


def score_longmemeval_sample(
    gen_text: str,
    expected_answer: str,
    is_unanswerable: bool = False,
    refusal_triggers: list[str] | None = None,
) -> bool:
    """Evaluate whether a generated answer satisfies a LongMemEval sample."""
    import re

    cleaned = gen_text.strip().lower()
    triggers = refusal_triggers if refusal_triggers is not None else DEFAULT_REFUSAL_TRIGGERS

    if is_unanswerable:
        return any(trig in cleaned for trig in triggers)

    candidates = extract_candidate_answers(expected_answer)
    if not candidates:
        return expected_answer.strip().lower() in cleaned

    for cand in candidates:
        if cand in cleaned:
            return True
        if re.search(r"\b" + re.escape(cand) + r"\b", cleaned):
            return True

    return False


def score_longmemeval_dataset(
    samples: list[dict[str, Any]],
    predictions: list[str],
    refusal_triggers: list[str] | None = None,
) -> dict[str, Any]:
    """Compute overall and category-level memory adherence accuracy and F1 scores."""
    category_counts: dict[str, int] = {}
    category_correct: dict[str, int] = {}
    category_f1s: dict[str, list[float]] = {}
    total_correct = 0
    all_f1s: list[float] = []
    total_samples = len(samples)

    for sample, pred in zip(samples, predictions):
        task = sample.get("task_type", "unknown")
        expected = sample.get("expected_answer", "")
        unans = is_ans_or_unans(sample)

        is_corr = score_longmemeval_sample(pred, expected, is_unanswerable=unans, refusal_triggers=refusal_triggers)
        category_counts[task] = category_counts.get(task, 0) + 1
        if is_corr:
            category_correct[task] = category_correct.get(task, 0) + 1
            total_correct += 1

        f1 = compute_token_f1(pred, "not mentioned" if unans else expected)
        all_f1s.append(f1)
        category_f1s.setdefault(task, []).append(f1)

    overall_acc = round(total_correct / total_samples, 4) if total_samples > 0 else 0.0
    overall_f1 = round(sum(all_f1s) / len(all_f1s), 4) if all_f1s else 0.0

    summary: dict[str, Any] = {
        "longmem_acc_overall": overall_acc,
        "longmem_f1_overall": overall_f1,
        "longmem_total_samples": total_samples,
        "longmem_correct_samples": total_correct,
    }
    for task, count in category_counts.items():
        corr = category_correct.get(task, 0)
        summary[f"longmem_acc_{task}"] = round(corr / count, 4) if count > 0 else 0.0
        f1_list = category_f1s.get(task, [])
        summary[f"longmem_f1_{task}"] = round(sum(f1_list) / len(f1_list), 4) if f1_list else 0.0
        summary[f"longmem_count_{task}"] = count

    return summary


def is_ans_or_unans(sample: dict[str, Any]) -> bool:
    """Check unanswerable flag safely from sample."""
    return bool(sample.get("unanswerable", False) or sample.get("task_type") == "unanswerable_refusal")

