"""LongMemEval Workload (Section 3.2 and 3.3.2 in proposal.md).

Evaluates Long-Term Memory Adherence across 5 core conversational capabilities:
1. Fact Recall
2. Cross-Session Reasoning
3. Information Update
4. Relative Timeframes
5. Unanswerable Question Refusal
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

from src.core.metrics import calc_percentile
from src.core.quality import (
    DEFAULT_REFUSAL_TRIGGERS,
    extract_step_logprobs,
    score_longmemeval_dataset,
    score_longmemeval_sample,
)
from src.core.workload import Workload, WorkloadOutput, extract_request_timings

BENCHMARK_DATASET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "longmemeval", "longmemeval_benchmark.json")
)
FALLBACK_DATASET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "longmemeval_subset.json")
)
DEFAULT_DATASET_PATH = (
    BENCHMARK_DATASET_PATH if os.path.exists(BENCHMARK_DATASET_PATH) else FALLBACK_DATASET_PATH
)

REFUSAL_TRIGGERS = DEFAULT_REFUSAL_TRIGGERS


def classify_length_tier(words: int) -> tuple[int, str]:
    """Classify context word count into approximate token count and length tier."""
    approx_tokens = int(words * 1.35)
    if approx_tokens <= 2500:
        return approx_tokens, "short"
    elif approx_tokens <= 8000:
        return approx_tokens, "medium"
    else:
        return approx_tokens, "long"


def _compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def format_longmem_prompt(context: str, question: str, tokenizer: Any = None) -> str:
    """Format a LongMemEval conversation + question using the model's chat template when available.

    Keeps the entire conversation history + question inside a single `user` message so
    `[Session Break]` markers remain intact for `diagnostic.py`, while avoiding raw
    non-chat prompting that triggers boilerplate refusals on Instruct models.
    """
    user_content = (
        "Answer the question concisely based on the following conversation history. "
        "If the information is not mentioned in the conversation, say that it was not mentioned.\n\n"
        f"{context}\n\n"
        f"Question: {question}"
    )

    chat_tpl = getattr(tokenizer, "chat_template", None) if tokenizer is not None else None
    if isinstance(chat_tpl, str) and chat_tpl and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": user_content}]
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if isinstance(rendered, str) and rendered:
                return rendered
        except TypeError:
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if isinstance(rendered, str) and rendered:
                    return rendered
            except Exception:
                pass
        except Exception:
            pass

    return f"{user_content}\nAnswer:"


class LongMemEvalWorkload(Workload):
    """Workload that executes multi-turn conversations from the LongMemEval subset
    and evaluates category-level and overall memory adherence accuracy.
    """

    def __init__(
        self,
        dataset_path: str | None = None,
        samples_per_task: int | None = None,
        max_samples: int | None = None,
        seed: int | None = None,
        length_tier: str | None = None,
        select_by_length: bool = False,
        eval_reference: str | None = None,
    ):
        import random

        if dataset_path is None:
            dataset_path = (
                BENCHMARK_DATASET_PATH if os.path.exists(BENCHMARK_DATASET_PATH) else FALLBACK_DATASET_PATH
            )
        self.dataset_path = dataset_path
        self.dataset_sha256 = _compute_file_sha256(dataset_path) if os.path.exists(dataset_path) else ""
        self.samples_per_task = samples_per_task
        self.max_samples = max_samples
        self.seed = seed
        self.length_tier = length_tier if length_tier != "all" else None
        self.select_by_length = select_by_length or (length_tier == "auto")
        self.eval_reference = eval_reference

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            all_samples: list[dict[str, Any]] = json.load(f)

        # Ensure all samples have length metadata
        for s in all_samples:
            if "word_count" not in s or "approx_tokens" not in s or "length_tier" not in s:
                words = len(s.get("context", "").split())
                approx_tok, tier = classify_length_tier(words)
                s["word_count"] = words
                s["approx_tokens"] = approx_tok
                s["length_tier"] = tier

        self.all_samples = all_samples

        # Filter by static length_tier if explicitly specified and not auto
        if self.length_tier and self.length_tier not in ("auto", "all"):
            filtered_samples = [s for s in all_samples if s.get("length_tier") == self.length_tier]
            if not filtered_samples:
                raise ValueError(
                    f"Requested length_tier='{self.length_tier}', but dataset '{self.dataset_path}' has 0 matching samples. "
                    "Run 'uv run python scripts/download_datasets.py' to stage the full benchmark dataset."
                )
        else:
            filtered_samples = all_samples

        if self.select_by_length:
            has_long_samples = any(s.get("approx_tokens", 0) >= 7000 for s in all_samples)
            if not has_long_samples:
                raise ValueError(
                    f"select_by_length=True requires multi-tier benchmark samples (up to 16k context, >= 7000 tokens), "
                    f"but dataset '{self.dataset_path}' only contains short samples. "
                    "Run 'uv run python scripts/download_datasets.py' to stage the full benchmark dataset."
                )

        effective_per_task = samples_per_task
        if effective_per_task is None and max_samples is not None and max_samples > 0:
            effective_per_task = max(1, max_samples // 5)

        if effective_per_task is not None and effective_per_task > 0:
            tasks: dict[str, list[dict[str, Any]]] = {}
            for s in filtered_samples:
                tasks.setdefault(s.get("task_type", "unknown"), []).append(s)

            rng = random.Random(seed) if seed is not None else random
            selected: list[dict[str, Any]] = []
            for task_type in sorted(tasks.keys()):
                category_samples = tasks[task_type]
                n = min(effective_per_task, len(category_samples))
                selected.extend(rng.sample(category_samples, n))
            self.samples = selected
        else:
            self.samples = filtered_samples

    def get_active_samples(self, context_length: int) -> list[dict[str, Any]]:
        """Resolve active samples for the current context length."""
        if not self.select_by_length and self.length_tier != "auto":
            return self.samples

        # Select samples that naturally fit the target context window without truncation
        import random
        rng = random.Random(self.seed) if self.seed is not None else random

        if context_length <= 1024:
            candidates = [s for s in self.all_samples if s.get("approx_tokens", 0) <= 1200]
        elif context_length <= 2048:
            candidates = [s for s in self.all_samples if s.get("approx_tokens", 0) <= 2500]
        elif context_length <= 4096:
            candidates = [s for s in self.all_samples if 2000 <= s.get("approx_tokens", 0) <= 6500]
        elif context_length <= 8192:
            candidates = [s for s in self.all_samples if 4000 <= s.get("approx_tokens", 0) <= 10000]
        else:  # 16384+
            candidates = [s for s in self.all_samples if s.get("approx_tokens", 0) >= 7000]

        if not candidates:
            raise ValueError(
                f"No matching samples found for context_length={context_length} in '{self.dataset_path}'. "
                "Ensure the benchmark dataset is staged via 'uv run python scripts/download_datasets.py'."
            )

        effective_per_task = self.samples_per_task
        if effective_per_task is None and self.max_samples is not None and self.max_samples > 0:
            effective_per_task = max(1, self.max_samples // 5)

        if effective_per_task is not None and effective_per_task > 0:
            tasks: dict[str, list[dict[str, Any]]] = {}
            for s in candidates:
                tasks.setdefault(s.get("task_type", "unknown"), []).append(s)
            selected: list[dict[str, Any]] = []
            for task_type in sorted(tasks.keys()):
                cat_samples = tasks[task_type]
                n = min(effective_per_task, len(cat_samples))
                selected.extend(rng.sample(cat_samples, n))
            return selected
        return candidates

    def run(self, llm, context_length: int, max_tokens: int) -> WorkloadOutput:
        try:
            from vllm import SamplingParams
            sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=5)
        except ImportError:
            sampling_params = None

        from src.core.workload import _get_tokenizer

        active_samples = self.get_active_samples(context_length)
        tok = _get_tokenizer(llm)

        prompts = [
            format_longmem_prompt(s.get("context", ""), s["question"], tokenizer=tok)
            for s in active_samples
        ]

        ref_correctness_by_id: dict[str, bool] = {}
        if self.eval_reference and os.path.exists(self.eval_reference):
            try:
                from src.core.quality import load_quality_reference
                ref_data = load_quality_reference(self.eval_reference)
                ctx_ref = ref_data.get("contexts", {}).get(str(context_length), {})
                ref_correctness_by_id = ctx_ref.get("sample_correctness", {})
            except Exception:
                ref_correctness_by_id = {}

        diagnostic_records = []
        diag_by_sample_id: dict[str, dict[str, Any]] = {}

        def _post_hoc_diag_hook(cache: Any, req_idx: int, req_out: Any) -> dict[str, Any] | None:
            if req_idx < len(active_samples):
                sample = active_samples[req_idx]
                sid = sample.get("id", f"sample_{req_idx}")
                pred_text = req_out.outputs[0].text.strip()
                from src.core.quality import is_ans_or_unans, score_longmemeval_sample
                is_correct = score_longmemeval_sample(
                    pred_text,
                    sample.get("expected_answer", ""),
                    is_unanswerable=is_ans_or_unans(sample),
                    refusal_triggers=REFUSAL_TRIGGERS,
                )
                baseline_correct = ref_correctness_by_id.get(
                    sid, sample.get("baseline_correct", None)
                )
                from src.core.diagnostic import inspect_sample_diagnostic
                diag = inspect_sample_diagnostic(
                    cache=cache,
                    prompt_text=req_out.prompt,
                    prompt_token_ids=req_out.prompt_token_ids,
                    sample_meta=sample,
                    is_correct=is_correct,
                    baseline_correct=baseline_correct,
                    tokenizer=tok,
                )
                diagnostic_records.append(diag)
                diag_dict = diag.to_dict()
                diag_by_sample_id[sid] = {
                    "overall_retention_ratio": diag_dict.get("overall_retention_ratio"),
                    "overall_effective_bitwidth": diag_dict.get("overall_effective_bitwidth"),
                    "min_session_effective_bitwidth": diag_dict.get("min_session_effective_bitwidth"),
                    "most_degraded_session_idx": diag_dict.get("most_degraded_session_idx"),
                    "num_sessions": len(diag_dict.get("session_token_counts", [])),
                }
                return diag_dict
            return None

        generate_kwargs: dict[str, Any] = {}
        import inspect
        gen_params = inspect.signature(llm.generate).parameters
        if "post_hoc_hook" in gen_params:
            generate_kwargs["post_hoc_hook"] = _post_hoc_diag_hook
        if "max_tokens" in gen_params:
            generate_kwargs["max_tokens"] = max_tokens

        start_wall = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, **generate_kwargs)
        end_wall = time.perf_counter()

        predictions: list[str] = []
        total_prompt_tok = 0
        total_gen_tok = 0
        ttfts: list[float] = []
        decodes: list[float] = []
        tpots: list[float] = []
        all_token_ids: list[int] = []
        all_step_logprobs: list[dict[int, float]] = []
        sample_outputs: dict[str, dict[str, Any]] = {}
        sample_records: list[dict[str, Any]] = []

        total_wall = end_wall - start_wall

        for idx, (sample, out) in enumerate(zip(active_samples, outputs)):
            sid = sample.get("id", f"sample_{idx}")
            comp = out.outputs[0]
            gen_text = comp.text.strip()
            predictions.append(gen_text)
            p_len = len(out.prompt_token_ids)
            total_prompt_tok += p_len
            gen_len = len(comp.token_ids)
            total_gen_tok += gen_len

            step_lps = extract_step_logprobs(out)
            all_token_ids.extend(comp.token_ids)
            all_step_logprobs.extend(step_lps)
            sample_outputs[sid] = {
                "token_ids": list(comp.token_ids),
                "step_logprobs": step_lps,
            }

            raw_finish = getattr(comp, "finish_reason", None)
            finish_reason = raw_finish if isinstance(raw_finish, str) else "unknown"

            rec: dict[str, Any] = {
                "id": sid,
                "task_type": sample.get("task_type", "unknown"),
                "unanswerable": bool(sample.get("unanswerable", False) or sample.get("task_type") == "unanswerable_refusal"),
                "prompt_tokens": p_len,
                "generated_tokens": gen_len,
                "finish_reason": finish_reason,
                "question": sample.get("question", ""),
                "expected_answer": sample.get("expected_answer", ""),
                "prediction": gen_text,
            }
            if sid in diag_by_sample_id:
                rec["diagnostic"] = diag_by_sample_id[sid]
            sample_records.append(rec)

            t, d, _ = extract_request_timings(out, total_wall, gen_len)
            ttfts.append(round(t, 4))
            decodes.append(round(d, 4))
            decode_tok = gen_len - 1 if gen_len > 1 else gen_len
            if d > 0 and decode_tok > 0:
                tpots.append(round(d / decode_tok * 1000.0, 2))

        # Evaluate quality metrics via unified quality scorer (fast heuristic fallback;
        # canonical semantic grading is performed offline over `sample_records`).
        eval_summary = score_longmemeval_dataset(active_samples, predictions, refusal_triggers=REFUSAL_TRIGGERS)
        eval_summary["dataset_sha256"] = self.dataset_sha256
        _tpl = getattr(tok, "chat_template", None)
        eval_summary["prompt_format"] = "chat_template" if (isinstance(_tpl, str) and bool(_tpl)) else "raw_qa"
        if self.samples_per_task is not None:
            eval_summary["longmem_samples_per_task"] = self.samples_per_task
        if self.seed is not None:
            eval_summary["longmem_seed"] = self.seed
        eval_summary["longmem_length_tier"] = self.length_tier or "all"
        eval_summary["longmem_select_by_length"] = self.select_by_length
        eval_summary["longmem_active_samples_count"] = len(active_samples)

        from src.core.quality import is_ans_or_unans

        # Calculate per-length-tier breakdown if multiple tiers are present
        tier_scores: dict[str, list[bool]] = {}
        for sample, pred in zip(active_samples, predictions):
            tier = sample.get("length_tier", "unknown")
            is_correct = score_longmemeval_sample(
                pred,
                sample.get("expected_answer", ""),
                is_unanswerable=is_ans_or_unans(sample),
                refusal_triggers=REFUSAL_TRIGGERS,
            )
            tier_scores.setdefault(tier, []).append(is_correct)
        eval_summary["tier_accuracy"] = {
            t: round(sum(scores) / len(scores), 4) for t, scores in sorted(tier_scores.items()) if scores
        }
        eval_summary["sample_correctness"] = {
            sample.get("id", f"sample_{idx}"): score_longmemeval_sample(
                pred,
                sample.get("expected_answer", ""),
                is_unanswerable=is_ans_or_unans(sample),
                refusal_triggers=REFUSAL_TRIGGERS,
            )
            for idx, (sample, pred) in enumerate(zip(active_samples, predictions))
        }
        eval_summary["sample_predictions"] = {
            sample.get("id", f"sample_{idx}"): pred
            for idx, (sample, pred) in enumerate(zip(active_samples, predictions))
        }
        eval_summary["sample_records"] = sample_records

        if diagnostic_records:
            from src.core.diagnostic import summarize_diagnostic_results
            diag_summary = summarize_diagnostic_results(diagnostic_records)
            eval_summary.update(diag_summary)

        total_samples = len(active_samples)
        mean_ttft = sum(ttfts) / len(ttfts) if ttfts else (total_wall / total_samples if total_samples > 0 else 0.0)
        p95_ttft = calc_percentile(ttfts, 95.0) if ttfts else mean_ttft
        total_decode = max(decodes) if decodes else max(0.0, total_wall - mean_ttft)

        decode_tokens = total_gen_tok - total_samples if total_gen_tok > total_samples else total_gen_tok
        fallback_tpot = (total_decode / decode_tokens * 1000.0) if (decode_tokens > 0 and total_decode > 0) else 0.0
        mean_tpot = sum(tpots) / len(tpots) if tpots else fallback_tpot
        p95_tpot = calc_percentile(tpots, 95.0) if tpots else round(mean_tpot, 2)

        out = WorkloadOutput(
            prompt_tokens=total_prompt_tok,
            generated_tokens=total_gen_tok,
            ttft_sec=round(mean_ttft, 4),
            decode_time_sec=round(total_decode, 4),
            total_time_sec=round(total_wall, 4),
            tpot_mean_ms=round(mean_tpot, 2),
            tpot_p95_ms=round(p95_tpot, 2),
            ttft_p95_sec=round(p95_ttft, 4),
            inter_token_latencies_ms=tpots,
            request_ttfts_sec=ttfts,
            generated_token_ids=all_token_ids,
            step_logprobs=all_step_logprobs,
            sample_outputs=sample_outputs,
            extra_summary=eval_summary,
        )
        return out

