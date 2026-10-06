"""LongMemEval Workload (Section 3.2 and 3.3.2 in proposal.md).

Evaluates Long-Term Memory Adherence across 5 core conversational capabilities:
1. Fact Recall
2. Cross-Session Reasoning
3. Information Update
4. Relative Timeframes
5. Unanswerable Question Refusal
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from src.core.metrics import calc_percentile
from src.core.workload import Workload, WorkloadOutput

DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "longmemeval_subset.json"
)

REFUSAL_TRIGGERS = [
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


class LongMemEvalWorkload(Workload):
    """Workload that executes multi-turn conversations from the LongMemEval subset
    and evaluates category-level and overall memory adherence accuracy.
    """

    def __init__(self, dataset_path: str = DEFAULT_DATASET_PATH):
        self.dataset_path = dataset_path
        with open(self.dataset_path, "r") as f:
            self.samples: list[dict[str, Any]] = json.load(f)

    def run(self, llm, context_length: int, max_tokens: int) -> WorkloadOutput:
        from vllm import SamplingParams

        from src.core.workload import _make_prompt

        prefix = ""
        if context_length > 256:
            prefix = _make_prompt(context_length - 150) + "\n\n[Chat History Continued]\n"

        prompts = [
            f"{prefix}{s['context']}\nUser: {s['question']}\nAssistant:" for s in self.samples
        ]

        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

        start_wall = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        end_wall = time.perf_counter()

        category_counts: dict[str, int] = {}
        category_correct: dict[str, int] = {}
        total_correct = 0
        total_prompt_tok = 0
        total_gen_tok = 0
        ttfts: list[float] = []

        for sample, out in zip(self.samples, outputs):
            task = sample["task_type"]
            expected = sample["expected_answer"].lower()
            is_unanswerable = sample.get("unanswerable", False)

            gen_text = out.outputs[0].text.strip().lower()
            total_prompt_tok += len(out.prompt_token_ids)
            total_gen_tok += len(out.outputs[0].token_ids)

            # Evaluate correctness
            if is_unanswerable:
                correct = any(trig in gen_text for trig in REFUSAL_TRIGGERS)
            else:
                correct = expected in gen_text

            category_counts[task] = category_counts.get(task, 0) + 1
            if correct:
                category_correct[task] = category_correct.get(task, 0) + 1
                total_correct += 1

            metrics = getattr(out, "metrics", None)
            if metrics and hasattr(metrics, "first_token_time") and hasattr(metrics, "arrival_time"):
                ttfts.append(round(metrics.first_token_time - metrics.arrival_time, 4))

        # Accuracies
        total_samples = len(self.samples)
        overall_acc = round(total_correct / total_samples, 4) if total_samples > 0 else 0.0

        eval_summary = {
            "longmem_acc_overall": overall_acc,
            "longmem_total_samples": total_samples,
        }
        for task, count in category_counts.items():
            corr = category_correct.get(task, 0)
            eval_summary[f"longmem_acc_{task}"] = round(corr / count, 4) if count > 0 else 0.0

        total_wall = end_wall - start_wall
        mean_ttft = sum(ttfts) / len(ttfts) if ttfts else (total_wall / total_samples if total_samples > 0 else 0.0)
        p95_ttft = calc_percentile(ttfts, 95.0) if ttfts else mean_ttft

        out = WorkloadOutput(
            prompt_tokens=total_prompt_tok,
            generated_tokens=total_gen_tok,
            ttft_sec=round(mean_ttft, 4),
            decode_time_sec=round(total_wall - mean_ttft, 4),
            total_time_sec=round(total_wall, 4),
            ttft_p95_sec=round(p95_ttft, 4),
            request_ttfts_sec=ttfts,
        )
        # Store category accuracies on out for harness extraction
        out.extra_summary = eval_summary  # type: ignore[attr-defined]
        return out
