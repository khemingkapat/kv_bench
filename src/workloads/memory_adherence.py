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
from src.core.quality import DEFAULT_REFUSAL_TRIGGERS, score_longmemeval_dataset
from src.core.workload import Workload, WorkloadOutput

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


class LongMemEvalWorkload(Workload):
    """Workload that executes multi-turn conversations from the LongMemEval subset
    and evaluates category-level and overall memory adherence accuracy.
    """

    def __init__(self, dataset_path: str = DEFAULT_DATASET_PATH, max_samples: int | None = None):
        self.dataset_path = dataset_path
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            self.samples: list[dict[str, Any]] = json.load(f)
        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]

    def run(self, llm, context_length: int, max_tokens: int) -> WorkloadOutput:
        from vllm import SamplingParams

        target_words = max(30, int(context_length / 1.3))
        prompts = []
        for s in self.samples:
            words = s.get("context", "").split()
            if len(words) > target_words:
                ctx_text = " ".join(words[-target_words:])
                prefix = ""
            else:
                ctx_text = s.get("context", "")
                remaining = target_words - len(words)
                prefix = (_make_prompt(remaining) + "\n\n[Chat History Continued]\n") if remaining > 20 else ""

            prompts.append(f"{prefix}{ctx_text}\nUser: {s['question']}\nAssistant:")


        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

        start_wall = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        end_wall = time.perf_counter()

        predictions: list[str] = []
        total_prompt_tok = 0
        total_gen_tok = 0
        ttfts: list[float] = []
        tpots: list[float] = []

        for out in outputs:
            gen_text = out.outputs[0].text.strip()
            predictions.append(gen_text)
            total_prompt_tok += len(out.prompt_token_ids)
            gen_len = len(out.outputs[0].token_ids)
            total_gen_tok += gen_len

            metrics = getattr(out, "metrics", None)
            if metrics and hasattr(metrics, "first_token_time") and hasattr(metrics, "arrival_time"):
                sample_ttft = metrics.first_token_time - metrics.arrival_time
                ttfts.append(round(sample_ttft, 4))
                if hasattr(metrics, "finished_time"):
                    sample_decode = metrics.finished_time - metrics.first_token_time
                    decode_tok = gen_len - 1 if gen_len > 1 else gen_len
                    if sample_decode > 0 and decode_tok > 0:
                        tpots.append(round(sample_decode / decode_tok * 1000.0, 2))

        # Evaluate quality metrics via unified quality scorer
        eval_summary = score_longmemeval_dataset(self.samples, predictions, refusal_triggers=REFUSAL_TRIGGERS)

        total_wall = end_wall - start_wall
        total_samples = len(self.samples)
        mean_ttft = sum(ttfts) / len(ttfts) if ttfts else (total_wall / total_samples if total_samples > 0 else 0.0)
        p95_ttft = calc_percentile(ttfts, 95.0) if ttfts else mean_ttft
        mean_tpot = sum(tpots) / len(tpots) if tpots else 0.0
        p95_tpot = calc_percentile(tpots, 95.0) if tpots else mean_tpot

        out = WorkloadOutput(
            prompt_tokens=total_prompt_tok,
            generated_tokens=total_gen_tok,
            ttft_sec=round(mean_ttft, 4),
            decode_time_sec=round(total_wall - mean_ttft, 4),
            total_time_sec=round(total_wall, 4),
            tpot_mean_ms=round(mean_tpot, 2),
            tpot_p95_ms=round(p95_tpot, 2),
            ttft_p95_sec=round(p95_ttft, 4),
            inter_token_latencies_ms=tpots,
            request_ttfts_sec=ttfts,
        )
        # Store category accuracies on out for harness extraction
        out.extra_summary = eval_summary  # type: ignore[attr-defined]
        return out

