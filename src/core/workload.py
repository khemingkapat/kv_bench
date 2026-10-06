"""A Workload defines *what gets sent to the engine and how it's timed* --
kept separate from Technique so several techniques can share the same
request pattern (e.g. every prefix-reuse method needs a cold/warm pair)
without copy-pasting the request logic into every technique file.

Add new workloads here as your evaluation needs grow -- e.g. a future
`NeedleInHaystack(Workload)` that buries a fact in the context and scores
retrieval accuracy would slot in exactly like SyntheticFiller/ColdWarm do,
and its score would flow into ResultEntry.extra via Technique.extra_metrics().
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from .metrics import calc_percentile

FILLER_SENTENCE = "The quick brown fox jumps over the lazy dog. "


@dataclass
class WorkloadOutput:
    prompt_tokens: int
    generated_tokens: int
    ttft_sec: float
    decode_time_sec: float
    total_time_sec: float
    # Cold-run metrics -- populated only by ColdWarm; zero for SyntheticFiller.
    # Having them on the same type means ResultEntry.cold_ttft_sec is always
    # present (just 0.0 for single-shot workloads), so CSV columns are stable.
    cold_ttft_sec: float = 0.0
    cold_total_sec: float = 0.0

    # Latency distributions & percentiles
    tpot_mean_ms: float = 0.0
    tpot_p95_ms: float = 0.0
    ttft_p95_sec: float = 0.0
    inter_token_latencies_ms: list[float] = field(default_factory=list)
    request_ttfts_sec: list[float] = field(default_factory=list)


class Workload(ABC):
    @abstractmethod
    def run(self, llm, context_length: int, max_tokens: int) -> WorkloadOutput:
        raise NotImplementedError


def _make_prompt(context_length: int) -> str:
    return FILLER_SENTENCE * max(1, context_length // 10)


def _timed_generate(llm, prompt: str, max_tokens: int) -> WorkloadOutput:
    from vllm import SamplingParams

    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    start_wall = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params)
    end_wall = time.perf_counter()

    out = outputs[0]
    prompt_tokens = len(out.prompt_token_ids)
    gen_tokens = len(out.outputs[0].token_ids)

    metrics = getattr(out, "metrics", None)
    has_engine_metrics = metrics and all(
        getattr(metrics, f, None) for f in ("first_token_time", "arrival_time", "finished_time")
    )
    if has_engine_metrics:
        ttft = metrics.first_token_time - metrics.arrival_time
        decode = metrics.finished_time - metrics.first_token_time
        total = metrics.finished_time - metrics.arrival_time
    else:
        total = end_wall - start_wall
        ttft = total
        decode = 0.0

    decode_tokens = gen_tokens - 1 if gen_tokens > 1 else gen_tokens
    tpot_mean = (decode / decode_tokens * 1000.0) if (decode > 0 and decode_tokens > 0) else 0.0

    return WorkloadOutput(
        prompt_tokens=prompt_tokens,
        generated_tokens=gen_tokens,
        ttft_sec=round(ttft, 4),
        decode_time_sec=round(decode, 4),
        total_time_sec=round(total, 4),
        tpot_mean_ms=round(tpot_mean, 2),
        tpot_p95_ms=round(tpot_mean, 2),
        ttft_p95_sec=round(ttft, 4),
        request_ttfts_sec=[round(ttft, 4)],
    )


class SyntheticFiller(Workload):
    """One cold prompt of the target length, one short generation. This is
    what baseline/flexgen-style techniques need -- pure memory/latency
    measurement, no reuse involved.

    cold_ttft_sec and cold_total_sec in the output will be 0.0 for this
    workload since there is no separate cold vs. warm distinction.
    """

    def run(self, llm, context_length: int, max_tokens: int) -> WorkloadOutput:
        return _timed_generate(llm, _make_prompt(context_length), max_tokens)


class ColdWarm(Workload):
    """Two-phase prefix-reuse workload for chat-style scenarios: a 'cold'
    request that populates the KV cache (simulating the first message in a
    conversation or a shared system prompt), then a 'warm' request reusing
    that prefix with a new instruction (simulating subsequent turns).

    This models the SME/team chat use-case: 10-15 people sharing context
    (e.g. a long document or thread) where repeated queries all reuse the
    same prefix. A warm TTFT that is meaningfully lower than cold TTFT is
    direct evidence that prefix reuse is working.

    Both cold and warm timings are returned so the harness can surface the
    delta as a first-class reported field rather than a derived annotation.
    """

    def run(self, llm, context_length: int, max_tokens: int) -> WorkloadOutput:
        prefix = _make_prompt(context_length)
        cold_prompt = prefix + "\n\nInstruction: Summarize."
        warm_prompt = prefix + "\n\nInstruction: What is the theme?"

        cold_out = _timed_generate(llm, cold_prompt, max_tokens)
        warm_out = _timed_generate(llm, warm_prompt, max_tokens)

        # Surface cold timing on the warm result so the harness gets both in
        # one return value and can compute the delta without a second output type.
        warm_out.cold_ttft_sec = cold_out.ttft_sec
        warm_out.cold_total_sec = cold_out.total_time_sec
        return warm_out


class ConcurrentChatWorkload(Workload):
    """Multi-user concurrent chat workload simulating 10-15 SME staff members
    sharing a single GPU (Section 1.3.4 & 3.3.1 in proposal.md).

    Issues N concurrent requests against a shared prefix/document, exercising
    continuous batching in vLLM. Measures individual TTFT distributions,
    P95 TTFT tail latency, and aggregated decode throughput.
    """

    def __init__(self, num_users: int = 10):
        self.num_users = num_users

    def run(self, llm, context_length: int, max_tokens: int) -> WorkloadOutput:
        from vllm import SamplingParams

        prefix = _make_prompt(context_length)
        queries = [
            f"User {i+1} inquiry: Answer query {i+1} concerning the document."
            for i in range(self.num_users)
        ]
        prompts = [f"{prefix}\n\n{q}" for q in queries]

        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

        start_wall = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        end_wall = time.perf_counter()

        ttfts = []
        decodes = []
        tot_prompt_tok = 0
        tot_gen_tok = 0

        for out in outputs:
            tot_prompt_tok += len(out.prompt_token_ids)
            gen_len = len(out.outputs[0].token_ids)
            tot_gen_tok += gen_len

            metrics = getattr(out, "metrics", None)
            has_metrics = metrics and all(
                getattr(metrics, f, None)
                for f in ("first_token_time", "arrival_time", "finished_time")
            )
            if has_metrics:
                t = metrics.first_token_time - metrics.arrival_time
                d = metrics.finished_time - metrics.first_token_time
            else:
                t = (end_wall - start_wall) / len(outputs)
                d = 0.0
            ttfts.append(round(t, 4))
            decodes.append(round(d, 4))

        mean_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
        p95_ttft = calc_percentile(ttfts, 95.0)

        total_wall = end_wall - start_wall
        total_decode = max(decodes) if decodes else 0.0

        decode_toks = tot_gen_tok - len(outputs) if tot_gen_tok > len(outputs) else tot_gen_tok
        tpot_mean = (total_decode / decode_toks * 1000.0) if (total_decode > 0 and decode_toks > 0) else 0.0

        return WorkloadOutput(
            prompt_tokens=tot_prompt_tok,
            generated_tokens=tot_gen_tok,
            ttft_sec=round(mean_ttft, 4),
            decode_time_sec=round(total_decode, 4),
            total_time_sec=round(total_wall, 4),
            tpot_mean_ms=round(tpot_mean, 2),
            tpot_p95_ms=round(tpot_mean, 2),
            ttft_p95_sec=round(p95_ttft, 4),
        )


from src.workloads.memory_adherence import LongMemEvalWorkload  # noqa: E402
