"""Unit tests for the LongMemEval harness overhaul (single-pass native length,
chat templating, multi-EOS stopping, per-sample KL alignment, BPE session parsing,
quantile dataset sampling, and decoupled offline grading).
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from scripts.download_datasets import select_quantile_samples
from scripts.longmem_grades import apply_grades, export_review_sheet
from src.core.diagnostic import parse_session_token_ranges
from src.core.pytorch_runner import PyTorchRunner
from src.core.quality import evaluate_fidelity_against_reference, load_quality_reference, save_quality_reference
from src.workloads.memory_adherence import LongMemEvalWorkload, format_longmem_prompt


class TestQuantileSampling(unittest.TestCase):
    def test_select_quantile_samples_evenly_spaced_and_deterministic(self):
        entries = [
            {"id": f"q{i}", "approx_tokens": i * 100, "word_count": i * 70}
            for i in range(1, 101)
        ]
        selected = select_quantile_samples(entries, 5)
        self.assertEqual(len(selected), 5)
        self.assertEqual([s["approx_tokens"] for s in selected], [100, 2600, 5100, 7500, 10000])
        # Deterministic regardless of input order
        selected_rev = select_quantile_samples(list(reversed(entries)), 5)
        self.assertEqual([s["id"] for s in selected], [s["id"] for s in selected_rev])


class TestChatTemplatingAndNativeLength(unittest.TestCase):
    def test_format_longmem_prompt_uses_chat_template_with_enable_thinking_guard(self):
        calls = []

        class _ThinkingTokenizer:
            chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
                calls.append((messages, tokenize, add_generation_prompt, enable_thinking))
                return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"

        out = format_longmem_prompt("User: Hi\nAssistant: Hello", "What did I say?", _ThinkingTokenizer())
        self.assertTrue(out.startswith("<|im_start|>user\n"))
        self.assertTrue(out.endswith("<|im_start|>assistant\n"))
        self.assertIn("Question: What did I say?", out)
        self.assertEqual(calls[0][3], False)

    def test_format_longmem_prompt_fallback_on_no_template_and_magicmock(self):
        class _BaseTok:
            chat_template = None

        out = format_longmem_prompt("ctx", "q?", _BaseTok())
        self.assertTrue(out.endswith("Question: q?\nAnswer:"))

        # MagicMock should not return a MagicMock string
        mock_tok = MagicMock()
        out_mock = format_longmem_prompt("ctx", "q?", mock_tok)
        self.assertIsInstance(out_mock, str)
        self.assertIn("Question: q?", out_mock)

    def test_workload_does_not_front_truncate_or_pad_pg19(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds_path = os.path.join(tmp, "ds.json")
            unique_prefix = "UNIQUE_SESSION_ONE_MARKER"
            long_ctx = unique_prefix + " " + ("word " * 2000)
            with open(ds_path, "w", encoding="utf-8") as f:
                json.dump([
                    {
                        "id": "s1",
                        "task_type": "cross_session_reasoning",
                        "context": long_ctx,
                        "question": "What was the marker?",
                        "expected_answer": unique_prefix,
                        "unanswerable": False,
                    }
                ], f)

            captured_prompts = []

            class _FakeLLM:
                def get_tokenizer(self):
                    return None

                def generate(self, prompts, sampling_params=None, max_tokens=64):
                    captured_prompts.extend(prompts)
                    comp = SimpleNamespace(text=unique_prefix, token_ids=[10, 11], logprobs=[{10: -0.1}, {11: -0.2}], finish_reason="stop")
                    return [SimpleNamespace(prompt=prompts[0], prompt_token_ids=[1, 2, 3], outputs=[comp], metrics=None)]

            wl = LongMemEvalWorkload(dataset_path=ds_path)
            # Even with a small context_length arg, native context must NEVER be front-truncated
            out = wl.run(_FakeLLM(), context_length=512, max_tokens=128)
            self.assertIn(unique_prefix, captured_prompts[0])
            self.assertNotIn("[Chat History Continued]", captured_prompts[0])
            self.assertIn("sample_records", out.extra_summary)
            rec = out.extra_summary["sample_records"][0]
            self.assertEqual(rec["id"], "s1")
            self.assertEqual(rec["prompt_tokens"], 3)
            self.assertEqual(rec["finish_reason"], "stop")
            self.assertEqual(rec["prediction"], unique_prefix)
            self.assertIn("s1", out.sample_outputs)
            self.assertTrue(len(out.extra_summary["dataset_sha256"]) == 64)


class TestPerSampleFidelityAlignment(unittest.TestCase):
    def test_per_sample_fidelity_immune_to_variable_length_phase_shift(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref_path = os.path.join(tmp, "ref.json")
            save_quality_reference(
                ref_path,
                {
                    "model": "qwen3-4b",
                    "technique": "baseline",
                    "contexts": {
                        "32768": {
                            "token_ids": [1, 2, 10, 20, 30],
                            "step_logprobs": [{1: -0.1}, {2: -0.1}, {10: -0.1}, {20: -0.1}, {30: -0.1}],
                            "samples": {
                                "q1": {"token_ids": [1, 2], "step_logprobs": [{1: -0.1}, {2: -0.1}]},
                                "q2": {"token_ids": [10, 20, 30], "step_logprobs": [{10: -0.1}, {20: -0.1}, {30: -0.1}]},
                            },
                        }
                    },
                },
            )
            loaded = load_quality_reference(ref_path)
            # Verify int key normalization inside samples
            self.assertIn(10, loaded["contexts"]["32768"]["samples"]["q2"]["step_logprobs"][0])

            # Target q1 generated 1 extra token [1, 2, 99], while q2 is IDENTICAL [10, 20, 30]
            target_samples = {
                "q1": {"token_ids": [1, 2, 99], "step_logprobs": [{1: -0.1}, {2: -0.1}, {99: -0.5}]},
                "q2": {"token_ids": [10, 20, 30], "step_logprobs": [{10: -0.1}, {20: -0.1}, {30: -0.1}]},
            }
            res = evaluate_fidelity_against_reference(
                loaded,
                32768,
                target_token_ids=[1, 2, 99, 10, 20, 30],
                target_step_logprobs=[{1: -0.1}, {2: -0.1}, {99: -0.5}, {10: -0.1}, {20: -0.1}, {30: -0.1}],
                target_samples=target_samples,
            )
            # All aligned positions (2 in q1 + 3 in q2 = 5) match 100% with 0.0 KL!
            self.assertEqual(res["fidelity_alignment"], "per_sample")
            self.assertEqual(res["top1_agreement"], 1.0)
            self.assertAlmostEqual(res["kl_div_mean"], 0.0, places=5)
            self.assertEqual(res["sample_fidelity"]["q1"]["top1"], 1.0)
            self.assertEqual(res["sample_fidelity"]["q2"]["top1"], 1.0)
            self.assertAlmostEqual(res["sample_fidelity"]["q2"]["kl_mean"], 0.0, places=5)


class TestPyTorchRunnerMultiEOS(unittest.TestCase):
    def test_step0_and_multi_eos_stopping(self):
        class _FakeTok:
            eos_token_id = 151645
            bos_token = None

            def encode(self, prompt, return_tensors="pt"):
                return torch.tensor([[1, 2, 3]], dtype=torch.long)

            def decode(self, ids, skip_special_tokens=True):
                return "decoded"

        class _FakeModel:
            def __init__(self, seq):
                self.seq = list(seq)
                self.idx = 0
                self.generation_config = SimpleNamespace(eos_token_id=[151645, 151643])

            def __call__(self, input_ids, past_key_values=None, position_ids=None, use_cache=True):
                vocab = 151650
                logits = torch.full((1, input_ids.shape[1], vocab), -10.0)
                tok = self.seq[min(self.idx, len(self.seq) - 1)]
                self.idx += 1
                logits[0, -1, tok] = 10.0
                return SimpleNamespace(logits=logits)

        runner = PyTorchRunner(model_name_or_path="mock", device="cpu", dry_run=True)
        runner.dry_run = False
        runner.tokenizer = _FakeTok()

        # Case 1: Step-0 emits secondary EOS (151643) -> stops immediately at 1 token
        runner.model = _FakeModel([151643, 5, 6])
        outs1 = runner.generate(["hi"], max_tokens=16)
        self.assertEqual(outs1[0].outputs[0].token_ids, [151643])
        self.assertEqual(outs1[0].outputs[0].finish_reason, "stop")

        # Case 2: Emits [42, 151643] -> stops at 2 tokens with finish_reason="stop"
        runner.model = _FakeModel([42, 151643, 99, 100])
        outs2 = runner.generate(["hi"], max_tokens=16)
        self.assertEqual(outs2[0].outputs[0].token_ids, [42, 151643])
        self.assertEqual(outs2[0].outputs[0].finish_reason, "stop")


class TestDiagnosticBPESessionOffsets(unittest.TestCase):
    def test_parse_session_token_ranges_handles_merged_bpe_delimiter(self):
        prompt = "Session 1 text.\n\n[Session Break]\n\nSession 2 text."

        class _OffsetBPETok:
            # Simulates Qwen3 BPE merging ']\n\n' into a single token
            def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
                chunks = ["Session 1 text.\n\n", "[Session", " Break", "]\n\n", "Session 2 text."]
                offsets = []
                pos = 0
                for c in chunks:
                    offsets.append((pos, pos + len(c)))
                    pos += len(c)
                return {"input_ids": [10, 58, 15623, 2533, 20], "offset_mapping": offsets}

            def encode(self, s, add_special_tokens=False):
                # Raw encode of "[Session Break]" uses token 60 (']') instead of 2533 (']\n\n')
                return [58, 15623, 60]

        ranges = parse_session_token_ranges(prompt, [10, 58, 15623, 2533, 20], tokenizer=_OffsetBPETok())
        self.assertEqual(ranges, [(0, 1), (4, 5)])


class TestOfflineGradingUtility(unittest.TestCase):
    def test_export_and_apply_grades_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            res_path = os.path.join(tmp, "qserve_longmem.json")
            raw_payload = {
                "technique": "qserve_kv4",
                "results": [
                    {
                        "requested_ctx": 32768,
                        "extra": {
                            "dataset_sha256": "abc123",
                            "sample_records": [
                                {"id": "q1", "task_type": "fact_recall", "unanswerable": False, "prompt_tokens": 500, "generated_tokens": 12, "finish_reason": "stop", "question": "Q1?", "expected_answer": "A1", "prediction": "A1"},
                                {"id": "q2", "task_type": "information_update", "unanswerable": False, "prompt_tokens": 6000, "generated_tokens": 15, "finish_reason": "stop", "question": "Q2?", "expected_answer": "A2", "prediction": "wrong"},
                            ],
                        },
                    }
                ],
            }
            with open(res_path, "w", encoding="utf-8") as f:
                json.dump(raw_payload, f)

            csv_out = os.path.join(tmp, "review.csv")
            n = export_review_sheet([res_path], csv_out, fmt="csv")
            self.assertEqual(n, 2)
            with open(csv_out, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["qserve_kv4_prediction"], "A1")
            self.assertEqual(rows[1]["prompt_tokens"], "6000")

            grades_path = os.path.join(tmp, "qserve_grades.json")
            base_grades_path = os.path.join(tmp, "base_grades.json")
            with open(grades_path, "w", encoding="utf-8") as f:
                json.dump({"q1": {"correct": True, "reason": "exact", "grader": "human"}, "q2": {"correct": False, "reason": "wrong", "grader": "human"}}, f)
            with open(base_grades_path, "w", encoding="utf-8") as f:
                json.dump({"q1": {"correct": True}, "q2": {"correct": True}}, f)

            judged_out = os.path.join(tmp, "qserve_judged.json")
            summary = apply_grades(res_path, grades_path, judged_out, baseline_grades_path=base_grades_path)
            self.assertEqual(summary["correct_samples"], 1)
            self.assertEqual(summary["total_samples"], 2)
            self.assertEqual(summary["causal_verdict_counts"]["PASS_FIDELITY_MAINTAINED"], 1)
            self.assertEqual(summary["causal_verdict_counts"]["QUALITY_REGRESSION"], 1)


if __name__ == "__main__":
    unittest.main()
