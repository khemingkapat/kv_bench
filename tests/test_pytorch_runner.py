import unittest
import weakref
from types import SimpleNamespace

import torch

from src.core.pytorch_runner import (
    PyTorchCompletionOutput,
    PyTorchRequestOutput,
    PyTorchRunner,
)
from src.core.quality import extract_step_logprobs
from src.core.workload import SyntheticFiller, extract_request_timings


class TestPyTorchRunner(unittest.TestCase):
    def setUp(self):
        self.runner = PyTorchRunner(
            model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
            revision="cdbee75f17c01a7cc42f958dc650907174af0554",
            dry_run=True,
        )

    def test_dry_run_generation_structure(self):
        prompts = ["Hello world, this is a test prompt."]
        outputs = self.runner.generate(prompts, max_tokens=16)

        self.assertEqual(len(outputs), 1)
        req_out = outputs[0]
        self.assertIsInstance(req_out, PyTorchRequestOutput)
        self.assertEqual(len(req_out.outputs), 1)

        comp = req_out.outputs[0]
        self.assertIsInstance(comp, PyTorchCompletionOutput)
        self.assertEqual(len(comp.token_ids), 16)
        self.assertIsNotNone(comp.logprobs)
        self.assertEqual(len(comp.logprobs), 16)

    def test_timing_extraction_compatibility(self):
        prompts = ["Timing test prompt."]
        outputs = self.runner.generate(prompts, max_tokens=32)
        req_out = outputs[0]

        ttft, decode, total = extract_request_timings(req_out, fallback_wall_sec=2.0, gen_tokens=32)
        self.assertGreater(ttft, 0.0)
        self.assertGreater(decode, 0.0)
        self.assertGreaterEqual(total, ttft + decode - 0.01)

    def test_logprob_extraction_compatibility(self):
        prompts = ["Logprob extraction test prompt."]
        outputs = self.runner.generate(prompts, max_tokens=10)
        req_out = outputs[0]

        step_lps = extract_step_logprobs(req_out)
        self.assertEqual(len(step_lps), 10)
        for step in step_lps:
            self.assertIsInstance(step, dict)
            self.assertTrue(len(step) > 0)

    def test_synthetic_filler_workload_compatibility(self):
        workload = SyntheticFiller()
        output = workload.run(self.runner, context_length=128, max_tokens=16)

        self.assertEqual(output.generated_tokens, 16)
        self.assertGreater(output.ttft_sec, 0.0)
        self.assertGreater(output.tpot_mean_ms, 0.0)
        self.assertEqual(len(output.generated_token_ids), 16)
        self.assertEqual(len(output.step_logprobs), 16)

    def test_multi_prompt_generation(self):
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        outputs = self.runner.generate(prompts, max_tokens=8)

        self.assertEqual(len(outputs), 3)
        for idx, out in enumerate(outputs):
            self.assertEqual(out.prompt, prompts[idx])
            self.assertEqual(len(out.outputs[0].token_ids), 8)

    def test_multi_sample_cache_reset_gc_and_telemetry(self):
        class _SpyCache:
            reset_calls = 0

            def __init__(self):
                self.bytes_transferred = 1024
                self._cycle = lambda: self  # intentional ref cycle like HF Cache

            def reset(self):
                _SpyCache.reset_calls += 1
                self.bytes_transferred = 0

        class _MockTokenizer:
            eos_token_id = 999

            def encode(self, prompt, return_tensors=None):
                del prompt, return_tensors
                return torch.tensor([[1, 2, 3]], dtype=torch.long)

            def decode(self, ids, skip_special_tokens=True):
                del ids, skip_special_tokens
                return "ok"

        class _MockModel:
            def __call__(self, input_ids, past_key_values=None, use_cache=True, position_ids=None):
                del past_key_values, use_cache, position_ids
                seq_len = input_ids.shape[1]
                logits = torch.zeros((1, seq_len, 16), dtype=torch.float32)
                logits[:, -1, 5] = 10.0
                return SimpleNamespace(logits=logits)

        cache_refs = []
        observed_bytes = []

        def _factory():
            c = _SpyCache()
            cache_refs.append(weakref.ref(c))
            return c

        def _on_complete(cache):
            observed_bytes.append(cache.bytes_transferred)

        runner = PyTorchRunner(
            model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
            device="cpu",
            dtype=torch.float32,
            cache_factory=_factory,
            dry_run=True,
            on_sample_complete=_on_complete,
        )
        runner.dry_run = False
        runner.model = _MockModel()
        runner.tokenizer = _MockTokenizer()

        outputs = runner.generate(["p1", "p2", "p3"], max_tokens=2)
        self.assertEqual(len(outputs), 3)
        self.assertEqual(_SpyCache.reset_calls, 3)
        self.assertEqual(observed_bytes, [1024, 1024, 1024])
        self.assertTrue(all(ref() is None for ref in cache_refs))

    def test_multi_sample_cleanup_runs_on_error(self):
        class _SpyCache:
            reset_calls = 0

            def reset(self):
                _SpyCache.reset_calls += 1

        class _MockTokenizer:
            eos_token_id = 999

            def encode(self, prompt, return_tensors=None):
                del prompt, return_tensors
                return torch.tensor([[1, 2]], dtype=torch.long)

        class _FailingModel:
            def __call__(self, *args, **kwargs):
                del args, kwargs
                raise RuntimeError("boom")

        cache_refs = []

        def _factory():
            c = _SpyCache()
            cache_refs.append(weakref.ref(c))
            return c

        runner = PyTorchRunner(
            model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
            device="cpu",
            dtype=torch.float32,
            cache_factory=_factory,
            dry_run=True,
        )
        runner.dry_run = False
        runner.model = _FailingModel()
        runner.tokenizer = _MockTokenizer()

        with self.assertRaises(RuntimeError):
            runner.generate(["p1"], max_tokens=2)
        self.assertEqual(_SpyCache.reset_calls, 1)
        self.assertIsNone(cache_refs[0]())

    def test_harness_calls_setup_per_context(self) -> None:
        from src.core.harness import run_benchmark
        from src.core.technique import Technique
        from src.core.workload import SyntheticFiller

        class _CountingTechnique(Technique):
            name = "_test_counting"
            axes = ()

            def __init__(self) -> None:
                self.setup_calls = 0
                self.count = 0

            def setup(self) -> None:
                self.setup_calls += 1
                self.count = 0

            def engine_kwargs(self, context_length: int, max_tokens: int):
                del context_length, max_tokens
                return {}

            def workload(self):
                return SyntheticFiller()

            def extra_metrics(self, llm, workload_output):
                del llm, workload_output
                self.count += 1
                return {"setup_calls": self.setup_calls, "count": self.count}

        tech = _CountingTechnique()
        report = run_benchmark(
            technique=tech,
            model="qwen3-4b",
            contexts=[512, 1024, 2048],
            max_tokens=4,
            engine="pytorch",
            dry_run=True,
        )
        # 1 initial setup before workload() + 3 per-context setups
        self.assertEqual(tech.setup_calls, 4)
        self.assertEqual(len(report.results), 3)
        for idx, entry in enumerate(report.results):
            if "count" in entry.extra:
                self.assertEqual(entry.extra["count"], 1, f"context {idx} leaked count")

    def test_turboquant_telemetry_persists_after_cache_gc(self) -> None:
        import gc
        from src.techniques.turboquant import TurboQuantTechnique

        class _MockTQCache:
            def get_shannon_phi_metrics(self):
                return {
                    "k_eps_snr": 0.91,
                    "k_eps_dir": 0.92,
                    "v_eps_snr": 0.85,
                    "v_eps_dir": 0.87,
                }

            def get_cache_size_bytes(self):
                return (123456, 234567)

        tech = TurboQuantTechnique()
        tech.setup()
        cache = _MockTQCache()
        cache_ref = weakref.ref(cache)
        tech.on_sample_complete(cache)
        del cache
        gc.collect()
        self.assertIsNone(cache_ref())

        metrics = tech.extra_metrics(llm=None, workload_output=None)
        self.assertEqual(metrics["physical_cache_bytes"], 123456)
        self.assertEqual(metrics["theoretical_cache_bytes"], 234567)
        self.assertAlmostEqual(metrics["shannon_k_eps_snr"], 0.91)
        self.assertAlmostEqual(metrics["shannon_v_eps_dir"], 0.87)


if __name__ == "__main__":
    unittest.main()
