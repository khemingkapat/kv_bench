"""Unit tests for WP3 data consistency fixes: physical bitwidth telemetry and InfiniGen pinned staging."""

import unittest

import torch

from src.core.diagnostic import inspect_sample_diagnostic
from src.techniques.cachegen_cache import CacheGenCache
from src.techniques.headq_cache import HeadQCache
from src.techniques.infinigen_cache import InfiniGenLayer
from src.techniques.qserve_cache import QServeKVCache
from src.techniques.turboquant_cache import TurboQuantCache


class _DummyLayer:
    def __init__(self, positions: list[int]) -> None:
        self.original_positions = positions


class _DummyQuantCache:
    def __init__(self, positions: list[int], bits: float) -> None:
        self.layers = [_DummyLayer(positions)]
        self.effective_bits_per_scalar = bits


class _DummyPlainCache:
    def __init__(self, positions: list[int]) -> None:
        self.layers = [_DummyLayer(positions)]


class TestPhysicalBitwidthTelemetry(unittest.TestCase):
    def test_quantized_cache_effective_bits_per_scalar(self) -> None:
        self.assertAlmostEqual(QServeKVCache(group_size=32).effective_bits_per_scalar, 4.50, places=4)
        self.assertAlmostEqual(TurboQuantCache().effective_bits_per_scalar, 4.375, places=4)
        self.assertAlmostEqual(HeadQCache().effective_bits_per_scalar, 3.25, places=4)
        self.assertAlmostEqual(CacheGenCache().effective_bits_per_scalar, 6.392, places=4)

    def test_diagnostic_reads_cache_effective_bits_per_scalar(self) -> None:
        sample = {"id": "s1", "task_type": "fact_recall"}
        toks = list(range(10))
        prompt = "hello world"

        diag_q = inspect_sample_diagnostic(
            cache=_DummyQuantCache(toks, 4.50),
            prompt_text=prompt,
            prompt_token_ids=toks,
            sample_meta=sample,
            is_correct=True,
            baseline_correct=True,
        )
        self.assertAlmostEqual(diag_q.overall_effective_bitwidth, 4.50, places=2)

        diag_plain = inspect_sample_diagnostic(
            cache=_DummyPlainCache(toks),
            prompt_text=prompt,
            prompt_token_ids=toks,
            sample_meta=sample,
            is_correct=True,
            baseline_correct=True,
        )
        self.assertAlmostEqual(diag_plain.overall_effective_bitwidth, 16.0, places=2)


class TestInfiniGenPinnedStaging(unittest.TestCase):
    def test_staging_buffers_allocated_and_reset(self) -> None:
        layer = InfiniGenLayer(block_size=4, max_budget_ratio=0.5)
        k = torch.randn(1, 2, 16, 8, dtype=torch.float32)
        v = torch.randn(1, 2, 16, 8, dtype=torch.float32)
        k_out, v_out = layer.update(k, v)
        self.assertEqual(k_out.shape, (1, 2, 16, 8))
        self.assertEqual(v_out.shape, (1, 2, 16, 8))
        self.assertIsNotNone(layer.staging_cpu_k)
        self.assertIsNotNone(layer.staging_cpu_v)

        # Trigger decode path (seq_len == 1, total_tokens_seen > block_size)
        q_rot = torch.randn(1, 2, 1, 8, dtype=torch.float32)
        k_dec = torch.randn(1, 2, 1, 8, dtype=torch.float32)
        v_dec = torch.randn(1, 2, 1, 8, dtype=torch.float32)
        k_d, v_d = layer.update(k_dec, v_dec, cache_kwargs={"query_states": q_rot})
        self.assertGreaterEqual(k_d.shape[2], 4)
        self.assertEqual(k_d.shape, v_d.shape)

        layer.reset()
        self.assertIsNone(layer.staging_cpu_k)
        self.assertIsNone(layer.staging_cpu_v)


if __name__ == "__main__":
    unittest.main()
