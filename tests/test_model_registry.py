import unittest

from src.core.config import (
    MODEL_REGISTRY,
    ModelSpec,
    get_model_spec,
    resolve_model_target,
)


class TestModelRegistry(unittest.TestCase):
    def test_registered_models_exist(self):
        expected_models = [
            "qwen3-4b",
            "phi4-mini",
            "ministral-3b",
            "smollm3-3b",
            "qwen3-1.7b",
            "qwen3.5-4b",
            "qwen3.5-2b",
            "qwen2.5-3b",
        ]
        for name in expected_models:
            self.assertIn(name, MODEL_REGISTRY)
            spec = MODEL_REGISTRY[name]
            self.assertIsInstance(spec, ModelSpec)
            self.assertTrue(len(spec.revision) == 40, f"Revision SHA for {name} must be 40 chars")
            self.assertGreater(spec.kv_bytes_per_token_bf16, 0)
            self.assertGreater(spec.native_max_context, 0)

    def test_dense_kv_cache_calculation(self):
        # Qwen 3 4B: L=36, KV=8, HeadDim=128
        # KV bytes / token = 4 * 36 * 8 * 128 = 147,456 bytes = 144 KiB
        qwen3 = MODEL_REGISTRY["qwen3-4b"]
        self.assertEqual(qwen3.kv_bytes_per_token_bf16, 147456)
        
        # 32K context single stream: (147456 * 32768) / (1024^3) = 4.5 GB
        kv_32k = qwen3.kv_cache_gb(32768, concurrent_sessions=1)
        self.assertAlmostEqual(kv_32k, 4.5, places=2)

    def test_hybrid_vs_dense_kv_footprint(self):
        # Qwen 3.5 4B: L_full_attn=8, KV=4, HeadDim=256
        # KV bytes / token = 4 * 8 * 4 * 256 = 32,768 bytes = 32 KiB
        qwen35 = MODEL_REGISTRY["qwen3.5-4b"]
        self.assertEqual(qwen35.kv_bytes_per_token_bf16, 32768)
        self.assertTrue(qwen35.is_hybrid_gdn)

        kv_32k = qwen35.kv_cache_gb(32768, concurrent_sessions=1)
        self.assertAlmostEqual(kv_32k, 1.0, places=2)

    def test_phi4_mini_specs(self):
        phi4 = MODEL_REGISTRY["phi4-mini"]
        # L=32, KV=8, HeadDim=128 -> 4 * 32 * 8 * 128 = 131,072 bytes = 128 KiB
        self.assertEqual(phi4.kv_bytes_per_token_bf16, 131072)
        kv_32k = phi4.kv_cache_gb(32768, concurrent_sessions=1)
        self.assertAlmostEqual(kv_32k, 4.0, places=2)

    def test_resolve_model_target(self):
        # Alias lookup
        repo_id, rev, extra_args = resolve_model_target("qwen3-4b")
        self.assertEqual(repo_id, "Qwen/Qwen3-4B-Instruct-2507")
        self.assertEqual(rev, MODEL_REGISTRY["qwen3-4b"].revision)
        self.assertEqual(extra_args, ())

        # Multimodal flag resolution
        m_repo, m_rev, m_extra = resolve_model_target("qwen3.5-4b")
        self.assertIn("--language-model-only", m_extra)

        # Unregistered fallback
        custom_repo, custom_rev, custom_extra = resolve_model_target("custom/my-model")
        self.assertEqual(custom_repo, "custom/my-model")
        self.assertIsNone(custom_rev)
        self.assertEqual(custom_extra, ())

    def test_parse_extra_vllm_args(self):
        from src.core.harness import _parse_extra_vllm_args

        self.assertEqual(_parse_extra_vllm_args(()), {})
        self.assertEqual(
            _parse_extra_vllm_args(("--language-model-only", "--mamba-cache-mode", "align")),
            {"language_model_only": True, "mamba_cache_mode": "align"},
        )

    def test_verify_mode_dry_run(self):
        import json
        import subprocess
        import sys
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = f"{tmpdir}/verify_out.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "--technique",
                    "baseline",
                    "--model",
                    "qwen3-4b",
                    "--verify",
                    "--dry-run",
                    "--max-vram-gb",
                    "16.0",
                    "--output",
                    out_path,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("[VERIFY OK]", proc.stdout)
            with open(out_path, "r") as f:
                data = json.load(f)
            self.assertEqual(len(data["results"]), 1)
            self.assertEqual(data["results"][0]["requested_ctx"], 128)
            self.assertEqual(data["results"][0]["status"], "DRY_RUN")
            self.assertEqual(data["results"][0]["extra"]["max_vram_gb"], 16.0)


if __name__ == "__main__":
    unittest.main()
