"""
Shared Pytest Fixtures for kv_bench Test Suite.

Provides robust mocking for vLLM, PyTorch CUDA memory, LMCache C++ hooks,
synthetic Chrome Profiler traces, and a CLI execution helper runner.
Ensures tests run deterministically without physical GPU hardware or active vLLM processes.
"""

import os
import sys
import json
import types
import io
import pytest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path so main and src can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Pre-populate mock c_ops and torch.distributed.tensor before patch.dict snapshots sys.modules
mock_c_ops = types.ModuleType("lmcache.c_ops")
sys.modules["lmcache.c_ops"] = mock_c_ops

import torch
import torch.distributed
try:
    import torch.distributed._tensor
    sys.modules["torch.distributed.tensor"] = torch.distributed._tensor
except ImportError:
    pass

# --- Mock Classes for vLLM ---

class MockCompletionOutput:
    """Simulated completion output choice for vLLM RequestOutput."""
    def __init__(self, text="Generated response text from mock vLLM", token_ids=None):
        self.text = text
        self.token_ids = token_ids if token_ids is not None else list(range(100, 164))

class MockRequestMetrics:
    """Simulated metrics tracking object for vLLM requests."""
    def __init__(self, arrival_time=100.0, first_token_time=100.25, finished_time=101.50):
        self.arrival_time = arrival_time
        self.first_token_time = first_token_time
        self.finished_time = finished_time

class MockRequestOutput:
    """Simulated vLLM RequestOutput containing metrics and generated tokens."""
    def __init__(
        self,
        prompt=None,
        prompt_token_ids=None,
        outputs=None,
        arrival_time=100.0,
        first_token_time=100.25,
        finished_time=101.50,
        request_id="mock-req-001"
    ):
        if prompt_token_ids is None:
            num_tokens = len(prompt.split()) if isinstance(prompt, str) and prompt else 512
            prompt_token_ids = list(range(num_tokens))
        
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs if outputs is not None else [MockCompletionOutput()]
        self.metrics = MockRequestMetrics(arrival_time, first_token_time, finished_time)
        
        # Direct attributes for multi-version vLLM compatibility
        self.arrival_time = arrival_time
        self.first_token_time = first_token_time
        self.finished_time = finished_time

class MockLLM:
    """Simulated vLLM engine instance."""
    def __init__(self, model="mock-model", **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.generate_count = 0

    def generate(self, prompts, sampling_params=None, **kwargs):
        self.generate_count += 1
        results = []
        for prompt in prompts:
            if isinstance(prompt, str):
                words = prompt.strip().split()
                token_count = max(len(words), 512)
            elif isinstance(prompt, dict) and "prompt" in prompt:
                token_count = max(len(prompt["prompt"].split()), 512)
            else:
                token_count = 512

            max_tokens = getattr(sampling_params, "max_tokens", 64) if sampling_params else 64
            
            # Cold run (first call) vs Warm run (subsequent calls) timing simulation
            ttft_delay = 0.25 if self.generate_count == 1 else 0.05
            arrival = 100.0 + (self.generate_count * 5.0)
            first_tok = arrival + ttft_delay
            finished = first_tok + (max_tokens * 0.01)

            res = MockRequestOutput(
                prompt=prompt if isinstance(prompt, str) else None,
                prompt_token_ids=list(range(token_count)),
                outputs=[MockCompletionOutput(token_ids=list(range(1000, 1000 + max_tokens)))],
                arrival_time=arrival,
                first_token_time=first_tok,
                finished_time=finished
            )
            results.append(res)
        return results

class MockSamplingParams:
    """Simulated vLLM SamplingParams object."""
    def __init__(self, temperature=0.0, max_tokens=64, **kwargs):
        self.temperature = temperature
        self.max_tokens = max_tokens
        for k, v in kwargs.items():
            setattr(self, k, v)


# --- Pytest Fixtures ---

@pytest.fixture
def mock_cuda():
    """
    Mocks torch.cuda functions to simulate an NVIDIA GeForce GTX 1660 SUPER GPU.
    Returns simulated memory stats and device info without requiring hardware.
    """
    cuda_patches = {
        "torch.cuda.is_available": patch("torch.cuda.is_available", return_value=True),
        "torch.cuda.max_memory_allocated": patch("torch.cuda.max_memory_allocated", return_value=3141592653), # ~3.00 GB
        "torch.cuda.max_memory_reserved": patch("torch.cuda.max_memory_reserved", return_value=3296000000), # ~3.14 GB
        "torch.cuda.memory_allocated": patch("torch.cuda.memory_allocated", return_value=2000000000),
        "torch.cuda.memory_reserved": patch("torch.cuda.memory_reserved", return_value=2500000000),
        "torch.cuda.get_device_name": patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce GTX 1660 SUPER"),
        "torch.cuda.synchronize": patch("torch.cuda.synchronize", return_value=None),
        "torch.cuda.empty_cache": patch("torch.cuda.empty_cache", return_value=None),
        "torch.cuda.reset_peak_memory_stats": patch("torch.cuda.reset_peak_memory_stats", return_value=None),
    }

    started_patches = [p.start() for p in cuda_patches.values()]

    try:
        yield {
            "device_name": "NVIDIA GeForce GTX 1660 SUPER",
            "max_memory_allocated": 3141592653,
            "max_memory_reserved": 3296000000,
            "memory_allocated": 2000000000,
            "memory_reserved": 2500000000,
        }
    finally:
        for p in cuda_patches.values():
            p.stop()


@pytest.fixture
def mock_vllm():
    """
    Mocks vllm.LLM and vllm.SamplingParams to return simulated RequestOutput with realistic metrics.
    Patches vllm module imports globally during test execution.
    """
    vllm_patch = patch("vllm.LLM", MockLLM)
    params_patch = patch("vllm.SamplingParams", MockSamplingParams)

    vllm_patch.start()
    params_patch.start()

    try:
        yield {
            "LLM": MockLLM,
            "SamplingParams": MockSamplingParams,
            "MockRequestOutput": MockRequestOutput,
            "MockCompletionOutput": MockCompletionOutput,
            "MockRequestMetrics": MockRequestMetrics,
        }
    finally:
        vllm_patch.stop()
        params_patch.stop()


@pytest.fixture
def mock_lmcache():
    """
    Mocks lmcache imports and c_ops C++ bindings to prevent symbol lookup errors on non-CUDA hosts.
    """
    mock_lmcache_mod = MagicMock()
    mock_c_ops_mod = types.ModuleType("lmcache.c_ops")
    mock_lmcache_vllm_mod = MagicMock()
    mock_injection_mod = MagicMock()

    mock_injection_mod.InitLMCacheEnvironment = MagicMock(return_value=None)
    mock_lmcache_vllm_mod.vllm_injection = mock_injection_mod

    mock_adapter_mod = MagicMock()
    mock_lmcache_vllm_mod.vllm_adapter = mock_adapter_mod

    modules_to_patch = {
        "lmcache": mock_lmcache_mod,
        "lmcache.c_ops": mock_c_ops_mod,
        "lmcache_vllm": mock_lmcache_vllm_mod,
        "lmcache_vllm.vllm_injection": mock_injection_mod,
        "lmcache_vllm.vllm_adapter": mock_adapter_mod,
        "lmcache.cache_engine": MagicMock(),
        "lmcache.storage_backend": MagicMock(),
    }

    with patch.dict("sys.modules", modules_to_patch):
        yield {
            "lmcache": mock_lmcache_mod,
            "c_ops": mock_c_ops_mod,
            "lmcache_vllm": mock_lmcache_vllm_mod,
            "injection": mock_injection_mod,
        }


@pytest.fixture
def synthetic_chrome_trace():
    """
    Returns a dictionary and helper object representing a valid Chrome Trace with
    'cat': 'kernel' and 'cat': 'gpu_memcpy' events for PyTorch Profiler timeline validation.
    """
    trace_data = {
        "schemaVersion": 1,
        "deviceProperties": [
            {
                "id": 0,
                "name": "NVIDIA GeForce GTX 1660 SUPER",
                "totalGlobalMem": 6442450944
            }
        ],
        "traceEvents": [
            {
                "name": "aten::matmul",
                "cat": "kernel",
                "ph": "X",
                "ts": 1000.0,
                "dur": 45.0,
                "pid": 1,
                "tid": 1,
                "args": {"external id": 1, "device": 0}
            },
            {
                "name": "flash_attention_forward_kernel",
                "cat": "kernel",
                "ph": "X",
                "ts": 1050.0,
                "dur": 110.0,
                "pid": 1,
                "tid": 1,
                "args": {"grid": [64, 1, 1], "block": [128, 1, 1]}
            },
            {
                "name": "Memcpy HtoD (Host to Device KV Cache)",
                "cat": "gpu_memcpy",
                "ph": "X",
                "ts": 1010.0,
                "dur": 80.0,
                "pid": 1,
                "tid": 2,
                "args": {"bytes": 4194304, "src": "host", "dst": "device"}
            },
            {
                "name": "Memcpy HtoD (Layer-wise Async Transfer)",
                "cat": "gpu_memcpy",
                "ph": "X",
                "ts": 1090.0,
                "dur": 65.0,
                "pid": 1,
                "tid": 2,
                "args": {"bytes": 8388608, "src": "host", "dst": "device"}
            }
        ],
        "otherData": {
            "version": "PyTorch Profiler v2.4",
            "profile_name": "Phase 3 Layer-wise Pipelining Profile"
        }
    }

    class ChromeTraceWrapper(dict):
        def to_json(self):
            return json.dumps(self, indent=2)

    return ChromeTraceWrapper(trace_data)


@pytest.fixture
def cli_runner():
    """
    Helper function to invoke main.py via subprocess.
    This guarantees VRAM is freed between tests.
    """
    class CLIResult:
        def __init__(self, exit_code, stdout, stderr, output_file=None, json_data=None):
            self.exit_code = exit_code
            self.stdout = stdout
            self.stderr = stderr
            self.output_file = output_file
            self.json_data = json_data

    def run_cli(args_list):
        import subprocess
        res = subprocess.run(
            [sys.executable, "main.py"] + args_list,
            capture_output=True,
            text=True
        )
        
        output_file = None
        if "--output" in args_list:
            idx = args_list.index("--output")
            if idx + 1 < len(args_list):
                output_file = args_list[idx + 1]
        elif "--mode" in args_list:
            idx = args_list.index("--mode")
            if idx + 1 < len(args_list):
                mode_name = args_list[idx + 1]
                defaults = {
                    "baseline": "results/baseline_phase1.json",
                    "lmcache": "results/lmcache_phase2.json",
                }
                output_file = defaults.get(mode_name, f"results/{mode_name}_results.json")

        json_data = None
        if output_file and os.path.exists(output_file):
            with open(output_file, "r") as f:
                try:
                    json_data = json.load(f)
                except json.JSONDecodeError:
                    json_data = None

        return CLIResult(res.returncode, res.stdout, res.stderr, output_file, json_data)

    return run_cli
