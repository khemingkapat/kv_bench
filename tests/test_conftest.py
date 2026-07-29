"""
Unit tests to verify fixture behavior in tests/conftest.py.
"""

import json
import torch
import sys

def test_mock_cuda(mock_cuda):
    """Verify mock_cuda fixture accurately mocks PyTorch CUDA device and memory functions."""
    assert torch.cuda.is_available() is True
    assert torch.cuda.get_device_name(0) == "NVIDIA GeForce GTX 1660 SUPER"
    assert torch.cuda.max_memory_allocated(0) == 3141592653
    assert torch.cuda.max_memory_reserved(0) == 3296000000
    assert torch.cuda.memory_allocated(0) == 2000000000
    assert torch.cuda.memory_reserved(0) == 2500000000
    # Operations should run cleanly without error
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def test_mock_vllm(mock_vllm):
    """Verify mock_vllm fixture simulates vLLM LLM and SamplingParams generation."""
    import vllm
    
    llm = vllm.LLM(model="Qwen/Qwen2.5-0.5B")
    sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=32)
    
    prompt = "Test prompt for vLLM simulation"
    outputs = llm.generate([prompt], sampling_params)
    
    assert len(outputs) == 1
    req_output = outputs[0]
    
    # Check attributes
    assert hasattr(req_output, "prompt_token_ids")
    assert hasattr(req_output, "outputs")
    assert hasattr(req_output, "metrics")
    
    # Check metrics
    assert req_output.metrics.arrival_time > 0
    assert req_output.metrics.first_token_time >= req_output.metrics.arrival_time
    assert req_output.metrics.finished_time >= req_output.metrics.first_token_time
    
    # Check completion tokens
    assert len(req_output.outputs[0].token_ids) == 32


def test_mock_lmcache(mock_lmcache):
    """Verify mock_lmcache fixture intercepts lmcache module and c_ops imports."""
    import lmcache
    import lmcache.c_ops
    import lmcache_vllm.vllm_injection as injection
    
    assert lmcache is not None
    assert lmcache.c_ops is not None
    injection.InitLMCacheEnvironment()
    mock_lmcache["injection"].InitLMCacheEnvironment.assert_called()


def test_synthetic_chrome_trace(synthetic_chrome_trace):
    """Verify synthetic_chrome_trace fixture returns valid Chrome Trace structure with kernel and memcpy events."""
    assert "traceEvents" in synthetic_chrome_trace
    events = synthetic_chrome_trace["traceEvents"]
    
    categories = {evt.get("cat") for evt in events}
    assert "kernel" in categories
    assert "gpu_memcpy" in categories
    
    # Check JSON serialization wrapper method
    json_str = synthetic_chrome_trace.to_json()
    parsed = json.loads(json_str)
    assert parsed["schemaVersion"] == 1
    assert len(parsed["traceEvents"]) >= 4


def test_cli_runner(cli_runner, tmp_path):
    """Verify cli_runner helper executes main.py CLI without process termination."""
    output_file = str(tmp_path / "baseline_results.json")
    res = cli_runner(["--mode", "baseline", "--contexts", "512", "--output", output_file])
    
    assert res.exit_code == 0
    assert res.output_file == output_file
    assert res.json_data is not None
    assert res.json_data["mode"] == "baseline"
    assert len(res.json_data["results"]) == 1
    assert res.json_data["results"][0]["requested_ctx"] == 512
