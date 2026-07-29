"""
Unit and integration tests for Milestone R1: Baseline & LMCache TTFT refinement and CLI routing.
"""

import json
import pytest
from main import parse_contexts

def test_parse_contexts():
    """Verify parse_contexts handles comma-separated, space-separated, and integer lists."""
    assert parse_contexts("512,1024,2048") == [512, 1024, 2048]
    assert parse_contexts(["512,1024", "2048"]) == [512, 1024, 2048]
    assert parse_contexts(["512", "1024", "2048"]) == [512, 1024, 2048]
    assert parse_contexts([512, 1024, 2048]) == [512, 1024, 2048]


def test_baseline_benchmark_metrics():
    """Verify run_benchmark in src/benchmark_baseline extracts TTFT, decode time, and throughput via RequestMetrics."""
    from src.benchmark_baseline import run_benchmark
    
    results = run_benchmark(
        model_name="Qwen/Qwen2.5-1.5B",
        context_lengths=[512],
        max_tokens=32
    )
    
    assert len(results) == 1
    res = results[0]
    
    assert res["status"] == "SUCCESS"
    assert res["requested_ctx"] == 512
    assert res["prompt_tokens"] > 0
    assert res["generated_tokens"] == 32
    assert "ttft_sec" in res
    assert "decode_time_sec" in res
    assert "total_time_sec" in res
    assert "throughput_tok_per_sec" in res
    
    assert res["ttft_sec"] > 0
    assert res["decode_time_sec"] > 0
    assert res["total_time_sec"] >= res["ttft_sec"]
    assert res["throughput_tok_per_sec"] > 0


def test_lmcache_benchmark_metrics():
    """Verify run_lmcache_benchmark in src/benchmark_lmcache uses distinct suffixes and calculates ttft_speedup_factor."""
    from src.benchmark_lmcache import run_lmcache_benchmark
    
    results = run_lmcache_benchmark(
        model_name="Qwen/Qwen2.5-0.5B",
        context_lengths=[512],
        max_tokens=32
    )
    
    assert len(results) == 1
    res = results[0]
    
    assert res["status"] == "SUCCESS"
    assert res["requested_ctx"] == 512
    assert res["prompt_tokens"] > 0
    assert res["generated_tokens"] == 32
    assert "cold_ttft_sec" in res
    assert "warm_ttft_sec" in res
    assert "ttft_speedup_factor" in res
    assert "cold_total_time_sec" in res
    assert "warm_total_time_sec" in res
    assert "total_speedup_factor" in res
    
    assert res["cold_ttft_sec"] > 0
    assert res["warm_ttft_sec"] > 0
    # In real execution, the warm run should ideally be faster, so speedup should be > 1.0 
    assert res["ttft_speedup_factor"] >= 1.0


def test_cli_baseline_execution(cli_runner, tmp_path):
    """Verify CLI baseline execution creates contract output file matching JSON schema."""
    output_file = str(tmp_path / "baseline_phase1.json")
    res = cli_runner(["--mode", "baseline", "--contexts", "512,1024", "--output", output_file])
    
    assert res.exit_code == 0
    assert res.json_data is not None
    assert res.json_data["model"] == "Qwen/Qwen2.5-1.5B"
    assert res.json_data["mode"] == "baseline"
    assert len(res.json_data["results"]) == 2
    
    r1 = res.json_data["results"][0]
    assert r1["requested_ctx"] == 512
    assert r1["status"] == "SUCCESS"
    assert "ttft_sec" in r1
    assert "decode_time_sec" in r1
    assert "throughput_tok_per_sec" in r1


def test_cli_lmcache_execution(cli_runner, tmp_path):
    """Verify CLI lmcache execution creates contract output file matching JSON schema."""
    output_file = str(tmp_path / "lmcache_phase2.json")
    res = cli_runner(["--mode", "lmcache", "--contexts", "512", "--output", output_file])
    
    if res.exit_code != 0:
        print(f"DEBUG EXIT CODE: {res.exit_code}\nSTDERR: {res.stderr}\nSTDOUT: {res.stdout}")
    assert res.exit_code == 0
    assert res.json_data is not None
    assert res.json_data["model"] == "Qwen/Qwen2.5-1.5B"
    assert res.json_data["mode"] == "lmcache"
    assert len(res.json_data["results"]) == 1
    
    r1 = res.json_data["results"][0]
    assert r1["requested_ctx"] == 512
    assert r1["status"] == "SUCCESS"
    assert "cold_ttft_sec" in r1
    assert "warm_ttft_sec" in r1
    assert "ttft_speedup_factor" in r1
