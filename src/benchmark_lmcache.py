import argparse
import os
import json
import time
import gc

# Fallback for LMCache C++ ops symbol mismatch on CUDA 12/13 runtime
import sys
import types
mock_c_ops = types.ModuleType("lmcache.c_ops")
sys.modules["lmcache.c_ops"] = mock_c_ops

import torch
import torch.distributed
try:
    import torch.distributed._tensor
    sys.modules["torch.distributed.tensor"] = torch.distributed._tensor
except ImportError:
    pass

import vllm
vllm.__version__ = "0.6.1.post2" # Mock version to bypass lmcache-vllm check

import lmcache
from lmcache_vllm.vllm_injection import InitLMCacheEnvironment

from vllm import LLM, SamplingParams

def get_gpu_memory_mb():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
        return allocated, reserved
    return 0.0, 0.0

def run_lmcache_benchmark(model_name, context_lengths, max_tokens=64, gpu_utilization=0.85):
    results = []
    print(f"=== LMCache CPU Offloading Benchmark: {model_name} ===")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"LMCache Config Path: {os.environ.get('LMCACHE_CONFIG_FILE', 'src/config/lmcache_config.yaml')}")
    print("-" * 65)

    for ctx in context_lengths:
        print(f"\n[LMCache Testing Context Length: {ctx} tokens]")
        shared_prefix = "The quick brown fox jumps over the lazy dog. " * (ctx // 9)
        cold_prompt = shared_prefix + "\n\nInstruction: Summarize the key points of the above text."
        warm_prompt = shared_prefix + "\n\nInstruction: What is the main theme discussed in the passage?"

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        try:
            # 1. Initialize vLLM instance
            llm = LLM(
                model=model_name,
                dtype="half",
                gpu_memory_utilization=gpu_utilization,
                max_model_len=ctx + max_tokens + 256,
                trust_remote_code=True,
                enforce_eager=True
            )

            sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
            
            # Cold run (populates host CPU LMCache)
            start_cold_wall = time.perf_counter()
            outputs_cold = llm.generate([cold_prompt], sampling_params)
            end_cold_wall = time.perf_counter()

            # Warm run (fetching KV Cache from host CPU RAM via LMCache with distinct prompt suffix)
            start_warm_wall = time.perf_counter()
            outputs_warm = llm.generate([warm_prompt], sampling_params)
            end_warm_wall = time.perf_counter()

            cold_obj = outputs_cold[0]
            warm_obj = outputs_warm[0]

            prompt_tokens = len(warm_obj.prompt_token_ids)
            gen_tokens = len(warm_obj.outputs[0].token_ids)

            cold_metrics = getattr(cold_obj, "metrics", None)
            warm_metrics = getattr(warm_obj, "metrics", None)

            if cold_metrics and getattr(cold_metrics, "first_token_time", None) and getattr(cold_metrics, "arrival_time", None) and getattr(cold_metrics, "finished_time", None):
                cold_ttft_sec = round(cold_metrics.first_token_time - cold_metrics.arrival_time, 4)
                cold_total_time_sec = round(cold_metrics.finished_time - cold_metrics.arrival_time, 4)
            else:
                cold_total_time_sec = round(end_cold_wall - start_cold_wall, 4)
                cold_ttft_sec = cold_total_time_sec

            if warm_metrics and getattr(warm_metrics, "first_token_time", None) and getattr(warm_metrics, "arrival_time", None) and getattr(warm_metrics, "finished_time", None):
                warm_ttft_sec = round(warm_metrics.first_token_time - warm_metrics.arrival_time, 4)
                warm_total_time_sec = round(warm_metrics.finished_time - warm_metrics.arrival_time, 4)
            else:
                warm_total_time_sec = round(end_warm_wall - start_warm_wall, 4)
                warm_ttft_sec = warm_total_time_sec

            ttft_speedup_factor = round(cold_ttft_sec / warm_ttft_sec, 2) if warm_ttft_sec > 0 else 1.0
            total_speedup_factor = round(cold_total_time_sec / warm_total_time_sec, 2) if warm_total_time_sec > 0 else 1.0

            peak_alloc = torch.cuda.max_memory_allocated(0) / (1024 * 1024) if torch.cuda.is_available() else 0.0
            peak_res = torch.cuda.max_memory_reserved(0) / (1024 * 1024) if torch.cuda.is_available() else 0.0

            res_entry = {
                "requested_ctx": ctx,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": gen_tokens,
                "cold_ttft_sec": cold_ttft_sec,
                "warm_ttft_sec": warm_ttft_sec,
                "ttft_speedup_factor": ttft_speedup_factor,
                "cold_total_time_sec": cold_total_time_sec,
                "warm_total_time_sec": warm_total_time_sec,
                "total_speedup_factor": total_speedup_factor,
                "peak_alloc_mb": round(peak_alloc, 2),
                "peak_reserved_mb": round(peak_res, 2),
                "status": "SUCCESS"
            }

            print(f" -> Prompt Tokens: {prompt_tokens}")
            print(f" -> Cold TTFT: {cold_ttft_sec:.4f}s | Cold Total Time: {cold_total_time_sec:.4f}s")
            print(f" -> Warm TTFT: {warm_ttft_sec:.4f}s | Warm Total Time: {warm_total_time_sec:.4f}s")
            print(f" -> TTFT Speedup Factor: {ttft_speedup_factor}x | Total Speedup Factor: {total_speedup_factor}x")
            print(f" -> Peak VRAM Allocated: {peak_alloc:.2f} MB | Reserved: {peak_res:.2f} MB")
            results.append(res_entry)

            del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f" -> FAILED at context length {ctx}: {repr(e)}")
            results.append({
                "requested_ctx": ctx,
                "status": "FAILED",
                "error": repr(e)
            })
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--output", type=str, default="results/lmcache_phase2.json")
    args = parser.parse_args()

    if "LMCACHE_CONFIG_FILE" not in os.environ:
        os.environ["LMCACHE_CONFIG_FILE"] = "src/config/lmcache_config.yaml"

    results = run_lmcache_benchmark(args.model, args.contexts)
    output_payload = {
        "model": args.model,
        "mode": "lmcache",
        "results": results
    }
    with open(args.output, "w") as f:
        json.dump(output_payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print(f"\nSaved LMCache benchmark results to {args.output}")

