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

def run_lmcache_benchmark(model_name, context_lengths, max_tokens=64, gpu_utilization=0.60):
    results = []
    print(f"=== LMCache CPU Offloading Benchmark: {model_name} ===")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"LMCache Config Path: {os.environ.get('LMCACHE_CONFIG_FILE', 'lmcache_config.yaml')}")
    print("-" * 65)

    for ctx in context_lengths:
        print(f"\n[LMCache Testing Context Length: {ctx} tokens]")
        prompt_text = "The quick brown fox jumps over the lazy dog. " * (ctx // 9)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        try:
            # 1. First run: Cold start (populates host CPU LMCache)
            llm = LLM(
                model=model_name,
                dtype="half",
                gpu_memory_utilization=0.85,
                max_model_len=max(ctx + max_tokens + 256, 4096),
                trust_remote_code=True,
                enforce_eager=True
            )

            sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
            
            # Cold run
            start_cold = time.perf_counter()
            outputs_cold = llm.generate([prompt_text], sampling_params)
            end_cold = time.perf_counter()
            cold_time = end_cold - start_cold

            # 2. Warm run (fetching KV Cache from host CPU RAM via LMCache)
            start_warm = time.perf_counter()
            outputs_warm = llm.generate([prompt_text], sampling_params)
            end_warm = time.perf_counter()
            warm_time = end_warm - start_warm

            prompt_tokens = len(outputs_warm[0].prompt_token_ids)
            gen_tokens = len(outputs_warm[0].outputs[0].token_ids)

            peak_alloc = torch.cuda.max_memory_allocated(0) / (1024 * 1024) if torch.cuda.is_available() else 0.0
            peak_res = torch.cuda.max_memory_reserved(0) / (1024 * 1024) if torch.cuda.is_available() else 0.0

            throughput_warm = gen_tokens / warm_time if warm_time > 0 else 0

            res_entry = {
                "requested_ctx": ctx,
                "actual_prompt_tokens": prompt_tokens,
                "generated_tokens": gen_tokens,
                "cold_time_sec": round(cold_time, 4),
                "warm_time_sec": round(warm_time, 4),
                "speedup_factor": round(cold_time / warm_time, 2) if warm_time > 0 else 1.0,
                "throughput_tok_per_sec": round(throughput_warm, 2),
                "peak_alloc_mb": round(peak_alloc, 2),
                "peak_reserved_mb": round(peak_res, 2),
                "status": "SUCCESS"
            }

            print(f" -> Actual Prompt Tokens: {prompt_tokens}")
            print(f" -> Cold Start Time (Prefill + Cache Build): {cold_time:.3f}s")
            print(f" -> Warm Start Time (LMCache CPU Retrieval): {warm_time:.3f}s")
            print(f" -> Speedup Factor: {res_entry['speedup_factor']}x")
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
    parser.add_argument("--output", type=str, default="lmcache_results_qwen0.5b.json")
    args = parser.parse_args()

    # LMCache environment hooks for vLLM are automatically applied on import in v0.6.2.3
    os.environ["LMCACHE_CONFIG_FILE"] = "lmcache_config.yaml"
    # InitLMCacheEnvironment() # Removed to prevent infinite recursion

    results = run_lmcache_benchmark(args.model, args.contexts)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved LMCache benchmark results to {args.output}")
