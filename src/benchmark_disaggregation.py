import argparse
import os
import gc
import json
import time
import subprocess
import torch
from vllm import LLM, SamplingParams

def run_engine(model_name, context_length, engine_type, max_tokens=64, gpu_utilization=0.85):
    shared_prefix = "The quick brown fox jumps over the lazy dog. " * (context_length // 9)
    cold_prompt = shared_prefix + "\n\nInstruction: Summarize."
    warm_prompt = shared_prefix + "\n\nInstruction: What is the theme?"
    
    print(f"[{engine_type}] Initializing vLLM (Phase 4)...")
    llm = LLM(
        model=model_name,
        dtype="half",
        gpu_memory_utilization=gpu_utilization,
        max_model_len=context_length + max_tokens + 256,
        trust_remote_code=True,
        enforce_eager=True
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    
    if engine_type == "EngineA":
        print(f"[{engine_type}] Executing Prefill (Cold Run) to build persistent cache...")
        start_time = time.perf_counter()
        llm.generate([cold_prompt], sampling_params)
        end_time = time.perf_counter()
        print(f"[{engine_type}] Completed in {end_time - start_time:.4f}s. Shutting down to release VRAM.")
        # We don't save stats for EngineA as it's just meant to populate the cache
    
    elif engine_type == "EngineB":
        print(f"[{engine_type}] Executing Decode (Warm Run) reading from persistent cache...")
        start_wall = time.perf_counter()
        outputs = llm.generate([warm_prompt], sampling_params)
        end_wall = time.perf_counter()
        
        warm_obj = outputs[0]
        metrics = getattr(warm_obj, "metrics", None)
        
        ttft_sec = end_wall - start_wall
        if metrics and getattr(metrics, "first_token_time", None) and getattr(metrics, "arrival_time", None):
            ttft_sec = metrics.first_token_time - metrics.arrival_time
            
        print(f"[{engine_type}] TTFT achieved: {ttft_sec:.4f}s")
        
        results = {
            "requested_ctx": context_length,
            "engine_b_ttft_sec": ttft_sec,
            "status": "SUCCESS"
        }
        
        # Save to a temporary file to be read by the orchestrator
        with open("results/phase4_engine_b_tmp.json", "w") as f:
            json.dump(results, f)

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_sequential_disaggregation(model_name, context_length):
    print(f"=== Phase 4: Sequential Context Disaggregation ({model_name}) ===")
    os.makedirs("results", exist_ok=True)
    
    # We enforce a local disk backend for LMCache to ensure persistence between processes
    env = os.environ.copy()
    env["LMCACHE_CONFIG_FILE"] = "src/config/lmcache_config.yaml"
    # Override config local_device to be a file path, or we can assume lmcache_config.yaml handles it.
    # We will pass LMCACHE_LOCAL_DEVICE to explicitly force disk if supported by LMCache version, 
    # but for safety we just use the default env vars.
    env["LMCACHE_LOCAL_DEVICE"] = "file://local_disk_cache"
    
    try:
        # Run Engine A
        print("\n--- Starting Engine A ---")
        res_a = subprocess.run(["python", __file__, "--engine", "EngineA", "--model", model_name, "--context", str(context_length)], env=env)
        if res_a.returncode != 0:
            raise RuntimeError(f"Engine A failed with exit code {res_a.returncode}")
        
        # Run Engine B
        print("\n--- Starting Engine B ---")
        res_b = subprocess.run(["python", __file__, "--engine", "EngineB", "--model", model_name, "--context", str(context_length)], env=env)
        if res_b.returncode != 0:
            raise RuntimeError(f"Engine B failed with exit code {res_b.returncode}")
        
        # Read the results back
        results = []
        if os.path.exists("results/phase4_engine_b_tmp.json"):
            with open("results/phase4_engine_b_tmp.json", "r") as f:
                results.append(json.load(f))
            os.remove("results/phase4_engine_b_tmp.json")
        
        return results

    except Exception as e:
        print(f" -> FAILED at context length {context_length}: {e}")
        return [{"status": "FAILED", "context": context_length, "error": str(e)}]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--engine", type=str, choices=["EngineA", "EngineB"], help="Internal use only")
    args = parser.parse_args()
    
    if args.engine:
        run_engine(args.model, args.context, args.engine)
    else:
        results = run_sequential_disaggregation(args.model, args.context)
        output_payload = {
            "model": args.model,
            "mode": "disaggregation",
            "results": results
        }
        with open("results/disaggregation_phase4.json", "w") as f:
            json.dump(output_payload, f, indent=2)
        print(f"\nSaved disaggregation benchmark results to results/disaggregation_phase4.json")
