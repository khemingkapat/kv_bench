import argparse
import os
import gc
import torch
from vllm import LLM, SamplingParams

def run_profiling(model_name, context_length, max_tokens=64, gpu_utilization=0.85, trace_dir="results/profiler_trace_phase3"):
    print(f"=== Phase 3: Hardware Profiling ({model_name}) ===")
    os.makedirs(trace_dir, exist_ok=True)
    
    try:
        # Initialize vLLM
        print("[1/4] Initializing vLLM for Profiling...")
        llm = LLM(
            model=model_name,
            dtype="half",
            gpu_memory_utilization=gpu_utilization,
            max_model_len=context_length + max_tokens + 256,
            trust_remote_code=True,
            enforce_eager=True
        )
        
        shared_prefix = "The quick brown fox jumps over the lazy dog. " * (context_length // 9)
        cold_prompt = shared_prefix + "\n\nInstruction: Summarize."
        warm_prompt = shared_prefix + "\n\nInstruction: What is the theme?"
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        
        # Cold Run (No profiling)
        print("[2/4] Executing Cold Run (Building Cache)...")
        llm.generate([cold_prompt], sampling_params)
        
        # Warm Run (With PyTorch Profiler)
        print("[3/4] Executing Warm Run with PyTorch Profiler...")
        trace_path = os.path.join(trace_dir, f"trace_ctx{context_length}")
        
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            llm.generate([warm_prompt], sampling_params)
            prof.step()
            
        print(f"[4/4] Profiling complete. Trace saved to {trace_path}")
        
        # Cleanup
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return [{"status": "SUCCESS", "trace_dir": trace_dir, "context": context_length}]

    except Exception as e:
        print(f" -> FAILED at context length {context_length}: {e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return [{"status": "FAILED", "trace_dir": trace_dir, "context": context_length, "error": str(e)}]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--context", type=int, default=2048)
    args = parser.parse_args()
    
    if "LMCACHE_CONFIG_FILE" not in os.environ:
        os.environ["LMCACHE_CONFIG_FILE"] = "src/config/lmcache_config.yaml"
        
    run_profiling(args.model, args.context)
