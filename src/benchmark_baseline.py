import argparse
import json
import time
import torch
import gc
from vllm import LLM, SamplingParams

def get_gpu_memory_mb():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
        return allocated, reserved
    return 0.0, 0.0

def run_benchmark(model_name, context_lengths, max_tokens=64, gpu_utilization=0.85):
    results = []
    print(f"=== Baseline Benchmark: {model_name} ===")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"GPU Memory Utilization Target: {gpu_utilization}")
    print("-" * 65)

    for ctx in context_lengths:
        print(f"\n[Testing Context Length: {ctx} tokens]")
        
        # Construct dummy prompt of approximate target context length
        # Using repeating word tokens
        prompt_text = "The quick brown fox jumps over the lazy dog. " * (ctx // 9)
        
        # Reset memory stats
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        initial_alloc, initial_res = get_gpu_memory_mb()
        
        try:
            # Initialize vLLM instance
            llm = LLM(
                model=model_name,
                dtype="half", # Turing GTX 1660 SUPER (sm_75) requires FP16 instead of BF16
                gpu_memory_utilization=gpu_utilization,
                max_model_len=max(ctx + max_tokens + 256, 16384), # Support up to 16k context window
                trust_remote_code=True,
                enforce_eager=True # Eager mode for precise memory tracking on Turing consumer GPUs
            )
            
            sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
            
            start_time = time.perf_counter()
            outputs = llm.generate([prompt_text], sampling_params)
            end_time = time.perf_counter()
            
            output_obj = outputs[0]
            prompt_tokens = len(output_obj.prompt_token_ids)
            gen_tokens = len(output_obj.outputs[0].token_ids)
            total_time = end_time - start_time
            
            peak_alloc = torch.cuda.max_memory_allocated(0) / (1024 * 1024) if torch.cuda.is_available() else 0.0
            peak_res = torch.cuda.max_memory_reserved(0) / (1024 * 1024) if torch.cuda.is_available() else 0.0
            
            tok_per_sec = gen_tokens / total_time if total_time > 0 else 0
            
            res_entry = {
                "requested_ctx": ctx,
                "actual_prompt_tokens": prompt_tokens,
                "generated_tokens": gen_tokens,
                "total_time_sec": round(total_time, 4),
                "throughput_tok_per_sec": round(tok_per_sec, 2),
                "peak_alloc_mb": round(peak_alloc, 2),
                "peak_reserved_mb": round(peak_res, 2),
                "status": "SUCCESS"
            }
            print(f" -> Actual Prompt Tokens: {prompt_tokens}")
            print(f" -> Peak VRAM Allocated: {peak_alloc:.2f} MB | Reserved: {peak_res:.2f} MB")
            print(f" -> Total Time: {total_time:.3f}s ({tok_per_sec:.2f} tok/s)")
            results.append(res_entry)

            # Cleanup vLLM worker process memory
            del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f" -> FAILED at context length {ctx}: {e}")
            results.append({
                "requested_ctx": ctx,
                "status": "FAILED",
                "error": str(e)
            })
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--output", type=str, default="baseline_results.json")
    args = parser.parse_args()

    results = run_benchmark(args.model, args.contexts)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {args.output}")
