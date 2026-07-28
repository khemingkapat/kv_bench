# KV Cache Optimization Benchmark

This repository contains tools and scripts for benchmarking Key-Value (KV) cache optimization techniques in Large Language Models (LLMs) on consumer-grade hardware.

## Project Goals

The main objective is to conduct a systematic, empirical evaluation of KV cache optimization methods, specifically focusing on the trade-offs between memory usage, latency, throughput, and model quality.

1. **Baseline Evaluation**: Measure GPU memory usage, latency, and throughput as context length increases using unoptimized inference (e.g., standard vLLM).
2. **LMCache CPU Offloading**: Experiment with [LMCache](https://github.com/lmcache/lmcache), a method for CPU offloading of the KV cache, and compare it against the baseline.
3. **Reproduce & Extend**: Reproduce published optimization methods from the literature and systematically benchmark them on consumer hardware.

## Repository Structure

*   `src/`: Contains the core Python scripts for running the benchmarks.
*   `results/`: Directory where the JSON output of benchmark runs are saved.
*   `docs/`: Contains project documentation, including the original task definitions (`TASK.md`).
*   `main.py`: The unified entry point for running all benchmarks.

## Usage

This project uses `uv` for package management. To run a benchmark, use the `main.py` script:

### Run Baseline Benchmark
```bash
uv run main.py --mode baseline --model Qwen/Qwen2.5-0.5B --contexts 512 1024 2048 4096
```

### Run LMCache Benchmark
```bash
uv run main.py --mode lmcache --model Qwen/Qwen2.5-0.5B --contexts 512 1024 2048 4096
```

By default, results are saved in the `results/` directory as `baseline_results.json` or `lmcache_results.json`. You can override the output location with `--output`.
