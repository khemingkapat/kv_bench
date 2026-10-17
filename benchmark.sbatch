#!/bin/bash
#SBATCH --job-name=kv_bench
#SBATCH --output=slurm-%j.log
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

source /etc/profile
set -euo pipefail

echo "========================================="
echo "=== Host & Execution Environment      ==="
echo "========================================="
hostname
date
uname -a

echo "========================================="
echo "=== Load Cluster CUDA Module          ==="
echo "========================================="
module purge
module load cuda/12.2
module list

echo "========================================="
echo "=== GPU Information                   ==="
echo "========================================="
nvidia-smi

echo "========================================="
echo "=== Toolchain & uv Setup             ==="
echo "========================================="
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version
uv sync --frozen

echo ">> Running Automated Test Suite on Compute Node..."
uv run --no-sync python -m unittest discover tests

echo "========================================="
echo "=== KV-Bench Suite (Capped at 16GB)   ==="
echo "========================================="
# The physical GPU is an RTX 4090 (24GB VRAM).
# --max-vram-gb 16.0 enforces the consumer hardware ceiling (both in PyTorch's
# allocator and vLLM's KV allocation pool) so results strictly mirror 16GB VRAM.
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0

# Default to primary pure dense model (qwen3-4b); accepts aliases (phi4-mini, qwen3.5-4b, etc.)
MODEL="${1:-qwen3-4b}"
RUN_FULL_SWEEP="${RUN_FULL_SWEEP:-0}"
CONTEXTS="512 1024 2048 4096 8192 16384"
EVAL_CONTEXTS="512 1024 2048 4096"

echo "Model Selected: $MODEL"

echo ">> 0. Running Fast Verification Sanity Check (128 ctx, 16 tokens, 16.0 GB VRAM cap)..."
uv run --no-sync main.py \
    --technique baseline \
    --model "$MODEL" \
    --verify \
    --max-vram-gb 16.0 \
    --output results/verify_16gb.json

if [[ "$RUN_FULL_SWEEP" == "1" ]]; then
    echo ">> 1. Running Baseline Context Sweep (512 - 16384)..."
    uv run --no-sync main.py \
        --technique baseline \
        --model "$MODEL" \
        --contexts $CONTEXTS \
        --max-vram-gb 16.0 \
        --output results/baseline_16gb.json

    echo ">> 2. Running Baseline Multi-User Concurrency Workload (12 Users across contexts)..."
    uv run --no-sync main.py \
        --technique baseline \
        --model "$MODEL" \
        --workload concurrent \
        --num-users 12 \
        --contexts $EVAL_CONTEXTS \
        --max-vram-gb 16.0 \
        --output results/concurrent_baseline_16gb.json

    echo ">> 3. Running Baseline Long-Term Memory Adherence (LongMemEval across contexts)..."
    uv run --no-sync main.py \
        --technique baseline \
        --model "$MODEL" \
        --workload longmem \
        --contexts $EVAL_CONTEXTS \
        --max-vram-gb 16.0 \
        --output results/longmem_baseline_16gb.json
fi

echo "========================================="
echo "=== Benchmarks Completed Successfully ==="
echo "========================================="
date
