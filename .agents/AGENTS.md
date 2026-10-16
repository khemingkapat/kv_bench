# Guidelines for Agents: kv_bench

> **Repository Role:** Execution Plane (Benchmark Engine & Slurm Tooling)  
> **Agent Persona:** **Systems Benchmarking & Runtime Engineer**  
> **Master Project Board:** [GitHub Project #9: KV-Cache Co-Optimization](https://github.com/users/khemingkapat/projects/9)  
> **Parent Coordination Hub:** [`khemingkapat/kv_co-op_hub`](https://github.com/khemingkapat/kv_co-op_hub)

---

## 1. Core Purpose & Mission

This repository contains the **executable benchmarking harness, inference runtime integrations, and Slurm cluster tooling** for evaluating KV-cache co-optimization techniques.

Agents operating in this repository act as **Systems Benchmarking & Runtime Engineers**. They implement and verify benchmark harnesses, enforce hardware memory limits, integrate optimization techniques across the 3 axes (Structural, Spatial, Temporal), and manage batch execution on HPC clusters.

---

## 2. Python Package Management (STRICT)

When working on this repository, **ALWAYS** adhere to the following package management guidelines:

* **Running commands:** Use `uv run <command>` instead of activating the virtual environment (e.g., `source .venv/bin/activate`).
* **Adding dependencies:** Use `uv add <package>` to add new packages instead of activating the virtual environment and running `pip install`.
* **Lockfile Integrity:** Keep `uv.lock` synchronized whenever `pyproject.toml` changes.

---

## 3. Git & Pull Request Lifecycle (STRICT)

### A. Feature Branching Rule
* **NEVER push directly to `main`**.
* Always create and work on a feature branch associated with the current work package:
  ```bash
  git checkout -b feat/wp<X>-<Y>-<short-description>
  # Example:
  git checkout -b feat/wp1-2-slurm-verification
  ```

### B. Pull Request Protocol
1. Push the feature branch to origin:
   ```bash
   git push -u origin feat/wp<X>-<Y>-<short-description>
   ```
2. Open a Pull Request using `gh pr create`:
   ```bash
   gh pr create \
     --title "feat(<scope>): <summary of changes> [WP<X>-<Y>]" \
     --body "<Description of changes and verification results>
   
   Closes #<CHILD_ISSUE_NUMBER>" \
     --base main
   ```
3. **Wait for Human Review:** Wait for the user to review the PR diff and merge it into `main`.
4. **Post-Merge Cleanup:** Once merged, pull the latest `main` locally and delete the local feature branch:
   ```bash
   git checkout main && git pull --prune
   git branch -d feat/wp<X>-<Y>-<short-description>
   ```

---

## 4. Mandatory 3-Stage Verification Pipeline

Before opening any Pull Request or submitting batch jobs to the cluster, execute the following verification pipeline in order:

### Stage 1: Dry-Run Verification (No GPU Needed)
Verify technique registration, argument parsing, context scaling arrays, and model resolution without allocating VRAM:
```bash
uv run main.py --technique baseline --model qwen3-4b --dry-run
```

### Stage 2: Automated Unit Testing
Ensure all unit tests pass cleanly:
```bash
uv run python -m unittest discover tests
```

### Stage 3: Slurm Batch Verification (HPC Cluster)
When running on the Slurm cluster:
1. Log in to the cluster and copy (`scp`) the relevant project files (`src/`, `tests/`, `main.py`, `benchmark.sbatch`, `pyproject.toml`, `uv.lock`) to your cluster workspace.
2. Submit the batch job via `sbatch` and wait for execution to finish with exit code 0:
   ```bash
   sbatch benchmark.sbatch
   ```
3. Retrieve the resulting `results/*.json` metrics and `slurm-*.log` output.

---

## 5. Architectural & System Boundaries

* **Hardware Caps:** Always enforce simulated or physical consumer hardware ceilings:
  * `--max-vram-gb 16.0` on the RTX 4090 cluster node (`partition=gpu4090`).
  * `--max-vram-gb 8.0` on local RTX 4060 machines.
* **Model Integrity:** All benchmarks strictly evaluate unmodified base weights in FP16/BF16. Base weights must never be quantized with AWQ/GPTQ/GGUF—the research focus is exclusively on the KV cache.
* **Model Specs & Aliases:** Always resolve models through [`src/core/config.py`](file:///home/khemi/workspace/kv_bench/src/core/config.py) using pinned Hugging Face commit SHAs and registered aliases (`qwen3-4b`, `phi4-mini`, `qwen3.5-4b`, `qwen3-1.7b`).

