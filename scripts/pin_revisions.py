#!/usr/bin/env python3
"""Helper script to query and verify Hugging Face commit SHAs for model version pinning."""

import argparse
import json
import urllib.request
import sys

DEFAULT_MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "microsoft/Phi-4-mini-instruct",
    "mistralai/Ministral-3-3B-Base-2512",
    "HuggingFaceTB/SmolLM3-3B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-2B",
    "google/gemma-4-E4B-it",
    "Qwen/Qwen2.5-3B-Instruct",
]


def fetch_model_sha(repo_id: str) -> str:
    url = f"https://huggingface.co/api/models/{repo_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "kv-bench-pinning"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return data.get("sha", "UNKNOWN")
    except Exception as e:
        return f"ERROR: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Query and verify HF model commit SHAs.")
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS, help="HF model repo IDs to inspect")
    args = parser.parse_args()

    print(f"{'Model Repository':<42} {'Pinned Commit SHA'}")
    print("-" * 84)
    for model in args.models:
        sha = fetch_model_sha(model)
        print(f"{model:<42} {sha}")


if __name__ == "__main__":
    main()
