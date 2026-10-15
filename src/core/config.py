"""Shared run config and model specifications.

Kept as plain dataclasses -- no YAML/pydantic needed.
Technique-specific tunables (e.g. LMCache's chunk size) belong on that
Technique's own __init__ instead of here; see techniques/lmcache_offload.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ModelSpec:
    """Architectural metadata and pinned checkpoint configuration for a model."""

    name: str
    hf_repo_id: str
    revision: str
    arch_class: str
    num_layers: int
    num_full_attn_layers: int
    hidden_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    native_max_context: int
    weight_size_gb: float
    is_hybrid_gdn: bool = False
    is_multimodal_backbone: bool = False
    extra_vllm_args: Tuple[str, ...] = ()

    @property
    def kv_bytes_per_token_bf16(self) -> int:
        """Full-attention KV bytes per token: 2 (K,V) * 2 bytes * full_attn_layers * kv_heads * head_dim."""
        return 4 * self.num_full_attn_layers * self.num_kv_heads * self.head_dim

    def kv_cache_gb(self, context_len: int, concurrent_sessions: int = 1) -> float:
        """Calculate dynamic KV-cache memory requirement in GB for a given sequence length and concurrency."""
        return (self.kv_bytes_per_token_bf16 * context_len * concurrent_sessions) / (1024**3)


# Pinned model registry for WP1-1 benchmark testbed
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "qwen3-4b": ModelSpec(
        name="qwen3-4b",
        hf_repo_id="Qwen/Qwen3-4B-Instruct-2507",
        revision="cdbee75f17c01a7cc42f958dc650907174af0554",
        arch_class="Qwen3ForCausalLM",
        num_layers=36,
        num_full_attn_layers=36,
        hidden_size=2560,
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        native_max_context=262144,
        weight_size_gb=8.06,
    ),
    "phi4-mini": ModelSpec(
        name="phi4-mini",
        hf_repo_id="microsoft/Phi-4-mini-instruct",
        revision="cfbefacb99257ffa30c83adab238a50856ac3083",
        arch_class="Phi3ForCausalLM",
        num_layers=32,
        num_full_attn_layers=32,
        hidden_size=3072,
        num_q_heads=24,
        num_kv_heads=8,
        head_dim=128,
        native_max_context=131072,
        weight_size_gb=7.65,
    ),
    "ministral-3b": ModelSpec(
        name="ministral-3b",
        hf_repo_id="mistralai/Ministral-3-3B-Base-2512",
        revision="6f9c4b12a95b139af68670a6713616b757923735",
        arch_class="Mistral3ForConditionalGeneration",
        num_layers=26,
        num_full_attn_layers=26,
        hidden_size=3072,
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        native_max_context=262144,
        weight_size_gb=6.80,
        is_multimodal_backbone=True,
        extra_vllm_args=("--language-model-only",),
    ),
    "smollm3-3b": ModelSpec(
        name="smollm3-3b",
        hf_repo_id="HuggingFaceTB/SmolLM3-3B",
        revision="a07cc9a04f16550a088caea529712d1d335b0ac1",
        arch_class="SmolLM3ForCausalLM",
        num_layers=36,
        num_full_attn_layers=36,
        hidden_size=2048,
        num_q_heads=16,
        num_kv_heads=4,
        head_dim=128,
        native_max_context=65536,
        weight_size_gb=6.00,
    ),
    "qwen3-1.7b": ModelSpec(
        name="qwen3-1.7b",
        hf_repo_id="Qwen/Qwen3-1.7B",
        revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        arch_class="Qwen3ForCausalLM",
        num_layers=28,
        num_full_attn_layers=28,
        hidden_size=2048,
        num_q_heads=16,
        num_kv_heads=8,
        head_dim=128,
        native_max_context=40960,
        weight_size_gb=3.45,
    ),
    "qwen3.5-4b": ModelSpec(
        name="qwen3.5-4b",
        hf_repo_id="Qwen/Qwen3.5-4B",
        revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        arch_class="Qwen3_5ForConditionalGeneration",
        num_layers=32,
        num_full_attn_layers=8,  # 8 full attention layers interleaved with 24 Gated DeltaNet layers
        hidden_size=2560,
        num_q_heads=16,
        num_kv_heads=4,
        head_dim=256,
        native_max_context=262144,
        weight_size_gb=8.10,
        is_hybrid_gdn=True,
        is_multimodal_backbone=True,
        extra_vllm_args=("--language-model-only", "--mamba-cache-mode", "align"),
    ),
    "qwen3.5-2b": ModelSpec(
        name="qwen3.5-2b",
        hf_repo_id="Qwen/Qwen3.5-2B",
        revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
        arch_class="Qwen3_5ForConditionalGeneration",
        num_layers=24,
        num_full_attn_layers=6,
        hidden_size=2048,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=256,
        native_max_context=262144,
        weight_size_gb=3.80,
        is_hybrid_gdn=True,
        is_multimodal_backbone=True,
        extra_vllm_args=("--language-model-only", "--mamba-cache-mode", "align"),
    ),
    "qwen2.5-3b": ModelSpec(
        name="qwen2.5-3b",
        hf_repo_id="Qwen/Qwen2.5-3B-Instruct",
        revision="aa8e72537993ba99e69dfaafa59ed015b17504d1",
        arch_class="Qwen2ForCausalLM",
        num_layers=36,
        num_full_attn_layers=36,
        hidden_size=2048,
        num_q_heads=16,
        num_kv_heads=2,
        head_dim=128,
        native_max_context=32768,
        weight_size_gb=6.10,
    ),
}


def get_model_spec(model_name_or_id: str) -> Optional[ModelSpec]:
    """Retrieve ModelSpec by alias or HF repo ID."""
    if model_name_or_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name_or_id]
    for spec in MODEL_REGISTRY.values():
        if spec.hf_repo_id == model_name_or_id:
            return spec
    return None


def resolve_model_target(model_name_or_id: str) -> Tuple[str, Optional[str], Tuple[str, ...]]:
    """Resolve model input into (hf_repo_id, revision_sha, extra_vllm_args)."""
    spec = get_model_spec(model_name_or_id)
    if spec is not None:
        return spec.hf_repo_id, spec.revision, spec.extra_vllm_args
    return model_name_or_id, None, ()


@dataclass
class BenchmarkConfig:
    model: str = "qwen3-4b"
    contexts: list[int] = field(default_factory=lambda: [512, 1024, 2048, 4096, 8192, 16384, 32768])
    max_tokens: int = 64
    gpu_utilization: float = 0.85
    output: str | None = None
