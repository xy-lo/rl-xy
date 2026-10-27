"""Inspect local HF decoder configuration and exact Qwen2 parameter shapes.

No weights are loaded. Importing model metadata does not imply backend support.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def inspect_hf_model(path: str | Path) -> dict:
    path = Path(path)
    raw = json.loads(path.read_text())
    if raw.get("model_type") != "qwen2" or raw.get("architectures") != ["Qwen2ForCausalLM"]:
        raise ValueError("current HF inspector supports Qwen2ForCausalLM only")
    keys = ("num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads",
            "num_key_value_heads", "vocab_size", "max_position_embeddings")
    for k in keys:
        if type(raw.get(k)) is not int or raw[k] <= 0:
            raise ValueError(f"invalid/missing {k}")
    d, q, kv = raw["hidden_size"], raw["num_attention_heads"], raw["num_key_value_heads"]
    if d % q or q % kv:
        raise ValueError("incompatible hidden/query/KV head dimensions")
    if raw.get("head_dim", d // q) != d // q:
        raise ValueError("nonstandard head_dim is not supported")
    if raw.get("hidden_act") != "silu" or raw.get("attention_dropout") != 0 or raw.get("use_sliding_window") is not False:
        raise ValueError("current inspector requires dense SwiGLU, zero attention dropout and no sliding window")
    if type(raw.get("tie_word_embeddings")) is not bool:
        raise ValueError("tie_word_embeddings must be explicit")
    if any(k in raw for k in ("quantization_config", "num_experts", "num_local_experts")):
        raise ValueError("quantized/MoE parameter layout requires another model adapter")
    width = kv * (d // q)
    layers, ffn, vocab = raw["num_hidden_layers"], raw["intermediate_size"], raw["vocab_size"]
    # Qwen2 has q/k/v projection biases and no o_proj/MLP biases.
    per_layer = {"qkv_weights": d * (d + 2 * width), "qkv_biases": d + 2 * width,
                 "o_proj": d * d, "ffn": 3 * d * ffn, "rmsnorm": 2 * d}
    unique = vocab * d + layers * sum(per_layer.values()) + d
    if not raw["tie_word_embeddings"]:
        unique += vocab * d
    blockers = ["formal H requires missing execution-rate assumptions and independent calibration; run-wafer supports explicit DP/multicore mesh scenarios",
                "full veRL algorithm/minibatch/epoch contract is not imported by model inspection"]
    if raw.get("rope_scaling") is not None:
        blockers.append("nondefault RoPE requires another execution template")
    return {
        "schema_version": "rlxy.hf-model/1", "model_type": "qwen2", "config": raw,
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "head_dim": d // q, "kv_width": width, "per_layer_parameter_elements": per_layer,
        "unique_parameter_elements": unique,
        "bf16_parameter_bytes": unique * 2,
        "bf16_kv_bytes_per_processed_token_all_layers": 2 * layers * width * 2,
        "bf16_adamw_resident_bytes_with_reference_and_gradient": unique * 20,
        "resident_scope": "actor2+reference2+master4+gradient4+adam_m4+adam_v4; excludes activation/KV/workspace",
        "current_full_grpo_executable": False, "backend_blockers": blockers,
        "limited_operator_replay": "schema1.1: TP2 single-logical-core, default RoPE/GQA/QKV-bias/tied state, one trajectory per microbatch; explicit deterministic optimizer minibatches/epochs with batch-frozen behavior",
        "scope": "parameter/shape metadata import; not numeric execution or a complete operator graph",
    }
