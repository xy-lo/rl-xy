"""Exact toy-Transformer parameter and persistent-state accounting."""

from __future__ import annotations

from .types import CaseConfig, ParameterRow


def enumerate_parameters(case: CaseConfig) -> list[ParameterRow]:
    model = case.model
    tp = case.strategy.tensor_parallel
    qkv_width = model.hidden_size + 2 * model.kv_heads * model.head_dim
    rows = [
        ParameterRow(
            "token_embedding",
            model.vocab_size * model.hidden_size,
            model.vocab_size * model.hidden_size // tp,
            False,
            "vocab_parallel",
        )
    ]
    for layer in range(model.layers):
        rows.extend(
            [
                ParameterRow(
                    f"layer{layer}.qkv",
                    qkv_width * model.hidden_size,
                    qkv_width * model.hidden_size // tp,
                    False,
                    "column_parallel",
                ),
                ParameterRow(
                    f"layer{layer}.o_proj",
                    model.hidden_size * model.hidden_size,
                    model.hidden_size * model.hidden_size // tp,
                    False,
                    "row_parallel",
                ),
                ParameterRow(
                    f"layer{layer}.ffn",
                    3 * model.hidden_size * model.ffn_size,
                    3 * model.hidden_size * model.ffn_size // tp,
                    False,
                    "gate_up_column_down_row",
                ),
                ParameterRow(
                    f"layer{layer}.rmsnorm",
                    2 * model.hidden_size,
                    2 * model.hidden_size,
                    True,
                    "replicated",
                ),
            ]
        )
        if model.qkv_bias:
            rows.append(ParameterRow(f"layer{layer}.qkv_bias", qkv_width, qkv_width // tp,
                                     False, "column_parallel"))
    rows.extend(
        [
            ParameterRow(
                "final_rmsnorm",
                model.hidden_size,
                model.hidden_size,
                True,
                "replicated",
            ),
            ParameterRow(
                "lm_head",
                model.hidden_size * model.vocab_size,
                model.hidden_size * model.vocab_size // tp,
                False,
                "vocab_parallel",
            ),
        ]
    )
    if model.tie_word_embeddings:
        rows = [row for row in rows if row.parameter_id != "lm_head"]
    return rows


def parameter_totals(case: CaseConfig) -> dict[str, int]:
    rows = enumerate_parameters(case)
    unique = sum(row.unique_elements for row in rows)
    local = sum(row.local_elements_per_die for row in rows)
    model = case.model
    long_term = local * (
        model.actor_dtype_bytes
        + model.reference_dtype_bytes
        + model.master_dtype_bytes
        + 2 * model.optimizer_dtype_bytes
    )
    gradient = local * model.gradient_dtype_bytes
    return {
        "unique_parameter_elements": unique,
        "local_parameter_elements_per_die": local,
        "long_term_bytes_per_die": long_term,
        "gradient_buffer_bytes_per_die": gradient,
        "resident_with_gradient_bytes_per_die": long_term + gradient,
    }


def kv_bytes_per_processed_token_per_die(case: CaseConfig) -> int:
    model = case.model
    local_kv_heads = model.kv_heads // case.strategy.tensor_parallel
    return (
        2
        * model.layers
        * local_kv_heads
        * model.head_dim
        * model.actor_dtype_bytes
    )

