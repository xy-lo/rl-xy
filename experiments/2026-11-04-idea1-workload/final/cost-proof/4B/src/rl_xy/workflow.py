"""Lower the frozen synchronous-GRPO contract into AstraSim-level operations."""

from __future__ import annotations

import math
from typing import Any

from .costs import compute_duration_ns, dense_linear_flops
from .model import parameter_totals
from .types import CaseConfig, Operation, Trajectory
from .operator_ir import KernelSpec, Matmul
from .execution_contract import prompt_for, group_for, demean_advantages, local_qkv_width


def linear_backward(tokens: int, inputs: int, outputs: int) -> tuple[Matmul, ...]:
    return (
        Matmul(tokens, inputs, outputs, a_bytes=4, b_bytes=2, output_bytes=4, direction="dinput"),
        Matmul(inputs, outputs, tokens, a_bytes=2, b_bytes=4, output_bytes=4, direction="dweight"),
    )


REQUIRED_PHASES = (
    "input",
    "rollout",
    "reward",
    "reference",
    "old_policy",
    "advantage",
    "train_A",
    "train_B",
    "clip",
    "optimizer",
    "publish",
    "cleanup",
)


class WorkflowBuilder:
    """Build deterministic rank-local chains with explicit communication sync.

    Each participating rank receives exactly its prior local node as a control
    dependency.  P2P and collective nodes join those local chains; rank-0-only
    host/control work is never mirrored onto rank 1.  This conservative policy
    is the frozen baseline S, not a limitation of the operation schema.
    """

    def __init__(self, case: CaseConfig) -> None:
        if case.model.qk_norm:
            raise ValueError("Qwen3 requires the qwen3_baselines.py execution contract")
        self.case = case
        self.operations: list[Operation] = []
        self.next_node_id = 1
        self.kernel_context: dict[str, Any] = {}
        self.last_node_by_rank: dict[int, int | None] = {
            rank: None for rank in range(case.hardware.dies)
        }

    def _append(
        self,
        *,
        round_id: int,
        phase: str,
        name: str,
        kind: str,
        duration_ns: int = 0,
        payload_bytes: int = 0,
        collective: str | None = None,
        src_rank: int | None = None,
        dst_rank: int | None = None,
        flops: int = 0,
        vector_ops: int = 0,
        tensor_bytes: int = 0,
        metadata: dict[str, Any] | None = None,
        participant_ranks: tuple[int, ...] | None = None,
        kernel_spec: KernelSpec | None = None,
    ) -> Operation:
        node_id = self.next_node_id
        self.next_node_id += 1
        participants = (
            tuple(range(self.case.hardware.dies))
            if participant_ranks is None
            else tuple(sorted(set(participant_ranks)))
        )
        if not participants:
            raise ValueError(f"{name}: operation must participate on at least one rank")
        dependencies_by_rank = {
            rank: ()
            if self.last_node_by_rank[rank] is None
            else (self.last_node_by_rank[rank],)
            for rank in participants
        }
        dependencies = tuple(
            sorted({item for values in dependencies_by_rank.values() for item in values})
        )
        op = Operation(
            node_id=node_id,
            op_id=f"r{round_id:02d}.n{node_id:05d}.{name}",
            round_id=round_id,
            phase=phase,
            name=name,
            kind=kind,  # type: ignore[arg-type]
            duration_ns=duration_ns,
            payload_bytes=payload_bytes,
            collective=collective,
            src_rank=src_rank,
            dst_rank=dst_rank,
            dependencies=dependencies,
            flops=flops,
            vector_ops=vector_ops,
            tensor_bytes=tensor_bytes,
            metadata={**self.kernel_context, **({} if metadata is None else metadata)},
            participant_ranks=participants,
            dependencies_by_rank=dependencies_by_rank,
            kernel_spec=kernel_spec,
        )
        self.operations.append(op)
        for rank in participants:
            self.last_node_by_rank[rank] = node_id
        return op

    def compute(
        self,
        round_id: int,
        phase: str,
        name: str,
        *,
        flops: int = 0,
        vector_ops: int = 0,
        tensor_bytes: int = 0,
        duration_ns: int | None = None,
        metadata: dict[str, Any] | None = None,
        participant_ranks: tuple[int, ...] | None = None,
        matmuls: tuple[Matmul, ...] = (),
    ) -> Operation:
        duration = (
            compute_duration_ns(self.case, flops=flops, vector_ops=vector_ops)
            if duration_ns is None
            else duration_ns
        )
        if duration <= 0:
            raise ValueError(f"{name}: compute duration must be positive")
        return self._append(
            round_id=round_id,
            phase=phase,
            name=name,
            kind="compute",
            duration_ns=duration,
            flops=flops,
            vector_ops=vector_ops,
            tensor_bytes=tensor_bytes,
            metadata=metadata,
            participant_ranks=participant_ranks,
            kernel_spec=KernelSpec(name.rsplit(".", 1)[-1], matmuls=matmuls, vector_ops=vector_ops)
            if matmuls else None,
        )

    def collective(
        self,
        round_id: int,
        phase: str,
        name: str,
        collective: str,
        payload_bytes: int,
        metadata: dict[str, Any] | None = None,
    ) -> Operation:
        if payload_bytes <= 0:
            raise ValueError(f"{name}: collective payload must be positive")
        if collective not in {"ALL_REDUCE", "ALL_GATHER", "REDUCE_SCATTER", "ALL_TO_ALL"}:
            raise ValueError(f"unsupported AstraSim collective {collective}")
        return self._append(
            round_id=round_id,
            phase=phase,
            name=name,
            kind="collective",
            payload_bytes=payload_bytes,
            collective=collective,
            metadata=metadata,
        )

    def p2p(
        self,
        round_id: int,
        phase: str,
        name: str,
        src_rank: int,
        dst_rank: int,
        payload_bytes: int,
        metadata: dict[str, Any] | None = None,
    ) -> Operation:
        if src_rank == dst_rank:
            raise ValueError("p2p endpoints must differ")
        if not (0 <= src_rank < self.case.hardware.dies):
            raise ValueError(f"invalid src_rank {src_rank}")
        if not (0 <= dst_rank < self.case.hardware.dies):
            raise ValueError(f"invalid dst_rank {dst_rank}")
        if payload_bytes <= 0:
            raise ValueError(f"{name}: p2p payload must be positive")
        return self._append(
            round_id=round_id,
            phase=phase,
            name=name,
            kind="p2p",
            payload_bytes=payload_bytes,
            src_rank=src_rank,
            dst_rank=dst_rank,
            metadata=metadata,
            participant_ranks=(src_rank, dst_rank),
        )

    def _forward(
        self,
        *,
        round_id: int,
        phase: str,
        prefix: str,
        tokens: int,
        context_tokens: int,
        target_positions: int,
        probability_mode: bool,
        sample_mode: bool,
        model_role: str,
    ) -> None:
        case = self.case
        model = case.model
        tp = case.strategy.tensor_parallel
        local_heads = model.query_heads // tp
        hidden_bf16 = tokens * model.hidden_size * model.actor_dtype_bytes
        self.kernel_context = {"tokens": tokens, "context_tokens": context_tokens,
                               "forward_prefix": prefix, "model_role": model_role,
                               "target_positions": target_positions,
                               "save_tape": phase.startswith("train_")}
        self.compute(
            round_id,
            phase,
            f"{prefix}.embedding",
            vector_ops=tokens * model.hidden_size,
            tensor_bytes=hidden_bf16,
            metadata={"model_role": model_role, "tokens": tokens},
        )
        self.collective(
            round_id,
            phase,
            f"{prefix}.embedding_tp_sum",
            "ALL_REDUCE",
            hidden_bf16,
            {"semantic": "vocab_parallel_embedding_partial_sum", "dtype": "bf16"},
        )
        if model.rotary_embedding:
            self.compute(round_id, phase, f"{prefix}.rope_table",
                         vector_ops=tokens * model.head_dim * 8,
                         tensor_bytes=tokens * model.head_dim * 2 * 4,
                         metadata={"dtype": "fp32", "semantic": "default_rope_sin_cos_explicit_analytic_cost"})
        for layer in range(model.layers):
            base = f"{prefix}.layer{layer}"
            self.compute(
                round_id,
                phase,
                f"{base}.rmsnorm_attn",
                vector_ops=tokens
                * model.hidden_size
                * case.costs.rmsnorm_vector_ops_per_element,
                tensor_bytes=hidden_bf16,
                metadata={"dtype": "bf16", "saved_for_backward": probability_mode and phase.startswith("train_")},
            )
            qkv_flops = dense_linear_flops(
                tokens, model.hidden_size, local_qkv_width(case)
            )
            self.compute(
                round_id,
                phase,
                f"{base}.qkv",
                flops=qkv_flops,
                matmuls=(Matmul(tokens, local_qkv_width(case), model.hidden_size),),
                tensor_bytes=tokens * local_qkv_width(case) * model.actor_dtype_bytes,
                metadata={"layout": "column_parallel", "dtype": "bf16"},
            )
            if model.qkv_bias:
                self.compute(round_id, phase, f"{base}.qkv_bias_add",
                             vector_ops=tokens * local_qkv_width(case),
                             tensor_bytes=tokens * local_qkv_width(case) * 2,
                             metadata={"dtype": "bf16", "local_bias_elements": local_qkv_width(case)})
            if model.rotary_embedding:
                qk = (model.query_heads + model.kv_heads) * model.head_dim // tp
                self.compute(round_id, phase, f"{base}.rope_apply",
                             vector_ops=tokens * qk * 3, tensor_bytes=tokens * qk * 2,
                             metadata={"dtype": "bf16", "semantic": "rotate_half_q_k_only"})
            if model.query_heads != model.kv_heads:
                self.compute(round_id, phase, f"{base}.gqa_expand_kv",
                             vector_ops=2 * context_tokens * local_heads * model.head_dim,
                             tensor_bytes=2 * context_tokens * local_heads * model.head_dim * 2,
                             metadata={"dtype": "bf16", "head_replication": model.query_heads // model.kv_heads,
                                       "semantic": "explicit_materialized_repeat_kv_not_extra_cache"})
            attention_flops = 2 * local_heads * tokens * context_tokens * model.head_dim
            self.compute(
                round_id,
                phase,
                f"{base}.attention_scores",
                flops=attention_flops,
                matmuls=(Matmul(tokens, context_tokens, model.head_dim, batches=local_heads, output_bytes=4),),
                tensor_bytes=local_heads * tokens * context_tokens * 4,
                metadata={"dtype": "fp32", "context_tokens": context_tokens},
            )
            if case.schema_version == "1.1":
                self.compute(round_id, phase, f"{base}.attention_scale_causal_mask",
                             vector_ops=2*local_heads*tokens*context_tokens,
                             tensor_bytes=local_heads*tokens*context_tokens*4,
                             metadata={"dtype":"fp32","semantic":"scale_by_inverse_sqrt_head_dim_and_offset_causal_mask",
                                       "query_position_offset":context_tokens-tokens})
            self.compute(
                round_id,
                phase,
                f"{base}.attention_softmax",
                vector_ops=local_heads
                * tokens
                * context_tokens
                * case.costs.softmax_vector_ops_per_element,
                tensor_bytes=local_heads * tokens * context_tokens * 4,
                metadata={"dtype": "fp32", "dense_attention": True},
            )
            self.compute(
                round_id,
                phase,
                f"{base}.attention_value",
                flops=attention_flops,
                matmuls=(Matmul(tokens, model.head_dim, context_tokens, batches=local_heads, a_bytes=4),),
                tensor_bytes=tokens * model.hidden_size // tp * model.actor_dtype_bytes,
                metadata={"dtype": "bf16"},
            )
            o_flops = dense_linear_flops(
                tokens, model.hidden_size // tp, model.hidden_size
            )
            self.compute(
                round_id,
                phase,
                f"{base}.o_proj_partial",
                flops=o_flops,
                matmuls=(Matmul(tokens, model.hidden_size, model.hidden_size // tp),),
                tensor_bytes=hidden_bf16,
                metadata={"layout": "row_parallel", "dtype": "bf16"},
            )
            self.collective(
                round_id,
                phase,
                f"{base}.o_proj_tp_sum",
                "ALL_REDUCE",
                hidden_bf16,
                {"semantic": "row_parallel_partial_sum", "dtype": "bf16"},
            )
            self.compute(
                round_id,
                phase,
                f"{base}.residual_attn",
                vector_ops=tokens * model.hidden_size,
                tensor_bytes=hidden_bf16,
                metadata={"dtype": "bf16"},
            )
            self.compute(
                round_id,
                phase,
                f"{base}.rmsnorm_ffn",
                vector_ops=tokens
                * model.hidden_size
                * case.costs.rmsnorm_vector_ops_per_element,
                tensor_bytes=hidden_bf16,
                metadata={"dtype": "bf16"},
            )
            gate_up_flops = 2 * dense_linear_flops(
                tokens, model.hidden_size, model.ffn_size // tp
            )
            self.compute(
                round_id,
                phase,
                f"{base}.gate_up",
                flops=gate_up_flops,
                matmuls=(Matmul(tokens, 2 * model.ffn_size // tp, model.hidden_size),),
                tensor_bytes=2 * tokens * model.ffn_size // tp * model.actor_dtype_bytes,
                metadata={"layout": "column_parallel", "dtype": "bf16"},
            )
            self.compute(
                round_id,
                phase,
                f"{base}.swiglu",
                vector_ops=tokens
                * (model.ffn_size // tp)
                * case.costs.swiglu_vector_ops_per_element,
                tensor_bytes=tokens * model.ffn_size // tp * model.actor_dtype_bytes,
                metadata={"dtype": "bf16"},
            )
            down_flops = dense_linear_flops(
                tokens, model.ffn_size // tp, model.hidden_size
            )
            self.compute(
                round_id,
                phase,
                f"{base}.down_partial",
                flops=down_flops,
                matmuls=(Matmul(tokens, model.hidden_size, model.ffn_size // tp),),
                tensor_bytes=hidden_bf16,
                metadata={"layout": "row_parallel", "dtype": "bf16"},
            )
            self.collective(
                round_id,
                phase,
                f"{base}.down_tp_sum",
                "ALL_REDUCE",
                hidden_bf16,
                {"semantic": "row_parallel_partial_sum", "dtype": "bf16"},
            )
            self.compute(
                round_id,
                phase,
                f"{base}.residual_ffn",
                vector_ops=tokens * model.hidden_size,
                tensor_bytes=hidden_bf16,
                metadata={"dtype": "bf16"},
            )
        self.compute(
            round_id,
            phase,
            f"{prefix}.final_rmsnorm",
            vector_ops=tokens
            * model.hidden_size
            * case.costs.rmsnorm_vector_ops_per_element,
            tensor_bytes=hidden_bf16,
            metadata={"dtype": "bf16"},
        )
        logits_flops = dense_linear_flops(
            tokens, model.hidden_size, model.vocab_size // tp
        )
        logits_bytes = tokens * model.vocab_size // tp * 4
        self.compute(
            round_id,
            phase,
            f"{prefix}.lm_head_local",
            flops=logits_flops,
            matmuls=(Matmul(tokens, model.vocab_size // tp, model.hidden_size, output_bytes=4),),
            tensor_bytes=logits_bytes,
            metadata={"layout": "vocab_parallel", "dtype": "fp32"},
        )
        if probability_mode:
            target_bytes = target_positions * 4
            self.compute(
                round_id,
                phase,
                f"{prefix}.distributed_logsoftmax_local",
                vector_ops=target_positions
                * (model.vocab_size // tp)
                * case.costs.softmax_vector_ops_per_element,
                tensor_bytes=target_positions * model.vocab_size // tp * 4,
                metadata={"dtype": "fp32", "target_positions": target_positions},
            )
            for reduction in ("max", "sumexp", "target_logit"):
                self.collective(
                    round_id,
                    phase,
                    f"{prefix}.logsoftmax_{reduction}",
                    "ALL_REDUCE",
                    target_bytes,
                    {"semantic": f"distributed_logsoftmax_{reduction}", "dtype": "fp32"},
                )
        if sample_mode:
            local_logits_bytes = model.vocab_size // tp * 4
            self.p2p(
                round_id,
                phase,
                f"{prefix}.sample_logits_rank1_to_rank0",
                1,
                0,
                local_logits_bytes,
                {"semantic": "single_owner_sampling_gather", "dtype": "fp32"},
            )
            self.compute(
                round_id,
                phase,
                f"{prefix}.sample_on_rank0",
                vector_ops=model.vocab_size * case.costs.sample_vector_ops_per_token,
                tensor_bytes=model.vocab_size * 4,
                metadata={
                    "logical_executor": "rank0",
                    "adapter_execution": "rank_local_synthetic_replay",
                },
                participant_ranks=(0,),
            )
            self.p2p(
                round_id,
                phase,
                f"{prefix}.sample_token_rank0_to_rank1",
                0,
                1,
                4,
                {"semantic": "replicate_sampled_token", "dtype": "int32"},
            )

        self.kernel_context = {}

    def _backward(
        self,
        *,
        round_id: int,
        phase: str,
        prefix: str,
        tokens: int,
        target_positions: int,
    ) -> None:
        case = self.case
        model = case.model
        tp = case.strategy.tensor_parallel
        hidden_grad_bytes = tokens * model.hidden_size * model.gradient_dtype_bytes
        self.kernel_context = {"tokens": tokens, "context_tokens": tokens,
                               "backward_prefix": prefix, "model_role": "actor",
                               "target_positions": target_positions}
        lm_forward = dense_linear_flops(tokens, model.hidden_size, model.vocab_size // tp)
        self.compute(
            round_id,
            phase,
            f"{prefix}.lm_head_backward_dinput_dweight",
            flops=2 * lm_forward,
            matmuls=linear_backward(tokens, model.hidden_size, model.vocab_size // tp),
            tensor_bytes=hidden_grad_bytes,
            metadata={"dtype": "fp32", "explicit_products": ["dinput", "dweight"]},
        )
        self.collective(
            round_id,
            phase,
            f"{prefix}.lm_head_input_grad_tp_sum",
            "ALL_REDUCE",
            hidden_grad_bytes,
            {"semantic": "column_parallel_input_gradient", "dtype": "fp32"},
        )
        self.compute(
            round_id,
            phase,
            f"{prefix}.final_rmsnorm_backward",
            vector_ops=tokens
            * model.hidden_size
            * case.costs.rmsnorm_vector_ops_per_element
            * 2,
            tensor_bytes=hidden_grad_bytes,
            metadata={"dtype": "fp32", "parameter_gradient": "final_rmsnorm"},
        )
        local_heads = model.query_heads // tp
        for layer in reversed(range(model.layers)):
            base = f"{prefix}.layer{layer}"
            down_forward = dense_linear_flops(tokens, model.ffn_size // tp, model.hidden_size)
            gate_up_forward = 2 * dense_linear_flops(
                tokens, model.hidden_size, model.ffn_size // tp
            )
            self.compute(
                round_id,
                phase,
                f"{base}.ffn_down_backward",
                flops=2 * down_forward,
                matmuls=linear_backward(tokens, model.ffn_size // tp, model.hidden_size),
                tensor_bytes=hidden_grad_bytes,
                metadata={"dtype": "fp32", "saved_tensor_required": True},
            )
            self.compute(
                round_id,
                phase,
                f"{base}.swiglu_backward",
                vector_ops=tokens
                * (model.ffn_size // tp)
                * case.costs.swiglu_vector_ops_per_element
                * 2,
                tensor_bytes=tokens * model.ffn_size // tp * 4,
                metadata={"dtype": "fp32", "saved_tensor_required": True},
            )
            self.compute(
                round_id,
                phase,
                f"{base}.gate_up_backward",
                flops=2 * gate_up_forward,
                matmuls=linear_backward(tokens, model.hidden_size, 2 * model.ffn_size // tp),
                tensor_bytes=hidden_grad_bytes,
                metadata={"dtype": "fp32", "explicit_products": ["dinput", "dweight"]},
            )
            self.collective(
                round_id,
                phase,
                f"{base}.gate_up_input_grad_tp_sum",
                "ALL_REDUCE",
                hidden_grad_bytes,
                {"semantic": "column_parallel_input_gradient", "dtype": "fp32"},
            )
            self.compute(
                round_id,
                phase,
                f"{base}.rmsnorm_ffn_backward",
                vector_ops=tokens
                * model.hidden_size
                * case.costs.rmsnorm_vector_ops_per_element
                * 2,
                tensor_bytes=hidden_grad_bytes,
                metadata={"dtype": "fp32"},
            )
            o_forward = dense_linear_flops(tokens, model.hidden_size // tp, model.hidden_size)
            qkv_forward = dense_linear_flops(
                tokens, model.hidden_size, local_qkv_width(case)
            )
            attention_forward = 4 * local_heads * tokens * tokens * model.head_dim
            self.compute(
                round_id,
                phase,
                f"{base}.o_proj_backward",
                flops=2 * o_forward,
                matmuls=linear_backward(tokens, model.hidden_size // tp, model.hidden_size),
                tensor_bytes=hidden_grad_bytes,
                metadata={"dtype": "fp32"},
            )
            self.compute(
                round_id,
                phase,
                f"{base}.attention_backward",
                flops=3 * attention_forward,
                matmuls=(
                    Matmul(tokens, tokens, model.head_dim, batches=local_heads, a_bytes=4, output_bytes=4, direction="attention_backward"),
                    Matmul(tokens, model.head_dim, tokens, batches=local_heads, a_bytes=4, b_bytes=4, output_bytes=4, direction="attention_backward"),
                    Matmul(tokens, model.head_dim, tokens, batches=local_heads, a_bytes=4, output_bytes=4, direction="attention_backward"),
                    Matmul(tokens, model.head_dim, tokens, batches=local_heads, a_bytes=4, output_bytes=4, direction="attention_backward"),
                ),
                vector_ops=local_heads
                * tokens
                * tokens
                * case.costs.softmax_vector_ops_per_element
                * 2,
                tensor_bytes=hidden_grad_bytes,
                metadata={"dtype": "fp32", "saved_softmax_required": True},
            )
            if model.query_heads != model.kv_heads:
                compact = 2 * tokens * model.kv_heads * model.head_dim // tp
                repeats = model.query_heads // model.kv_heads
                self.compute(round_id, phase, f"{base}.gqa_reduce_kv_gradients",
                             vector_ops=compact * (repeats - 1), tensor_bytes=compact * 4,
                             metadata={"dtype": "fp32", "semantic": "sum_repeated_heads_before_compact_kv_projection"})
            if model.rotary_embedding:
                qk = (model.query_heads + model.kv_heads) * model.head_dim // tp
                self.compute(round_id, phase, f"{base}.rope_backward",
                             vector_ops=tokens * qk * 3, tensor_bytes=tokens * qk * 4,
                             metadata={"dtype": "fp32", "semantic": "transpose_rotation_q_k"})
            if model.qkv_bias:
                self.compute(round_id, phase, f"{base}.qkv_bias_backward",
                             vector_ops=max(1, (tokens - 1) * local_qkv_width(case)),
                             tensor_bytes=local_qkv_width(case) * 4,
                             metadata={"dtype": "fp32", "semantic": "reduce_token_axis_for_qkv_bias"})
            self.compute(
                round_id,
                phase,
                f"{base}.qkv_backward",
                flops=2 * qkv_forward,
                matmuls=linear_backward(tokens, model.hidden_size, local_qkv_width(case)),
                tensor_bytes=hidden_grad_bytes,
                metadata={"dtype": "fp32", "explicit_products": ["dinput", "dweight"]},
            )
            self.collective(
                round_id,
                phase,
                f"{base}.qkv_input_grad_tp_sum",
                "ALL_REDUCE",
                hidden_grad_bytes,
                {"semantic": "column_parallel_input_gradient", "dtype": "fp32"},
            )
            self.compute(
                round_id,
                phase,
                f"{base}.rmsnorm_attn_backward",
                vector_ops=tokens
                * model.hidden_size
                * case.costs.rmsnorm_vector_ops_per_element
                * 2,
                tensor_bytes=hidden_grad_bytes,
                metadata={"dtype": "fp32"},
            )
        self.compute(
            round_id,
            phase,
            f"{prefix}.embedding_backward",
            vector_ops=tokens * model.hidden_size * 2,
            tensor_bytes=hidden_grad_bytes,
            metadata={
                "dtype": "fp32",
                "parameter_gradient": "token_embedding",
                "semantic": "vocab_parallel_scatter_add",
            },
        )
        local_params = parameter_totals(case)["local_parameter_elements_per_die"]
        if model.tie_word_embeddings:
            self.compute(round_id, phase, f"{prefix}.tied_embedding_gradient_merge",
                         vector_ops=model.vocab_size * model.hidden_size // tp,
                         tensor_bytes=model.vocab_size * model.hidden_size // tp * 4,
                         metadata={"dtype": "fp32", "semantic": "lm_head_plus_embedding_into_one_gradient_storage",
                                   "gradient_alias": "token_embedding"})
        self.compute(
            round_id,
            phase,
            f"{prefix}.gradient_accumulate_rmw",
            vector_ops=local_params * 2,
            tensor_bytes=local_params * model.gradient_dtype_bytes,
            metadata={
                "dtype": "fp32",
                "semantic": "read_modify_write_gradient_buffer",
                "global_loss_denominator": sum(
                    len(item.response_token_ids) for item in case.workload.trajectories
                ),
                "microbatch_target_positions": target_positions,
            },
        )

        self.kernel_context = {}

    def build_round(self, round_offset: int) -> None:
        case = self.case
        round_id = case.workload.round_id + round_offset
        version = case.workload.behavior_version + round_offset
        prompt_len = len(case.workload.prompt_token_ids)
        advantages = demean_advantages(case)
        prompt_bytes = (sum(len(prompt_for(case, group[0])) for group in
                        [[t for t in case.workload.trajectories if group_for(case, t) == gid]
                         for gid in dict.fromkeys(group_for(case, t) for t in case.workload.trajectories)]) * 4)
        input_ns = case.hardware.host_latency_ns + math.ceil(
            prompt_bytes / case.hardware.host_bytes_per_ns
        )
        self.compute(
            round_id,
            "input",
            "host_to_rank0_prompt",
            duration_ns=input_ns,
            tensor_bytes=prompt_bytes,
            metadata={
                "logical_executor": "host_link",
                "adapter_execution": "rank0_synthetic_host_replay",
                "behavior_version": version,
            },
            participant_ranks=(0,),
        )
        self.p2p(
            round_id,
            "input",
            "prompt_rank0_to_rank1",
            0,
            1,
            prompt_bytes,
            {"dtype": "int32", "behavior_version": version},
        )

        for trajectory in case.workload.trajectories:
            prompt_len = len(prompt_for(case, trajectory))
            prefix = f"rollout_{trajectory.trajectory_id}_prefill"
            self._forward(
                round_id=round_id,
                phase="rollout",
                prefix=prefix,
                tokens=prompt_len,
                context_tokens=prompt_len,
                target_positions=0,
                probability_mode=False,
                sample_mode=True,
                model_role="actor",
            )
        max_response = max(len(item.response_token_ids) for item in case.workload.trajectories)
        for decode_index in range(1, max_response):
            for trajectory in case.workload.trajectories:
                if decode_index >= len(trajectory.response_token_ids):
                    continue
                self._forward(
                    round_id=round_id,
                    phase="rollout",
                    prefix=f"rollout_{trajectory.trajectory_id}_decode{decode_index}",
                    tokens=1,
                    context_tokens=len(prompt_for(case, trajectory)) + decode_index,
                    target_positions=0,
                    probability_mode=False,
                    sample_mode=True,
                    model_role="actor",
                )

        response_bytes = sum(
            len(item.response_token_ids) * 4 for item in case.workload.trajectories
        )
        reward_transfer_ns = case.hardware.host_latency_ns + math.ceil(
            response_bytes / case.hardware.host_bytes_per_ns
        )
        self.compute(
            round_id,
            "reward",
            "device_to_host_trajectory_payload",
            duration_ns=reward_transfer_ns,
            tensor_bytes=response_bytes,
            metadata={
                "logical_executor": "host_link",
                "adapter_execution": "rank0_synthetic_host_replay",
            },
            participant_ranks=(0,),
        )
        self.compute(
            round_id,
            "reward",
            "rule_reward",
            duration_ns=case.costs.reward_cpu_ns,
            tensor_bytes=len(case.workload.trajectories) * 4,
            metadata={
                "logical_executor": "host_controller",
                "adapter_execution": "rank0_synthetic_host_replay",
                "rewards": [item.reward for item in case.workload.trajectories],
            },
            participant_ranks=(0,),
        )
        self.p2p(
            round_id,
            "reward",
            "reward_rank0_to_rank1",
            0,
            1,
            len(case.workload.trajectories) * 4,
            {"dtype": "fp32", "semantic": "reward_result_and_phase_fence"},
        )

        for model_role, phase in (("reference", "reference"), ("actor", "old_policy")):
            for trajectory in case.workload.trajectories:
                response_len = len(trajectory.response_token_ids)
                teacher_tokens = len(prompt_for(case, trajectory)) + response_len - 1
                self._forward(
                    round_id=round_id,
                    phase=phase,
                    prefix=f"{phase}_{trajectory.trajectory_id}",
                    tokens=teacher_tokens,
                    context_tokens=teacher_tokens,
                    target_positions=response_len,
                    probability_mode=True,
                    sample_mode=False,
                    model_role=model_role,
                )

        self.compute(
            round_id,
            "advantage",
            "group_advantage_demean",
            duration_ns=case.costs.advantage_cpu_ns,
            tensor_bytes=len(case.workload.trajectories) * 4,
            metadata={
                "logical_executor": "host_controller",
                "adapter_execution": "rank0_synthetic_host_replay",
                "advantages": [
                    advantages[item.trajectory_id]
                    for item in case.workload.trajectories
                ],
            },
            participant_ranks=(0,),
        )
        self.p2p(
            round_id,
            "advantage",
            "advantage_rank0_to_rank1",
            0,
            1,
            len(case.workload.trajectories) * 4,
            {"dtype": "fp32", "group_id": case.workload.group_id},
        )

        local_params = parameter_totals(case)["local_parameter_elements_per_die"]
        self.compute(
            round_id,
            f"train_{case.workload.trajectories[0].trajectory_id}",
            "zero_gradient_buffer",
            vector_ops=local_params,
            tensor_bytes=local_params * case.model.gradient_dtype_bytes,
            metadata={
                "writes": ["gradient"],
                "initial_gradient_value": 0,
                "reset_each_round": True,
                "dtype": "fp32",
            },
        )

        for trajectory in case.workload.trajectories:
            phase = f"train_{trajectory.trajectory_id}"
            response_len = len(trajectory.response_token_ids)
            teacher_tokens = len(prompt_for(case, trajectory)) + response_len - 1
            self._forward(
                round_id=round_id,
                phase=phase,
                prefix=f"{phase}_forward",
                tokens=teacher_tokens,
                context_tokens=teacher_tokens,
                target_positions=response_len,
                probability_mode=True,
                sample_mode=False,
                model_role="actor",
            )
            self.compute(
                round_id,
                phase,
                f"{phase}_loss",
                vector_ops=response_len * 12,
                tensor_bytes=response_len * 4,
                metadata={
                    "dtype": "fp32",
                    "global_denominator": sum(
                        len(item.response_token_ids)
                        for item in case.workload.trajectories
                    ),
                    "advantage": advantages[trajectory.trajectory_id],
                },
            )
            local_vocab = case.model.vocab_size // case.strategy.tensor_parallel
            self.compute(
                round_id,
                phase,
                f"{phase}_loss_backward",
                vector_ops=response_len * 12,
                tensor_bytes=response_len * local_vocab * 4,
                metadata={
                    "dtype": "fp32",
                    "semantic": "distributed_loss_to_local_logits_gradient",
                    "local_vocab_size": local_vocab,
                    "target_positions": response_len,
                    "global_denominator": sum(
                        len(item.response_token_ids)
                        for item in case.workload.trajectories
                    ),
                },
            )
            self._backward(
                round_id=round_id,
                phase=phase,
                prefix=f"{phase}_backward",
                tokens=teacher_tokens,
                target_positions=response_len,
            )

        local_params = parameter_totals(case)["local_parameter_elements_per_die"]
        self.compute(
            round_id,
            "clip",
            "local_gradient_square_sum",
            vector_ops=local_params
            * case.costs.gradient_clip_vector_ops_per_local_parameter,
            tensor_bytes=local_params * case.model.gradient_dtype_bytes,
            metadata={"replicated_parameter_rule": "owner_only"},
        )
        self.collective(
            round_id,
            "clip",
            "global_gradient_norm_sum",
            "ALL_REDUCE",
            4,
            {"dtype": "fp32", "semantic": "global_norm_scalar"},
        )
        self.compute(
            round_id,
            "clip",
            "apply_gradient_clip",
            vector_ops=local_params,
            tensor_bytes=local_params * case.model.gradient_dtype_bytes,
            metadata={"dtype": "fp32"},
        )
        self.compute(
            round_id,
            "optimizer",
            "adamw_read_modify_write",
            vector_ops=local_params
            * case.costs.optimizer_vector_ops_per_local_parameter,
            tensor_bytes=local_params
            * (
                case.model.gradient_dtype_bytes
                + case.model.master_dtype_bytes
                + 2 * case.model.optimizer_dtype_bytes
                + case.model.actor_dtype_bytes
            ),
            metadata={
                "optimizer": "adamw_shape_only",
                "actor_version_in": version,
                "actor_version_out": version + 1,
                "reference_written": False,
                "reads": ["gradient", "master", "adam_m", "adam_v"],
                "writes": ["master", "adam_m", "adam_v", "actor"],
            },
        )
        self.collective(
            round_id,
            "publish",
            "all_shards_updated_barrier",
            "ALL_REDUCE",
            4,
            {"semantic": "version_publish_barrier", "new_version": version + 1},
        )
        self.compute(
            round_id,
            "publish",
            "publish_actor_version",
            duration_ns=case.costs.control_marker_ns,
            tensor_bytes=4,
            metadata={
                "published_version": version + 1,
                "cost_source": "synthetic_control_marker_ns",
            },
        )
        self.compute(
            round_id,
            "cleanup",
            "release_round_temporaries",
            duration_ns=case.costs.control_marker_ns,
            tensor_bytes=1,
            metadata={
                "kv_remaining_bytes": 0,
                "reference_version": case.workload.behavior_version,
                "cost_source": "synthetic_control_marker_ns",
            },
        )


def build_workflow(case: CaseConfig) -> list[Operation]:
    builder = WorkflowBuilder(case)
    for round_offset in range(case.rounds):
        builder.build_round(round_offset)
    from .training_schedule import expand_updates
    return expand_updates(case, builder.operations)

