"""Strict JSON loading and cross-field validation for W/H/S contracts."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .types import (
    CaseConfig,
    CostConfig,
    HardwareConfig,
    ModelConfig,
    StrategyConfig,
    Trajectory,
    WorkloadConfig,
    TokenExtent,
)


class ConfigError(ValueError):
    """Raised when a case is ambiguous or violates the frozen MVP contract."""


ROOT_FIELDS = {
    "schema_version",
    "case_id",
    "evidence_kind",
    "rounds",
    "model",
    "workload",
    "hardware",
    "strategy",
    "costs",
}
MODEL_FIELDS = set(ModelConfig.__dataclass_fields__)
MODEL_OPTIONS = {"qkv_bias", "rotary_embedding", "tie_word_embeddings"}
MODEL_EXTENSIONS = {"qk_norm"}
WORKLOAD_FIELDS = {
    "round_id",
    "group_id",
    "prompt_token_ids",
    "max_response_length",
    "behavior_version",
    "trajectories",
    "advantage_rule",
    "loss_reduction",
    "optimizer",
}
TRAJECTORY_FIELDS = {
    "trajectory_id",
    "response_token_ids",
    "finish_reason",
    "reward",
}
HARDWARE_FIELDS = set(HardwareConfig.__dataclass_fields__)
HARDWARE_OPTIONS = {'gemm_fp32_flops_per_ns'}
STRATEGY_FIELDS = set(StrategyConfig.__dataclass_fields__)
STRATEGY_OPTIONS = {"training_epochs", "optimizer_minibatch_trajectories"}
COST_FIELDS = set(CostConfig.__dataclass_fields__)


def _exact_fields(value: dict[str, Any], allowed: set[str], scope: str) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(allowed - set(value))
    if unknown or missing:
        raise ConfigError(f"{scope}: unknown={unknown}, missing={missing}")


def _positive(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer, got {value!r}")


def _non_negative(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"{name} must be a non-negative integer, got {value!r}")


def _token_ids(name: str, values: list[Any], vocab_size: int) -> tuple[int, ...]:
    if not values:
        raise ConfigError(f"{name} must not be empty")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in values):
        raise ConfigError(f"{name} must contain integers")
    if any(item < 0 or item >= vocab_size for item in values):
        raise ConfigError(f"{name} contains token outside [0, {vocab_size})")
    return tuple(values)


def _validate_case(case: CaseConfig) -> None:
    if case.schema_version not in {"1.0", "1.1"}:
        raise ConfigError(f"unsupported schema_version {case.schema_version!r}")
    if case.evidence_kind != "synthetic_fixture":
        raise ConfigError(
            "this uncalibrated toy MVP requires evidence_kind='synthetic_fixture'"
        )
    if not isinstance(case.case_id, str) or not case.case_id:
        raise ConfigError("case_id must be a non-empty string")
    _positive("rounds", case.rounds)
    model = case.model
    hardware = case.hardware
    strategy = case.strategy
    for name, value in asdict(model).items():
        if name in MODEL_OPTIONS | MODEL_EXTENSIONS:
            if type(value) is not bool:
                raise ConfigError(f"model.{name} must be boolean")
        else:
            _positive(f"model.{name}", value)
    for name, value in asdict(hardware).items():
        if name in HARDWARE_OPTIONS and value is None:continue
        _positive(f"hardware.{name}", value)
    for name, value in asdict(case.costs).items():
        _positive(f"costs.{name}", value)
    for name in (
        "tensor_parallel",
        "pipeline_parallel",
        "data_parallel",
        "training_microbatch_trajectories",
        "rollout_max_concurrency",
    ):
        _positive(f"strategy.{name}", getattr(strategy, name))
    for name in ("share_prompt_kv", "actor_train_inference_layout_compatible"):
        if not isinstance(getattr(strategy, name), bool):
            raise ConfigError(f"strategy.{name} must be boolean")
    _positive("strategy.training_epochs", strategy.training_epochs)
    _non_negative("strategy.optimizer_minibatch_trajectories", strategy.optimizer_minibatch_trajectories)
    if case.schema_version == "1.0" and (strategy.training_epochs != 1 or strategy.optimizer_minibatch_trajectories != 0):
        raise ConfigError("multi-update training requires schema 1.1")
    if not model.qk_norm and model.hidden_size != model.query_heads * model.head_dim:
        raise ConfigError("hidden_size must equal query_heads * head_dim")
    if case.schema_version == "1.0" and (any(getattr(model, k) for k in MODEL_OPTIONS)
                                        or model.kv_heads != model.query_heads):
        raise ConfigError("this first dense attention template supports MHA only")
    if model.query_heads % model.kv_heads:
        raise ConfigError("GQA requires query_heads divisible by kv_heads")
    if model.rotary_embedding and model.head_dim % 2:
        raise ConfigError("RoPE requires even head_dim")
    if (model.actor_dtype_bytes, model.reference_dtype_bytes, model.master_dtype_bytes,
            model.gradient_dtype_bytes, model.optimizer_dtype_bytes) != (2, 2, 4, 4, 4):
        raise ConfigError("current typed kernel templates require BF16 actor/reference and FP32 state/gradients")
    if hardware.dies != 2 or (case.schema_version == '1.0' and hardware.cores_per_die != 1):
        raise ConfigError("base case fixes two TP ranks; schema1.0 fixes one logical core")
    if model.query_heads % strategy.tensor_parallel != 0:
        raise ConfigError("query_heads must be divisible by tensor_parallel")
    if model.kv_heads % strategy.tensor_parallel != 0:
        raise ConfigError("kv_heads must be divisible by tensor_parallel in this MVP")
    if model.hidden_size % strategy.tensor_parallel != 0:
        raise ConfigError("hidden_size must be divisible by tensor_parallel")
    if model.ffn_size % strategy.tensor_parallel != 0:
        raise ConfigError("ffn_size must be divisible by tensor_parallel")
    if model.vocab_size % strategy.tensor_parallel != 0:
        raise ConfigError("vocab_size must be divisible by tensor_parallel")
    if strategy.tensor_parallel != hardware.dies:
        raise ConfigError("MVP requires one TP rank per die")
    if strategy.pipeline_parallel != 1 or strategy.data_parallel != 1:
        raise ConfigError("MVP fixes PP=1 and DP=1")
    if strategy.training_microbatch_trajectories != 1:
        raise ConfigError("MVP fixes one trajectory per training microbatch")
    if case.schema_version == "1.0" and strategy.rollout_max_concurrency != 2:
        raise ConfigError("toy MVP admits exactly two live rollout trajectories")
    if strategy.share_prompt_kv:
        raise ConfigError("toy MVP requires share_prompt_kv=false")
    if not strategy.actor_train_inference_layout_compatible:
        raise ConfigError(
            "toy MVP requires actor_train_inference_layout_compatible=true"
        )
    if strategy.schedule != "strict_synchronous":
        raise ConfigError("MVP only supports strict_synchronous schedule")
    if strategy.kv_admission != "reserve_public_max_length":
        raise ConfigError("MVP only supports public-max KV admission")
    workload = case.workload
    _non_negative("workload.round_id", workload.round_id)
    _positive("workload.max_response_length", workload.max_response_length)
    _non_negative("workload.behavior_version", workload.behavior_version)
    if not isinstance(workload.group_id, str) or not workload.group_id:
        raise ConfigError("workload.group_id must be a non-empty string")
    if case.schema_version == "1.0" and len(workload.trajectories) != 2:
        raise ConfigError("toy MVP requires exactly two trajectories")
    if case.schema_version == "1.1":
        from .execution_contract import prompt_for, group_for
        if not workload.trajectories or strategy.rollout_max_concurrency != len(workload.trajectories):
            raise ConfigError("1.1 reserves all trajectories; concurrency must equal trajectory count")
        groups = {}
        for t in workload.trajectories:
            if not t.trajectory_id.isalnum() or not t.trajectory_id.isascii():
                raise ConfigError("1.1 trajectory IDs must be ASCII alphanumeric to avoid prefix ambiguity")
            g = groups.setdefault(group_for(case, t), [])
            g.append(t)
        for group in groups.values():
            if len(group) < 2 or len({prompt_for(case, t) for t in group}) != 1:
                raise ConfigError("each group requires >=2 responses to the identical prompt")
    ids = [item.trajectory_id for item in workload.trajectories]
    if len(set(ids)) != len(ids):
        raise ConfigError("trajectory_id values must be unique")
    if any(len(item.response_token_ids) > workload.max_response_length for item in workload.trajectories):
        raise ConfigError("response exceeds public max_response_length")
    if any(item.reward is not None and not math.isfinite(item.reward) for item in workload.trajectories):
        raise ConfigError("trajectory rewards must be finite")
    if case.schema_version == "1.0" and any(t.reward is None for t in workload.trajectories):
        raise ConfigError("1.0 requires numeric rewards")
    if case.schema_version == "1.1":
        for group in groups.values():
            if len({t.reward is None for t in group})!=1:
                raise ConfigError("a group cannot mix known and unknown reward values")
    if workload.advantage_rule != "demean_only":
        raise ConfigError("toy MVP freezes advantage_rule=demean_only")
    if workload.loss_reduction != "global_response_token_mean":
        raise ConfigError("toy MVP freezes global response-token mean loss")
    if workload.optimizer != "adamw_shape_only":
        raise ConfigError("toy MVP performs shape/cost replay, not numeric AdamW")
    if hardware.nic_tx_bytes_per_die < hardware.d2d_packet_bytes:
        raise ConfigError("NIC TX capacity must hold one configured D2D packet")
    if hardware.nic_rx_bytes_per_die < hardware.d2d_packet_bytes:
        raise ConfigError("NIC RX capacity must hold one configured D2D packet")


def load_case(path: str | Path, rounds_override: int | None = None) -> CaseConfig:
    source = Path(path).resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("case root must be an object")
    _exact_fields(raw, ROOT_FIELDS, "root")
    model_raw = raw["model"]
    workload_raw = raw["workload"]
    hardware_raw = raw["hardware"]
    strategy_raw = raw["strategy"]
    costs_raw = raw["costs"]
    for value, scope in (
        (model_raw, "model"),
        (workload_raw, "workload"),
        (hardware_raw, "hardware"),
        (strategy_raw, "strategy"),
        (costs_raw, "costs"),
    ):
        if not isinstance(value, dict):
            raise ConfigError(f"{scope} must be an object")
    allowed_model = (MODEL_FIELDS - (MODEL_EXTENSIONS - set(model_raw))) if raw["schema_version"] == "1.1" else MODEL_FIELDS - MODEL_OPTIONS - MODEL_EXTENSIONS
    _exact_fields(model_raw, allowed_model, "model")
    _exact_fields(workload_raw, WORKLOAD_FIELDS, "workload")
    if not HARDWARE_FIELDS-HARDWARE_OPTIONS <= set(hardware_raw) or set(hardware_raw)-HARDWARE_FIELDS:
        raise ConfigError('hardware: invalid required/optional fields')
    required_strategy = STRATEGY_FIELDS - STRATEGY_OPTIONS
    if not required_strategy <= set(strategy_raw) or set(strategy_raw) - STRATEGY_FIELDS:
        raise ConfigError("strategy: invalid required/optional fields")
    _exact_fields(costs_raw, COST_FIELDS, "costs")

    model = ModelConfig(**model_raw)
    prompt = (() if raw["schema_version"] == "1.1" and workload_raw["prompt_token_ids"] == [] else
              _token_ids("prompt_token_ids", workload_raw["prompt_token_ids"], model.vocab_size))
    trajectory_values = workload_raw["trajectories"]
    if not isinstance(trajectory_values, list):
        raise ConfigError("workload.trajectories must be a list")
    trajectories: list[Trajectory] = []
    for index, item in enumerate(trajectory_values):
        if not isinstance(item, dict):
            raise ConfigError(f"trajectory[{index}] must be an object")
        lengths_only=raw["schema_version"] == "1.1" and "response_length" in item
        allowed_trajectory = ((TRAJECTORY_FIELDS-{"response_token_ids"}) | {"prompt_length","response_length","group_id"}
                              if lengths_only else TRAJECTORY_FIELDS | ({"prompt_token_ids", "group_id"} if raw["schema_version"] == "1.1" else set()))
        _exact_fields(item, allowed_trajectory, f"trajectory[{index}]")
        if not isinstance(item["trajectory_id"], str) or not item["trajectory_id"]:
            raise ConfigError(f"trajectory[{index}].trajectory_id must be a non-empty string")
        if not isinstance(item["finish_reason"], str) or not item["finish_reason"]:
            raise ConfigError(f"trajectory[{index}].finish_reason must be a non-empty string")
        if not (raw["schema_version"] == "1.1" and item["reward"] is None) and (not isinstance(item["reward"], (int, float)) or isinstance(item["reward"], bool)):
            raise ConfigError(f"trajectory[{index}].reward must be numeric")
        if lengths_only:
            _positive("prompt_length",item["prompt_length"])
            _positive("response_length",item["response_length"])
        tokens = TokenExtent(item["response_length"]) if lengths_only else _token_ids(
            f"trajectory[{index}].response_token_ids",
            item["response_token_ids"],
            model.vocab_size,
        )
        trajectories.append(
            Trajectory(
                trajectory_id=item["trajectory_id"],
                response_token_ids=tokens,
                finish_reason=item["finish_reason"],
                reward=float(item["reward"]) if item["reward"] is not None else None,
                prompt_token_ids=TokenExtent(item["prompt_length"]) if lengths_only else
                    _token_ids("trajectory.prompt_token_ids", item["prompt_token_ids"], model.vocab_size)
                    if raw["schema_version"] == "1.1" else (),
                group_id=item.get("group_id", ""),
            )
        )
        if raw["schema_version"] == "1.1" and (not isinstance(item["group_id"], str) or not item["group_id"]):
            raise ConfigError("trajectory.group_id must be a nonempty string")
    workload = WorkloadConfig(
        round_id=workload_raw["round_id"],
        group_id=workload_raw["group_id"],
        prompt_token_ids=prompt,
        max_response_length=workload_raw["max_response_length"],
        behavior_version=workload_raw["behavior_version"],
        trajectories=tuple(trajectories),
        advantage_rule=workload_raw["advantage_rule"],
        loss_reduction=workload_raw["loss_reduction"],
        optimizer=workload_raw["optimizer"],
    )
    rounds = raw["rounds"] if rounds_override is None else rounds_override
    resolved = json.loads(json.dumps(raw))
    resolved["rounds"] = rounds
    case = CaseConfig(
        schema_version=raw["schema_version"],
        case_id=raw["case_id"],
        evidence_kind=raw["evidence_kind"],
        rounds=rounds,
        model=model,
        workload=workload,
        hardware=HardwareConfig(**hardware_raw),
        strategy=StrategyConfig(**strategy_raw),
        costs=CostConfig(**costs_raw),
        source_path=source,
        resolved=resolved,
    )
    _validate_case(case)
    return case

