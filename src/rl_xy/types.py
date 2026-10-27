"""Typed contracts shared by the planner, ledger, adapter, and reporter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from .operator_ir import KernelSpec


EvidenceKind = Literal["synthetic_fixture", "analytic_assumption", "calibrated"]
OperationKind = Literal["compute", "collective", "p2p"]


@dataclass(frozen=True)
class ModelConfig:
    layers: int
    hidden_size: int
    query_heads: int
    kv_heads: int
    head_dim: int
    ffn_size: int
    vocab_size: int
    actor_dtype_bytes: int
    reference_dtype_bytes: int
    master_dtype_bytes: int
    gradient_dtype_bytes: int
    optimizer_dtype_bytes: int
    qkv_bias: bool = False
    rotary_embedding: bool = False
    tie_word_embeddings: bool = False


@dataclass(frozen=True)
class TokenExtent:
    """A length without token values; deliberately not a sequence of fake IDs."""
    length: int

    def __len__(self):
        return self.length


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    response_token_ids: tuple[int, ...] | TokenExtent
    finish_reason: str
    reward: float | None
    prompt_token_ids: tuple[int, ...] | TokenExtent = ()
    group_id: str = ""


@dataclass(frozen=True)
class WorkloadConfig:
    round_id: int
    group_id: str
    prompt_token_ids: tuple[int, ...]
    max_response_length: int
    behavior_version: int
    trajectories: tuple[Trajectory, ...]
    advantage_rule: str
    loss_reduction: str
    optimizer: str


@dataclass(frozen=True)
class HardwareConfig:
    dies: int
    cores_per_die: int
    dram_bytes_per_die: int
    sram_bytes_per_core: int
    nic_tx_bytes_per_die: int
    nic_rx_bytes_per_die: int
    gemm_flops_per_ns: int
    vector_ops_per_ns: int
    dram_bytes_per_ns: int
    module_path_bytes_per_ns: int
    sram_bytes_per_ns: int
    d2d_bytes_per_ns_per_direction: int
    d2d_latency_ns: int
    d2d_packet_bytes: int
    host_bytes_per_ns: int
    host_latency_ns: int
    astra_time_scale: int
    gemm_fp32_flops_per_ns: int | None = None


@dataclass(frozen=True)
class StrategyConfig:
    tensor_parallel: int
    pipeline_parallel: int
    data_parallel: int
    training_microbatch_trajectories: int
    rollout_max_concurrency: int
    share_prompt_kv: bool
    actor_train_inference_layout_compatible: bool
    schedule: str
    kv_admission: str
    training_epochs: int = 1
    # 0 means the full collected batch. A final short minibatch is retained.
    optimizer_minibatch_trajectories: int = 0


@dataclass(frozen=True)
class CostConfig:
    reward_cpu_ns: int
    advantage_cpu_ns: int
    sample_vector_ops_per_token: int
    softmax_vector_ops_per_element: int
    rmsnorm_vector_ops_per_element: int
    swiglu_vector_ops_per_element: int
    optimizer_vector_ops_per_local_parameter: int
    gradient_clip_vector_ops_per_local_parameter: int
    control_marker_ns: int


@dataclass(frozen=True)
class CaseConfig:
    schema_version: str
    case_id: str
    evidence_kind: EvidenceKind
    rounds: int
    model: ModelConfig
    workload: WorkloadConfig
    hardware: HardwareConfig
    strategy: StrategyConfig
    costs: CostConfig
    source_path: Path
    resolved: dict[str, Any]


@dataclass(frozen=True)
class ParameterRow:
    parameter_id: str
    unique_elements: int
    local_elements_per_die: int
    replicated: bool
    layout: str


@dataclass(frozen=True)
class Operation:
    node_id: int
    op_id: str
    round_id: int
    phase: str
    name: str
    kind: OperationKind
    duration_ns: int = 0
    payload_bytes: int = 0
    collective: str | None = None
    src_rank: int | None = None
    dst_rank: int | None = None
    dependencies: tuple[int, ...] = ()
    flops: int = 0
    vector_ops: int = 0
    tensor_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    participant_ranks: tuple[int, ...] = ()
    dependencies_by_rank: dict[int, tuple[int, ...]] = field(default_factory=dict)
    kernel_spec: KernelSpec | None = None


@dataclass(frozen=True)
class ObjectRecord:
    storage_id: str
    tensor_id: str
    kind: str
    location: str
    nbytes: int
    dtype: str
    version: int | None
    created_at: str
    ready_at: str
    last_consumer: str
    released_at: str | None
    persistent: bool
    owner_rank: int | None


@dataclass(frozen=True)
class MemorySnapshot:
    step: int
    phase: str
    location: str
    allocated_bytes: int
    valid_bytes: int
    unmaterialized_bytes: int
    capacity_bytes: int
    reason: str


@dataclass(frozen=True)
class TransferRecord:
    round_id: int
    op_id: str
    phase: str
    transfer_type: str
    src: str
    dst: str
    route: str
    payload_bytes: int
    node_id: int
    source: str = "astrasim"


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    passed: bool
    expected: Any
    actual: Any
    evidence: str
    scope: str


@dataclass(frozen=True)
class AstraEvent:
    """One issue/callback pair reconstructed from AstraSim's debug log."""

    rank: int
    node_id: int
    op_id: str
    round_id: int
    phase: str
    node_type: int
    issue_tick: int
    callback_tick: int
    duration_ticks: int
    issue_ns: float
    callback_ns: float
    duration_ns: float


@dataclass(frozen=True)
class AstraRankSummary:
    """Mandatory completion and utilization counters for one simulated rank."""

    rank: int
    cycles: int
    exposed_communication_cycles: int
    wall_cycles: int
    gpu_cycles: int
    communication_cycles: int


@dataclass(frozen=True)
class AstraRunResult:
    """Raw-process provenance returned before semantic validation."""

    command: tuple[str, ...]
    return_code: int
    started_at_utc: str
    finished_at_utc: str
    stdout_path: Path
    stderr_path: Path
    debug_log_path: Path
    binary_path: Path
    binary_sha256: str
    astra_commit: str

