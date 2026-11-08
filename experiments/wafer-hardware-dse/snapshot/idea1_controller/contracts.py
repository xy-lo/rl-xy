"""Immutable controller contracts; no placement or scheduling algorithms."""
from dataclasses import dataclass, fields
from collections.abc import Mapping as MappingABC
from types import MappingProxyType
from typing import FrozenSet, Literal, Mapping, Optional, Tuple

def _freeze(value):
    if isinstance(value, MappingABC):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


class _FrozenRecord:
    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(self, field.name, _freeze(getattr(self, field.name)))


Role = Literal['ROLLOUT', 'TRAINING', 'UNASSIGNED']
Activity = Literal['OWN', 'HELP', 'TRANSFER', 'WAIT', 'TRANSITION']
Phase = Literal['PREFILL', 'DECODE', 'OLD_LOGPROB', 'REFERENCE', 'ADVANTAGE', 'BACKWARD', 'GRAD_READY', 'REDUCE', 'OPTIMIZE', 'PUBLISH']
ActionKind = Literal['PREPARE_INITIAL_STATE', 'CONTINUE_OWN', 'TAKE_PENDING_RESPONSE', 'HELP_ATTENTION', 'ENTER_PREPARATION', 'START_BACKWARD', 'TAKE_TRAIN_BATCH', 'RELEASE_LEASE', 'WAIT', 'FINALIZE_UPDATE']

@dataclass(frozen=True)
class Version(_FrozenRecord):
    iteration: int
    actor: str
    reference: str
    optimizer_step: int

@dataclass(frozen=True)
class RequestView(_FrozenRecord):
    request_id: str
    prompt_key: str
    grpo_group_id: str
    prompt_tokens: int
    response_token_cap: int
    decoding_signature: str
    rng_stream_id: str
    # 未完成任务不暴露真实response_length/EOS或未来耗时。

@dataclass(frozen=True)
class MeshView(_FrozenRecord):
    die_ids: Tuple[int, ...]
    coordinates: Mapping[int, Tuple[int, int]]
    capacity_bytes: Mapping[int, int]
    link_bytes_per_ns: Mapping[Tuple[int, int], float]
    link_latency_ns: Mapping[Tuple[int, int], float]

@dataclass(frozen=True)
class ModelContract(_FrozenRecord):
    model_id: str
    parameter_schema_digest: str
    supported_rollout_tp: Tuple[int, ...]
    supported_training_tp: Tuple[int, ...]
    dtype_signature: str
    backend_capability_id: str

@dataclass(frozen=True)
class AlgorithmContract(_FrozenRecord):
    version: Version
    expected_request_ids: FrozenSet[str]
    group_size: int
    loss_weighting_digest: str
    global_normalizer: Optional[float]
    normalizer_recipe_digest: str
    clip_and_optimizer_digest: str
    stochastic_replay_digest: str
    # 权重/分母由算法确定；不能自动改为DP数或die数。

@dataclass(frozen=True)
class TensorSlice(_FrozenRecord):
    tensor_id: str
    flat_start: int
    flat_end: int
    dtype: str

@dataclass(frozen=True)
class StatePlacement(_FrozenRecord):
    actor_slices: Mapping[int, Tuple[TensorSlice, ...]]
    reference_slices: Mapping[int, Tuple[TensorSlice, ...]]
    optimizer_slices: Mapping[int, Tuple[TensorSlice, ...]]
    migration_plan_digest: str

@dataclass(frozen=True)
class ParallelCandidate(_FrozenRecord):
    candidate_id: str
    rollout_tp: int
    rollout_replica_count: int
    training_tp_by_region: Mapping[str, int]
    rollout_cap: int
    training_microbatch_by_region: Mapping[str, int]
    idle_die_count_by_phase: Mapping[str, int]

@dataclass(frozen=True)
class InferenceInstance(_FrozenRecord):
    instance_id: str
    die_ids: Tuple[int, ...]
    tp: int

@dataclass(frozen=True)
class TrainingRegion(_FrozenRecord):
    region_id: str
    die_ids: Tuple[int, ...]
    tp: int
    # 本合同一region对应一份逻辑DP副本；|die_ids|=tp。

@dataclass(frozen=True)
class PlacementPlan(_FrozenRecord):
    plan_id: str
    candidate: ParallelCandidate
    inference: Tuple[InferenceInstance, ...]
    training: Tuple[TrainingRegion, ...]
    admission_domains: Mapping[str, Tuple[int, ...]]
    drain_dependencies: Mapping[str, Tuple[str, ...]]
    state_placement: StatePlacement
    physical_neighbors: Mapping[str, Tuple[str, ...]]
    unused_die_ids_by_phase: Mapping[str, Tuple[int, ...]]

@dataclass(frozen=True)
class ValidationReport(_FrozenRecord):
    valid: bool
    reasons: Tuple[str, ...]
    peak_bytes_by_die_phase: Mapping[Tuple[int, str], int]
    proof_digest: str

@dataclass(frozen=True)
class HistorySnapshot(_FrozenRecord):
    snapshot_id: str
    before_iteration: int
    version_signature: str
    completed_observation_count: int

@dataclass(frozen=True)
class Prediction(_FrozenRecord):
    request_id: str
    predicted_rollout_ns: float
    predicted_training_ns_by_tp: Mapping[int, float]
    upper_quantile_rollout_ns: float
    confidence: float
    history_snapshot_id: str

@dataclass(frozen=True)
class AssignmentPlan(_FrozenRecord):
    plan_id: str
    home_region_by_group: Mapping[str, str]
    initial_instance_by_request: Mapping[str, str]
    ordered_requests_by_instance: Mapping[str, Tuple[str, ...]]
    long_group_quota_by_region: Mapping[str, float]
    fallback_group_ids: FrozenSet[str]
    prediction_digest: str

@dataclass(frozen=True)
class StepPlan(_FrozenRecord):
    step_plan_id: str
    placement: PlacementPlan
    assignment: AssignmentPlan
    algorithm: AlgorithmContract
    feasibility_proof_digest: str
    gradient_routing_digest: str
    publication_plan_digest: str
    capability_report_digest: str

@dataclass(frozen=True)
class DieState(_FrozenRecord):
    die_id: int
    assigned_role: Role
    transition_target: Optional[Role]
    activity: Activity
    phase: Optional[Phase]
    execution_group_id: Optional[str]
    version: Version
    reserved_bytes: int
    lease_ids: FrozenSet[str]

@dataclass(frozen=True)
class RuntimeEvent(_FrozenRecord):
    event_id: str
    tick_ps: int
    kind: str
    command_id: str
    payload_digest: str
    version: Version

@dataclass(frozen=True)
class WorldSnapshot(_FrozenRecord):
    epoch: int
    tick_ps: int
    step_plan: StepPlan
    dies: Mapping[int, DieState]
    task_ledger_digest: str
    sample_ledger_digest: str
    state_catalog_digest: str
    backend_capability_id: str
    # 策略获得只读投影；真值EOS只允许后端注入完成事件。

@dataclass(frozen=True)
class ActionProposal(_FrozenRecord):
    proposal_id: str
    kind: ActionKind
    task_ids: Tuple[str, ...]
    die_ids: Tuple[int, ...]
    source_group: Optional[str]
    target_group: Optional[str]
    based_on_epoch: int
    expected_makespan_gain_ns: float
    transfer_bytes: int
    release_delay_ns: float
    prediction_confidence: float
    version: Version

@dataclass(frozen=True)
class Readiness(_FrozenRecord):
    allowed: bool
    blockers: Tuple[str, ...]
    proof_digest: str

@dataclass(frozen=True)
class Decision(_FrozenRecord):
    decision_id: str
    accepted: Tuple[ActionProposal, ...]
    rejected_reasons: Mapping[str, Tuple[str, ...]]
    based_on_epoch: int
    reservation_digest: str

@dataclass(frozen=True)
class CommandBatch(_FrozenRecord):
    command_id: str
    decision_id: str
    native_program_digest: str
    completion_event_ids: Tuple[str, ...]
    version: Version

@dataclass(frozen=True)
class PreparedBatch(_FrozenRecord):
    batch_id: str
    home_region_id: str
    request_ids: Tuple[str, ...]
    algorithm_digest: str
    version: Version
    payload_bytes: int
    input_ready_proof: str

@dataclass(frozen=True)
class GradientFragment(_FrozenRecord):
    contribution_id: str
    execution_region_id: str
    request_ids: Tuple[str, ...]
    parameter_slice: TensorSlice
    version: Version
    accumulation_dtype: str
    local_scale: float
    tp_replication_rule: str

@dataclass(frozen=True)
class GradientRoutePlan(_FrozenRecord):
    plan_digest: str
    canonical_chunk_owners: Mapping[str, int]
    input_layout_digests: Mapping[str, str]
    normalization_digest: str
    transfer_schedule_digest: str

@dataclass(frozen=True)
class UpdateTicket(_FrozenRecord):
    ticket_id: str
    iteration: int
    completed_tasks_digest: str
    reduced_gradients_digest: str
    optimizer_owner_digest: str
    next_actor_version: str
    resolved_global_normalizer: float

@dataclass(frozen=True)
class CapabilityReport(_FrozenRecord):
    supported: bool
    supported_actions: FrozenSet[ActionKind]
    heterogeneous_training_tp: bool
    training_help_before_global_eos: bool
    pending_response_takeover: bool
    reasons: Tuple[str, ...]
