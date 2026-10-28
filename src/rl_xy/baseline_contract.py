"""Preflight a declared veRL configuration and its literal cross-hardware transfer.

A valid configuration is not proof that the current IR reproduces its runtime.
The two gates deliberately remain separate. No timing result is selected here.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Iterable

from .model import enumerate_parameters

UPSTREAM_COMMIT = '1876b06d0a3e4e71e06230be10af14492ca8a75b'
UPSTREAM_ROOT = f'https://github.com/verl-project/verl/blob/{UPSTREAM_COMMIT}/'


@dataclass(frozen=True)
class BaselinePolicy:
    # veRL trainer batch and actor ppo mini-batch are measured in prompts;
    # ray_trainer._update_actor multiplies the latter by rollout.n.
    prompt_batch_size: int = 48
    responses_per_prompt: int = 2
    ppo_mini_batch_prompts: int = 48
    train_tp: int = 2
    rollout_tp: int = 1
    pipeline_parallel: int = 1
    train_micro_batch_trajectories: int = 1
    logprob_micro_batch_trajectories: int = 1
    ppo_epochs: int = 1
    balance_batch: bool = False
    shuffle: bool = False
    dynamic_batching: bool = False
    norm_adv_by_std: bool = True
    loss_aggregation: str = 'token-mean'
    policy_loss: str = 'vanilla'
    clip_ratio: float = 0.2
    max_grad_norm: float = 1.0
    use_kl_loss: bool = False
    use_kl_in_reward: bool = False
    kl_coefficient: float = 0.0
    entropy_coefficient: float = 0.0
    distributed_optimizer: bool = True
    offload: bool = False
    sequence_parallel: bool = False
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False
    weight_sync: str = 'naive_per_tensor_export_distinct_rollout'
    kv_sharding: str = 'partition_no_replication'

    @property
    def trajectory_batch_size(self):
        return self.prompt_batch_size * self.responses_per_prompt

    @property
    def trajectory_mini_batch_size(self):
        return self.ppo_mini_batch_prompts * self.responses_per_prompt

    @property
    def needs_reference(self):
        return self.use_kl_loss or self.use_kl_in_reward


def policy_fingerprint(policy):
    return hashlib.sha256(json.dumps(asdict(policy), sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def _positive_int(x):
    return type(x) is int and x > 0


def audit_policy(model, policy, *, world_size=48, runtime_capabilities: Iterable[str] = ()):
    """Return official batch constraints, selected no-KV-replication layout,
    and independent IR capability gates. Unsupported scope is not mislabeled
    as an upstream engine limitation.
    """
    errors = []
    official = []
    selected = []
    integer_fields = ['prompt_batch_size', 'responses_per_prompt', 'ppo_mini_batch_prompts',
        'train_tp', 'rollout_tp', 'pipeline_parallel', 'train_micro_batch_trajectories',
        'logprob_micro_batch_trajectories', 'ppo_epochs']
    for key in integer_fields:
        if not _positive_int(getattr(policy, key)):
            errors.append('positive_integer_required:' + key)
    if not _positive_int(world_size):
        errors.append('positive_integer_required:world_size')
    if errors:
        return {'status': 'INVALID_CONFIGURATION', 'blockers': errors,
            'official_batch_constraints_pass': False, 'selected_layout_constraints_pass': False,
            'runtime_alignment_ready': False, 'policy': asdict(policy)}
    tr, ge, pp = policy.train_tp, policy.rollout_tp, policy.pipeline_parallel
    if world_size % (tr * pp):
        official.append('world_not_divisible_by_train_tp_times_pp')
    if world_size % ge:
        official.append('world_not_divisible_by_rollout_tp')
    dp = world_size // (tr * pp)
    if policy.ppo_mini_batch_prompts > policy.prompt_batch_size:
        official.append('ppo_minibatch_prompts_exceeds_train_batch_prompts')
    if policy.prompt_batch_size % policy.ppo_mini_batch_prompts:
        selected.append('selected_full_minibatch_contract_does_not_drop_tail')
    if dp and policy.trajectory_mini_batch_size % dp:
        official.append('trajectory_minibatch_not_divisible_by_train_dp')
    local_mini = policy.trajectory_mini_batch_size // dp if dp else 0
    local_batch = policy.trajectory_batch_size // dp if dp else 0
    if dp and policy.trajectory_batch_size % dp:
        official.append('trajectory_batch_not_divisible_by_train_dp')
    if not policy.dynamic_batching and local_mini % policy.train_micro_batch_trajectories:
        official.append('local_minibatch_not_divisible_by_static_train_microbatch')
    if not policy.dynamic_batching and local_batch % policy.logprob_micro_batch_trajectories:
        official.append('local_batch_not_divisible_by_static_logprob_microbatch')
    for key in ['hidden_size', 'query_heads', 'ffn_size', 'vocab_size']:
        if getattr(model, key) % tr:
            selected.append('no_padding_train_tp_does_not_divide_' + key)
        if getattr(model, key) % ge:
            selected.append('no_padding_rollout_tp_does_not_divide_' + key)
    if policy.kv_sharding != 'partition_no_replication':
        selected.append('selected_contract_has_no_KV_replication_implementation')
    elif model.kv_heads % tr or model.kv_heads % ge:
        selected.append('selected_no_KV_replication_TP_does_not_divide_KV_heads')
    if model.layers % pp:
        selected.append('selected_equal_layer_PP_does_not_divide_layers')
    if tr % ge:
        selected.append('selected_contiguous_subgroup_layout_requires_rollout_TP_divides_train_TP')
    if policy.loss_aggregation != 'token-mean' or policy.policy_loss != 'vanilla':
        selected.append('selected_loss_scope_is_vanilla_token_mean')
    if not (0 < policy.clip_ratio < 1) or policy.max_grad_norm <= 0:
        selected.append('invalid_clip_ratio_or_gradient_norm_limit')
    if policy.kl_coefficient < 0 or policy.entropy_coefficient < 0:
        selected.append('negative_KL_or_entropy_coefficient')
    if policy.weight_sync != 'naive_per_tensor_export_distinct_rollout':
        selected.append('selected_weight_sync_scope_is_naive_full_export')
    # These are implementer assertions to be backed by tests/source matching,
    # not knobs which improve predicted performance.
    required = {'equal_size_trajectory_dispatch', 'global_GRPO_uid_advantages',
        'PPO_ratio_clipping', 'global_token_loss_reduction', 'static_microbatch_execution',
        'conditional_reference_stage', 'TP_owner_gradient_norm', 'actor_version_fence',
        'explicit_naive_weight_export', 'BF16_GEMM_FP32_accumulation'}
    if policy.train_micro_batch_trajectories > 1:
        required.add('multi_trajectory_train_microbatch')
    if policy.logprob_micro_batch_trajectories > 1:
        required.add('multi_trajectory_logprob_microbatch')
    if policy.norm_adv_by_std:
        required.add('GRPO_sample_std_normalization')
    if policy.needs_reference:
        required.add('configured_KL_term')
    if policy.balance_batch:
        required.add('pinned_verl_equal_count_token_balancer')
    if policy.shuffle:
        required.add('pinned_verl_minibatch_shuffle')
    if policy.dynamic_batching:
        required.add('pinned_verl_dynamic_microbatch_packing')
    if policy.ppo_epochs != 1:
        required.add('multi_epoch_update_schedule')
    if pp != 1:
        required.add('pipeline_parallel_schedule')
    if policy.distributed_optimizer:
        required.add('distributed_optimizer_state_and_collectives')
    if policy.offload:
        required.add('host_offload_transfers_and_capacity')
    if policy.sequence_parallel:required.add('sequence_parallel_execution')
    if policy.overlap_grad_reduce:required.add('gradient_reduce_overlap_execution')
    if policy.overlap_param_gather:required.add('parameter_gather_overlap_execution')
    missing = sorted(required - set(runtime_capabilities))
    errors = ['upstream_batch:' + x for x in official] + ['selected_scope:' + x for x in selected]
    return {'status': ('VALID_CONFIGURATION_RUNTIME_ALIGNED_COST_PENDING' if not missing else 'VALID_CONFIGURATION_PENDING_RUNTIME_ALIGNMENT') if not errors else 'INVALID_CONFIGURATION',
        'blockers': errors, 'official_batch_constraints_pass': not official,
        'selected_layout_constraints_pass': not selected, 'runtime_alignment_ready': not errors and not missing,
        'runtime_capabilities_missing': missing, 'world_size': world_size, 'train_dp': dp,
        'rollout_dp': world_size // ge, 'trajectory_batch_size': policy.trajectory_batch_size,
        'trajectory_mini_batch_size': policy.trajectory_mini_batch_size,
        'trajectory_batch_per_train_dp': local_batch, 'trajectory_minibatch_per_train_dp': local_mini,
        'gradient_accumulation_steps': None if policy.dynamic_batching else local_mini // policy.train_micro_batch_trajectories,
        'reference_required': policy.needs_reference,
        'optimizer_alignment': {'upstream_Megatron_use_distributed_optimizer_default': True,
            'selected_use_distributed_optimizer': policy.distributed_optimizer,
            'legacy_v07_has_only_replicated_master_Adam_and_full_gradient_SUM': True,
            'required_if_distributed': ['DP sharded FP32 master and Adam states',
                'gradient reduce-scatter with declared bucket working buffers',
                'updated parameter all-gather and selected overlap policy'],
            'do_not_divide_all_live_buffers_by_DP': True,
            'source': UPSTREAM_ROOT + 'verl/trainer/config/engine/megatron.yaml#L32'},
        'policy': asdict(policy),
        'policy_sha256': policy_fingerprint(policy),
        'source': {'commit': UPSTREAM_COMMIT,
            'prompt_to_trajectory_units': UPSTREAM_ROOT + 'verl/trainer/ppo/ray_trainer.py#L1352-L1368',
            'DP_minibatch_divisibility': UPSTREAM_ROOT + 'verl/workers/engine_workers.py#L243-L277',
            'microbatch_divisibility': UPSTREAM_ROOT + 'verl/workers/engine/utils.py#L82-L118'},
        'scope': 'source-configuration and implemented-feature preflight; not a native run or hardware accuracy validation',
        'paper_performance_ready': False}


def capacity_lower_bounds(case, policy, capacities):
    """Declared resident weights/states only, with replicated tensors counted.
    Failure proves this selected no-offload layout cannot fit; passing never
    establishes fit because activation/KV/export/communication buffers are absent.
    """
    if policy.pipeline_parallel != 1 or policy.distributed_optimizer or policy.offload:
        return {'scope_supported': False, 'platforms': {},
            'reason': 'this resident lower bound only models PP1 conventional optimizer without offload',
            'passing_lower_bound_is_not_capacity_admission': True}
    if not all(_positive_int(getattr(policy, key)) for key in ['train_tp', 'rollout_tp']):
        raise ValueError('positive TP sizes required for capacity bound')
    train_case = replace(case, strategy=replace(case.strategy, tensor_parallel=policy.train_tp))
    roll_case = replace(case, strategy=replace(case.strategy, tensor_parallel=policy.rollout_tp))
    train_params = sum(p.local_elements_per_die for p in enumerate_parameters(train_case))
    rollout_params = sum(p.local_elements_per_die for p in enumerate_parameters(roll_case))
    # Explicit selected conventional optimizer: BF16 actor, FP32 grad/master/m/v,
    # plus frozen BF16 reference iff algorithm actually requests it.
    role_bytes = {'actor': 2, 'gradient': 4, 'master': 4, 'adam_m': 4, 'adam_v': 4}
    if policy.needs_reference:
        role_bytes['reference'] = 2
    persistent = train_params * sum(role_bytes.values())
    rollover = rollout_params * 2
    result = {}
    for name, capacity in capacities.items():
        result[name] = {'capacity_bytes_per_device': capacity, 'persistent_bytes': persistent,
            'distinct_rollout_weight_bytes': rollover,
            'rollout_phase_resident_lower_bound_bytes': persistent + rollover,
            'lower_bound_exceeds_capacity': persistent + rollover > capacity}
    return {'scope_supported': True, 'train_local_parameter_elements': train_params, 'rollout_local_parameter_elements': rollout_params,
        'bytes_per_train_local_parameter_by_role': role_bytes, 'platforms': result,
        'excludes': ['activations', 'KV cache', 'complete export scratch', 'collective buffers',
            'allocator overhead', 'temporary conversion tensors'],
        'passing_lower_bound_is_not_capacity_admission': True}


def candidate_policies(model, *, prompt_batch_size=48, responses_per_prompt=2,
        ppo_mini_batch_prompts=None, train_tps=(1, 2), rollout_tps=(1, 2),
        microbatches=(1, 2, 4), world_size=48, **common):
    """Small declared candidate space; includes invalid rows with reasons.
    This is enumeration, not timing optimization or an optimality claim.
    """
    for tr in train_tps:
        for ge in rollout_tps:
            if tr % ge:
                continue
            for mb in microbatches:
                policy = BaselinePolicy(prompt_batch_size=prompt_batch_size,
                    responses_per_prompt=responses_per_prompt,
                    ppo_mini_batch_prompts=ppo_mini_batch_prompts or prompt_batch_size,
                    train_tp=tr, rollout_tp=ge, train_micro_batch_trajectories=mb,
                    logprob_micro_batch_trajectories=mb, **common)
                yield policy, audit_policy(model, policy, world_size=world_size)


def audit_direct_transfer(gpu_policy, wafer_policy, *, logical_world=48,
        gpu_allocated=48, wafer_allocated=54, wafer_active_ranks=tuple(range(48)),
        barrier_participants=None):
    """Keep software policy and logical device count identical; the six extra
    wafer dies are budgeted idle, not extra DP replicas/barrier participants.
    """
    errors = []
    if not _positive_int(logical_world) or not _positive_int(gpu_allocated) or not _positive_int(wafer_allocated):
        errors.append('positive_logical_world_and_budgets_required')
    if policy_fingerprint(gpu_policy) != policy_fingerprint(wafer_policy):
        errors.append('software_policy_changed_during_transfer')
    if len(set(wafer_active_ranks)) != logical_world or len(wafer_active_ranks) != logical_world:
        errors.append('wafer_active_device_count_differs_from_GPU_logical_world')
    if any(type(r) is not int or not 0 <= r < wafer_allocated for r in wafer_active_ranks):
        errors.append('invalid_wafer_active_rank')
    if min(gpu_allocated, wafer_allocated) < logical_world:
        errors.append('allocated_device_budget_smaller_than_logical_world')
    if barrier_participants is None:
        errors.append('explicit_active_only_barrier_participants_required')
    elif list(barrier_participants) != list(wafer_active_ranks):
        errors.append('barrier_participants_differ_from_selected_active_world')
    return {'status': 'STRICT_TRANSFER_CONFIGURATION' if not errors else 'NOT_STRICT_TRANSFER',
        'blockers': errors, 'policy_sha256': policy_fingerprint(gpu_policy), 'logical_world_size': logical_world,
        'gpu_allocated': gpu_allocated, 'wafer_allocated': wafer_allocated,
        'wafer_active_ranks_in_logical_order': list(wafer_active_ranks),
        'wafer_idle_ranks': sorted(set(range(wafer_allocated)) - set(wafer_active_ranks)),
        'native_graph_active_groups_and_idle_trace_markers_still_require_verification': True,
        'budget_extension_to_54_active_is_a_separate_experiment': True}
