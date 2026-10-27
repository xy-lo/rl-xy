"""Shared workload semantics; no fabricated numeric training or timestamps."""
from collections import defaultdict


def prompt_for(case, trajectory):
    return trajectory.prompt_token_ids or case.workload.prompt_token_ids


def group_for(case, trajectory):
    return trajectory.group_id or case.workload.group_id


def demean_advantages(case):
    groups = defaultdict(list)
    for t in case.workload.trajectories:
        groups[group_for(case, t)].append(t)
    return {t.trajectory_id: None if t.reward is None else t.reward - sum(x.reward for x in group) / len(group)
            for group in groups.values() for t in group}


def local_qkv_width(case):
    m = case.model
    return (m.hidden_size + 2 * m.kv_heads * m.head_dim) // case.strategy.tensor_parallel
