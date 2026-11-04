"""Deterministic optimizer minibatches over a single frozen rollout batch.

This is graph scheduling, not numerical PPO/GRPO. Group advantages are computed
before partitioning; behavior/reference data survive all optimizer updates.
Each microbatch contains one trajectory; no padding/packing is implied.
"""
from dataclasses import replace


def enabled(case):
    return case.strategy.training_epochs != 1 or case.strategy.optimizer_minibatch_trajectories != 0


def minibatches(case):
    trajectories = case.workload.trajectories
    size = case.strategy.optimizer_minibatch_trajectories or len(trajectories)
    return tuple(trajectories[i:i + size] for i in range(0, len(trajectories), size))


def updates_per_round(case):
    return len(minibatches(case)) * case.strategy.training_epochs


def training_visits(case):
    """Expected visits independently usable by tensor and shape validators."""
    for offset in range(case.rounds):
        for epoch in range(case.strategy.training_epochs):
            for batch_index, batch in enumerate(minibatches(case)):
                update = offset * updates_per_round(case) + epoch * len(minibatches(case)) + batch_index
                for trajectory in batch:
                    phase = (f"train_u{update}_{trajectory.trajectory_id}" if enabled(case)
                             else f"train_{trajectory.trajectory_id}")
                    yield case.workload.round_id + offset, update, epoch, batch_index, trajectory, phase


def expand_updates(case, operations):
    """Expand the verified one-update template, then rebuild rank-local chains.

    Template dependencies contain no dataflow beyond strict per-rank ordering;
    communication synchronization remains explicit. Tensor storage is planned
    only after expansion, so no allocation IDs or leases are reused.
    """
    if not enabled(case):
        return operations
    expanded = []
    batches = minibatches(case)
    for offset in range(case.rounds):
        rid = case.workload.round_id + offset
        original = [o for o in operations if o.round_id == rid]
        zero_index = next(i for i, o in enumerate(original) if o.name == "zero_gradient_buffer")
        clip_index = next(i for i, o in enumerate(original) if o.name == "local_gradient_square_sum")
        cleanup_index = next(i for i, o in enumerate(original) if o.name == "release_round_temporaries")
        behavior = case.workload.behavior_version + offset * updates_per_round(case)
        for op in original[:zero_index]:
            meta = dict(op.metadata)
            if "behavior_version" in meta:
                meta["behavior_version"] = behavior
            expanded.append(replace(op, metadata=meta))
        train = original[zero_index + 1:clip_index]
        for epoch in range(case.strategy.training_epochs):
            for batch_index, batch in enumerate(batches):
                update = offset * updates_per_round(case) + epoch * len(batches) + batch_index
                version = case.workload.behavior_version + update
                denominator = sum(len(t.response_token_ids) for t in batch)
                context = {"update_id": update, "training_epoch": epoch, "minibatch_index": batch_index,
                           "behavior_version": behavior, "actor_version": version,
                           "reference_version": case.workload.behavior_version,
                           "minibatch_denominator": denominator}
                rename = {f"train_{t.trajectory_id}": f"train_u{update}_{t.trajectory_id}" for t in batch}
                zero = original[zero_index]
                expanded.append(replace(zero, phase=rename[f"train_{batch[0].trajectory_id}"],
                    metadata={**zero.metadata, **context, "reset_each_round": False, "reset_each_update": True}))
                for trajectory in batch:
                    old_phase = f"train_{trajectory.trajectory_id}"
                    for op in train:
                        if op.phase != old_phase:
                            continue
                        meta = {**op.metadata, **context, "trajectory_id": trajectory.trajectory_id}
                        for key in ("forward_prefix", "backward_prefix"):
                            if key in meta:
                                meta[key] = meta[key].replace(old_phase, rename[old_phase], 1)
                        for key in ("global_denominator", "global_loss_denominator"):
                            if key in meta:
                                meta[key] = denominator
                        expanded.append(replace(op, phase=rename[old_phase],
                            name=op.name.replace(old_phase, rename[old_phase], 1), metadata=meta))
                for op in original[clip_index:cleanup_index]:
                    meta = {**op.metadata, **context}
                    for key, value in (("actor_version_in", version), ("actor_version_out", version + 1),
                                       ("new_version", version + 1), ("published_version", version + 1)):
                        if key in meta:
                            meta[key] = value
                    expanded.append(replace(op, metadata=meta))
        expanded.append(original[cleanup_index])
    result = []
    previous = {}
    for node_id, op in enumerate(expanded, 1):
        by_rank = {rank: (previous[rank],) if rank in previous else () for rank in op.participant_ranks}
        result.append(replace(op, node_id=node_id, op_id=f"r{op.round_id:02d}.n{node_id:05d}.{op.name}",
            dependencies_by_rank=by_rank, dependencies=tuple(sorted({n for ns in by_rank.values() for n in ns}))))
        previous.update({rank: node_id for rank in op.participant_ranks})
    return result
