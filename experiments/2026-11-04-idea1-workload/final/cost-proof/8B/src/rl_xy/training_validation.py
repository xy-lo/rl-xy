"""Independent multi-update schedule checks, including mutation rejection.

Coverage and ordering are reconstructed from the case, not accepted from the
planner's update labels. Existing parameter-bearing forward/backward and Adam
checks are applied to a normalized view of every minibatch.
"""
from collections import Counter
from dataclasses import replace

from .execution_contract import demean_advantages
from .types import CheckResult
from .validation import _dependency_checks, _training_checks


def multi_update_checks(case, operations):
    checks = _dependency_checks(case, operations)
    errors = []
    trajectories = case.workload.trajectories
    batch_size = case.strategy.optimizer_minibatch_trajectories or len(trajectories)
    batches = [trajectories[i:i + batch_size] for i in range(0, len(trajectories), batch_size)]
    count = len(batches) * case.strategy.training_epochs
    advantages = demean_advantages(case)
    expected_updates = set(range(count * case.rounds))
    labels = {o.metadata.get('update_id') for o in operations if 'update_id' in o.metadata}
    if labels != expected_updates:
        errors.append(['update coverage', sorted(labels), sorted(expected_updates)])
    last_publish = 0
    for offset in range(case.rounds):
        rid = case.workload.round_id + offset
        round_ops = [o for o in operations if o.round_id == rid]
        before = [o for o in round_ops if 'update_id' not in o.metadata and o.phase != 'cleanup']
        after = [o for o in round_ops if o.phase == 'cleanup']
        inputs = [o for o in before if o.name == 'host_to_rank0_prompt']
        behavior = case.workload.behavior_version + offset * count
        if (len(inputs) != 1 or inputs[0].metadata.get('behavior_version') != behavior
                or inputs[0].node_id <= last_publish):
            errors.append([rid, 'round rollout/version boundary'])
        adv_ops = [o for o in before if o.name == 'group_advantage_demean']
        expected_adv = [advantages[t.trajectory_id] for t in trajectories]
        if len(adv_ops) != 1 or adv_ops[0].metadata.get('advantages') != expected_adv:
            errors.append([rid, 'advantages must use original complete groups'])
        if len(after) != 1 or after[0].name != 'release_round_temporaries' or after[0].metadata.get('reference_version') != case.workload.behavior_version:
            errors.append([rid, 'one cleanup with frozen reference'])
        for o in operations:
            if o.metadata.get('reference_written') is True:
                errors.append([o.op_id, 'reference mutation'])
        for epoch in range(case.strategy.training_epochs):
            for mb, batch in enumerate(batches):
                update = offset * count + epoch * len(batches) + mb
                version = case.workload.behavior_version + update
                selected = [o for o in operations if o.metadata.get('update_id') == update]
                prefix = f'r{rid}.u{update}'
                names = Counter(o.name for o in selected)
                required = ['zero_gradient_buffer', 'local_gradient_square_sum', 'global_gradient_norm_sum',
                            'apply_gradient_clip', 'adamw_read_modify_write', 'all_shards_updated_barrier', 'publish_actor_version']
                if any(names[n] != 1 for n in required):
                    errors.append([prefix, 'control coverage', {n:names[n] for n in required}])
                else:
                    controls = [next(o for o in selected if o.name == n) for n in required]
                    ordered = [o.node_id for o in controls]
                    if ordered != sorted(ordered) or ordered[0] <= max([last_publish] + [o.node_id for o in before]):
                        errors.append([prefix, 'control order'])
                    zero, norm, _, _, adam, barrier, publish = controls
                    train_ops = [o for o in selected if 'trajectory_id' in o.metadata]
                    if not train_ops or not all(zero.node_id < o.node_id < norm.node_id for o in train_ops):
                        errors.append([prefix, 'gradient accumulation/update boundary'])
                    if (adam.metadata.get('actor_version_in'), adam.metadata.get('actor_version_out'),
                            barrier.metadata.get('new_version'), publish.metadata.get('published_version')) != (version,version+1,version+1,version+1):
                        errors.append([prefix, 'version chain'])
                    if not zero.metadata.get('reset_each_update'):
                        errors.append([prefix, 'reset flag'])
                    last_publish = publish.node_id
                denominator = sum(len(t.response_token_ids) for t in batch)
                for o in selected:
                    if (o.round_id != rid or o.metadata.get('training_epoch') != epoch or
                            o.metadata.get('minibatch_index') != mb or o.metadata.get('behavior_version') != behavior or
                            o.metadata.get('actor_version') != version or o.metadata.get('reference_version') != case.workload.behavior_version or
                            o.metadata.get('minibatch_denominator') != denominator):
                        errors.append([o.op_id, 'update context'])
                normalized = []
                for o in selected:
                    tid = o.metadata.get('trajectory_id')
                    old = f'train_u{update}_{tid}' if tid else ''
                    phase = f'train_{tid}' if tid else o.phase
                    if tid and tid not in {t.trajectory_id for t in batch}:
                        errors.append([o.op_id, 'unexpected trajectory'])
                    if o.name.endswith('_loss') and o.metadata.get('advantage') != advantages.get(tid):
                        errors.append([o.op_id, 'advantage changed after partition'])
                    if o.name.endswith('_loss_backward') and o.metadata.get('global_denominator') != denominator:
                        errors.append([o.op_id, 'loss backward denominator'])
                    normalized.append(replace(o, name=o.name.replace(old, phase, 1) if old else o.name, phase=phase))
                if adv_ops:
                    actual = adv_ops[0].metadata.get('advantages', [])
                    selected_adv = [actual[i] for i,t in enumerate(trajectories) if t in batch and i < len(actual)]
                    normalized.append(replace(adv_ops[0], metadata={**adv_ops[0].metadata, 'advantages': selected_adv}))
                virtual = replace(case, rounds=1, workload=replace(case.workload, round_id=rid, trajectories=tuple(batch)))
                for check in _training_checks(virtual, normalized, advantages_override=advantages):
                    checks.append(replace(check, check_id=f'{prefix}.{check.check_id}', scope='optimizer_update'))
        if after and after[0].node_id <= last_publish:
            errors.append([rid, 'cleanup before last publish'])
    checks.append(CheckResult('training.multi_update_schedule', not errors,
        'exact update/epoch/minibatch coverage; frozen batch behavior/reference; per-update normalization/reset/clip/Adam/publish',
        errors, 'independently reconstructed schedule and metadata; not numerical RL validation', 'run'))
    return checks
