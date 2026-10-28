#!/usr/bin/env python3
"""Enumerate valid selected veRL configurations; run no GPU/native experiment."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rl_xy.baseline_contract import candidate_policies, capacity_lower_bounds, audit_direct_transfer
from rl_xy.config import load_case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', type=Path, required=True, help='model/length template, not a new run')
    p.add_argument('--prompt-batch-size', type=int, default=48)
    p.add_argument('--responses-per-prompt', type=int, default=2)
    p.add_argument('--ppo-mini-batch-prompts', type=int)
    p.add_argument('--train-tps', type=int, nargs='+', default=[1, 2])
    p.add_argument('--rollout-tps', type=int, nargs='+', default=[1, 2])
    p.add_argument('--microbatches', type=int, nargs='+', default=[1, 2, 4])
    p.add_argument('--world-size', type=int, default=48)
    p.add_argument('--use-kl-loss', action='store_true')
    p.add_argument('--kl-coefficient', type=float, default=0.0)
    p.add_argument('--conventional-optimizer', action='store_true',
        help='explicit override of upstream distributed-optimizer default; off-default sensitivity only')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error('output already exists; preserve earlier preflight')
    case = load_case(a.case)
    rows = []
    for policy, audit in candidate_policies(case.model, prompt_batch_size=a.prompt_batch_size,
        responses_per_prompt=a.responses_per_prompt, ppo_mini_batch_prompts=a.ppo_mini_batch_prompts,
        train_tps=a.train_tps, rollout_tps=a.rollout_tps, microbatches=a.microbatches,
        world_size=a.world_size, use_kl_loss=a.use_kl_loss, kl_coefficient=a.kl_coefficient,
        distributed_optimizer=not a.conventional_optimizer):
        row = {'configuration': audit, 'capacity_lower_bounds':
            capacity_lower_bounds(case, policy, {'GPU_A100_80GB': 80_000_000_000, 'wafer_die_64GB': 64_000_000_000})
            if not audit['blockers'] else {'scope_supported': False, 'reason': 'invalid candidate; no capacity estimate'}}
        if not audit['blockers'] and policy.distributed_optimizer:
            row['conventional_optimizer_counterfactual_only'] = capacity_lower_bounds(case,
                replace(policy, distributed_optimizer=False),
                {'GPU_A100_80GB': 80_000_000_000, 'wafer_die_64GB': 64_000_000_000})
        row['strict_transfer'] = audit_direct_transfer(policy, policy, logical_world=a.world_size,
            gpu_allocated=a.world_size, wafer_allocated=54,
            wafer_active_ranks=tuple(range(a.world_size)), barrier_participants=tuple(range(a.world_size)))
        rows.append(row)
    result = {'schema_version': 'rlxy.baseline_preflight/1', 'status': 'PREFLIGHT_ONLY',
        'model_template': str(a.case), 'template_trajectory_count': len(case.workload.trajectories),
        'declared_candidate_trajectory_count': a.prompt_batch_size * a.responses_per_prompt,
        'case_workload_must_be_regenerated_or_verified_before_execution': True,
        'upstream_default_distributed_optimizer_selected': not a.conventional_optimizer,
        'candidate_count': len(rows), 'valid_configuration_count': sum(not x['configuration']['blockers'] for x in rows),
        'global_optimum_claimed': False, 'performance_results_generated': False,
        'current_v07_protocol_identity': 'custom_sync_grpo_like_not_full_verl_runtime', 'candidates': rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'candidates'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
