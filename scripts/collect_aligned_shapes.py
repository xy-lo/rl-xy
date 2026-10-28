#!/usr/bin/env python3
"""Exact CPU-only GEMM census for a selected aligned graph, never timings.

Contracted components are read from contraction.original_plan and weighted by
every physical invocation and TP participant. Noncomponent global kernels are
also included. The placeholder provider is for shape materialization only.
The resulting ``shapes`` mapping is accepted by benchmark_gemm.py.
"""
from collections import Counter
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rl_xy.cost_model import CostEstimate


from rl_xy.gemm_census import signature,CensusProvider,census_graph,measured_domain


def source_snapshot():
    files = sorted((ROOT / 'src' / 'rl_xy').rglob('*.py')) + [Path(__file__).resolve(), ROOT / 'scripts/benchmark_gemm.py']
    records = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    return {'tree_sha256': signature(records), 'files': records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['case', 'scenario', 'policy', 'gpu-profile', 'out']:
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--active', type=int, default=48)
    parser.add_argument('--measurements', type=Path, help='optional matching complete hotGEMM labels for coverage audit')
    parser.add_argument('--require-covered', action='store_true', help='exit2 if exact measured domain is incomplete; census is still saved')
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('output exists; preserve previous census')
    from rl_xy.config import load_case
    from rl_xy.baseline_contract import BaselinePolicy, audit_policy, policy_fingerprint
    from rl_xy.wafer_run import effective_case
    from rl_xy.gpu_model import effective_gpu_case, load_gpu_profile
    from rl_xy.aligned_graph import build_aligned_graph
    snapshot = source_snapshot()
    inputs = {'case': args.case, 'scenario': args.scenario, 'policy': args.policy,
        'GPU_profile': args.gpu_profile}
    input_snapshot = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k,v in inputs.items()}
    case, scenario, source = effective_case(load_case(args.case), args.scenario)
    inputs['wafer_hardware_source'] = source
    profile = load_gpu_profile(args.gpu_profile)
    if args.measurements:
        inputs['measurements'] = args.measurements
    if profile.get('gemm_profile'):
        inputs['GPU_gemm_profile'] = (args.gpu_profile.resolve().parent / profile['gemm_profile']).resolve()
    input_snapshot.update({k: hashlib.sha256(v.read_bytes()).hexdigest() for k,v in inputs.items()
        if k not in input_snapshot})
    if args.active > profile['gpu_count']:
        raise ValueError('active count exceeds GPU profile allocation')
    case = effective_gpu_case(case, profile)
    policy = BaselinePolicy(**json.loads(args.policy.read_text()))
    policy_check = audit_policy(case.model, policy, world_size=args.active)
    if not policy_check['official_batch_constraints_pass'] or not policy_check['selected_layout_constraints_pass']:
        raise ValueError('selected policy invalid: ' + str(policy_check['blockers']))
    if len(case.workload.trajectories) != policy.trajectory_batch_size:
        raise ValueError('selected case/policy trajectory count differs')
    graph = build_aligned_graph(case, profile['gpu_count'], policy,
        CensusProvider(), '/CENSUS_ONLY_NOT_EXECUTED', active=args.active,
        compact_components=True)
    result = census_graph(graph, active=args.active)
    measurements = profile.get('gemm_measurements')
    if args.measurements:
        from validate_gpu_costs import validate_labels
        measurements = validate_labels(json.loads(args.measurements.read_text()))
    result['measured_domain'] = measured_domain(result, measurements)
    if source_snapshot() != snapshot:
        raise ValueError('source mutated during census; rerun against a fixed source snapshot')
    if input_snapshot != {k: hashlib.sha256(v.read_bytes()).hexdigest() for k,v in inputs.items()}:
        raise ValueError('input mutated during census; rerun with fixed inputs')
    result.update(schema_version='rlxy.aligned_gemm_census/1', status='CENSUS_COMPLETE',
        scope='CPU shape census only; no mapping/benchmark/native simulation/performance prediction',
        placeholder_provider='CENSUS_ONLY_NOT_A_PERFORMANCE_ESTIMATE; internal integer duration1 is never exported as kernel time',
        policy=asdict(policy), policy_sha256=policy_fingerprint(policy),
        workload={'case_id': case.case_id, 'model': asdict(case.model),
            'rounds': case.rounds, 'trajectories': len(case.workload.trajectories),
            'prompt_tokens': sum(len(t.prompt_token_ids or case.workload.prompt_token_ids) for t in case.workload.trajectories),
            'response_tokens': sum(len(t.response_token_ids) for t in case.workload.trajectories),
            'length_counts': {str(k): v for k,v in sorted(Counter((len(t.prompt_token_ids or case.workload.prompt_token_ids),len(t.response_token_ids)) for t in case.workload.trajectories).items())}},
        active_ranks=args.active, GPU_allocated=profile['gpu_count'],
        wafer_allocated=scenario['mesh'][0]*scenario['mesh'][1],
        source=snapshot, inputs={k: {'path': str(v.resolve()),
            'sha256': input_snapshot[k]} for k,v in inputs.items()},
        runtime_or_paper_release=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'totals': result['totals'],
        'missing_count': result['measured_domain']['missing_count'], 'out': str(args.out)}, indent=2))
    return 2 if args.require_covered and not result['measured_domain']['exact_domain_complete'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
