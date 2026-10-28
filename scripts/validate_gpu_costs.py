#!/usr/bin/env python3
"""CPU-only analytic GEMM audit against independently captured exact A100 labels.

No fitting/scaling, no GPU calls, no native run. Device kernel labels include
memory service; the declared host launch is subtracted from the analytic side
for the hot-kernel comparison. This cannot validate fused vLLM/veRL runtime.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rl_xy.config import load_case
from rl_xy.gpu_model import load_gpu_profile, effective_gpu_case, GPUCostProvider
from rl_xy.operator_ir import Matmul, KernelSpec


def signature(problem):
    return hashlib.sha256(json.dumps(problem, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def shape_family(problem):
    # Nearby shape sizes share a family; direction/precision remain separate.
    # The partition is derived from geometry only, never from observed error.
    value = {'direction': problem['direction'], 'log2_mnk': [int(problem[k]).bit_length() - 1
        for k in ('m', 'n', 'k')], 'batches': problem['batches'],
        'precision': [problem[k] for k in ('a_bytes', 'b_bytes', 'output_bytes', 'accumulator_bytes')]}
    return signature(value)


def split_for(problem):
    return 'holdout' if int(shape_family(problem)[:8], 16) % 5 == 0 else 'development'


def error_statistics(rows):
    if not rows:
        return {'count': 0}
    ratios = [r['predicted_hot_ns'] / r['observed_hot_ns'] for r in rows]
    signed = [(r['predicted_hot_ns'] - r['observed_hot_ns']) / r['observed_hot_ns'] for r in rows]
    absolute = sorted(abs(x) for x in signed)
    def quantile(q):
        at = q * (len(absolute) - 1);low = math.floor(at);high = math.ceil(at)
        return absolute[low] * (high - at) + absolute[high] * (at - low) if low != high else absolute[low]
    predicted = sum(r['predicted_hot_ns'] for r in rows)
    observed = sum(r['observed_hot_ns'] for r in rows)
    return {'count': len(rows), 'MAPE': statistics.mean(absolute), 'p50_APE': quantile(.5),
        'p90_APE': quantile(.9), 'p95_APE': quantile(.95), 'max_APE': max(absolute),
        'mean_signed_relative_error': statistics.mean(signed),
        'WAPE_unique_shapes': sum(abs(r['predicted_hot_ns'] - r['observed_hot_ns']) for r in rows) / observed,
        'sum_predicted_over_observed_unique_shapes': predicted / observed,
        'median_predicted_over_observed': statistics.median(ratios),
        'predicted_hot_ns_sum': predicted, 'observed_hot_ns_sum': observed,
        'weighting_warning': 'unique signatures counted once; not frequency-weighted end-to-end error'}


def validate_labels(data):
    if data.get('schema_version') != 'rlxy.gemm_measurements/1' or data.get('status') != 'PASS':
        raise ValueError('require a complete independently captured measurement profile')
    if data.get('device') != 'NVIDIA A100-SXM4-80GB':
        raise ValueError('only the matching A100 measurement domain is supported')
    checks = data.get('numeric_fixture_checks', [])
    if not checks or not all(c.get('passed') is True for c in checks):
        raise ValueError('nonempty passing GEMM numeric fixtures required')
    measurements = data.get('measurements', {})
    if not measurements:
        raise ValueError('empty measured domain')
    for key, row in measurements.items():
        if signature(row['problem']) != key:
            raise ValueError('measurement signature integrity failure')
        Matmul(**row['problem'])
        if type(row['duration_ns']) is not int or row['duration_ns'] <= 0:
            raise ValueError('positive integer measured duration required')
        if not 0 <= row['repeat_relative_difference'] <= .15:
            raise ValueError('unstable label exceeds declared 15% repeatability gate')
    return measurements


def audit(data, profile, hardware, *, max_shapes=0, required_shapes=None):
    labels = validate_labels(data)
    if profile.get('gemm_measurements') or profile.get('gemm_profile'):
        raise ValueError('audit an analytic profile; measured lookup would compare labels with themselves')
    if type(max_shapes) is not int or max_shapes < 0:
        raise ValueError('max_shapes must be nonnegative')
    keys = sorted(labels)
    if max_shapes and max_shapes < len(keys):
        # Deterministic direction-stratified SHA order, disclosed as a subset.
        directions = sorted({r['problem']['direction'] for r in labels.values()})
        bins = {d: [k for k in keys if labels[k]['problem']['direction'] == d] for d in directions}
        chosen = []
        index = 0
        while len(chosen) < max_shapes:
            advanced = False
            for d in directions:
                if index < len(bins[d]) and len(chosen) < max_shapes:
                    chosen.append(bins[d][index]);advanced = True
            if not advanced:
                break
            index += 1
        keys = sorted(chosen)
    provider = GPUCostProvider(profile)
    rows = []
    for key in keys:
        record = labels[key];problem = record['problem']
        estimate = provider.estimate(KernelSpec('independent_GEMM_cost_audit',
            matmuls=(Matmul(**problem),)), hardware)
        predicted = max(1, estimate.duration_ns - profile['kernel_launch_ns'])
        observed = record['duration_ns']
        rows.append({'signature_sha256': key, 'problem': problem, 'family_sha256': shape_family(problem),
            'split': split_for(problem), 'predicted_hot_ns': predicted, 'observed_hot_ns': observed,
            'predicted_over_observed': predicted / observed,
            'absolute_relative_error': abs(predicted - observed) / observed,
            'repeat_relative_difference': record['repeat_relative_difference']})
    groups = defaultdict(list)
    for row in rows:
        groups['all'].append(row)
        groups['direction:' + row['problem']['direction']].append(row)
        groups['split:' + row['split']].append(row)
        groups['split:' + row['split'] + '/direction:' + row['problem']['direction']].append(row)
    required = required_shapes or {}
    for key, problem in required.items():
        if signature(problem) != key:
            raise ValueError('required shape signature integrity failure')
    missing = sorted(set(required) - set(labels))
    return {'schema_version': 'rlxy.gpu_cost_validation/1', 'status': 'DIAGNOSTIC_ONLY',
        'measurement_domain_count': len(labels), 'audited_shape_count': len(rows),
        'all_measured_shapes_audited': len(rows) == len(labels),
        'sampling': 'all signatures' if len(rows) == len(labels) else 'direction-stratified SHA order, no outcome selection',
        'statistics': {k: error_statistics(v) for k, v in sorted(groups.items())},
        'required_domain': {'count': len(required), 'missing_count': len(missing),
            'covered_signature_fraction': (len(required) - len(missing)) / len(required) if required else None,
            'missing_signature_sha256': missing,
            'exact_domain_complete': not missing if required else None},
        'comparison_scope': 'unfused hot cuBLAS BF16 GEMM device time incl memory vs analytic service minus declared host launch',
        'host_launch_subtracted_ns': profile['kernel_launch_ns'],
        'split_rule': 'shape geometry log2 buckets + direction/precision/batches family SHA modulo5; one fold held out',
        'fitting_or_global_rescaling_performed': False,
        'holdout_interpretation': 'predeclared geometry partition for later development; no model is fitted in this audit',
        'whole_veRL_or_48GPU_cluster_validated': False, 'wafer_model_validated': False,
        'paper_performance_ready': False,
        'limitations': ['hot labels omit host launch and fused attention/MLP, caching and dynamic scheduling effects',
            'repeated hot GEMMs can retain inputs/weights in L2; analytic model omits cross-CTA L2',
            'cold/streaming and representative operator-sequence validation are required before full-step lookup replacement',
            'these mismatch metrics are not measured whole-veRL runtime error or a global correction factor',
            'unique-signature WAPE is not execution-frequency-weighted step error',
            'label repeatability alone is not an independent generalization/error bound',
            'unseen exact signatures cannot be supplied to the measured lookup without new labels'],
        'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements', type=Path, required=True)
    parser.add_argument('--gpu-profile', type=Path, default=Path('configs/gpu/a100_48_reference.json'))
    parser.add_argument('--case', type=Path, default=Path('configs/wafer4_fixture/case.json'))
    parser.add_argument('--required-shapes', type=Path)
    parser.add_argument('--max-shapes', type=int, default=0, help='0 audits complete label domain')
    parser.add_argument('--max-holdout-wape', type=float, help='optional explicitly chosen gate, not an established hardware accuracy threshold')
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    if a.out.exists():
        parser.error('output exists; preserve earlier validation')
    profile = load_gpu_profile(a.gpu_profile)
    case = effective_gpu_case(load_case(a.case), profile)
    data = json.loads(a.measurements.read_text())
    required = None
    if a.required_shapes:
        source = json.loads(a.required_shapes.read_text())
        required = source['shapes']
        if 'required_packed_domain' in source:
            required = {k: required[k] for k in source['required_packed_domain']}
    result = audit(data, profile, case.hardware, max_shapes=a.max_shapes, required_shapes=required)
    result['input_sha256'] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in [('measurements', a.measurements), ('gpu_profile', a.gpu_profile),
            ('case_template', a.case), ('script', Path(__file__)),
            ('analytic_gpu_model_source', Path(__file__).resolve().parents[1] / 'src/rl_xy/gpu_model.py'),
            ('operator_ir_source', Path(__file__).resolve().parents[1] / 'src/rl_xy/operator_ir.py')]}
    result['device'] = data['device']
    result['measurement_provenance'] = {k: data.get(k) for k in ['started_at_utc', 'torch_version', 'cuda_version',
        'source_shapes_sha256', 'benchmark_source_sha256', 'numeric_fixture_checks']}
    failed = bool(result['required_domain']['missing_count'])
    if a.max_holdout_wape is not None:
        if not 0 < a.max_holdout_wape < 10:
            parser.error('explicit holdout WAPE threshold must be positive and below10')
        metric = result['statistics'].get('split:holdout', {}).get('WAPE_unique_shapes')
        result['declared_holdout_gate'] = {'max_WAPE': a.max_holdout_wape, 'observed_WAPE': metric,
            'passed': metric is not None and metric <= a.max_holdout_wape}
        failed |= not result['declared_holdout_gate']['passed']
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['rows', 'measurement_provenance']}, ensure_ascii=False, indent=2))
    return 2 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
