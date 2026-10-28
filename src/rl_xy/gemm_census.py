"""Exact GEMM domain/call census shared by runtime and CLI, never timings."""
from collections import Counter
from dataclasses import asdict
import hashlib,json
from .cost_model import CostEstimate

def signature(problem):
    return hashlib.sha256(json.dumps(problem, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


class CensusProvider:
    """Zero workspace; integer duration1 exists only to satisfy DAG contracts.

    This is deliberately NOT a GPU/wafer cost provider. No generated graph or
    placeholder durations are exported or eligible for performance reporting.
    """
    def estimate(self, spec, hardware):
        zeros = (0,) * hardware.cores_per_die
        # Preserve mandatory gradient IO lower bounds for batching validators.
        # These are logical matrix bytes, not tiled traffic or timing costs.
        reads = spec.read_bytes + sum(m.batches * (m.m*m.k*m.a_bytes + m.k*m.n*m.b_bytes) for m in spec.matmuls)
        writes = spec.write_bytes + sum(m.batches*m.m*m.n*m.output_bytes for m in spec.matmuls)
        return CostEstimate(1, 0, 1, reads, writes, 0, (),
            source='CENSUS_ONLY_NOT_A_PERFORMANCE_ESTIMATE',
            core_workspace_bytes=zeros, core_compute_ns=zeros)


def census_graph(graph, *, active):
    """Count rank-local GEMM invocations, not unique shapes or macro kernels."""
    if type(active) is not int or active <= 0:
        raise ValueError('active must be a positive integer')
    rows = {};templates = [];expansions = getattr(graph, 'component_expansions', [])
    grouped = {}
    for expansion in expansions:
        contraction = expansion['contraction']
        if contraction is None:
            raise ValueError('component expansion has no original plan')
        physical = tuple(expansion['physical'])
        if not physical or len(set(physical)) != len(physical) or any(type(r) is not int or not 0 <= r < active for r in physical):
            raise ValueError('component physical participants include invalid/idle ranks')
        key = id(contraction)
        if key not in grouped:
            grouped[key] = {'plan': contraction.original_plan, 'invocations': 0,
                'physical_width': len(physical)}
        group = grouped[key]
        if group['physical_width'] != len(physical):
            raise ValueError('same component template invoked at different TP widths')
        group['invocations'] += 1

    def add(op, multiplier, domain, width=None):
        if op.kind != 'compute':
            return
        if op.kernel_spec is None:
            if op.flops:
                raise ValueError('nonzero GEMM FLOPs without shape-bearing spec: ' + op.name)
            return
        participants = tuple(op.participant_ranks)
        limit = active if width is None else width
        if not participants or len(set(participants)) != len(participants) or any(type(r) is not int or not 0 <= r < limit for r in participants):
            raise ValueError('GEMM/compute must have explicit legal participant ranks')
        if type(multiplier) is not int or multiplier <= 0:
            raise ValueError('integer occurrence multiplier required')
        for matmul in op.kernel_spec.matmuls:
            problem = asdict(matmul);key = signature(problem)
            row = rows.setdefault(key, {'signature_sha256': key, 'problem': problem,
                'occurrences': 0, 'logical_matmul_occurrences': 0,
                'weighted_flops': 0, 'directions': Counter(), 'phases': Counter(),
                'kernel_variants': Counter(), 'domains': Counter()})
            if row['problem'] != problem:
                raise ValueError('signature collision')
            occurrence = multiplier * len(participants)
            row['occurrences'] += occurrence
            row['logical_matmul_occurrences'] += multiplier
            row['weighted_flops'] += matmul.flops * occurrence
            row['directions'][matmul.direction] += occurrence
            row['phases'][op.metadata.get('colocation_stage', op.phase)] += occurrence
            row['kernel_variants'][op.kernel_spec.kernel_variant] += occurrence
            row['domains'][domain] += occurrence

    for group in grouped.values():
        plan = group['plan'];identity = []
        for op in plan.operations:
            add(op, group['invocations'], 'component_original_plan', group['physical_width'])
            if op.kernel_spec and op.kernel_spec.matmuls:
                identity.append({'participants': list(op.participant_ranks),
                    'phase': op.phase, 'spec': asdict(op.kernel_spec)})
        templates.append({'template_shapes_sha256': signature(identity),
            'invocations': group['invocations'], 'physical_width': group['physical_width'],
            'original_operation_count': len(plan.operations),
            'original_matmul_count': sum(len(o.kernel_spec.matmuls) for o in plan.operations if o.kernel_spec)})
    globals_count = 0
    for op in graph.plan.operations:
        if expansions and op.metadata.get('stage_component') is not None:
            continue
        globals_count += 1
        add(op, 1, 'global_noncomponent' if expansions else 'uncontracted_graph')
    for row in rows.values():
        for field in ['directions', 'phases', 'kernel_variants', 'domains']:
            row[field] = dict(sorted(row[field].items()))
    return {'shapes': {k: rows[k]['problem'] for k in sorted(rows)},
        'signatures': {k: rows[k] for k in sorted(rows)},
        'totals': {'unique_signatures': len(rows),
            'rank_local_matmul_occurrences': sum(r['occurrences'] for r in rows.values()),
            'logical_matmul_occurrences': sum(r['logical_matmul_occurrences'] for r in rows.values()),
            'weighted_flops': sum(r['weighted_flops'] for r in rows.values()),
            'component_templates': len(grouped), 'component_invocations': len(expansions),
            'global_noncomponent_operation_count': globals_count},
        'component_templates': sorted(templates, key=lambda r: r['template_shapes_sha256'])}


def measured_domain(result, measurements):
    """Coverage uses exact full Matmul signatures; no analytic fallback."""
    shapes = result['shapes'];valid = {}
    for key, row in (measurements or {}).items():
        if signature(row['problem']) != key:
            raise ValueError('measurement signature integrity failure')
        if type(row.get('duration_ns')) is not int or row['duration_ns'] <= 0:
            raise ValueError('positive integer measured duration required')
        if not 0 <= row.get('repeat_relative_difference', 1) <= .15:
            raise ValueError('measurement stability gate failed')
        valid[key] = row
    missing = sorted(set(shapes) - set(valid))
    covered = [k for k in shapes if k in valid]
    return {'exact_domain_complete': bool(shapes) and not missing,
        'profile_signature_count': len(valid), 'required_signature_count': len(shapes),
        'missing_count': len(missing), 'missing_signature_sha256': missing,
        'missing_shapes': {k: shapes[k] for k in missing},
        'covered_rank_local_occurrences': sum(result['signatures'][k]['occurrences'] for k in covered),
        'missing_rank_local_occurrences': sum(result['signatures'][k]['occurrences'] for k in missing),
        'covered_weighted_flops': sum(result['signatures'][k]['weighted_flops'] for k in covered),
        'missing_weighted_flops': sum(result['signatures'][k]['weighted_flops'] for k in missing),
        'fallback_permitted': False, 'scope': 'Exact shape coverage only, not cache/fusion/fullGPU calibration'}
