"""Explicit protocol residuals and complete runtime cost identities.

Protocol terms complement native packet/fabric timing and the existing endpoint
ledger. They do not replace transport with a fitted end-to-end constant.
"""
from dataclasses import asdict
import hashlib
import json
import math


def protocol_overhead_ns(op, ranks, kind, payload):
    if ranks <= 1 or not getattr(op, 'profile', None):
        return 0.
    model = op.profile.get('communication_service_overhead')
    if not model:
        return 0.
    if model.get('schema_version') != 'rlxy.gpu_protocol_overhead/1':
        raise ValueError('unsupported GPU protocol model')
    if model.get('status') not in ('FROZEN', 'VALIDATION_CANDIDATE'):
        raise ValueError('GPU protocol model has no declared validation state')
    calibrated_ranks = model.get('calibrated_ranks', [])
    if ranks not in calibrated_ranks and not model.get('allow_rank_extrapolation', False):
        raise ValueError('GPU collective rank count outside calibrated domain')
    terms = model.get('by_kind', {}).get(kind)
    if terms is None:
        raise ValueError('GPU protocol cost missing collective '+kind)
    coefficients = [terms.get(k, 0.) for k in
                    ('fixed_ns', 'per_peer_ns', 'per_input_byte_ns', 'per_wire_byte_ns')]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
           for v in coefficients):
        raise ValueError('invalid GPU protocol coefficient')
    if payload < 0:
        raise ValueError('negative communication payload')
    wire_factor = (2*(ranks-1)/ranks if kind == 'ALL_REDUCE' else
                   (ranks-1)/ranks if kind == 'REDUCE_SCATTER' else ranks-1)
    knots = terms.get('wire_residual_knots')
    if knots is not None:
        if any(coefficients):
            raise ValueError('lookup residual cannot also charge linear residual terms')
        if terms.get('wire_residual_boundary_policy') != 'constant_below_nonnegative_last_above':
            raise ValueError('lookup residual requires an explicit boundary policy')
        if len(knots) < 2 or any(len(k) != 2 or not all(
                isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in k)
                for k in knots):
            raise ValueError('invalid residual lookup knots')
        if any(a[0] >= b[0] for a, b in zip(knots, knots[1:])):
            raise ValueError('residual lookup knots must have distinct increasing sizes')
        x = wire_factor*payload
        if x <= knots[0][0]:
            return float(knots[0][1])
        for a, b in zip(knots, knots[1:]):
            if x <= b[0]:
                return max(0., a[1]+(b[1]-a[1])*(x-a[0])/(b[0]-a[0]))
        a, b = knots[-2:]
        return b[1]+max(0., (b[1]-a[1])/(b[0]-a[0]))*(x-b[0])
    return (coefficients[0] + coefficients[1]*(ranks-1) +
            coefficients[2]*payload + coefficients[3]*wire_factor*payload)


def operator_cost_identity(op):
    """Include runtime overrides as well as source/binary identity in caches."""
    fields = {
        'hardware': asdict(op.hw),
        'gpu_hardware_and_protocol_profile': getattr(op, 'profile', None),
        'local_timing_identity': getattr(op, 'local_timing_identity', None),
        'runtime_cost_identity': getattr(op, 'runtime_cost_identity', None),
    }
    raw = json.dumps(fields, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return {'schema_version': 'rlxy.operator_cost_identity/1',
            'sha256': hashlib.sha256(raw.encode()).hexdigest(), 'fields': json.loads(raw)}
