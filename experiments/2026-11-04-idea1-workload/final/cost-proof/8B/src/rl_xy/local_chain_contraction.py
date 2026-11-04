"""Exact finalized-clock contraction of strictly rank-serial local chains.

Network operations remain native. Native resources use a preallocated per-rank
arena; original object lifetimes are independently replayed at expanded native
logical clocks. Arena hooks are NOT the original object-level native hooks.
"""
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json

from .tensor_plan import TensorPlan, preflight_contract
from .types import AstraEvent


def _digest(value):
    h = hashlib.sha256()
    # The C encoder emits the same canonical bytes as the Python iterator.
    # Template contracts contain many private-core integers; per-token Python
    # callbacks made fingerprinting dominate planning despite unchanged IR.
    # Keep this at template boundaries; this does not materialize a global DAG.
    for part in json.JSONEncoder(sort_keys=True, separators=(',', ':')).iterencode(value, _one_shot=True):
        h.update(part.encode())
    return h.hexdigest()


def _ops_digest(operations):
    h = hashlib.sha256()
    for op in operations:
        row = {f.name: getattr(op, f.name) for f in fields(op) if f.name not in ('metadata', 'kernel_spec')}
        row['metadata'] = {k: v for k, v in op.metadata.items() if k != 'cost_estimate'}
        row['kernel_spec'] = asdict(op.kernel_spec) if op.kernel_spec is not None else None
        h.update(bytes.fromhex(_digest(row)))
    return h.hexdigest()


def _participants(op, ranks):
    return tuple(op.participant_ranks) or ranks


def _deps(op, rank):
    return tuple(op.dependencies_by_rank.get(rank, op.dependencies))


def _validate_serial(plan):
    ranks = tuple(sorted(int(r) for r in plan.contract['ranks']))
    if not ranks or len(set(ranks)) != len(ranks):
        raise ValueError('explicit rank contract required')
    valid_ranks = set(ranks);counts = {r: 0 for r in ranks}
    last = {};seen = {};previous_id = 0
    for op in plan.operations:
        if type(op.node_id) is not int or op.node_id <= previous_id:
            raise ValueError('contraction requires unique increasing logical node IDs')
        previous_id = op.node_id
        participants = _participants(op, ranks)
        if len(set(participants)) != len(participants) or not set(participants) <= valid_ranks:
            raise ValueError('invalid logical participant set')
        if op.kind == 'compute' and (type(op.duration_ns) is not int or op.duration_ns <= 0):
            raise ValueError('exact contraction requires positive integer finalized model durations')
        for rank in participants:
            counts[rank] += 1
            deps = _deps(op, rank)
            if rank in last and last[rank] not in deps:
                raise ValueError('rank interleaving/overlap: immediate predecessor dependency missing')
            for dependency in deps:
                if dependency not in seen or rank not in _participants(seen[dependency], ranks):
                    raise ValueError('cross-rank/nonpreceding dependency cannot be contracted')
            if str(op.node_id) not in plan.contract['ranks'][str(rank)]['nodes']:
                raise ValueError('logical operation has no original resource node')
            last[rank] = op.node_id
        seen[op.node_id] = op
    for rank, row in plan.contract['ranks'].items():
        if len(row['nodes']) != counts[int(rank)]:
            raise ValueError('resource contract has missing/extra logical operations')
    return ranks


def replay_logical_resources(plan, events=None, *, _validated_ranks=None):
    """Independent serial logical ledger, optionally bound to real clocks.

    Reports original allocated peaks separately from reservation/admission
    peaks. No native snapshots are invented for internal macro objects.
    """
    ranks = _validated_ranks or _validate_serial(plan)
    rank_operations = {r: [] for r in ranks}
    for operation in plan.operations:
        for rank in _participants(operation, ranks):
            rank_operations[rank].append(operation)
    ordinary_preflight = preflight_contract(plan.contract)
    event_by_key = None
    if events is not None:
        event_by_key = {}
        for e in events:
            key = (e.rank, e.node_id)
            if key in event_by_key:
                raise ValueError('duplicate expanded logical event')
            event_by_key[key] = e
        expected = {(r, o.node_id) for o in plan.operations for r in _participants(o, ranks)}
        if set(event_by_key) != expected:
            raise ValueError('expanded logical event coverage mismatch')
    result = {}
    for rank in ranks:
        row = plan.contract['ranks'][str(rank)]
        live = dict(row['initial_objects']);ready = set(live);reserved = 0
        used = sum(live.values());allocated_peak = admission_peak = used
        cores = row.get('cores_per_die', 1);core_peaks = [0] * cores;sram_peak = 0;last_tick = None
        count = 0
        for op in rank_operations[rank]:
            count += 1;n = row['nodes'][str(op.node_id)]
            if event_by_key is not None:
                e = event_by_key[rank, op.node_id]
                if last_tick is not None and e.issue_tick < last_tick:
                    raise ValueError('expanded logical clocks overlap on a rank')
                if e.callback_tick < e.issue_tick:
                    raise ValueError('negative logical duration')
                for dependency in _deps(op, rank):
                    if event_by_key[rank, dependency].callback_tick > e.issue_tick:
                        raise ValueError('expanded clocks violate logical dependency')
                last_tick = e.callback_tick
            if not ready.issuperset(n['reads']):
                raise ValueError('logical boundary/internal read absent or not ready')
            if any(k in live for k in n['allocations']):
                raise ValueError('logical duplicate allocation')
            if any(type(v) is not int or v < 0 for v in n['allocations'].values()):
                raise ValueError('invalid logical allocation size')
            reserved += n['reserve_before']
            if reserved < 0:
                raise ValueError('negative logical reservation')
            live.update(n['allocations']);used += sum(n['allocations'].values())
            ready.difference_update(n['ready_outputs'])
            allocated_peak = max(allocated_peak, used);admission_peak = max(admission_peak, used + reserved)
            if used + reserved > row['dram_capacity']:
                raise ValueError('logical DRAM admission overflow')
            vector = n.get('core_workspace_bytes', [n['workspace_bytes']])
            if len(vector) != cores or sum(vector) != n['workspace_bytes']:
                raise ValueError('logical private SRAM vector invalid')
            capacity = row.get('sram_bytes_per_core', n['sram_capacity'])
            if any(type(x) is not int or not 0 <= x <= capacity for x in vector):
                raise ValueError('logical private SRAM capacity exceeded')
            core_peaks = [max(a, b) for a, b in zip(core_peaks, vector)]
            sram_peak = max(sram_peak, sum(vector))
            if any(k not in live for k in n['ready_outputs']):
                raise ValueError('logical READY without storage')
            if any('.reference.' in k for k in n['ready_outputs']):
                raise ValueError('logical immutable Reference written')
            ready.update(n['ready_outputs'])
            for key in n['frees']:
                if key not in live:
                    raise ValueError('logical double free')
                used -= live.pop(key);ready.discard(key)
            reserved += n['reserve_after']
            if reserved < 0:
                raise ValueError('negative logical reservation at callback')
        if live != row['initial_objects'] or reserved:
            raise ValueError('logical temporary/reservation leak')
        if ordinary_preflight[f'd{rank}.dram'] != allocated_peak:
            raise ValueError('independent logical ledger disagrees with original preflight')
        result[str(rank)] = {'allocated_dram_peak_bytes': allocated_peak,
            'admission_dram_peak_including_reservations': admission_peak,
            'logical_sram_peak_bytes': sram_peak, 'logical_private_core_peak_bytes': core_peaks,
            'logical_resource_nodes_replayed': count}
    return {'status': 'PASS', 'clock_bound_to_expanded_native_events': events is not None,
        'original_contract_sha256': _digest(plan.contract), 'ranks': result,
        'native_per_object_hook_claimed': False}


@dataclass(frozen=True)
class ContractionBlock:
    native_node_id: int
    logical_node_ids: tuple[int, ...]
    participants: tuple[int, ...]
    duration_prefix_model_ns: tuple[int, ...]


@dataclass
class ContractionResult:
    original_plan: TensorPlan
    macro_plan: TensorPlan
    blocks: tuple[ContractionBlock, ...]
    certificate: dict


def contract_local_chains(plan, *, astra_time_scale=1000, allow_multi_rank_compute=True, min_chain=2,
        retain_initial_objects=True):
    if type(astra_time_scale) is not int or astra_time_scale <= 0 or type(min_chain) is not int or min_chain < 2:
        raise ValueError('positive integer time scale and min_chain>=2 required')
    ranks = _validate_serial(plan)
    proof = replay_logical_resources(plan, _validated_ranks=ranks)
    def eligible(op):
        participants = _participants(op, ranks)
        return (op.kind == 'compute' and not op.metadata.get('contraction_boundary')
            and op.metadata.get('execution_domain') != 'host'
            and (op.kernel_spec is None or op.kernel_spec.execution_domain == 'device')
            and (allow_multi_rank_compute or len(participants) == 1))
    groups = [];current = [];current_ids = set()
    for op in plan.operations:
        same_chain_inputs = not current or all(set(_deps(op, r)) <= current_ids
            for r in _participants(op, ranks))
        if current and (not eligible(op) or not eligible(current[-1])
            or _participants(op, ranks) != _participants(current[-1], ranks)
            or op.round_id != current[-1].round_id or not same_chain_inputs):
            groups.append(current);current = [];current_ids = set()
        current.append(op);current_ids.add(op.node_id)
        if not eligible(op):
            groups.append(current);current = [];current_ids = set()
    if current:
        groups.append(current)
    expanded_groups = []
    for group in groups:
        expanded_groups.extend([group] if len(group) >= min_chain else [[o] for o in group])
    mapping = {};blocks = []
    for group in expanded_groups:
        prefix = [0]
        for op in group:
            prefix.append(prefix[-1] + (op.duration_ns if op.kind == 'compute' else 0))
            mapping[op.node_id] = group[0].node_id
        blocks.append(ContractionBlock(group[0].node_id, tuple(o.node_id for o in group),
            _participants(group[0], ranks), tuple(prefix)))
    macro_ops = []
    arena_contract = {k: v for k, v in plan.contract.items() if k != 'ranks'}
    arena_contract['ranks'] = {};arena_tensors = [];arena_sizes = {}
    if retain_initial_objects:
        initial_ids = {k for r in plan.contract['ranks'].values() for k in r['initial_objects']}
        arena_tensors.extend(t for t in plan.tensors if t['storage_id'] in initial_ids)
    for rank in ranks:
        original = plan.contract['ranks'][str(rank)]
        initial_resident = sum(original['initial_objects'].values()) if retain_initial_objects else 0
        arena = proof['ranks'][str(rank)]['admission_dram_peak_including_reservations'] - initial_resident
        arena_sizes[str(rank)] = {'retained_initial_resident_bytes': initial_resident,
            'temporary_arena_bytes': arena, 'native_reserved_dram_bytes': initial_resident + arena}
        sid = f'd{rank}.logical_plan_arena'
        arena_contract['ranks'][str(rank)] = {k: v for k, v in original.items() if k not in ('initial_objects', 'nodes')}
        initial = dict(original['initial_objects']) if retain_initial_objects else {}
        if sid in initial:
            raise ValueError('arena storage identity collides with original persistent object')
        if arena:
            initial[sid] = arena
        arena_contract['ranks'][str(rank)].update(initial_objects=initial, nodes={})
        if arena:
            arena_tensors.append({'storage_id': sid, 'tensor_id': sid, 'rank': rank, 'kind': 'preallocated_arena',
                'nbytes': arena, 'shape': [arena], 'dtype_bytes': 1, 'persistent': True,
                'created_node': None, 'ready_node': None, 'released_node': None})
    for group, block in zip(expanded_groups, blocks):
        first = group[0]
        deps = {rank: tuple(sorted({mapping[x] for x in _deps(first, rank)})) for rank in block.participants}
        if any(block.native_node_id in values for values in deps.values()):
            raise ValueError('invalid macro self-dependency')
        op = replace(first, dependencies_by_rank=deps,
            dependencies=tuple(sorted({x for values in deps.values() for x in values})))
        if len(group) > 1:
            op = replace(op, op_id=f'local_chain_macro.n{first.node_id}', phase='local_chain_macro',
                name='finalized_local_chain', duration_ns=block.duration_prefix_model_ns[-1], kernel_spec=None,
                flops=sum(o.flops for o in group), vector_ops=sum(o.vector_ops for o in group),
                tensor_bytes=sum(o.tensor_bytes for o in group),
                metadata={'execution_domain': 'device', 'timing_finalized': True,
                    'cost_source': 'exact_sum_of_finalized_logical_durations',
                    'logical_node_ids': list(block.logical_node_ids),
                    'resource_hook_scope': 'preallocated_arena_and_private_workspace; logical objects independently replayed'})
        macro_ops.append(op)
        for rank in block.participants:
            source = plan.contract['ranks'][str(rank)]
            rows = [source['nodes'][str(o.node_id)] for o in group]
            cores = source.get('cores_per_die', 1)
            vectors = [n.get('core_workspace_bytes', [n['workspace_bytes']]) for n in rows]
            maximum = [max(v[c] for v in vectors) for c in range(cores)]
            sid = f'd{rank}.logical_plan_arena'
            initial = source['initial_objects'] if retain_initial_objects else {}
            initial_keys = set(initial)
            if any(set(x['frees']) & initial_keys or set(x['allocations']) & initial_keys for x in rows):
                raise ValueError('retained initial-object lifetime changes need an explicit outer-stage contract')
            reads = sorted({k for x in rows for k in x['reads'] if k in initial})
            if arena_sizes[str(rank)]['temporary_arena_bytes']:
                reads.append(sid)
            writes = sorted({k for x in rows for k in x['ready_outputs'] if k in initial})
            n = {'reads': reads,
                'allocations': {}, 'ready_outputs': writes, 'frees': [],
                'workspace_bytes': sum(maximum), 'sram_capacity': min(x['sram_capacity'] for x in rows),
                'local_memory': any(x['local_memory'] for x in rows), 'reserve_before': 0, 'reserve_after': 0}
            if cores > 1:
                n['core_workspace_bytes'] = maximum
            arena_contract['ranks'][str(rank)]['nodes'][str(op.node_id)] = n
    arena_peaks = preflight_contract(arena_contract)
    certificate = {'schema_version': 'rlxy.local_chain_contraction/1',
        'logical_operation_count': len(plan.operations), 'native_operation_count': len(macro_ops),
        'logical_operations_sha256': _ops_digest(plan.operations), 'logical_resources': proof,
        'astra_time_scale': astra_time_scale, 'strict_per_rank_serialization_proved': True,
        'default_only_single_rank_compute': not allow_multi_rank_compute,
        'native_resource_scope': 'preallocated per-rank DRAM arena; private-core workspace maximum per macro',
        'logical_object_scope': 'original alloc/ready/read/free/reservation ledger independently replayed at expanded clocks',
        'retain_initial_objects': retain_initial_objects, 'arena_reservations_by_rank': arena_sizes,
        'logical_temporary_objects_are_views_into_preallocated_arena': True,
        'retained_persistent_objects_keep_original_native_storage_identity': retain_initial_objects,
        'native_SRAM_reservation_sum_can_exceed_logical_simultaneous_peak': True,
        'zero_additional_per_node_compute_scheduler_delay_required': True,
        'operations_digest_excludes_only_derived_cost_estimate_payload': True,
        'network_operations_deleted': 0, 'blocks': [asdict(b) for b in blocks]}
    return ContractionResult(plan, TensorPlan(arena_contract, arena_tensors, macro_ops, arena_peaks), tuple(blocks), certificate)


def expand_native_events(result, native_events):
    """Expand observed macro clocks; require an exact duration tick identity.

    Uncontracted network issue/callback ticks are copied without estimation.
    This is a clock/resource proof, not a claim of simulated numerical values.
    """
    plan = result.original_plan
    if _ops_digest(plan.operations) != result.certificate['logical_operations_sha256']:
        raise ValueError('logical operations changed after contraction')
    if _digest(plan.contract) != result.certificate['logical_resources']['original_contract_sha256']:
        raise ValueError('logical resource contract changed after contraction')
    scale = result.certificate['astra_time_scale']
    original = {o.node_id: o for o in plan.operations}
    blocks = {b.native_node_id: b for b in result.blocks}
    wanted = {(r, b.native_node_id) for b in result.blocks for r in b.participants}
    observed = {};expanded = []
    for event in native_events:
        if any(type(t) is not int or t < 0 for t in (event.issue_tick, event.callback_tick)):
            raise ValueError('native ticks must be nonnegative integers')
        key = (event.rank, event.node_id)
        if key in observed or key not in wanted:
            raise ValueError('duplicate/unknown native macro event')
        observed[key] = event
        block = blocks[event.node_id]
        if len(block.logical_node_ids) > 1 and event.callback_tick - event.issue_tick != block.duration_prefix_model_ns[-1] * scale:
            raise ValueError('native macro duration differs from exact finalized compute sum')
        for index, logical_id in enumerate(block.logical_node_ids):
            op = original[logical_id]
            start = event.issue_tick + block.duration_prefix_model_ns[index] * scale if len(block.logical_node_ids) > 1 else event.issue_tick
            end = event.issue_tick + block.duration_prefix_model_ns[index + 1] * scale if len(block.logical_node_ids) > 1 else event.callback_tick
            expanded.append(AstraEvent(event.rank, logical_id, op.op_id, op.round_id, op.phase,
                event.node_type, start, end, end - start, start / scale, end / scale, (end - start) / scale))
    if set(observed) != wanted:
        raise ValueError('incomplete native macro event coverage')
    proof = replay_logical_resources(plan, expanded)
    if proof['ranks'] != result.certificate['logical_resources']['ranks']:
        raise ValueError('expanded-clock logical resource peaks changed')
    return sorted(expanded, key=lambda e: (e.rank, e.node_id)), proof
