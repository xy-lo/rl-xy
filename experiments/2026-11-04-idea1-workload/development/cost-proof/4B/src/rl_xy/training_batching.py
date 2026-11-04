"""Packed dense training microbatches with explicit summed-gradient storage.

Linear forward/dinput concatenate token rows; linear dweight concatenates the
reduction dimension and produces ONE FP32 matrix. Non-GEMM contributions use
independent partials followed by a charged sum. Tied embeddings transpose/sum
head and embedding aggregates before one persistent-gradient RMW. Attention
retains independent ragged request blocks. This is an analytic operator/tensor
contract, not a claim to reproduce a particular fused CUDA implementation.
"""
from collections import OrderedDict
from dataclasses import asdict, replace
import copy
import hashlib
import re

from .cost_model import CostEstimate
from .mapped_costs import apply_mapped_costs
from .model import enumerate_parameters, parameter_totals
from .operator_ir import KernelSpec
from .tensor_plan import TensorPlan, finalize_tensor_costs, preflight_contract
from .types import CheckResult, Operation


SUPPORTED_TRAINING_MICROBATCHES = (1, 2, 4, 8, 16, 32, 64)
_LINEAR_FORWARD = {'qkv', 'o_proj_partial', 'gate_up', 'down_partial', 'lm_head_local'}
_GRADIENT_IO_VERSION = 'rlxy.explicit_gradient_output_io/1'


class TrainingBatchingUnsupported(ValueError):
    def __init__(self, audit):
        self.audit = audit
        super().__init__('training batching contract rejected: '+str(audit['blockers']))


class _SpecOnly:
    def estimate(self, spec, hardware):
        return CostEstimate(1, 0, 1, 0, 0, 0, (), source='temporary_spec_materialization_not_executed')


def _validate_size(size):
    if type(size) is not int or size not in SUPPORTED_TRAINING_MICROBATCHES:
        raise ValueError('training microbatch must be a supported positive integer: '+str(SUPPORTED_TRAINING_MICROBATCHES))


def training_batching_eligibility(plan, case, microbatch_trajectories):
    _validate_size(microbatch_trajectories)
    blockers = []
    if microbatch_trajectories != 1:
        if case.strategy.pipeline_parallel != 1 or case.strategy.training_epochs != 1 or case.strategy.optimizer_minibatch_trajectories != 0:
            blockers.append('packed contract currently requires PP1 and one full-batch update per round')
        if case.model.gradient_dtype_bytes != 4:
            blockers.append('summed parameter gradients require FP32 storage')
    return dict(schema_version='rlxy.training_batching_readiness/2',
        requested_microbatch_trajectories=microbatch_trajectories,
        implemented_microbatch_trajectories=list(SUPPORTED_TRAINING_MICROBATCHES),
        eligible=not blockers, runtime_ready=not blockers, blockers=blockers,
        transformation='exact_identity' if microbatch_trajectories == 1 else 'packed_token_rows_and_summed_parameter_gradients',
        trajectory_count=len(case.workload.trajectories))


def pack_training_matmuls(problems, *, linear=False):
    """Pack corresponding products; attention is grouped by exact shape only."""
    if not problems:
        return ()
    first = problems[0]
    types = (first.a_bytes, first.b_bytes, first.output_bytes, first.accumulator_bytes, first.direction)
    if any((p.a_bytes, p.b_bytes, p.output_bytes, p.accumulator_bytes, p.direction) != types for p in problems):
        raise ValueError('incompatible matmul precision/direction in one packed product')
    if linear or first.direction in ('dinput', 'dweight'):
        if any(p.batches != 1 for p in problems):
            raise ValueError('packed linear products require one matrix per trajectory')
        if first.direction == 'dweight':
            if any((p.m, p.n) != (first.m, first.n) for p in problems) or first.output_bytes != 4:
                raise ValueError('dweight must share FP32 parameter output shape')
            packed = (replace(first, k=sum(p.k for p in problems)),)
        else:
            if any((p.n, p.k) != (first.n, first.k) for p in problems):
                raise ValueError('linear forward/dinput output width and reduction shape differ')
            packed = (replace(first, m=sum(p.m for p in problems)),)
    else:
        groups = OrderedDict()
        for p in problems:
            key = (p.m, p.n, p.k)
            groups[key] = replace(p, batches=p.batches+groups[key].batches if key in groups else p.batches)
        packed = tuple(groups.values())
    if sum(p.flops for p in packed) != sum(p.flops for p in problems):
        raise ValueError('packed GEMM FLOP conservation failed')
    return packed


def _component(op, case):
    """Parameter components independently partition the persistent row table."""
    name = op.name.rsplit('.', 1)[-1]
    layer = re.search(r'\.layer(\d+)\.', op.name)
    if layer:
        prefix = 'layer'+layer.group(1)
        mapping = {
            'ffn_down_backward': (prefix+'.ffn', 'down'),
            'gate_up_backward': (prefix+'.ffn', 'gate_up'),
            'o_proj_backward': (prefix+'.o_proj', 'weight'),
            'qkv_backward': (prefix+'.qkv', 'weight'),
            'rmsnorm_ffn_backward': (prefix+'.rmsnorm', 'ffn_scale'),
            'rmsnorm_attn_backward': (prefix+'.rmsnorm', 'attn_scale'),
            'qkv_bias_backward': (prefix+'.qkv_bias', 'bias'),
        }
        if name in mapping:
            return mapping[name]
    mapping = {
        'lm_head_backward_dinput_dweight': ('token_embedding' if case.model.tie_word_embeddings else 'lm_head', 'head'),
        'embedding_backward': ('token_embedding', 'embedding'),
        'final_rmsnorm_backward': ('final_rmsnorm', 'scale'),
    }
    if name not in mapping:
        raise ValueError('unknown parameter-gradient producer: '+op.name)
    return mapping[name]


def _expected_components(case):
    result = {}
    for p in enumerate_parameters(case):
        name = p.parameter_id;count = p.local_elements_per_die
        if name.endswith('.ffn'):
            result[name, 'gate_up'] = count//3*2*4
            result[name, 'down'] = count//3*4
        elif name.endswith('.rmsnorm'):
            result[name, 'ffn_scale'] = count//2*4
            result[name, 'attn_scale'] = count//2*4
        elif name == 'token_embedding':
            result[name, 'merged' if case.model.tie_word_embeddings else 'embedding'] = count*4
        elif name == 'lm_head': result[name, 'head'] = count*4
        elif name == 'final_rmsnorm': result[name, 'scale'] = count*4
        elif name.endswith('.qkv_bias'): result[name, 'bias'] = count*4
        else: result[name, 'weight'] = count*4
    return result


def prepare_training_gradient_io(plan, case):
    """Opt-in, idempotent compiler pass shared by ALL aligned candidates.

    Apply to the ordinary plan before microbatch packing. Dense dweight
    outputs already have Matmul FP32 write traffic. Norm vector proxies need
    their additional parameter output; embedding scatter needs dense partial
    zero-init in addition to token updates. Bias output was already charged.
    Changed cost estimates are invalidated; the caller must subsequently run
    finalize_tensor_costs with the platform provider (as for any tensor plan).
    Legacy callers and fuse_training(...,1) remain exact identities.
    """
    if plan.contract.get('training_batching'):
        raise ValueError('prepare gradient output IO before training batching')
    outputs = {}
    for t in plan.tensors:
        if t['kind']=='parameter_gradient' and not t['persistent']:
            outputs.setdefault(t['created_node'],{}).setdefault(t['rank'],0)
            outputs[t['created_node']][t['rank']]+=t['nbytes']
    result=[]
    for op in plan.operations:
        sizes=outputs.get(op.node_id)
        if not sizes or (op.kernel_spec and any(p.direction=='dweight' for p in op.kernel_spec.matmuls)):
            result.append(op);continue
        if len(set(sizes.values()))!=1 or set(sizes)!=set(op.participant_ranks):
            raise ValueError('gradient output IO rank/size coverage differs')
        size=next(iter(sizes.values()));suffix=op.name.rsplit('.',1)[-1]
        _,component=_component(op,case)
        extra=0 if component=='bias' else size
        spec=op.kernel_spec or KernelSpec(suffix,vector_ops=op.vector_ops,
                                          read_bytes=op.tensor_bytes*2,write_bytes=op.tensor_bytes)
        existing=op.metadata.get('gradient_output_io')
        if existing:
            if (existing['schema_version']!=_GRADIENT_IO_VERSION or existing['parameter_partial_bytes_per_rank']!=size or
                existing['additional_write_bytes_per_rank']!=extra or
                spec.write_bytes!=existing['base_kernel_write_bytes']+extra):
                raise ValueError('gradient output IO marker/spec was changed')
            result.append(op);continue
        marker=dict(schema_version=_GRADIENT_IO_VERSION,parameter_partial_bytes_per_rank=size,
            additional_write_bytes_per_rank=extra,base_kernel_write_bytes=spec.write_bytes,
            semantic='dense_embedding_zero_init_then_scatter' if suffix=='embedding_backward' else
                     'bias_output_already_in_vector_stream' if component=='bias' else 'activation_and_parameter_outputs')
        meta={k:v for k,v in op.metadata.items() if k not in ('cost_estimate','cost_profile_sha256')}
        meta['gradient_output_io']=marker
        result.append(replace(op,kernel_spec=replace(spec,write_bytes=spec.write_bytes+extra),metadata=meta))
    plan.operations=result
    plan.contract['training_gradient_io']=dict(schema_version=_GRADIENT_IO_VERSION,
        policy='dense partials explicitly initialized/written; platform costs finalized by caller')
    return plan


def validate_training_gradient_io(plan, case):
    """Verify explicit vector-gradient writes and finalized platform costs."""
    errors=[];outputs={}
    for t in plan.tensors:
        if t['kind'] in ('parameter_gradient','parameter_gradient_partial') and not t['persistent']:
            outputs.setdefault(t['created_node'],{}).setdefault(t['rank'],0)
            outputs[t['created_node']][t['rank']]+=t['nbytes']
    ops={o.node_id:o for o in plan.operations}
    for node,sizes in outputs.items():
        op=ops[node];spec=op.kernel_spec
        if spec and any(p.direction=='dweight' for p in spec.matmuls):continue
        marker=op.metadata.get('gradient_output_io')
        if not marker or not spec:
            errors.append([node,'missing explicit vector-gradient output IO']);continue
        required=next(iter(sizes.values()))
        extra=0 if op.name.rsplit('.',1)[-1]=='qkv_bias_backward' else required
        if (len(set(sizes.values()))!=1 or marker.get('schema_version')!=_GRADIENT_IO_VERSION or
            marker.get('parameter_partial_bytes_per_rank')!=required or
            marker.get('additional_write_bytes_per_rank')!=extra or
            spec.write_bytes!=marker['base_kernel_write_bytes']+marker['additional_write_bytes_per_rank']):
            errors.append([node,'gradient output bytes/IO spec mismatch'])
        if any(spec.write_bytes<sum(plan.contract['ranks'][str(rank)]['nodes'][str(node)]['allocations'].values())
               for rank in sizes):
            errors.append([node,'kernel write IO cannot cover allocated gradient/activation outputs'])
        cost=op.metadata.get('cost_estimate')
        if not cost or cost.get('source','').startswith('temporary_'):
            errors.append([node,'gradient output IO has not received platform costs'])
        elif cost.get('dram_write_bytes',-1)<spec.write_bytes:
            errors.append([node,'gradient output IO missing from cost write traffic'])
    return [CheckResult('training_gradient_io.explicit_partial_writes',not errors,
        'all vector parameter partials have declared full writes/initialization and platform costs',errors,
        'gradient tensor producer bytes against kernel IO and cost provenance','shape_contract')]


def fuse_training(plan, case, microbatch_trajectories, *, cost_provider=None):
    """Return mb1 unchanged; otherwise construct, validate and cost a new plan.

    Call after tensor-plan construction and before any global rank remapping.
    The caller must pass its GPU provider for GPU costs. Communication padding
    can subsequently run normally using logical_original_op_id/parent_op.
    """
    eligibility = training_batching_eligibility(plan, case, microbatch_trajectories)
    if not eligibility['runtime_ready']:
        raise TrainingBatchingUnsupported(eligibility)
    if microbatch_trajectories == 1:
        return plan
    # A batching transform must not hide an invalid source allocation/free or
    # READY edge by coalescing duplicate destinations into an aggregate.
    preflight_contract(plan.contract)
    original = apply_mapped_costs(case, plan.operations, provider=_SpecOnly())
    old_ops = {o.node_id: o for o in original}
    source_tensors = {t['storage_id']: t for t in plan.tensors}
    tensors = {s: dict(t) for s, t in source_tensors.items()}
    phase_tid = {'train_'+t.trajectory_id: t.trajectory_id for t in case.workload.trajectories}
    owners = {s: phase_tid.get(old_ops[t['created_node']].phase)
              for s, t in source_tensors.items() if not t['persistent'] and t['created_node'] in old_ops}
    contract = {k: copy.deepcopy(v) for k, v in plan.contract.items() if k not in ('ranks', 'training_batching')}
    contract['ranks'] = {r: {**{k: copy.deepcopy(v) for k, v in c.items() if k != 'nodes'}, 'nodes': {}}
                         for r, c in plan.contract['ranks'].items()}
    operations = [];last = {};aliases = {};groups = [];removed = 0;removed_examples = [];removed_hash = hashlib.sha256()
    live = {int(r): set(c['initial_objects']) for r, c in contract['ranks'].items()}
    aggregates = {};new_allocations = set();cleanup_filtered = 0

    def resolve(s):
        seen = set()
        while s in aliases:
            if s in seen: raise ValueError('gradient storage alias cycle')
            seen.add(s);s = aliases[s]
        return s

    def blank(rank):
        c = contract['ranks'][str(rank)]
        n = dict(reads=[], allocations={}, ready_outputs=[], frees=[], workspace_bytes=0,
                 sram_capacity=case.hardware.sram_bytes_per_core, local_memory=True, reserve_before=0, reserve_after=0)
        if c.get('cores_per_die', 1) > 1: n['core_workspace_bytes'] = [0]*c['cores_per_die']
        return n

    def append(op, nodes):
        nonlocal cleanup_filtered
        node = len(operations)+1;deps = {r: (last[r],) if r in last else () for r in op.participant_ranks}
        metadata = {k: v for k, v in op.metadata.items() if k not in ('cost_estimate', 'cost_profile_sha256')}
        op = replace(op, node_id=node, dependencies_by_rank=deps,
                     dependencies=tuple(sorted({d for ds in deps.values() for d in ds})), metadata=metadata)
        for r in op.participant_ranks:
            n = nodes[r]
            if op.name == 'release_round_temporaries':
                before = len(n['frees']);n['frees'] = [s for s in n['frees'] if s in live[r]]
                cleanup_filtered += before-len(n['frees'])
            if any(s in live[r] for s in n['allocations']): raise ValueError('packed allocation aliases live storage')
            live[r].update(n['allocations']);new_allocations.update(n['allocations'])
            for s, size in n['allocations'].items():
                t = tensors[s]
                if t['nbytes'] != size: raise ValueError('packed tensor byte mismatch')
                t['created_node'] = node;t['ready_node'] = None;t['released_node'] = None
            for s in n['ready_outputs']:
                if s in tensors and not tensors[s]['persistent']: tensors[s]['ready_node'] = node
            for s in n['frees']:
                if s not in live[r]: raise ValueError('packed storage released twice: '+s)
                live[r].remove(s)
                if s in tensors and not tensors[s]['persistent']: tensors[s]['released_node'] = node
            contract['ranks'][str(r)]['nodes'][str(node)] = n;last[r] = node
        operations.append(op)
        return op

    def mapped_node(old, rank, tid=None):
        nonlocal removed
        source = plan.contract['ranks'][str(rank)]['nodes'][str(old.node_id)]
        n = {**source, 'allocations': dict(source['allocations'])}
        reads = []
        for s in source['reads']:
            # Only a training trajectory's unrelated temporary is removable.
            # Persistent states and non-training phase outputs are retained.
            if tid is not None and owners.get(s) is not None and owners[s] != tid:
                known_request = (old.metadata.get('forward_prefix') == 'train_'+tid+'_forward' or
                    old.metadata.get('backward_prefix') == 'train_'+tid+'_backward' or
                    old.name in ('train_'+tid+'_loss', 'train_'+tid+'_loss_backward') or
                    old.name.endswith('.gradient_accumulate_rmw'))
                if not known_request:
                    raise ValueError('unproven cross-trajectory temporary dependency: '+old.name)
                if len(removed_examples) < 8: removed_examples.append(dict(operation=old.op_id, rank=rank, storage_id=s, owner=owners[s]))
                removed += 1;removed_hash.update((old.op_id+'\0'+str(rank)+'\0'+s+'\n').encode())
                continue
            reads.append(resolve(s))
        n['reads'] = sorted(set(reads))
        for field in ('ready_outputs', 'frees'): n[field] = sorted({resolve(s) for s in source[field]})
        return n

    def aggregate(rid, batch, rank, parameter, component, size, ids, *, shape=None):
        key = (rid, batch, rank, parameter, component)
        if key in aggregates: raise ValueError('duplicate microbatch parameter component')
        sid = f'trainmb.r{rid}.b{batch}.d{rank}.{parameter}.{component}'
        kind = ('parameter_gradient_component' if parameter == 'token_embedding' and case.model.tie_word_embeddings and component != 'merged'
                else 'parameter_gradient_aggregate')
        tensors[sid] = dict(storage_id=sid, tensor_id=sid, shape=list(shape or (size//4,)), dtype_bytes=4,
            nbytes=size, rank=rank, kind=kind, round_id=rid, created_node=None, ready_node=None,
            released_node=None, persistent=False, parameter_id=parameter, parameter_component=component,
            contributor_ids=list(ids), microbatch_id=batch)
        aggregates[key] = sid
        return sid

    def merged_operation(members, rid, batch, ids, spec, *, suffix=None, metadata=None):
        first = members[0]
        meta = {**first.metadata, 'logical_original_op_id': first.op_id, 'training_microbatch_id': batch,
                'training_microbatch_contributors': list(ids), 'training_batch_request_count': len(ids),
                'training_batch_source_operations': [o.op_id for o in members],
                'independent_request_attention': True, **(metadata or {})}
        meta['training_batch_terms'] = [dict(trajectory_id=tid, **{k:o.metadata[k] for k in
            ('advantage','tokens','context_tokens','target_positions','global_denominator','global_loss_denominator',
             'local_loss_numerator_scale','pre_RS_scale_denominator')
            if k in o.metadata}) for tid,o in zip(ids,members)]
        markers=[o.metadata.get('gradient_output_io') for o in members]
        if metadata and metadata.get('training_batch_partial_reduce'):
            meta.pop('gradient_output_io',None)
        elif any(markers):
            if not all(markers):raise ValueError('mixed prepared/unprepared gradient output IO')
            meta['gradient_output_io']={**markers[0],**{key:sum(m[key] for m in markers) for key in
                ('parameter_partial_bytes_per_rank','additional_write_bytes_per_rank','base_kernel_write_bytes')}}
        if first.name in (first.phase+'_loss',first.phase+'_loss_backward'):
            meta['training_batch_loss_kind'] = 'loss_backward' if first.name.endswith('_loss_backward') else 'loss'
        # An advantage or causal mask belongs to one request. Never advertise
        # the first request's scalar as a value applied to the packed batch.
        for key in ('advantage','trajectory_id','forward_prefix','backward_prefix'):
            meta.pop(key,None)
        return replace(first, op_id=f'trainmb.r{rid}.b{batch}.n{len(operations)+1}.{first.op_id}',
            name=suffix or first.name, kernel_spec=spec, duration_ns=1,
            flops=sum(p.flops for p in spec.matmuls) if spec else sum(o.flops for o in members),
            vector_ops=spec.vector_ops if spec else sum(o.vector_ops for o in members),
            tensor_bytes=sum(o.tensor_bytes for o in members), payload_bytes=sum(o.payload_bytes for o in members), metadata=meta)

    i = 0
    while i < len(original):
        first = original[i]
        if first.phase not in phase_tid or first.name == 'zero_gradient_buffer':
            append(first, {r: mapped_node(first, r) for r in first.participant_ranks});i += 1;continue
        rid = first.round_id;blocks = OrderedDict()
        while i < len(original) and original[i].round_id == rid and original[i].phase in phase_tid and original[i].name != 'zero_gradient_buffer':
            op = original[i];blocks.setdefault(phase_tid[op.phase], []).append(op);i += 1
        ids_all = list(blocks)
        expected_ids = [t.trajectory_id for t in case.workload.trajectories]
        if ids_all != expected_ids: raise ValueError('packed training requires one ordered visit of every local trajectory')
        for batch, start in enumerate(range(0, len(ids_all), microbatch_trajectories)):
            ids = ids_all[start:start+microbatch_trajectories];templates = [blocks[t] for t in ids]
            if len({len(v) for v in templates}) != 1: raise ValueError('training operator templates differ')
            groups.append(dict(round_id=rid, microbatch_id=batch, contributor_ids=ids))
            for members in zip(*templates):
                first = members[0];participants = first.participant_ranks
                def canonical(op): return op.name.replace('train_'+phase_tid[op.phase], 'train_REQUEST', 1)
                if any(o.kind != first.kind or o.participant_ranks != participants or canonical(o) != canonical(first) for o in members):
                    raise ValueError('training microbatch operator templates differ')
                for key in ('actor_version','behavior_version','reference_version','global_denominator','global_loss_denominator',
                            'local_loss_numerator_scale','pre_RS_scale_denominator'):
                    if len({o.metadata.get(key) for o in members})!=1:
                        raise ValueError('training microbatch version/normalization differs: '+key)
                specs = [o.kernel_spec for o in members]
                suffix = first.name.rsplit('.', 1)[-1]
                if suffix == 'gradient_accumulate_rmw':
                    params = parameter_totals(case)['local_parameter_elements_per_die'];nodes = {}
                    for r in participants:
                        pieces = [mapped_node(o, r, tid) for o, tid in zip(members, ids)]
                        own = [s for (rr, bb, rank, _, _), s in aggregates.items()
                               if rr == rid and bb == batch and rank == r and tensors[s]['kind'] == 'parameter_gradient_aggregate']
                        if sum(tensors[s]['nbytes'] for s in own) != params*4: raise ValueError('incomplete aggregate before gradient RMW')
                        n = blank(r);n['reads'] = sorted({s for p in pieces for s in p['reads']} | set(own))
                        n['ready_outputs'] = sorted({s for p in pieces for s in p['ready_outputs']})
                        n['frees'] = sorted(({s for p in pieces for s in p['frees']} | set(own)) & live[r])
                        nodes[r] = n
                    denominators = {o.metadata.get('global_loss_denominator') for o in members}
                    if len(denominators) != 1 or None in denominators: raise ValueError('training loss denominators differ or are absent')
                    spec = KernelSpec('microbatch_gradient_accumulate_rmw', vector_ops=params*2, read_bytes=params*8, write_bytes=params*4)
                    op = merged_operation(members, rid, batch, ids, spec, metadata={'training_batch_rmw': True, 'sum_not_mean': True})
                    append(replace(op, tensor_bytes=params*4), nodes);continue
                if suffix == 'tied_embedding_gradient_merge':
                    if not case.model.tie_word_embeddings: raise ValueError('unexpected tied-gradient merge')
                    elements = case.model.vocab_size*case.model.hidden_size//case.strategy.tensor_parallel;nodes = {}
                    for r in participants:
                        head = aggregates[rid, batch, r, 'token_embedding', 'head'];embedding = aggregates[rid, batch, r, 'token_embedding', 'embedding']
                        sid = aggregate(rid, batch, r, 'token_embedding', 'merged', elements*4, ids)
                        n = blank(r);n.update(reads=[head, embedding], allocations={sid: elements*4}, ready_outputs=[sid], frees=[head, embedding]);nodes[r] = n
                        aliases[head] = sid;aliases[embedding] = sid
                        for old in members:
                            source = plan.contract['ranks'][str(r)]['nodes'][str(old.node_id)]
                            for storage in source['allocations']: aliases[storage] = sid
                    spec = KernelSpec('tied_embedding_gradient_transpose_sum', vector_ops=elements, read_bytes=elements*8, write_bytes=elements*4)
                    append(merged_operation(members, rid, batch, ids, spec,
                        metadata={'transpose_head_before_embedding_sum': True}), nodes);continue
                spec = None
                if first.kind == 'compute' and any(s is not None for s in specs):
                    if (any(s is None for s in specs) or len({len(s.matmuls) for s in specs}) != 1 or
                        len({(s.kernel_variant.replace('train_'+phase_tid[o.phase],'train_REQUEST',1),s.execution_domain)
                             for s,o in zip(specs,members)})!=1):
                        raise ValueError('incompatible packed kernel specs')
                    matrices = tuple(p for index in range(len(specs[0].matmuls))
                        for p in pack_training_matmuls([s.matmuls[index] for s in specs], linear=suffix in _LINEAR_FORWARD))
                    spec = KernelSpec('training_batch_'+specs[0].kernel_variant, matmuls=matrices,
                        vector_ops=sum(s.vector_ops for s in specs), read_bytes=sum(s.read_bytes for s in specs), write_bytes=sum(s.write_bytes for s in specs))
                nodes = {};reductions = []
                for r in participants:
                    sources = [plan.contract['ranks'][str(r)]['nodes'][str(o.node_id)] for o in members]
                    dense = []
                    for o, source in zip(members, sources):
                        dense.append([s for s in source['allocations'] if source_tensors[s]['kind'] == 'parameter_gradient'])
                    if any(dense):
                        if any(len(v) != 1 for v in dense): raise ValueError('unknown multi-gradient producer')
                        parameter, component = _component(first, case);size = source_tensors[dense[0][0]]['nbytes']
                        if any(source_tensors[v[0]]['nbytes'] != size for v in dense): raise ValueError('parameter partial sizes differ')
                        is_dense = bool(specs[0] and any(p.direction == 'dweight' for p in specs[0].matmuls))
                        sid = aggregate(rid, batch, r, parameter, component, size, ids)
                        if is_dense:
                            for values in dense: aliases[values[0]] = sid
                        else:
                            for values,tid in zip(dense,ids):
                                tensors[values[0]]['kind'] = 'parameter_gradient_partial'
                                tensors[values[0]].update(parameter_id=parameter, parameter_component=component,
                                    contributor_ids=[tid], microbatch_id=batch)
                            reductions.append((r, [v[0] for v in dense], sid, size))
                    pieces = [mapped_node(o, r, tid) for o, tid in zip(members, ids)]
                    view_groups = OrderedDict()
                    for tid,source in zip(ids,sources):
                        for storage,size in source['allocations'].items():
                            if resolve(storage)!=storage:
                                continue # one real dweight aggregate, not N output views
                            label=storage.rsplit('.',1)[-1]
                            view_groups.setdefault(label,[]).append((storage,tid,size))
                    for label,views in view_groups.items():
                        offset=0;total=sum(size for _,_,size in views)
                        buffer=f'trainmb.r{rid}.b{batch}.n{len(operations)+1}.d{r}.views.{label}'
                        for storage,tid,size in views:
                            # A logical contiguous allocation split into
                            # separately tracked views, never an additional
                            # parent allocation charged a second time.
                            tensors[storage]['packed_output_view']=dict(logical_buffer_id=buffer,
                                byte_offset=offset,byte_count=size,total_buffer_bytes=total,
                                contributor_id=tid,independent_attention_block=True,scope='local_tensor_plan')
                            offset+=size
                    n = blank(r)
                    for field in ('reads', 'ready_outputs', 'frees'): n[field] = sorted({s for p in pieces for s in p[field]})
                    allocations = {}
                    for p in pieces:
                        for s, size in p['allocations'].items():
                            target = resolve(s)
                            if target in allocations and target == s: raise ValueError('packed output storage collision')
                            allocations[target] = size
                    n['allocations'] = allocations
                    n['reserve_before'] = sum(p['reserve_before'] for p in pieces);n['reserve_after'] = sum(p['reserve_after'] for p in pieces)
                    nodes[r] = n
                op = merged_operation(members, rid, batch, ids, spec)
                if reductions:
                    # The old vector proxies mainly described activation
                    # outputs. Charge every full FP32 partial-gradient write
                    # as well; those are real allocations, including dense
                    # embedding scatter buffers rather than fictitious views.
                    # Bias backward's original output already IS its full
                    # parameter partial. Norms also emit activation gradients;
                    # dense embedding scatter additionally needs zero-init of
                    # every untouched parameter entry before accumulation.
                    prepared=op.metadata.get('gradient_output_io')
                    partial_bytes = 0 if suffix=='qkv_bias_backward' else reductions[0][3]*len(ids)
                    marker=prepared or dict(schema_version=_GRADIENT_IO_VERSION,
                        parameter_partial_bytes_per_rank=reductions[0][3]*len(ids),
                        additional_write_bytes_per_rank=partial_bytes,base_kernel_write_bytes=spec.write_bytes,
                        semantic='dense_embedding_zero_init_then_scatter' if suffix=='embedding_backward' else
                                 'bias_output_already_in_vector_stream' if suffix=='qkv_bias_backward' else 'activation_and_parameter_outputs')
                    op = replace(op, kernel_spec=replace(spec, write_bytes=spec.write_bytes+(0 if prepared else partial_bytes)),
                        metadata={**op.metadata,'explicit_partial_gradient_write_bytes':partial_bytes,
                                  'gradient_output_io':marker,'gradient_output_io_source_prepared':bool(prepared)})
                append(op, nodes)
                if reductions:
                    by_rank = {r: (partials, sid, size) for r, partials, sid, size in reductions}
                    if set(by_rank) != set(participants): raise ValueError('partial-gradient reduction rank coverage differs')
                    nodes = {}
                    for r in participants:
                        partials, sid, size = by_rank[r];n = blank(r)
                        n.update(reads=partials, allocations={sid: size}, ready_outputs=[sid], frees=partials);nodes[r] = n
                    size = reductions[0][3]
                    spec = KernelSpec('microbatch_parameter_gradient_sum', vector_ops=(len(ids)-1)*(size//4), read_bytes=len(ids)*size, write_bytes=size)
                    reduction = merged_operation(members, rid, batch, ids, spec,
                        suffix=first.name+'.microbatch_parameter_sum', metadata={'training_batch_partial_reduce': True,
                            'sum_not_mean':True,'parameter_id':parameter,'parameter_component':component})
                    append(reduction, nodes)
                    for _, partials, sid, _ in reductions:
                        for storage in partials: aliases[storage] = sid
    # Every remaining temporary must actually be backed by a submitted
    # allocation; old per-trajectory dweights/merge outputs are not phantom views.
    tensors = {s: t for s, t in tensors.items() if t['persistent'] or s in new_allocations}
    audit = {**eligibility, 'groups': groups, 'removed_conservative_read_count': removed,
        'removed_conservative_read_examples': removed_examples, 'removed_conservative_reads_sha256': removed_hash.hexdigest(),
        'filtered_already_released_cleanup_count': cleanup_filtered,
        'original_gemm_flops': sum(p.flops for o in original if o.kernel_spec for p in o.kernel_spec.matmuls),
        'packed_gemm_flops': sum(p.flops for o in operations if o.kernel_spec for p in o.kernel_spec.matmuls),
        'attention_contract': 'independent ragged request blocks; only equal shapes batch together',
        'gradient_contract': 'one FP32 summed component per parameter/microbatch; one persistent RMW'}
    audit['source_loss_denominators'] = sorted({o.metadata[k] for o in original for k in
        ('global_denominator','global_loss_denominator') if k in o.metadata})
    audit['source_loss_terms'] = [dict(round_id=o.round_id,trajectory_id=phase_tid[o.phase],
        kind='loss_backward' if o.name.endswith('_loss_backward') else 'loss',
        terms=dict(trajectory_id=phase_tid[o.phase],**{k:o.metadata[k] for k in
            ('advantage','tokens','context_tokens','target_positions','global_denominator','global_loss_denominator',
             'local_loss_numerator_scale','pre_RS_scale_denominator') if k in o.metadata}))
        for o in original if o.phase in phase_tid and o.name in (o.phase+'_loss',o.phase+'_loss_backward')]
    audit['boundary_operations'] = [[o.round_id,o.name] for o in original if o.name in
        ('zero_gradient_buffer','local_gradient_square_sum','apply_gradient_clip','adamw_read_modify_write','release_round_temporaries')]
    if audit['original_gemm_flops'] != audit['packed_gemm_flops']: raise ValueError('training GEMM coverage changed')
    contract['training_batching'] = audit
    result = TensorPlan(contract, list(tensors.values()), operations, {})
    result.preflight_peaks = preflight_contract(contract)
    finalize_tensor_costs(result, case, provider=cost_provider)
    checks = validate_training_batching(result, case)+validate_training_gradient_io(result,case)
    if not all(c.passed for c in checks):
        raise TrainingBatchingUnsupported({**eligibility,'eligible':False,'runtime_ready':False,
            'blockers':[c.actual for c in checks if not c.passed]})
    return result


def validate_training_batching(plan, case):
    """Independent parameter/component, contributor, byte and RMW coverage."""
    audit = plan.contract.get('training_batching')
    if not audit: return []
    expected = _expected_components(case);errors = [];params = parameter_totals(case)['local_parameter_elements_per_die']
    aggregates = [t for t in plan.tensors if t['kind'] == 'parameter_gradient_aggregate']
    view_groups = {}
    for t in plan.tensors:
        if 'packed_output_view' in t:
            view=t['packed_output_view'];view_groups.setdefault(view['logical_buffer_id'],[]).append((t,view))
    allocated_ids={t['storage_id'] for t in plan.tensors}
    for buffer,views in view_groups.items():
        offset=0;totals=set()
        for t,view in sorted(views,key=lambda row:row[1]['byte_offset']):
            if view['byte_offset']!=offset or view['byte_count']!=t['nbytes']:
                errors.append(['overlapping/missing packed output view',buffer])
            offset+=t['nbytes'];totals.add(view['total_buffer_bytes'])
        if totals!={offset} or buffer in allocated_ids:
            errors.append(['packed view byte accounting/parent double allocation',buffer])
    expected_ids = [t.trajectory_id for t in case.workload.trajectories]
    round_groups = OrderedDict();keys = set()
    for group in audit['groups']:
        key=(group['round_id'],group['microbatch_id'])
        if key in keys: errors.append(['duplicate microbatch identity',key])
        keys.add(key);round_groups.setdefault(group['round_id'],[]).extend(group['contributor_ids'])
        if not group['contributor_ids'] or len(group['contributor_ids'])>audit['requested_microbatch_trajectories']:
            errors.append(['invalid microbatch contributor count',key])
    if not round_groups or any(ids!=expected_ids for ids in round_groups.values()):
        errors.append(['local trajectory contributor coverage differs'])
    for t in aggregates:
        if (t['round_id'],t['microbatch_id']) not in keys or t['rank'] not in range(case.hardware.dies):
            errors.append(['aggregate outside declared microbatch/ranks',t['storage_id']])
    if sum(p.flops for o in plan.operations if o.kernel_spec for p in o.kernel_spec.matmuls)!=audit['original_gemm_flops']:
        errors.append(['GEMM FLOP coverage changed'])
    boundaries = [[o.round_id,o.name] for o in plan.operations if o.name in
        ('zero_gradient_buffer','local_gradient_square_sum','apply_gradient_clip','adamw_read_modify_write','release_round_temporaries')]
    if boundaries!=audit['boundary_operations']: errors.append(['training update boundary changed'])
    if len(audit['source_loss_denominators'])!=1 or type(audit['source_loss_denominators'][0]) is not int or audit['source_loss_denominators'][0]<=0:
        errors.append(['global loss denominator is not one explicit positive value'])
    for group in audit['groups']:
        rid = group['round_id'];batch = group['microbatch_id'];ids = group['contributor_ids']
        for kind in ('loss','loss_backward'):
            losses = [o for o in plan.operations if o.round_id==rid and
                o.metadata.get('training_microbatch_id')==batch and o.metadata.get('training_batch_loss_kind')==kind]
            terms = {t['trajectory_id']:t['terms'] for t in audit['source_loss_terms'] if t['round_id']==rid and t['kind']==kind}
            if len(losses)!=1 or set(ids)-set(terms) or losses[0].metadata['training_batch_terms']!=[terms[t] for t in ids]:
                errors.append([rid,batch,'loss terms/masks/normalization changed',kind])
        for rank in range(case.hardware.dies):
            selected = [t for t in aggregates if t['rank'] == rank and t['round_id'] == rid and t['microbatch_id'] == batch]
            seen = {}
            for t in selected:
                key = (t['parameter_id'], t['parameter_component'])
                if key in seen or key not in expected or t['nbytes'] != expected.get(key) or t['dtype_bytes'] != 4 or t['contributor_ids'] != ids:
                    errors.append([rid, batch, rank, 'component/contributors/bytes', key])
                seen[key] = t
                elements=1
                for dim in t['shape']:
                    if type(dim) is not int or dim<=0:errors.append([rid,batch,rank,'invalid aggregate dimension',key])
                    elements*=dim
                if elements*t['dtype_bytes']!=t['nbytes']:errors.append([rid,batch,rank,'aggregate shape/bytes',key])
                rank_nodes = plan.contract['ranks'][str(rank)]['nodes'];created = rank_nodes[str(t['created_node'])]
                if created['allocations'].get(t['storage_id']) != t['nbytes'] or t['storage_id'] not in created['ready_outputs'] or t['released_node'] is None:
                    errors.append([rid, batch, rank, 'missing aggregate producer/READY/free', key])
            if set(seen) != set(expected) or sum(t['nbytes'] for t in selected) != params*4:
                errors.append([rid, batch, rank, 'incomplete unique parameter coverage'])
            updates = [o for o in plan.operations if o.round_id == rid and o.metadata.get('training_microbatch_id') == batch and
                       o.metadata.get('training_batch_rmw') and rank in o.participant_ranks]
            if len(updates) != 1: errors.append([rid, batch, rank, 'one RMW required'])
            else:
                op = updates[0];node = plan.contract['ranks'][str(rank)]['nodes'][str(op.node_id)]
                if op.metadata.get('training_microbatch_contributors')!=ids or [op.metadata.get('global_loss_denominator')]!=audit['source_loss_denominators']:
                    errors.append([rid,batch,rank,'RMW contributors/global denominator'])
                if not {t['storage_id'] for t in selected} <= set(node['reads']) or not {t['storage_id'] for t in selected} <= set(node['frees']):
                    errors.append([rid, batch, rank, 'RMW missing aggregate read/free'])
                spec = op.kernel_spec
                if (spec.vector_ops, spec.read_bytes, spec.write_bytes) != (params*2, params*8, params*4):
                    errors.append([rid, batch, rank, 'RMW cost contract'])
    return [CheckResult('training_batching.aggregate_parameter_coverage', not errors,
        'each parameter component once per microbatch, exact contributors/FP32 bytes, one RMW', errors,
        'independent parameter row enumeration plus submitted aggregate tensor allocation/READY/free edges', 'shape_contract')]
