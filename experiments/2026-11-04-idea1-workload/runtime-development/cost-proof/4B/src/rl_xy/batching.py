"""Packed generation GEMMs with ragged per-request attention and KV views.

Only independent requests at the SAME prefill/decode step are fused. Tensor
allocations remain disjoint views with their exact byte counts. Attention
matrices are never concatenated across requests as if they shared keys.
"""
from dataclasses import replace
from collections import OrderedDict
from .cost_model import CostEstimate
from .mapped_costs import apply_mapped_costs
from .operator_ir import KernelSpec


class _SpecOnly:
    def estimate(self,spec,hardware):return CostEstimate(1,0,1,0,0,0,(),source='temporary_spec_materialization_not_executed')


def fuse_generation(plan,case):
    original=apply_mapped_costs(case,plan.operations,provider=_SpecOnly())
    frames=OrderedDict();before=[];after=[]
    for op in original:
        prefix=op.metadata.get('forward_prefix')
        if prefix:
            frame=prefix.rsplit('_',1)[1]
            frames.setdefault(frame,OrderedDict()).setdefault(prefix,[]).append(op)
        elif op.name=='release_round_temporaries':after.append(op)
        else:before.append(op)
    units=[([o],None) for o in before]
    for frame,prefixes in frames.items():
        values=list(prefixes.values())
        if len({len(v) for v in values})!=1:raise ValueError('generation frame has unequal operator templates')
        units.extend((list(rows),frame) for rows in zip(*values))
    units.extend(([o],None) for o in after)
    nodes={str(r):{} for r in range(case.hardware.dies)};operations=[];last={}
    tensor_by_id={t['storage_id']:t for t in plan.tensors}
    linear={'qkv','o_proj_partial','gate_up','down_partial','lm_head_local'}
    total_before=sum(o.flops for o in original);samples_before=sum(o.name.endswith('.sample_on_rank0') for o in original)
    sample_count=0
    for members,frame in units:
        first=members[0];participants=first.participant_ranks
        suffix=first.name.split('.',1)[-1] if frame else first.name
        if any(o.kind!=first.kind or o.participant_ranks!=participants or
            (frame and o.name.split('.',1)[-1]!=suffix) for o in members):raise ValueError('incompatible generation batch operators')
        meta={**first.metadata,'batch_request_count':len(members),'batch_frame':frame,
            'logical_original_op_id':first.op_id,
            'batch_source_prefixes':[o.metadata.get('forward_prefix') for o in members]}
        spec=None
        if first.kind=='compute' and any(o.kernel_spec is not None for o in members):
            specs=[o.kernel_spec for o in members]
            if any(s is None for s in specs):raise ValueError('mixed explicit-control/kernel batch')
            mats=[p for s in specs for p in s.matmuls]
            combined=[]
            if mats and suffix.rsplit('.',1)[-1] in linear:
                if any((p.n,p.k,p.batches,p.a_bytes,p.b_bytes,p.output_bytes,p.accumulator_bytes,p.direction)!=
                    (mats[0].n,mats[0].k,1,mats[0].a_bytes,mats[0].b_bytes,mats[0].output_bytes,mats[0].accumulator_bytes,mats[0].direction) for p in mats):
                    raise ValueError('incompatible packed linear shapes')
                combined=[replace(mats[0],m=sum(p.m for p in mats))]
            else:
                grouped=OrderedDict()
                for p in mats:
                    key=(p.m,p.n,p.k,p.a_bytes,p.b_bytes,p.output_bytes,p.accumulator_bytes,p.direction)
                    grouped[key]=replace(p,batches=p.batches+(grouped[key].batches if key in grouped else 0))
                combined=list(grouped.values())
            spec=KernelSpec('batched_'+specs[0].kernel_variant,matmuls=tuple(combined),
                vector_ops=sum(s.vector_ops for s in specs),read_bytes=sum(s.read_bytes for s in specs),
                write_bytes=sum(s.write_bytes for s in specs))
            if sum(p.flops for p in mats)!=sum(p.flops for p in spec.matmuls):raise ValueError('batched FLOP conservation')
        node=len(operations)+1
        deps={r:(last[r],) if r in last else () for r in participants}
        op=replace(first,node_id=node,op_id=f'batched.n{node}.{first.op_id}',
            name=('batch_'+str(frame)+'.'+suffix if frame else first.name),
            flops=sum(o.flops for o in members),vector_ops=sum(o.vector_ops for o in members),
            tensor_bytes=sum(o.tensor_bytes for o in members),payload_bytes=sum(o.payload_bytes for o in members),
            kernel_spec=spec,metadata=meta,dependencies_by_rank=deps,
            dependencies=tuple(sorted({n for ds in deps.values() for n in ds})))
        for r in participants:
            pieces=[]
            for old in members:
                n=dict(plan.contract['ranks'][str(r)]['nodes'][str(old.node_id)])
                prefix=old.metadata.get('forward_prefix')
                if prefix:
                    trajectory=prefix.split('_')[1]
                    # The older serial builder conservatively read every live
                    # sampled token, including unrelated earlier requests.
                    # Packed batching retains only this request's token edges.
                    n['reads']=[s for s in n['reads'] if tensor_by_id[s]['kind']!='token' or
                        f'rollout_{trajectory}_' in tensor_by_id[s]['tensor_id']]
                pieces.append(n)
            allocations={k:v for p in pieces for k,v in p['allocations'].items()}
            if len(allocations)!=sum(len(p['allocations']) for p in pieces):raise ValueError('batch storage alias collision')
            n={**pieces[0],'allocations':allocations,
                'reserve_before':sum(p['reserve_before'] for p in pieces),'reserve_after':sum(p['reserve_after'] for p in pieces)}
            for field in ('reads','ready_outputs','frees'):n[field]=sorted({s for p in pieces for s in p[field]})
            nodes[str(r)][str(node)]=n;last[r]=node
        operations.append(op)
        if suffix.endswith('sample_on_rank0'):sample_count+=len(members)
    if total_before!=sum(o.flops for o in operations) or samples_before!=sample_count:raise ValueError('generation coverage/FLOPs changed')
    for r in nodes:plan.contract['ranks'][r]['nodes']=nodes[r]
    plan.operations=operations
    return plan
