"""Sparse contributor RS, with unchanged canonical optimizer owners.

After regional TPp->TP2 restoration, only each region root has a nonzero
canonical gradient. RS these roots, then copy exact range intersections into
the original DP27 owner views. This is an optional execution graph; it is not
an alteration of the sample denominator, Adam ownership or actor layout.
"""
from math import lcm
from dataclasses import replace
from .workload import digest


def range_plan(elements, regions, owners=27):
    if type(elements) is not int or elements<1 or elements%owners:
        raise ValueError('canonical owner-aligned padded extent required')
    groups={int(k):tuple(v) for k,v in regions.items()}
    if (sorted(w for ws in groups.values() for w in ws)!=list(range(owners)) or
        any(not ws or k!=min(ws) for k,ws in groups.items())):
        raise ValueError('complete nonoverlapping regions with canonical roots required')
    roots=sorted(groups);n=len(roots);owner_extent=elements//owners
    if not 2<=n<owners:
        return dict(mode='original',reason='sparse RS requires 2..owners-1 contributor roots',roots=roots)
    padded=((elements+lcm(n,owners)-1)//lcm(n,owners))*lcm(n,owners)
    source_extent=padded//n;staging_start=elements-owner_extent
    if source_extent>staging_start:raise ValueError('RS output and staging alias')
    moves=[]
    for i,root in enumerate(roots):
        for owner in range(owners):
            lo=max(i*source_extent,owner*owner_extent)
            hi=min((i+1)*source_extent,(owner+1)*owner_extent,elements)
            if hi>lo:
                moves.append(dict(index=len(moves),source_root=root,owner=owner,
                    global_start=lo,global_end=hi,source_start=lo-i*source_extent,
                    destination_stage_start=staging_start+lo-owner*owner_extent,
                    owner_view_start=owner*owner_extent+lo-owner*owner_extent,elements=hi-lo))
    value=dict(schema='idea1.sparse-gradient-range/1',mode='sparse-root-rs',regions={str(k):list(ws) for k,ws in groups.items()},
        roots=roots,owners=owners,elements=elements,rs_padded_elements=padded,
        extra_allocation_bytes_per_contributor=4*(padded-elements),
        source_extent=source_extent,owner_extent=owner_extent,staging_start=staging_start,moves=moves,
        semantics='canonical TP2 lane: sum each nonzero regional contribution once; QK TP merge precedes RS; normalize/norm/clip/Adam unchanged',
        alias_contract='RS output compact prefix; receive into disjoint tail; all outgoing reads complete before final owner view copy; no second gradient shard allocation')
    value['digest']=digest(value)
    return value


def sparse_optimizer_plan(op,policy,regions,*,gradient_restore_peak_bytes,capacity_bytes,contributors_are_region_roots=False):
    """Compile the existing optimizer suffix, replacing only its RS phase.

    Uses the existing native collective implementation and endpoint cost
    contract. Every redistribution is an explicit native P2P plus read/write.
    """
    from rl_xy.qwen3_experiment import optimizer_plan
    from rl_xy.qwen3_native import Plan
    from rl_xy.types import Operation
    original=optimizer_plan(op,policy)
    # Training donations can accumulate NEW gradients on a non-root TP2 pair
    # after regional restoration. Static root membership is not a certificate
    # of final zero buffers. The production runtime must retain original RS
    # unless that contribution set is explicitly proven before compilation.
    if not contributors_are_region_roots:
        return original,dict(mode='original',reason='non-root gradients may be created by training helpers; no final contributor certificate')
    if op.active_dies!=54 or policy.train_tp!=2:
        return original,dict(mode='original',reason='unsupported canonical owner geometry')
    layout=op.state(2,policy.rollout_tp)['layout']
    proof=range_plan(layout.padded_elements,regions)
    if proof['mode']=='original':return original,proof
    peak=gradient_restore_peak_bytes+proof['extra_allocation_bytes_per_contributor']
    if peak>capacity_bytes:
        return original,dict(mode='original',reason='extra padding exceeds live arena capacity',required_bytes=peak,capacity_bytes=capacity_bytes)
    proof['capacity']=dict(required_bytes=peak,capacity_bytes=capacity_bytes,passed=True,
        scope='conservative existing gradient-restore phase bound plus extra RS zero padding; receive staging aliases retired full gradient arena')
    plan=Plan(op);inserted=False;owners=proof['owners'];length=proof['elements'];shard=proof['owner_extent']
    def register_append(operation,stats=None):
        if operation.kind=='collective':
            ranks=operation.participant_ranks
            if ranks not in plan.groupids:
                gid=str(len(plan.groups)+1);plan.groupids[ranks]=gid;plan.groups[gid]=list(ranks)
            operation=replace(operation,metadata={'comm_group':plan.groupids[ranks]})
        added=plan.append(operation)
        if operation.kind=='compute':plan.stats[added.node_id]=stats
    for operation in original.operations[1:]:
        if 'dp_gradient_RS' in operation.op_id:
            if inserted:continue
            inserted=True
            if proof['rs_padded_elements']>length:
                plan.local(tuple(2*r+h for r in proof['roots'] for h in (0,1)),
                    op.vec(0,0,proof['extra_allocation_bytes_per_contributor'],'sparse_rs_extra_zero'),'sparse_rs_extra_zero')
            for lane in (0,1):
                ranks=tuple(2*r+lane for r in proof['roots'])
                plan.comm(ranks,'REDUCE_SCATTER',proof['rs_padded_elements']*4,'dp_gradient_RS.roots')
                for move in proof['moves']:
                    src=2*move['source_root']+lane;dst=2*move['owner']+lane;size=move['elements']*4
                    name='sparse_owner_route.lane%d.move%d'%(lane,move['index'])
                    if src==dst:
                        plan.local((src,),op.vec(0,size,size,'sparse_local_stage'),name+'.local')
                    else:
                        plan.local((src,),op.vec(0,size,0,'sparse_send_read'),name+'.read')
                        plan.append(Operation(0,name,1,name,name,'p2p',payload_bytes=size,src_rank=src,dst_rank=dst,participant_ranks=(src,dst)))
                        plan.local((dst,),op.vec(0,0,size,'sparse_receive_write'),name+'.write')
                # Last-per-rank dependencies enforce all of that root's sends
                # and all owner receives before any final view can overwrite
                # its compact RS output. Tail staging stays disjoint until then.
                for owner in range(owners-1):
                    rank=2*owner+lane
                    plan.local((rank,),op.vec(0,shard*4,shard*4,'sparse_finalize_owner_view'),
                        'sparse_owner_view.lane%d.owner%d'%(lane,owner))
            continue
        register_append(operation,original.stats.get(operation.node_id))
    if not inserted:raise ValueError('original suffix has no RS block')
    proof['compiled_p2p_messages']=2*sum(m['source_root']!=m['owner'] for m in proof['moves'])
    proof['logical_payload_bytes']=2*sum(m['elements']*4 for m in proof['moves'] if m['source_root']!=m['owner'])
    proof['status']='COMPILED_REQUIRES_NATIVE_AND_INDEPENDENT_AUDIT'
    return plan,proof
