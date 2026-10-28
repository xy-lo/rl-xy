"""Exact homogeneous-request batching without first building N serial graphs.

The selected fixed-length workload has independent requests with identical
shapes and one behavior version. Linear GEMMs concatenate tokens, attention
batches independent head matrices, and all temporary storages remain disjoint.
No actual responses, rewards, or throughput measurements are manufactured.
"""
from dataclasses import replace
from .tensor_plan import TensorPlan
from .mapped_costs import apply_mapped_costs
from .batching import _SpecOnly
from .mapping import ceildiv


def replicate_uniform_generation(template, template_case, batch_case):
    n=len(batch_case.workload.trajectories)
    if len(template_case.workload.trajectories)!=1 or n<1:raise ValueError('one-request template and nonempty batch required')
    proto=template_case.workload.trajectories[0]
    def shape(c,t):return (len(t.prompt_token_ids or c.workload.prompt_token_ids),len(t.response_token_ids))
    if any(shape(batch_case,t)!=shape(template_case,proto) for t in batch_case.workload.trajectories):raise ValueError('heterogeneous lengths require ragged generation planner')
    if template_case.model!=batch_case.model or template_case.hardware!=batch_case.hardware:raise ValueError('template model/hardware mismatch')
    if template_case.workload.behavior_version!=batch_case.workload.behavior_version:raise ValueError('cannot mix behavior versions')
    ops=apply_mapped_costs(template_case,template.operations,provider=_SpecOnly())
    initial={s for r in template.contract['ranks'].values() for s in r['initial_objects']}
    # One contiguous buffer per prototype value, with a disjoint request
    # axis. This avoids N copies of all read/dependency lists while retaining
    # exactly N independent request storages and N times the live bytes.
    tensors=[]
    for t in template.tensors:
        if t['persistent']:tensors.append(dict(t));continue
        tensors.append({**t,'nbytes':t['nbytes']*n,'shape':[n]+list(t['shape']),
            'trajectory_id':None,'request_axis':0,'request_count':n,
            'request_stride_bytes':t['nbytes'],'homogeneous_template_storage':t['storage_id']})
    ranks={}
    for r,old in template.contract['ranks'].items():
        nodes={}
        for node,v in old['nodes'].items():
            z={**v,'allocations':{k:size*n for k,size in v['allocations'].items()},
               'reserve_before':v['reserve_before']*n,'reserve_after':v['reserve_after']*n}
            for field in ['reads','ready_outputs','frees']:z[field]=list(v[field])
            nodes[node]=z
        ranks[r]={**old,'initial_objects':dict(old['initial_objects']),'nodes':nodes}
    result=[];linear={'qkv','o_proj_partial','gate_up','down_partial','lm_head_local'}
    for op in ops:
        spec=op.kernel_spec
        if spec:
            mats=[]
            for m in spec.matmuls:
                if op.name.rsplit('.',1)[-1] in linear:
                    if m.batches!=1:raise ValueError('linear homogeneous template must have one matrix')
                    mats.append(replace(m,m=m.m*n))
                else:mats.append(replace(m,batches=m.batches*n))
            spec=replace(spec,matmuls=tuple(mats),vector_ops=spec.vector_ops*n,
                read_bytes=spec.read_bytes*n,write_bytes=spec.write_bytes*n)
            if sum(m.flops for m in mats)!=n*sum(m.flops for m in op.kernel_spec.matmuls):raise ValueError('uniform batch FLOP mismatch')
        duration=op.duration_ns
        if op.metadata.get('logical_executor')=='host_link' and op.name=='host_to_rank0_prompt':
            duration=batch_case.hardware.host_latency_ns+ceildiv(op.tensor_bytes*n,batch_case.hardware.host_bytes_per_ns)
        result.append(replace(op,duration_ns=duration,flops=op.flops*n,vector_ops=op.vector_ops*n,tensor_bytes=op.tensor_bytes*n,
            payload_bytes=op.payload_bytes*n,kernel_spec=spec,metadata={**op.metadata,
                'batch_request_count':n,'batch_trajectory_ids':[t.trajectory_id for t in batch_case.workload.trajectories],
                'homogeneous_generation_template':True,'prompt_transport_policy':'independent per-trajectory prompt bytes; no host input or prefix reuse','attention_keys_shared_between_requests':False}))
    return TensorPlan({**template.contract,'ranks':ranks,'generation_batching':{
        'schema_version':'rlxy.uniform_generation/1','requests':n,'prototype_trajectory':proto.trajectory_id,
        'FLOP_multiplier':n,'independent_attention_and_KV':True,'storage_layout':'disjoint_request_axis','prototype_object_count_unchanged':True}},tensors,result,{})
