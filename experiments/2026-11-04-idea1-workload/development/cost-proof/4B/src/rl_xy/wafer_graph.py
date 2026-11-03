"""Synchronous DP replicas of a TP2 graph, native communicator groups.

Complete prompt groups are assigned round-robin. Local losses are normalized
by the global token count BEFORE native gradient SUM, never by local token
counts followed by an unweighted average. No independent-run max-time shortcut.
"""
import copy
from dataclasses import dataclass, replace

from .communication_lowering import lower_communication_endpoints
from .cost_model import AnalyticalCostProvider
from .execution_contract import group_for
from .mapping import ceildiv
from .model import parameter_totals
from .operator_ir import KernelSpec
from .tensor_plan import TensorPlan, build_tensor_plan, finalize_tensor_costs, preflight_contract
from .types import Operation
from .workflow import build_workflow


@dataclass
class WaferGraph:
    plan: TensorPlan
    groups: dict
    cases: list
    local_operations: list
    mapping: dict
    placements: list


def build_wafer_graph(case, mesh, replicas, log_prefix, *, cost_provider=None):
    if len(mesh)!=2 or any(type(x) is not int or x<2 for x in mesh) or mesh[0]*mesh[1]!=2*replicas:
        raise ValueError('mesh must contain exactly DP*TP2 dies')
    if case.schema_version!='1.1' or case.strategy.optimizer_minibatch_trajectories!=0:
        raise ValueError('hierarchical baseline requires schema1.1 and full collected batch per update')
    gids=list(dict.fromkeys(group_for(case,t) for t in case.workload.trajectories))
    if len(gids)<replicas:raise ValueError('at least one complete prompt group per DP replica required')
    assignments=[gids[i::replicas] for i in range(replicas)]
    cases=[];plans=[];groups={};placements=[];local_ops=[]
    denom=sum(len(t.response_token_ids) for t in case.workload.trajectories)
    world=2*replicas
    for i,assigned in enumerate(assignments):
        trajectories=tuple(t for t in case.workload.trajectories if group_for(case,t) in assigned)
        local=replace(case,workload=replace(case.workload,trajectories=trajectories),
            strategy=replace(case.strategy,rollout_max_concurrency=len(trajectories)))
        cases.append(local)
        ops=[]
        for op in build_workflow(local):
            meta=dict(op.metadata)
            for key in ('global_denominator','global_loss_denominator','minibatch_denominator'):
                if key in meta:meta[key]=denom
            ops.append(replace(op,metadata=meta))
        p=build_tensor_plan(local,lower_communication_endpoints(local,ops),log_prefix)
        finalize_tensor_costs(p,local,provider=cost_provider);plans.append(p);local_ops.append(p.operations)
        groups[str(i+1)]=[2*i,2*i+1]
        placements.append({'replica':i,'tp_ranks':[2*i,2*i+1],'groups':assigned,
            'trajectories':[t.trajectory_id for t in trajectories],
            'local_response_tokens':sum(len(t.response_token_ids) for t in trajectories),
            'global_loss_denominator':denom})
    for shard in (0,1):groups[str(replicas+1+shard)]=list(range(shard,world,2))
    ranks={};operations=[];tensors=[];mapping={};last={}
    def sid(i,name):return f'replica{i}.{name}'
    for i,p in enumerate(plans):
        for rank,r in p.contract['ranks'].items():
            physical=2*i+int(rank)
            ranks[str(physical)]={**r,'nodes':{},'initial_objects':{sid(i,k):v for k,v in r['initial_objects'].items()}}
    def append(op,contracts):
        node=len(operations)+1
        deps={r:(last[r],) if r in last else () for r in op.participant_ranks}
        op=replace(op,node_id=node,dependencies_by_rank=deps,
            dependencies=tuple(sorted({n for ns in deps.values() for n in ns})))
        operations.append(op)
        for rank in op.participant_ranks:
            last[rank]=node;ranks[str(rank)]['nodes'][str(node)]=contracts[rank]
        return node
    def translate(i,op):
        contracts={}
        for rank in op.participant_ranks:
            c=copy.deepcopy(plans[i].contract['ranks'][str(rank)]['nodes'][str(op.node_id)])
            for field in ('reads','ready_outputs','frees'):c[field]=[sid(i,k) for k in c[field]]
            c['allocations']={sid(i,k):v for k,v in c['allocations'].items()}
            contracts[2*i+rank]=c
        meta={**op.metadata,'replica_id':i,'logical_node_id':op.node_id}
        if op.kind=='collective':
            meta['comm_group']=i+1
            # Native Sys pads small collectives to total_nodes; explicitly pad
            # to whole FP32 elements and group size rather than hide mutation.
            transport=ceildiv(max(op.payload_bytes,world),8)*8
            meta['transport_payload_bytes']=transport
            for rank,c in contracts.items():
                buffer=f'network_scratch.{i}.{op.node_id}.{rank}'
                c['allocations'][buffer]=transport
                c['frees'].append(buffer)
        mapped=replace(op,op_id=f'dp{i}.{op.op_id}',participant_ranks=tuple(2*i+r for r in op.participant_ranks),
            src_rank=None if op.src_rank is None else 2*i+op.src_rank,
            dst_rank=None if op.dst_rank is None else 2*i+op.dst_rank,metadata=meta)
        mapping[i,op.node_id]=append(mapped,contracts)
    def sync(rid,update):
        params=parameter_totals(case)['local_parameter_elements_per_die']
        size=ceildiv(max(params*4,world),replicas*4)*replicas*4
        for shard in (0,1):
            participants=tuple(range(shard,world,2));contracts={}
            for rank in participants:
                gradient=sorted(name for name in ranks[str(rank)]['initial_objects'] if '.gradient.' in name)
                temp=f'dp_network_buffer.r{rid}.u{update}.d{rank}'
                contracts[rank]={'reads':gradient,'allocations':{temp:size},'ready_outputs':gradient,
                    'frees':[temp],'workspace_bytes':0,'sram_capacity':case.hardware.sram_bytes_per_core,
                    'local_memory':False,'reserve_before':0,'reserve_after':0}
                if case.hardware.cores_per_die>1:contracts[rank]['core_workspace_bytes']=[0]*case.hardware.cores_per_die
            meta={'comm_group':replicas+1+shard,'dp_update':update,'sum_not_mean':True,
                  'global_loss_denominator':denom,'transport_payload_bytes':size}
            op=Operation(0,f'dp_gradient_sum.r{rid}.u{update}.s{shard}',rid,'dp_sync','dp_gradient_sum',
                'collective',payload_bytes=params*4,collective='ALL_REDUCE',participant_ranks=participants,metadata=meta)
            append(op,contracts)
            # Ring's reduction on padded equal chunks: DP-1 received chunks.
            # Endpoint traffic charged separately from native network transit.
            words=size//4//replicas*(replicas-1)
            spec=KernelSpec('dp_sum_local_service',vector_ops=words,
                read_bytes=words*8+size,write_bytes=words*4+size)
            estimate=(cost_provider or AnalyticalCostProvider()).estimate(spec,case.hardware)
            from dataclasses import asdict
            local_contracts={}
            for rank in participants:
                c=copy.deepcopy(contracts[rank]);c['allocations']={};c['frees']=[]
                c.update(workspace_bytes=estimate.sram_workspace_bytes,local_memory=True)
                if case.hardware.cores_per_die>1:c['core_workspace_bytes']=list(estimate.core_workspace_bytes)
                local_contracts[rank]=c
            append(Operation(0,f'dp_gradient_service.r{rid}.u{update}.s{shard}',rid,'dp_sync','dp_gradient_local_service',
                'compute',duration_ns=estimate.duration_ns,participant_ranks=participants,kernel_spec=spec,
                vector_ops=words,tensor_bytes=params*4,metadata={k:v for k,v in {**meta,'execution_domain':'device',
                    'cost_estimate':asdict(estimate)}.items() if k!='comm_group'}),local_contracts)
    for offset in range(case.rounds):
        rid=case.workload.round_id+offset
        remaining=[[o for o in p.operations if o.round_id==rid] for p in plans]
        for epoch in range(case.strategy.training_epochs):
            for i,queue in enumerate(remaining):
                split=next(k for k,o in enumerate(queue) if o.name=='local_gradient_square_sum')
                for op in queue[:split]:translate(i,op)
                remaining[i]=queue[split:]
            sync(rid,offset*case.strategy.training_epochs+epoch)
            for i,queue in enumerate(remaining):
                boundary=next((k for k,o in enumerate(queue) if o.name in ('zero_gradient_buffer','release_round_temporaries')),len(queue))
                for op in queue[:boundary]:translate(i,op)
                remaining[i]=queue[boundary:]
        for i,queue in enumerate(remaining):
            for op in queue:translate(i,op)
        # A global round fence is explicit P2P fan-in/fan-out, avoiding a
        # hidden multidimensional default collective. All publishers enter
        # before any replica begins the next rollout batch.
        for src,dst in [(r,0) for r in range(1,world)]+[(0,r) for r in range(1,world)]:
            contracts={}
            for rank in (src,dst):
                c={'reads':[],'allocations':{},'ready_outputs':[],'frees':[],
                   'workspace_bytes':0,'sram_capacity':case.hardware.sram_bytes_per_core,
                   'local_memory':False,'reserve_before':0,'reserve_after':0}
                if case.hardware.cores_per_die>1:c['core_workspace_bytes']=[0]*case.hardware.cores_per_die
                contracts[rank]=c
            append(Operation(0,f'round_fence.r{rid}.{src}_to_{dst}',rid,'round_fence','round_version_fence',
                'p2p',payload_bytes=4,src_rank=src,dst_rank=dst,participant_ranks=tuple(sorted((src,dst))),
                metadata={'semantic':'all replicas published before next rollout'}),contracts)
    for i,p in enumerate(plans):
        for tensor in p.tensors:
            t={**tensor,'storage_id':sid(i,tensor['storage_id']),'tensor_id':sid(i,tensor['tensor_id']),
               'rank':2*i+tensor['rank']}
            for key in ('created_node','released_node','ready_node'):
                if t[key] is not None:t[key]=mapping[i,t[key]]
            tensors.append(t)
    contract={'schema_version':'1.0','policy':'DP_SUM_global_token_mean_private_SRAM','log_prefix':log_prefix,'ranks':ranks}
    plan=TensorPlan(contract,tensors,operations,preflight_contract(contract))
    return WaferGraph(plan,groups,cases,local_ops,mapping,placements)
