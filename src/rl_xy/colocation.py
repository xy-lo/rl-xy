"""Global synchronous phases and explicit Megatron-to-rollout TP layouts.

Selected contract: dense PP=1, Megatron replicated optimizer, no CPU offload,
vLLM sleep/resume with distinct rollout weights and per-tensor gathered export.
This is a trace/cost implementation of the pinned contract, not veRL runtime.
"""
from dataclasses import dataclass,replace,asdict
import copy
import json
from pathlib import Path
from functools import lru_cache

from .workflow import WorkflowBuilder
from .model import enumerate_parameters,parameter_totals
from .mapping import ceildiv
from .operator_ir import KernelSpec
from .types import Operation,CheckResult
from .tensor_plan import TensorPlan,build_tensor_plan,finalize_tensor_costs,preflight_contract
from .communication_lowering import lower_communication_endpoints
from .cost_model import AnalyticalCostProvider
from .execution_contract import group_for,prompt_for


def legal_tp(model,tp):
    return type(tp) is int and tp>0 and all(x%tp==0 for x in
        (model.query_heads,model.kv_heads,model.hidden_size,model.ffn_size,model.vocab_size))


def load_colocation(path,case,world):
    p=json.loads(Path(path).read_text())
    required={'schema_version','reference','train_tp','rollout_tp','pipeline_parallel','weight_sync',
        'engine_switch_ns','offload','mapping','training_precision','provenance'}
    if set(p)!=required or p['schema_version']!='rlxy.colocation/1':raise ValueError('invalid colocation schema')
    if p['reference']!='verl_v0.9.1_megatron_naive_sync':raise ValueError('pin a supported reference contract')
    if p['pipeline_parallel']!=1 or p['offload'] is not False:raise ValueError('selected contract is PP1 without host offload')
    if p['weight_sync']!='per_tensor_full_export_distinct_rollout':raise ValueError('unsupported weight ownership contract')
    if not all(legal_tp(case.model,p[k]) for k in ('train_tp','rollout_tp')):raise ValueError('TP must divide all attention/model dimensions')
    if p['train_tp']%p['rollout_tp'] or world//p['train_tp']<2:raise ValueError('rollout TP must divide training TP; DP>=2 required')
    if type(p['engine_switch_ns']) is not int or p['engine_switch_ns']<1:raise ValueError('explicit positive engine switch cost required')
    if p['mapping'] not in ('reference','capacity_aware'):raise ValueError('unknown local mapping policy')
    if p['training_precision']!='bf16_mixed_fp32_accumulation':raise ValueError('selected backend uses BF16 GEMM inputs and FP32 accumulated parameter gradients')
    if case.rounds<1 or case.strategy.training_epochs!=1 or case.strategy.optimizer_minibatch_trajectories!=0:
        raise ValueError('selected contract uses one full-batch update per round')
    return p


class StageBuilder(WorkflowBuilder):
    def __init__(self,case,selected):super().__init__(case);self.selected=selected
    def wanted(self,phase):return phase in self.selected or ('train' in self.selected and phase.startswith('train_'))
    def _append(self,**kwargs):
        if not self.wanted(kwargs['phase']):return None
        spec=kwargs.get('kernel_spec')
        if spec and spec.matmuls:
            mats=tuple(replace(p,a_bytes=2,b_bytes=2,
                output_bytes=(4 if p.direction=='dweight' else 2) if p.direction in ('dinput','dweight') else p.output_bytes)
                for p in spec.matmuls)
            kwargs['kernel_spec']=replace(spec,matmuls=mats)
            kwargs['metadata']={**(kwargs.get('metadata') or {}),'precision_contract':'bf16_GEMM_FP32_dweight_accumulation'}
        return super()._append(**kwargs)
    def _forward(self,**kwargs):
        if self.wanted(kwargs['phase']):return super()._forward(**kwargs)
    def _backward(self,**kwargs):
        if self.wanted(kwargs['phase']):return super()._backward(**kwargs)
    def collective(self,round_id,phase,name,collective,payload_bytes,metadata=None):
        if not self.wanted(phase):return None
        if self.case.hardware.dies==1:
            return self.compute(round_id,phase,name,duration_ns=1,tensor_bytes=payload_bytes,
                metadata={**(metadata or {}),'single_rank_collective':True,'cost_source':'single_rank_identity_view'})
        return super().collective(round_id,phase,name,collective,payload_bytes,metadata)
    def compute(self,round_id,phase,name,**kwargs):
        if self.case.hardware.dies==1 and name.endswith('.sample_on_rank0'):
            kwargs['metadata']={**kwargs.get('metadata',{}),'terminal_ranks':[0],'sampling_terminal':True}
        return super().compute(round_id,phase,name,**kwargs)
    def p2p(self,round_id,phase,name,src_rank,dst_rank,payload_bytes,metadata=None):
        if not self.wanted(phase) or self.case.hardware.dies==1:return None
        tp=self.case.hardware.dies;result=None
        if name.endswith('.sample_logits_rank1_to_rank0'):
            for src in range(1,tp):
                result=super().p2p(round_id,phase,name.replace('rank1_to_rank0',f'rank{src}_to_rank0'),src,0,payload_bytes,metadata)
        elif src_rank==0 and dst_rank==1:
            for dst in range(1,tp):
                meta=dict(metadata or {})
                if '.sample_token_' in name:meta.update(terminal_ranks=[dst]+([0] if dst==tp-1 else []),sampling_terminal=True)
                result=super().p2p(round_id,phase,name.replace('rank0_to_rank1',f'rank0_to_rank{dst}'),0,dst,payload_bytes,meta)
        else:result=super().p2p(round_id,phase,name,src_rank,dst_rank,payload_bytes,metadata)
        return result


@dataclass
class ColocatedGraph:
    plan:TensorPlan
    groups:dict
    placements:list
    audit:dict
    component_checks:list


def pad_plan_collectives(plan,world):
    padded={}
    for op in plan.operations:
        if op.kind!='collective':continue
        transport=ceildiv(max(op.payload_bytes,world),len(op.participant_ranks)*4)*len(op.participant_ranks)*4
        padded[op.metadata.get('logical_original_op_id',op.op_id)]=transport
        for rank in op.participant_ranks:
            n=plan.contract['ranks'][str(rank)]['nodes'][str(op.node_id)]
            sid=f'collective_scratch.n{op.node_id}.d{rank}'
            n['allocations'][sid]=transport;n['ready_outputs'].append(sid);n['frees'].append(sid)
    result=[]
    for op in plan.operations:
        meta=dict(op.metadata)
        key=meta.get('logical_original_op_id',op.op_id)
        if op.kind=='collective':meta['transport_payload_bytes']=padded[key]
        parent=meta.get('parent_op')
        if parent in padded and meta.get('local_transfer'):
            width=padded[parent];spec=op.kernel_spec
            spec=replace(spec,read_bytes=width if meta['local_transfer']=='read' else 0,
                write_bytes=width if meta['local_transfer']=='write' else 0)
            op=replace(op,kernel_spec=spec)
        result.append(replace(op,metadata=meta))
    plan.operations=result
    return plan


def build_colocated_graph(case,world,config,log_prefix,*,cost_provider=None):
    tr=config['train_tp'];ge=config['rollout_tp'];dp=world//tr;active=dp*tr;micro=tr//ge
    provider=cost_provider or AnalyticalCostProvider()
    gids=list(dict.fromkeys(group_for(case,t) for t in case.workload.trajectories))
    if len(gids)<dp:raise ValueError('at least one complete group per training replica required')
    # Deterministic longest-total-token list scheduling, same algorithm on both
    # platforms; no use of either platform's final simulation outcome.
    assigned=[[] for _ in range(dp)];loads=[0]*dp
    for gid in sorted(gids,key=lambda g:(-sum(len(prompt_for(case,t))+len(t.response_token_ids)
                for t in case.workload.trajectories if group_for(case,t)==g),g)):
        i=min(range(dp),key=lambda i:(loads[i],i));assigned[i].append(gid)
        loads[i]+=sum(len(prompt_for(case,t))+len(t.response_token_ids) for t in case.workload.trajectories if group_for(case,t)==gid)
    train_case=replace(case,hardware=replace(case.hardware,dies=tr),strategy=replace(case.strategy,tensor_parallel=tr))
    train_params=enumerate_parameters(train_case)
    gen_case=replace(case,hardware=replace(case.hardware,dies=ge),strategy=replace(case.strategy,tensor_parallel=ge))
    gen_params={p.parameter_id:p for p in enumerate_parameters(gen_case)}
    contract={'schema_version':'1.1','log_prefix':log_prefix,'ranks':{}}
    tensors=[];ops=[];groups={};group_ids={};last={};placements=[];component_checks=[];boundaries=[]
    for rank in range(world):
        initial={}
        if rank<active:
            for p in train_params:
                for role,width in [('actor',2),('reference',2),('gradient',4),('master',4),('adam_m',4),('adam_v',4)]:
                    sid=f'd{rank}.{role}.{p.parameter_id}';size=p.local_elements_per_die*width;initial[sid]=size
                    tensors.append(dict(storage_id=sid,tensor_id=sid,shape=[p.local_elements_per_die],dtype_bytes=width,
                        nbytes=size,rank=rank,kind=role,round_id=None,created_node=None,ready_node=None,released_node=None,persistent=True))
        r={'initial_objects':initial,'dram_capacity':case.hardware.dram_bytes_per_die,'nodes':{}}
        if case.hardware.cores_per_die>1:r.update(cores_per_die=case.hardware.cores_per_die,sram_bytes_per_core=case.hardware.sram_bytes_per_core)
        contract['ranks'][str(rank)]=r
    def group(participants):
        participants=tuple(participants)
        if participants not in group_ids:
            idx=len(groups)+1;groups[str(idx)]=list(participants);group_ids[participants]=idx
        return group_ids[participants]
    def blank():
        n=dict(reads=[],allocations={},ready_outputs=[],frees=[],workspace_bytes=0,
            sram_capacity=case.hardware.sram_bytes_per_core,local_memory=False,reserve_before=0,reserve_after=0)
        if case.hardware.cores_per_die>1:n['core_workspace_bytes']=[0]*case.hardware.cores_per_die
        return n
    def append(op,nodes):
        node=len(ops)+1;deps={r:(last[r],) if r in last else () for r in op.participant_ranks}
        op=replace(op,node_id=node,dependencies_by_rank=deps,dependencies=tuple(sorted({x for ds in deps.values() for x in ds})))
        ops.append(op)
        for rank in op.participant_ranks:last[rank]=node;contract['ranks'][str(rank)]['nodes'][str(node)]=nodes[rank]
        return op
    def control(rid,phase,name,participants,*,reads=None,allocations=None,frees=None,writes=None,ready_alloc=True,duration=1,meta=None,spec=None):
        ns={}
        estimate=provider.estimate(spec,case.hardware) if spec else None
        for rank in participants:
            n=blank();n['reads']=list((reads or {}).get(rank,[]));n['allocations']=dict((allocations or {}).get(rank,{}))
            n['ready_outputs']=(list(n['allocations']) if ready_alloc else [])+list((writes or {}).get(rank,[]));n['frees']=list((frees or {}).get(rank,[]))
            if estimate:
                n.update(workspace_bytes=estimate.sram_workspace_bytes,local_memory=True)
                if case.hardware.cores_per_die>1:n['core_workspace_bytes']=list(estimate.core_workspace_bytes)
            ns[rank]=n
        return append(Operation(0,f'r{rid}.{name}',rid,phase,name,'compute',duration_ns=estimate.duration_ns if estimate else duration,
            participant_ranks=tuple(participants),kernel_spec=spec,
            metadata={'execution_domain':'device' if estimate else 'host',**(meta or {}),
                'cost_estimate':asdict(estimate) if estimate else {'duration_ns':duration,'source':'explicit_engine_control_assumption'}}),ns)
    def transfer(rid,phase,name,participants,size,*,collective=None,reads=None,outputs=None,writes=None,retain_output=False):
        participants=tuple(participants);ns={r:blank() for r in participants}
        # Explicit scratch and endpoint service; no unlimited hidden network buffer.
        transport=ceildiv(max(size,world),len(participants)*4)*len(participants)*4
        buffers={}
        for r in participants:
            ns[r]['reads']=list((reads or {}).get(r,[]));buf=f'network.{len(ops)}.d{r}'
            buffers[r]=buf
            ns[r]['allocations']={buf:transport,**(outputs or {}).get(r,{})}
            ns[r]['ready_outputs']=list(ns[r]['allocations'])+list((writes or {}).get(r,[]));ns[r]['frees']=[] if retain_output else [buf]
        for direction in ['read']:
            control(rid,phase,name+'.endpoint_'+direction,participants,reads=reads,
                spec=KernelSpec('network_endpoint_read',read_bytes=transport))
        meta={'transport_payload_bytes':transport,'logical_payload_bytes':size}
        if collective:meta['comm_group']=group(participants)
        op=append(Operation(0,f'r{rid}.{name}',rid,phase,name,'collective' if collective else 'p2p',payload_bytes=size,
            collective=collective,src_rank=None if collective else participants[0],dst_rank=None if collective else participants[1],
            participant_ranks=participants,metadata=meta),ns)
        control(rid,phase,name+'.endpoint_write',participants,spec=KernelSpec('network_endpoint_write',write_bytes=transport))
        return buffers if retain_output else op
    def fence(rid,label):
        # Binary tree barrier: all arrivals reach root before any release. This
        # avoids a serial O(world) fan-in while using real native p2p events.
        for child in range(world-1,0,-1):
            parent=(child-1)//2;append(Operation(0,f'r{rid}.barrier.{label}.up{child}',rid,'barrier',f'barrier_{label}_arrive','p2p',
                payload_bytes=4,src_rank=child,dst_rank=parent,participant_ranks=(child,parent),metadata={'stage_barrier':label}),{child:blank(),parent:blank()})
        for child in range(1,world):
            parent=(child-1)//2;append(Operation(0,f'r{rid}.barrier.{label}.down{child}',rid,'barrier',f'barrier_{label}_release','p2p',
                payload_bytes=4,src_rank=parent,dst_rank=child,participant_ranks=(parent,child),metadata={'stage_barrier':label}),{child:blank(),parent:blank()})
        boundaries.append(label)
    bridges={r:[] for r in range(world)}
    def carrier(rid,rank,tid):return f'trajectory.r{rid}.d{rank}.{tid}'
    def component(local,physical,selected,label,rollout=False):
        b=StageBuilder(local,selected|{'cleanup'});b.build_round(0)
        p=build_tensor_plan(local,lower_communication_endpoints(local,b.operations),log_prefix)
        if rollout:
            from .batching import fuse_generation
            p=fuse_generation(p,local)
        p=pad_plan_collectives(p,world)
        for r,c in p.contract['ranks'].items():
            used={s for n in c['nodes'].values() for key in ('reads','ready_outputs') for s in n[key]}
            c['initial_objects']={s:v for s,v in c['initial_objects'].items() if s in used}
        finalize_tensor_costs(p,local,provider=provider)
        # Initial stage views alias the persistent training state or the
        # explicitly allocated distinct rollout weights, never another pool.
        prefix=f'{label}.'
        initial_ids={s for c in p.contract['ranks'].values() for s in c['initial_objects']}
        @lru_cache(maxsize=None)
        def sid(name):
            if name in initial_ids:
                parts=name.split('.',2);rank=physical[int(parts[0][1:])];role=parts[1]
                return f'rollout.r{local.workload.round_id}.d{rank}.{parts[2]}' if rollout and role=='actor' else f'd{rank}.{role}.{parts[2]}'
            return prefix+name
        for t in p.tensors:
            if t['persistent']:continue
            v=dict(t);v.update(storage_id=sid(t['storage_id']),tensor_id=sid(t['tensor_id']),rank=physical[t['rank']]);tensors.append(v)
        for op in p.operations:
            if rollout and op.name=='release_round_temporaries':
                rid=local.workload.round_id
                reads={physical[lr]:[sid(t['storage_id']) for t in p.tensors if t['rank']==lr and t['kind'] in ('token','trajectory')]
                       for lr in range(local.hardware.dies)}
                alloc={r:{carrier(rid,r,t.trajectory_id):(len(prompt_for(local,t))+len(t.response_token_ids))*4
                         for t in local.workload.trajectories} for r in physical}
                size=sum(alloc[physical[0]].values())
                control(rid,'rollout',label+'.collect_trajectory_outputs',physical,reads=reads,allocations=alloc,
                    spec=KernelSpec('trajectory_output_pack',read_bytes=size,write_bytes=size))
                for r in physical:bridges[r].extend(alloc[r])
            elif op.name=='release_round_temporaries' and len(selected)==1:
                phase=next(iter(selected));rid=local.workload.round_id
                count=(sum(len(t.response_token_ids) for t in local.workload.trajectories)
                       if phase in ('reference','old_policy') else len(local.workload.trajectories))
                kinds={'log_prob'} if phase in ('reference','old_policy') else {phase}
                reads={physical[lr]:[sid(t['storage_id']) for t in p.tensors if t['rank']==lr and t['kind'] in kinds]
                       for lr in range(local.hardware.dies)}
                alloc={r:{f'state.r{rid}.d{r}.{phase}':count*4} for r in physical}
                control(rid,phase,label+'.retain_phase_outputs',physical,reads=reads,allocations=alloc,
                    spec=KernelSpec('phase_output_copy',read_bytes=count*4,write_bytes=count*4))
                for r in physical:bridges[r].extend(alloc[r])
            nodes={}
            for lr in op.participant_ranks:
                n=copy.deepcopy(p.contract['ranks'][str(lr)]['nodes'][str(op.node_id)])
                for field in ('reads','ready_outputs','frees'):n[field]=[sid(k) for k in n[field]]
                n['allocations']={sid(k):v for k,v in n['allocations'].items()};nodes[physical[lr]]=n
                if not rollout and op.name!='release_round_temporaries':n['reads']=sorted(set(n['reads']+bridges[physical[lr]]))
            stage='generation' if rollout else next(iter(selected))
            meta={**op.metadata,'stage_component':label,'tp_size':local.strategy.tensor_parallel,
                  'colocation_stage':stage,'reference_version':case.workload.behavior_version,
                  'actor_version':local.workload.behavior_version}
            participants=tuple(physical[r] for r in op.participant_ranks)
            if op.kind=='collective':meta['comm_group']=group(participants)
            append(replace(op,op_id=prefix+op.op_id,participant_ranks=participants,
                src_rank=None if op.src_rank is None else physical[op.src_rank],
                dst_rank=None if op.dst_rank is None else physical[op.dst_rank],metadata=meta),nodes)
        if 'train' in selected:
            from .execution_validation import shape_checks
            component_checks.extend(replace(c,check_id=label+'.'+c.check_id) for c in shape_checks(local,p.operations))
        return p
    denom=sum(len(t.response_token_ids) for t in case.workload.trajectories)
    weight_network_bytes=0
    for offset in range(case.rounds):
        rid=case.workload.round_id+offset;version=case.workload.behavior_version+offset
        bridges={r:[] for r in range(world)}
        if active<world:
            control(rid,'standby','standby_rank_marker',tuple(range(active,world)),
                meta={'execution_domain':'device','semantic':'explicit one-tick trace marker for allocated idle ranks; no model arithmetic'})
        locals=[]
        for i,g in enumerate(assigned):
            ts=tuple(t for t in case.workload.trajectories if group_for(case,t) in g)
            local=replace(train_case,rounds=1,workload=replace(case.workload,round_id=rid,behavior_version=version,trajectories=ts),
                strategy=replace(train_case.strategy,rollout_max_concurrency=len(ts)))
            locals.append(local)
            if offset==0:placements.append({'replica':i,'tp_ranks':list(range(i*tr,(i+1)*tr)),'groups':g,
                'rollout_tp_groups':[[i*tr+j*ge+k for k in range(ge)] for j in range(micro)],
                'trajectories':[t.trajectory_id for t in ts],'global_loss_denominator':denom})
        fence(rid,'before_weight_sync')
        # Distinct vLLM storage; conservative per-tensor full HF export. Norm
        # weights are already replicated and do not require TP all-gather.
        for i in range(dp):
            physical=tuple(range(i*tr,(i+1)*tr));alloc={r:{} for r in physical}
            for p in train_params:
                for r in physical:alloc[r][f'rollout.r{rid}.d{r}.{p.parameter_id}']=gen_params[p.parameter_id].local_elements_per_die*2
            control(rid,'switch_to_rollout',f'resume_rollout_{i}',physical,allocations=alloc,ready_alloc=False,duration=config['engine_switch_ns'],meta={'actor_version':version})
            for p in train_params:
                reads={r:[f'd{r}.actor.{p.parameter_id}'] for r in physical}
                buffers={}
                if not p.replicated and tr>1:
                    buffers=transfer(rid,'weight_reshard',f'export_{i}_{p.parameter_id}',physical,p.unique_elements*2,collective='ALL_GATHER',reads=reads,retain_output=True)
                    weight_network_bytes+=p.unique_elements*2
                # Read complete exported BF16 tensor, pack/write only the
                # destination inference shard; buffer is explicit per tensor.
                control(rid,'weight_reshard',f'pack_{i}_{p.parameter_id}',physical,
                    reads={r:reads[r]+([buffers[r]] if r in buffers else []) for r in physical},
                    frees={r:([buffers[r]] if r in buffers else []) for r in physical},
                    writes={r:[f'rollout.r{rid}.d{r}.{p.parameter_id}'] for r in physical},
                    spec=KernelSpec('export_convert_load',read_bytes=p.unique_elements*2,write_bytes=gen_params[p.parameter_id].local_elements_per_die*2),
                    meta={'source_train_tp':tr,'destination_rollout_tp':ge,'bf16_no_numeric_cast':True})
        fence(rid,'weights_ready')
        for i,local in enumerate(locals):
            ts=local.workload.trajectories
            for j in range(micro):
                chosen=ts[j::micro]
                if not chosen:continue
                physical=tuple(i*tr+j*ge+k for k in range(ge))
                gen=replace(local,hardware=replace(local.hardware,dies=ge),strategy=replace(local.strategy,tensor_parallel=ge,rollout_max_concurrency=len(chosen)),
                    workload=replace(local.workload,trajectories=chosen))
                component(gen,physical,{'input','rollout'},f'r{rid}.rollout{i}.{j}',rollout=True)
        fence(rid,'rollout_done')
        for i,local in enumerate(locals):
            for k,t in enumerate(local.workload.trajectories):
                size=(len(prompt_for(local,t))+len(t.response_token_ids))*4
                for u in range(ge):
                    owner=i*tr+(k%micro)*ge+u
                    for j in range(micro):
                        dest=i*tr+j*ge+u
                        if dest!=owner:
                            storage=carrier(rid,dest,t.trajectory_id)
                            transfer(rid,'data_redistribution',f'samples_{i}_{t.trajectory_id}_{j}_{u}',(owner,dest),size,
                                reads={owner:[carrier(rid,owner,t.trajectory_id)]},outputs={dest:{storage:size}})
                            bridges[dest].append(storage)
            physical=tuple(range(i*tr,(i+1)*tr))
            control(rid,'switch_to_train',f'sleep_rollout_{i}',physical,
                frees={r:[f'rollout.r{rid}.d{r}.{p.parameter_id}' for p in train_params] for r in physical},
                duration=config['engine_switch_ns'],meta={'restore_training_layout':'retained_actor_shards_no_reverse_copy','actor_version':version})
        fence(rid,'train_layout_ready')
        for phase in ['reward','old_policy','reference','advantage']:
            for i,local in enumerate(locals):component(local,tuple(range(i*tr,(i+1)*tr)),{phase},f'r{rid}.{phase}{i}')
            fence(rid,phase+'_done')
        # Training tape and gradients remain live through DP SUM and clip.
        tails=[]
        for i,local in enumerate(locals):
            b=StageBuilder(local,{'train','clip','optimizer','publish','cleanup'});b.build_round(0)
            # Global token denominator is applied before native DP SUM.
            b.operations=[replace(o,metadata={**o.metadata,**{k:denom for k in ('global_denominator','global_loss_denominator','minibatch_denominator') if k in o.metadata}}) for o in b.operations]
            # Reuse the ordinary component builder with an intercept at clip:
            # split stage operations while keeping one lifetime plan.
            p=build_tensor_plan(local,lower_communication_endpoints(local,b.operations),log_prefix);finalize_tensor_costs(p,local,provider=provider)
            # Recompute explicit endpoint service after native transport padding.
            p=pad_plan_collectives(p,world);finalize_tensor_costs(p,local,provider=provider)
            tails.append((i,local,p))
        # Translate training plans in two pieces; all ranks join DP SUM before clip.
        for i,local,p in tails:
            physical=tuple(range(i*tr,(i+1)*tr));label=f'r{rid}.training{i}.'
            for t in p.tensors:
                if not t['persistent']:tensors.append({**t,'storage_id':label+t['storage_id'],'tensor_id':label+t['tensor_id'],'rank':physical[t['rank']]})
            from .execution_validation import shape_checks
            component_checks.extend(replace(c,check_id=label+c.check_id) for c in shape_checks(local,p.operations))
            # One canonical renamed string per storage within this replica.
            # Capturing sid in trans's defaults also preserves it when tails
            # execute after the loop has advanced to another replica.
            @lru_cache(maxsize=None)
            def sid(s,physical=physical,label=label):
                return f'd{physical[int(s.split(".")[0][1:])]}.{s.split(".",1)[1]}' if s.startswith('d') else label+s
            def trans(op,i=i,p=p,label=label,physical=physical,sid=sid):
                nodes={}
                for lr in op.participant_ranks:
                    n=copy.deepcopy(p.contract['ranks'][str(lr)]['nodes'][str(op.node_id)])
                    for key in ('reads','ready_outputs','frees'):n[key]=[sid(s) for s in n[key]]
                    n['allocations']={sid(s):v for s,v in n['allocations'].items()};nodes[physical[lr]]=n
                    if op.name!='release_round_temporaries':n['reads']=sorted(set(n['reads']+bridges[physical[lr]]))
                participants=tuple(physical[r] for r in op.participant_ranks);meta={**op.metadata,'stage_component':label,'replica_id':i,'tp_size':tr,
                    'colocation_stage':'train' if op.phase.startswith('train_') else op.phase,'actor_version':local.workload.behavior_version,
                    'reference_version':case.workload.behavior_version}
                if op.kind=='collective':meta['comm_group']=group(participants)
                return append(replace(op,op_id=label+op.op_id,participant_ranks=participants,src_rank=None if op.src_rank is None else physical[op.src_rank],
                    dst_rank=None if op.dst_rank is None else physical[op.dst_rank],metadata=meta),nodes)
            split=next(k for k,o in enumerate(p.operations) if o.name=='local_gradient_square_sum')
            gradient_errors=[]
            expected=parameter_totals(local)['local_parameter_elements_per_die']*4
            if local.model.tie_word_embeddings:expected+=local.model.vocab_size*local.model.hidden_size//tr*4
            for lr in range(tr):
                for t in local.workload.trajectories:
                    count=sum(x['nbytes'] for x in p.tensors if x['rank']==lr and x['kind']=='parameter_gradient'
                        and f'train_{t.trajectory_id}_backward' in x['tensor_id'])
                    if count!=expected:gradient_errors.append([lr,t.trajectory_id,count,expected])
            component_checks.append(CheckResult(label+'gradient_contributions',not gradient_errors,expected,gradient_errors,
                'each local trajectory dweight/norm/bias/embedding against independently counted parameters','shape_contract'))
            for op in p.operations[:split]:trans(op)
            tails[i]=(trans,p.operations[split:])
        fence(rid,'backward_done')
        params=parameter_totals(train_case)['local_parameter_elements_per_die']
        for shard in range(tr):
            participants=tuple(range(shard,active,tr))
            transfer(rid,'dp_sync',f'dp_gradient_sum_{shard}',participants,params*4,collective='ALL_REDUCE',
                reads={r:[s for s in contract['ranks'][str(r)]['initial_objects'] if '.gradient.' in s] for r in participants})
            control(rid,'dp_sync',f'dp_gradient_local_reduce_{shard}',participants,
                spec=KernelSpec('dp_gradient_sum',vector_ops=ceildiv(params,dp)*(dp-1),read_bytes=params*8,write_bytes=params*4),
                meta={'sum_not_mean':True,'global_loss_denominator':denom})
        fence(rid,'gradient_sum_done')
        for trans,tail in tails:
            for op in tail:trans(op)
        control(rid,'cleanup','release_colocation_trajectories',tuple(range(active)),frees=bridges)
        fence(rid,'update_published')
    # Resolve all temporary identities against the final global node IDs,
    # including switch/network buffers not emitted by component tensor plans.
    indexed={t['storage_id']:t for t in tensors}
    for rank,c in contract['ranks'].items():
        for node,n in c['nodes'].items():
            for sid,size in n['allocations'].items():
                if sid not in indexed:indexed[sid]=dict(storage_id=sid,tensor_id=sid,shape=[size],dtype_bytes=1,
                    nbytes=size,rank=int(rank),kind='runtime_buffer',round_id=None,persistent=False,released_node=None)
                indexed[sid].update(created_node=int(node),ready_node=int(node))
            for sid in n['ready_outputs']:
                if sid in indexed and not indexed[sid]['persistent']:indexed[sid]['ready_node']=int(node)
            for sid in n['frees']:
                if sid in indexed:indexed[sid]['released_node']=int(node)
    plan=TensorPlan(contract,list(indexed.values()),ops,{})
    plan.preflight_peaks=preflight_contract(contract)
    return ColocatedGraph(plan,groups,placements,
        {'train_tp':tr,'rollout_tp':ge,'train_dp':dp,'rollout_dp':dp*micro,'active_devices':active,'allocated_devices':world,
         'idle_devices':list(range(active,world)),'global_barriers':boundaries,'weight_export_logical_payload_bytes':weight_network_bytes,
         'weight_sync':config['weight_sync'],'reference':config['reference'],'hardware_calibrated':False,'rollout_rank_order':'contiguous_TP_subgroups_within_training_TP',
         'dispatch':'complete-GRPO-group preserving longest-token-first; common controlled restriction, not default veRL trajectory balancer',
         'batching':'packed linear GEMMs per prefill/decode frame, ragged unfused attention; no exact vLLM fusion/continuous-admission claim'},component_checks)
