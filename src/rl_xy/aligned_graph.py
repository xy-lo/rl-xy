"""Selected veRL synchronous PP1 protocol with explicit DO and active ranks.

This is a cost/dataflow model of a declared configuration, not a model training
runtime. Legacy colocation remains frozen; this builder has separate gates.
"""
from dataclasses import replace,asdict
from functools import lru_cache
import copy,math
from .types import Operation,CheckResult
from .operator_ir import KernelSpec
from .model import enumerate_parameters
from .execution_contract import group_for,prompt_for
from .colocation import StageBuilder,ColocatedGraph,pad_plan_collectives
from .tensor_plan import TensorPlan,build_tensor_plan,finalize_tensor_costs,preflight_contract
from .communication_lowering import lower_communication_endpoints
from .distributed_optimizer import layout_for_case,global_loss_scaling,validate_publication
from .grpo_math import group_advantages
from .mapping import ceildiv


def lower_aligned_endpoints(case,operations):
    """Ring SUM arithmetic/staging is explicit; native local-mem proxy is off.

    For AR each rank reduces (TP-1) chunks of N/TP elements. Native ring
    transit remains in COMM. This unfused reference is shared by both sides.
    """
    return lower_communication_endpoints(case,operations)


def pad_aligned_collectives(plan,world):
    plan=pad_plan_collectives(plan,world)
    parents={o.metadata.get('logical_original_op_id',o.op_id):o for o in plan.operations if o.kind=='collective'}
    result=[]
    for op in plan.operations:
        parent=parents.get(op.metadata.get('parent_op'))
        if parent and parent.collective=='ALL_REDUCE' and op.metadata.get('local_transfer')=='write':
            tp=len(parent.participant_ranks);width=4 if parent.metadata.get('dtype')=='fp32' else 2
            received=ceildiv(parent.metadata['transport_payload_bytes'],tp*width)*(tp-1)*width
            spec=replace(op.kernel_spec,vector_ops=received//width,read_bytes=received*2,write_bytes=op.kernel_spec.write_bytes+received)
            op=replace(op,kernel_spec=spec,vector_ops=spec.vector_ops,metadata={**op.metadata,
                'ring_local_reduction':True,'reduction_op':'MAX' if parent.metadata.get('semantic')=='distributed_logsoftmax_max' else 'SUM','reduction_element_bytes':width,'native_local_mem_delay_disabled':True,
                'ring_reduction_additional_read_bytes':received*2,'ring_reduction_additional_write_bytes':received})
        result.append(op)
    plan.operations=result
    return plan


def select_generation_logits_rows(case,operations):
    """vLLM sample-only prefill selects one last hidden row per request.

    Dense teacher-forcing heads retain the declared Megatron full sequence.
    A BF16 row materialization cost is explicit rather than free selection.
    """
    result=[]
    for op in operations:
        if op.phase=='rollout' and op.name.endswith('.lm_head_local') and op.metadata.get('tokens',1)>1:
            if not op.kernel_spec or len(op.kernel_spec.matmuls)!=1:raise ValueError('one explicit generation vocabulary projection required')
            old=op.kernel_spec.matmuls[0];selected=replace(old,m=1)
            size=case.model.hidden_size*2
            selection=KernelSpec('last_hidden_row_select',read_bytes=size,write_bytes=size)
            meta={**op.metadata,'selected_hidden_rows':1,'prefill_hidden_rows':old.m,
                'logits_selection':'last_prompt_position_per_request',
                'reference':'vllm_v0.10.1_logits_processor_hidden_state_pruning'}
            result.append(replace(op,op_id=op.op_id+'.hidden_select',name=op.name.rsplit('.',1)[0]+'.last_hidden_row_select',
                flops=0,vector_ops=0,tensor_bytes=size,kernel_spec=selection,metadata={**meta,'dtype':'bf16'}))
            op=replace(op,flops=selected.flops,tensor_bytes=selected.n*selected.output_bytes,
                kernel_spec=replace(op.kernel_spec,matmuls=(selected,)),metadata=meta)
        result.append(op)
    return result


def audit_immutable_carriers(contract,carriers,*,cleanup_nodes=None):
    carrier_records={s:dict(alloc=[],ready=[],free=[]) for s in carriers}
    for rank,c in contract['ranks'].items():
        for node,n in c['nodes'].items():
            for field,values in [('alloc',n['allocations']),('ready',n['ready_outputs']),('free',n['frees'])]:
                for sid in values:
                    if sid in carrier_records:carrier_records[sid][field].append(int(node))
    invalid=[s for s,v in carrier_records.items() if len(v['alloc'])!=1 or len(v['ready'])!=1 or v['ready'][0]<v['alloc'][0] or len(v['free'])!=1 or v['free'][0]<=v['ready'][0] or (cleanup_nodes is not None and v['free'][0] not in cleanup_nodes)]
    if invalid:raise ValueError('external stage input carrier mutation/lifetime prevents read-proof pruning: '+str(invalid[:4]))
    return carrier_records


class AlignedStageBuilder(StageBuilder):
    def __init__(self,case,selected,policy):
        super().__init__(case,selected);self.policy=policy
        ts=case.workload.trajectories
        self.advantages=None if any(t.reward is None for t in ts) else dict(zip(
            [t.trajectory_id for t in ts],group_advantages([t.reward for t in ts],
            [group_for(case,t) for t in ts],normalize_std=policy.norm_adv_by_std)))
    def compute(self,round_id,phase,name,**kw):
        meta=dict(kw.get('metadata') or {});special=None
        if phase=='advantage' and name=='group_advantage_demean':
            n=len(self.case.workload.trajectories);g=len({group_for(self.case,t) for t in self.case.workload.trajectories})
            # Count the explicit scalar reference expression for semantic
            # coverage. The host_controller path retains its declared fixture
            # duration; this is not a calibrated CPU/vector timing estimate.
            count=5*n+4*g if self.policy.norm_adv_by_std else 2*n+g
            kw.update(vector_ops=count);special=KernelSpec('grpo_group_sample_std',vector_ops=count,read_bytes=4*n,write_bytes=4*n)
            meta.update(norm_adv_by_std=self.policy.norm_adv_by_std,std_unbiased=True,epsilon=1e-6,
                advantages=None if self.advantages is None else [self.advantages[t.trajectory_id] for t in self.case.workload.trajectories],
                numeric_values_supplied=self.advantages is not None)
        if phase.startswith('train_') and name.endswith('_loss'):
            tid=phase.removeprefix('train_');t=next(t for t in self.case.workload.trajectories if t.trajectory_id==tid);n=len(t.response_token_ids)
            count=n*(18+(7 if self.policy.use_kl_loss else 0))
            kw.update(vector_ops=count);special=KernelSpec('grpo_clipped_surrogate_k3',vector_ops=count,read_bytes=n*(17 if self.policy.use_kl_loss else 13),write_bytes=n*4)
            meta.update(policy_loss='vanilla_dual_clip',clip_low=self.policy.clip_ratio,clip_high=self.policy.clip_ratio,
                dual_clip=3.,kl_type='low_var_kl',kl_coefficient=self.policy.kl_coefficient,
                loss_aggregation=self.policy.loss_aggregation,entropy_coefficient=self.policy.entropy_coefficient,
                advantage=None if self.advantages is None else self.advantages[tid],
                local_loss_numerator_scale=self.policy._dp_size,
                semantic_inputs=['current_logprob','old_logprob','advantage','response_mask']+(['reference_logprob'] if self.policy.needs_reference else []))
        kw['metadata']=meta
        op=super().compute(round_id,phase,name,**kw)
        if op is not None and special is not None:
            op=replace(op,kernel_spec=special);self.operations[-1]=op
        return op


class AlignedGraphBuilder:
    def __init__(self,case,allocated,policy,provider,log_prefix,*,active=48,compact_components=True,conservative_bridge_reads=False):
        self.case=case;self.world=allocated;self.policy=policy;self.provider=provider;self.active=active
        self.compact_components=compact_components;self.component_cache={};self.consumer_cache={};self.workspace_vectors={};self.component_expansions=[]
        self.conservative_bridge_reads=conservative_bridge_reads;self.bridge_carriers=set();self.bridge_read_counts={"before":0,"after":0}
        self.global_advantages=None if any(t.reward is None for t in case.workload.trajectories) else dict(zip([t.trajectory_id for t in case.workload.trajectories],group_advantages([t.reward for t in case.workload.trajectories],[group_for(case,t) for t in case.workload.trajectories],normalize_std=policy.norm_adv_by_std)))
        self.tr=policy.train_tp;self.ge=policy.rollout_tp;self.dp=active//self.tr
        if active>allocated or active%self.tr or self.tr%self.ge or self.dp<2:raise ValueError('invalid active TP/DP/rollout layout')
        if policy.sequence_parallel or policy.overlap_grad_reduce or policy.overlap_param_gather:raise ValueError('selected implementation has SP/communication overlap disabled')
        if policy.pipeline_parallel!=1 or not policy.distributed_optimizer or policy.offload:raise ValueError('selected implementation requires PP1/DO/no-offload')
        if policy.dynamic_batching or policy.balance_batch or policy.shuffle:raise ValueError('selected contiguous/static no-shuffle contract required')
        if policy.loss_aggregation!='token-mean' or policy.ppo_epochs!=1 or policy.trajectory_mini_batch_size!=len(case.workload.trajectories):
            raise ValueError('selected implementation uses token-mean full-batch one update')
        if not policy.needs_reference or policy.use_kl_in_reward or not policy.use_kl_loss:raise ValueError('selected KL-in-loss/reference contract required')
        if policy.entropy_coefficient!=0:raise ValueError('entropy cost not selected')
        if len(case.workload.trajectories)!=policy.trajectory_batch_size or len(case.workload.trajectories)%self.dp:raise ValueError('trajectory batch/DP mismatch')
        self.local_case=replace(case,hardware=replace(case.hardware,dies=self.tr),strategy=replace(case.strategy,tensor_parallel=self.tr))
        self.params=enumerate_parameters(self.local_case);self.layout=layout_for_case(self.local_case,dp_size=self.dp)
        self.policy_runtime=copy.copy(policy);object.__setattr__(self.policy_runtime,'_dp_size',self.dp)
        self.log_prefix=log_prefix;self.ops=[];self.last={};self.groups={};self.gids={};self.tensors=[];self.placements=[];self.checks=[];self.barriers=[]
        self.contract={'schema_version':'1.1','policy':'selected_verl_PP1_DO_no_overlap','log_prefix':log_prefix,'ranks':{}}
        for rank in range(allocated):
            initial={}
            if rank<active:
                dpr=rank//self.tr;owned={p['parameter_id']:p['elements'] for p in self.layout.intersections(dpr)}
                for p in self.params:
                    for role,width in [('actor',2),('reference',2),('gradient',4)]:initial[f'd{rank}.{role}.{p.parameter_id}']=p.local_elements_per_die*width
                    for role in ['master','adam_m','adam_v']:
                        if p.parameter_id in owned:initial[f'd{rank}.{role}.{p.parameter_id}']=owned[p.parameter_id]*4
                for role,width in [('actor',2),('gradient',4)]:
                    if self.layout.padding_elements:initial[f'd{rank}.{role}.__padding']=self.layout.padding_elements*width
                for sid,size in initial.items():self.tensors.append(dict(storage_id=sid,tensor_id=sid,rank=rank,kind=sid.split('.')[1],
                    shape=[size],dtype_bytes=1,nbytes=size,persistent=True,round_id=None,created_node=None,ready_node=None,released_node=None))
            c={'initial_objects':initial,'dram_capacity':case.hardware.dram_bytes_per_die,'nodes':{}}
            if case.hardware.cores_per_die>1:c.update(cores_per_die=case.hardware.cores_per_die,sram_bytes_per_core=case.hardware.sram_bytes_per_core)
            self.contract['ranks'][str(rank)]=c
    def blank(self):
        n=dict(reads=[],allocations={},ready_outputs=[],frees=[],workspace_bytes=0,sram_capacity=self.case.hardware.sram_bytes_per_core,local_memory=False,reserve_before=0,reserve_after=0)
        if self.case.hardware.cores_per_die>1:n['core_workspace_bytes']=[0]*self.case.hardware.cores_per_die
        return n
    def append(self,op,nodes):
        nid=len(self.ops)+1;deps={r:(self.last[r],) if r in self.last else () for r in op.participant_ranks}
        op=replace(op,node_id=nid,dependencies_by_rank=deps,dependencies=tuple(sorted({x for ds in deps.values() for x in ds})))
        self.ops.append(op)
        for r in op.participant_ranks:self.last[r]=nid;self.contract['ranks'][str(r)]['nodes'][str(nid)]=nodes[r]
        return op
    def group(self,ps):
        ps=tuple(ps)
        if ps not in self.gids:self.gids[ps]=len(self.groups)+1;self.groups[str(self.gids[ps])]=list(ps)
        return self.gids[ps]
    def control(self,rid,phase,name,ps,*,reads=None,alloc=None,ready=None,frees=None,spec=None,duration=1,meta=None):
        ps=tuple(ps);estimate=self.provider.estimate(spec,self.case.hardware) if spec else None;nodes={}
        for r in ps:
            n=self.blank();n.update(reads=list((reads or {}).get(r,[])),allocations=dict((alloc or {}).get(r,{})),
                ready_outputs=list((ready or {}).get(r,[])),frees=list((frees or {}).get(r,[])))
            if estimate:
                n.update(workspace_bytes=estimate.sram_workspace_bytes,local_memory=True)
                if self.case.hardware.cores_per_die>1:n['core_workspace_bytes']=list(estimate.core_workspace_bytes)
            nodes[r]=n
        return self.append(Operation(0,f'r{rid}.{name}',rid,phase,name,'compute',duration_ns=estimate.duration_ns if estimate else duration,
            participant_ranks=ps,kernel_spec=spec,metadata={'colocation_stage':phase,'execution_domain':'device' if estimate else 'host',
                'cost_estimate':(self.provider.metadata_for(spec,self.case.hardware) if hasattr(self.provider,'metadata_for') else asdict(estimate)) if estimate else {'duration_ns':duration,'source':'explicit_control_assumption'},**(meta or {})}),nodes)
    def collective(self,rid,phase,name,ps,size,kind,*,reads,ready=None,source_bytes=None,destination_bytes=None,meta=None):
        ps=tuple(ps);transport=size if kind=='ALL_GATHER' else ceildiv(max(size,self.world),len(ps)*4)*len(ps)*4
        self.control(rid,phase,name+'.endpoint_read',ps,reads=reads,spec=KernelSpec('collective_endpoint_read',read_bytes=source_bytes if source_bytes is not None else transport))
        nodes={r:self.blank() for r in ps}
        for r in ps:nodes[r]['reads']=list(reads.get(r,[]))
        self.append(Operation(0,f'r{rid}.{name}',rid,phase,name,'collective',payload_bytes=size,collective=kind,participant_ranks=ps,
            metadata={'colocation_stage':phase,'comm_group':self.group(ps),'transport_payload_bytes':transport,
                'buffer_contract':'source/destination existing model buffers; no additional full-size bounce copy',
                'native_comm_size_semantics':'input_shard_bytes_per_rank' if kind=='ALL_GATHER' else 'full_input_bytes_per_rank',**(meta or {})}),nodes)
        received=ceildiv(transport,len(ps)*4)*(len(ps)-1)*4 if kind=='ALL_REDUCE' else 0
        self.control(rid,phase,name+'.endpoint_write',ps,ready=ready,
            spec=KernelSpec('collective_endpoint_write_and_SUM',vector_ops=received//4,read_bytes=received*2,
                write_bytes=(destination_bytes if destination_bytes is not None else transport)+received),
            meta={'ring_local_reduction':bool(received),'native_local_mem_delay_disabled':True})
    def fence(self,rid,label):
        for child in range(self.active-1,0,-1):
            par=(child-1)//2;self.append(Operation(0,f'r{rid}.barrier.{label}.up{child}',rid,'barrier','barrier_'+label,'p2p',payload_bytes=4,
                src_rank=child,dst_rank=par,participant_ranks=(child,par),metadata={'stage_barrier':label}),{child:self.blank(),par:self.blank()})
        for child in range(1,self.active):
            par=(child-1)//2;self.append(Operation(0,f'r{rid}.barrier.{label}.down{child}',rid,'barrier','barrier_'+label,'p2p',payload_bytes=4,
                src_rank=par,dst_rank=child,participant_ranks=(par,child),metadata={'stage_barrier':label}),{child:self.blank(),par:self.blank()})
        self.barriers.append(label)
    def roles(self,r,role,*,owned=False,norm=False):
        rows=self.params if not owned else self.layout.intersections(r//self.tr,tp_rank=r%self.tr,norm_only=norm)
        ids=[p.parameter_id if not owned else p['parameter_id'] for p in rows]
        return [f'd{r}.{role}.{p}' for p in ids]
    def stage(self,local,physical,selected,label,*,rollout=False,bridges=None,training=False,reset_gradient=True):
        source_local=local
        groupmap={}
        canonical=[]
        for i,t in enumerate(local.workload.trajectories):
            g=group_for(local,t)
            if g not in groupmap:groupmap[g]='G'+str(len(groupmap))
            canonical.append(replace(t,trajectory_id='T'+str(i),group_id=groupmap[g]))
        local=replace(local,workload=replace(local.workload,trajectories=tuple(canonical)))
        key=(tuple(sorted(selected)),rollout,training,reset_gradient,local.strategy.tensor_parallel,
            local.workload.round_id,local.workload.behavior_version,
            tuple((len(prompt_for(local,t)),len(t.response_token_ids),t.group_id,t.reward) for t in canonical),
            None if self.global_advantages is None else tuple(self.global_advantages[t.trajectory_id] for t in source_local.workload.trajectories))
        cached=self.component_cache.get(key)
        if cached is None:
            uniform=(rollout or selected in ({'old_policy'},{'reference'})) and len({(len(prompt_for(local,t)),len(t.response_token_ids)) for t in local.workload.trajectories})==1
            template_case=replace(local,workload=replace(local.workload,trajectories=local.workload.trajectories[:1])) if uniform else local
            b=AlignedStageBuilder(template_case,selected|{'cleanup'},self.policy_runtime)
            if self.global_advantages is not None:
                b.advantages={c.trajectory_id:self.global_advantages[t.trajectory_id] for c,t in zip(canonical,source_local.workload.trajectories)}
            b.build_round(0)
            b.operations=select_generation_logits_rows(template_case,b.operations)
            if training and not reset_gradient:b.operations=[o for o in b.operations if o.name!='zero_gradient_buffer']
            T=sum(len(t.response_token_ids) for t in self.case.workload.trajectories)
            b.operations=[replace(o,metadata={**o.metadata,**{k:T for k in ['global_denominator','global_loss_denominator','minibatch_denominator'] if k in o.metadata},
                **({'local_loss_numerator_scale':self.dp,'pre_RS_scale_denominator':self.dp} if o.phase.startswith('train_') else {})}) for o in b.operations]
            p=build_tensor_plan(template_case,lower_aligned_endpoints(template_case,b.operations),self.log_prefix,inference_layer_scope=uniform)
            if uniform:
                from .inference_lifetimes import trim_inference_lifetimes
                p=trim_inference_lifetimes(p,template_case)
                from .uniform_generation import replicate_uniform_generation
                p=replicate_uniform_generation(p,template_case,local)
            # Stage-local plans include only the states actually touched. The
            # global contract still retains Reference and DP-sharded optimizer.
            for c in p.contract['ranks'].values():
                used={s for n in c['nodes'].values() for key in ['reads','ready_outputs'] for s in n[key]}
                c['initial_objects']={s:v for s,v in c['initial_objects'].items() if s in used}
            if rollout and not uniform:
                from .batching import fuse_generation
                p=fuse_generation(p,local)
            if training:
                from .training_batching import fuse_training,validate_training_batching,prepare_training_gradient_io,validate_training_gradient_io
                p=prepare_training_gradient_io(p,local)
                p=fuse_training(p,local,self.policy.train_micro_batch_trajectories,cost_provider=self.provider)
                if self.policy.train_micro_batch_trajectories>1:self.checks.extend(replace(ch,check_id=label+"."+ch.check_id) for ch in validate_training_batching(p,local))
            p=pad_aligned_collectives(p,self.world);finalize_tensor_costs(p,local,provider=self.provider)
            if training:self.checks.extend(replace(ch,check_id=label+"."+ch.check_id) for ch in validate_training_gradient_io(p,local))
            contraction=None
            if self.compact_components:
                from .local_chain_contraction import contract_local_chains
                p.operations=[replace(o,metadata={**o.metadata,'contraction_boundary':True}) if o.name=='release_round_temporaries' else o for o in p.operations]
                contraction=contract_local_chains(p)
                p=contraction.macro_plan
            cached=(p,contraction)
            self.component_cache[key]=cached
        p,contraction=cached
        # Native hooks reserve the complete temporary arena. Original tensor
        # lifetimes are replayed independently from the observed macro clocks.
        arena={r:{label+'.'+s:size for s,size in p.contract['ranks'][str(lr)]['initial_objects'].items() if s.endswith('.logical_plan_arena')}
            for lr,r in enumerate(physical)}
        arena_initialized=set()
        expansion={'label':label,'contraction':contraction,'physical':physical,'global_to_local':{},
            'source_to_canonical_trajectories':{t.trajectory_id:c.trajectory_id for t,c in zip(source_local.workload.trajectories,canonical)}}
        if contraction is not None:self.component_expansions.append(expansion)
        initial={s for c in p.contract['ranks'].values() for s in c['initial_objects']}
        @lru_cache(maxsize=None)
        def sid(s):
            if s.endswith('.logical_plan_arena'):return label+'.'+s
            if s in initial:
                _,role,pid=s.split('.',2);r=physical[int(s.split('.',1)[0][1:])]
                return f'rollout.r{local.workload.round_id}.d{r}.{pid}' if rollout and role=='actor' else f'd{r}.{role}.{pid}'
            return label+'.'+s
        for t in p.tensors:
            if not t['persistent']:self.tensors.append({**t,'storage_id':sid(t['storage_id']),'tensor_id':sid(t['tensor_id']),'rank':physical[t['rank']]})
        cleanup=[]
        bridge_snapshot={r:tuple(bridges[r]) for r in physical};self.bridge_carriers.update(s for values in bridge_snapshot.values() for s in values)
        first_input_seen=set()
        def consumes_external(op):
            return (op.phase in ('reward','advantage') or '.embedding' in op.name or '.logsoftmax_target_logit' in op.name
                or op.name.endswith(('_loss','_loss_backward')))
        consumers=self.consumer_cache.get(id(p))
        if consumers is None:
            if rollout:consumers=set()
            elif contraction is None:consumers={o.node_id for o in p.operations if consumes_external(o)}
            else:
                original={o.node_id:o for o in contraction.original_plan.operations}
                consumers={block.native_node_id for block in contraction.blocks if any(consumes_external(original[n]) for n in block.logical_node_ids)}
            self.consumer_cache[id(p)]=consumers
        for op in p.operations:
            if op.name=='release_round_temporaries':
                if rollout:
                    rid=local.workload.round_id
                    alloc={r:{f'trajectory.r{rid}.d{r}.{t.trajectory_id}':(len(prompt_for(local,t))+len(t.response_token_ids))*4 for t in source_local.workload.trajectories} for r in physical}
                    reads={physical[lr]:[sid(t['storage_id']) for t in p.tensors if t['rank']==lr and (t['kind'] in ['token','trajectory'] or t['kind']=='preallocated_arena')] for lr in range(local.hardware.dies)}
                    size=sum(alloc[physical[0]].values());self.control(rid,'generation',label+'.retain_trajectories',physical,reads=reads,alloc=alloc,
                        ready={r:list(alloc[r]) for r in physical},spec=KernelSpec('trajectory_pack',read_bytes=size,write_bytes=size))
                    for r in physical:bridges[r].extend(alloc[r])
                elif not training:
                    phase=next(iter(selected));count=sum(len(t.response_token_ids) for t in local.workload.trajectories) if phase in ['old_policy','reference'] else len(local.workload.trajectories)
                    kinds={'log_prob'} if phase in ['old_policy','reference'] else {phase}
                    reads={physical[lr]:[sid(t['storage_id']) for t in p.tensors if t['rank']==lr and (t['kind'] in kinds or t['kind']=='preallocated_arena')] for lr in range(local.hardware.dies)}
                    alloc={r:{f'state.{label}.d{r}.{phase}':count*4} for r in physical}
                    self.control(local.workload.round_id,phase,label+'.retain_outputs',physical,reads=reads,alloc=alloc,ready={r:list(alloc[r]) for r in physical},spec=KernelSpec('phase_output_pack',read_bytes=count*4,write_bytes=count*4))
                    for r in physical:bridges[r].extend(alloc[r])
            nodes={}
            for lr in op.participant_ranks:
                source_node=p.contract['ranks'][str(lr)]['nodes'][str(op.node_id)]
                # The four mutable ID containers are rebuilt immediately below.
                # Share only an immutable copy of the read-only workspace vector.
                n=dict(source_node)
                if 'core_workspace_bytes' in n:
                    vec=source_node['core_workspace_bytes'];key=id(vec)
                    if key not in self.workspace_vectors:self.workspace_vectors[key]=tuple(vec)
                    n['core_workspace_bytes']=self.workspace_vectors[key]
                for key in ['reads','ready_outputs','frees']:n[key]=[sid(s) for s in n[key]]
                n['allocations']={sid(s):v for s,v in n['allocations'].items()}
                if physical[lr] not in arena_initialized and arena[physical[lr]]:
                    n['allocations'].update(arena[physical[lr]]);n['reads']=[s for s in n['reads'] if s not in arena[physical[lr]]]
                    n['ready_outputs'].extend(arena[physical[lr]]);arena_initialized.add(physical[lr])
                if op.name=='release_round_temporaries':n['frees'].extend(arena[physical[lr]])
                if not rollout and op.name!='release_round_temporaries':
                    external=bridge_snapshot[physical[lr]]
                    self.bridge_read_counts['before']+=len(external)
                    consume=self.conservative_bridge_reads or lr not in first_input_seen or op.node_id in consumers
                    if consume:
                        n['reads']=sorted(set(n['reads']).union(external));self.bridge_read_counts['after']+=len(external)
                    first_input_seen.add(lr)
                nodes[physical[lr]]=n
            ps=tuple(physical[r] for r in op.participant_ranks);meta={**op.metadata,'stage_component':label,'tp_size':local.strategy.tensor_parallel,
                'colocation_stage':'train' if training and op.name!='release_round_temporaries' else 'cleanup' if training else 'generation' if rollout else next(iter(selected)),
                'source_to_canonical_trajectories':expansion['source_to_canonical_trajectories'],
                'actor_version':local.workload.behavior_version,'reference_version':self.case.workload.behavior_version}
            if op.kind=='collective':meta['comm_group']=self.group(ps)
            translated=replace(op,op_id=label+'.'+op.op_id,participant_ranks=ps,src_rank=None if op.src_rank is None else physical[op.src_rank],
                dst_rank=None if op.dst_rank is None else physical[op.dst_rank],metadata=meta)
            appended=self.append(translated,nodes)
            expansion['global_to_local'][appended.node_id]=op.node_id
        return cleanup
    def build(self):
        count=len(self.case.workload.trajectories)//self.dp
        if count%self.policy.responses_per_prompt:raise ValueError('selected contiguous dispatcher requires complete equal-size groups per DP')
        ts=self.case.workload.trajectories
        for i in range(self.dp):
            selected=ts[i*count:(i+1)*count];gids={group_for(self.case,t) for t in selected}
            if any(sum(group_for(self.case,t)==g for t in selected)!=self.policy.responses_per_prompt for g in gids):raise ValueError('input order/group cardinality mismatch')
        total_tokens=sum(len(t.response_token_ids) for t in ts)
        for off in range(self.case.rounds):
            rid=self.case.workload.round_id+off;version=self.case.workload.behavior_version+off;bridges={r:[] for r in range(self.world)}
            if self.active<self.world:self.control(rid,'standby','inactive_rank_marker',range(self.active,self.world),meta={'execution_domain':'device','idle_rank_marker':True})
            locals=[]
            for i in range(self.dp):
                local_ts=ts[i*count:(i+1)*count]
                local=replace(self.local_case,rounds=1,workload=replace(self.case.workload,round_id=rid,behavior_version=version,trajectories=local_ts),
                    strategy=replace(self.local_case.strategy,rollout_max_concurrency=count))
                locals.append(local)
                if off==0:self.placements.append({'replica':i,'tp_ranks':list(range(i*self.tr,(i+1)*self.tr)),
                    'groups':list(dict.fromkeys(group_for(local,t) for t in local_ts)),
                    'trajectories':[t.trajectory_id for t in local_ts],
                    'rollout_tp_groups':[list(range(i*self.tr+j*self.ge,i*self.tr+(j+1)*self.ge)) for j in range(self.tr//self.ge)],
                    'global_loss_denominator':total_tokens,'distributed_optimizer_shard':self.layout.shard_interval(i)})
            if self.layout.padding_elements:
                self.control(rid,'gradient_initialize','zero_gradient_padding',range(self.active),
                    ready={r:[f'd{r}.gradient.__padding'] for r in range(self.active)},
                    spec=KernelSpec('zero_gradient_padding',write_bytes=self.layout.padding_elements*4),
                    meta={'padding_elements':self.layout.padding_elements,'padding_value':0})
            self.fence(rid,'before_weight_sync')
            gen_case=replace(self.local_case,hardware=replace(self.local_case.hardware,dies=self.ge),strategy=replace(self.local_case.strategy,tensor_parallel=self.ge))
            gen_params={p.parameter_id:p for p in enumerate_parameters(gen_case)}
            for i in range(self.dp):
                ps=tuple(range(i*self.tr,(i+1)*self.tr));alloc={r:{f'rollout.r{rid}.d{r}.{p.parameter_id}':gen_params[p.parameter_id].local_elements_per_die*2 for p in self.params} for r in ps}
                self.control(rid,'switch_to_rollout',f'resume_rollout_{i}',ps,alloc=alloc,duration=1000,meta={'actor_version':version})
                for p in self.params:
                    reads={r:[f'd{r}.actor.{p.parameter_id}'] for r in ps};scratch={}
                    if not p.replicated and self.tr>1:
                        for r in ps:scratch[r]={f'export.r{rid}.d{r}.{p.parameter_id}':p.unique_elements*2}
                        self.control(rid,'weight_reshard',f'export_allocate_{i}_{p.parameter_id}',ps,alloc=scratch)
                        self.collective(rid,'weight_reshard',f'export_{i}_{p.parameter_id}',ps,p.local_elements_per_die*2,'ALL_GATHER',reads=reads,
                            ready={r:list(scratch[r]) for r in ps},source_bytes=p.local_elements_per_die*2,destination_bytes=p.unique_elements*2)
                    self.control(rid,'weight_reshard',f'load_rollout_{i}_{p.parameter_id}',ps,
                        reads={r:reads[r]+list(scratch.get(r,{})) for r in ps},ready={r:[f'rollout.r{rid}.d{r}.{p.parameter_id}'] for r in ps},
                        frees={r:list(scratch.get(r,{})) for r in ps},
                        spec=KernelSpec('export_pack_load',read_bytes=p.unique_elements*2,write_bytes=gen_params[p.parameter_id].local_elements_per_die*2),
                        meta={'source_train_TP':self.tr,'destination_rollout_TP':self.ge,'actor_version':version})
            self.fence(rid,'weights_ready')
            for i,local in enumerate(locals):
                micro=self.tr//self.ge
                for j in range(micro):
                    chosen=local.workload.trajectories[j::micro]
                    if not chosen:continue
                    physical=tuple(range(i*self.tr+j*self.ge,i*self.tr+(j+1)*self.ge))
                    gen=replace(local,hardware=replace(local.hardware,dies=self.ge),strategy=replace(local.strategy,tensor_parallel=self.ge,rollout_max_concurrency=len(chosen)),workload=replace(local.workload,trajectories=chosen))
                    self.stage(gen,physical,{'input','rollout'},f'r{rid}.generation{i}.{j}',rollout=True,bridges=bridges)
            self.fence(rid,'rollout_done')
            for i,local in enumerate(locals):
                ps=tuple(range(i*self.tr,(i+1)*self.tr));micro=self.tr//self.ge
                for k,t in enumerate(local.workload.trajectories):
                    size=(len(prompt_for(local,t))+len(t.response_token_ids))*4
                    for r in ps:
                        if f'trajectory.r{rid}.d{r}.{t.trajectory_id}' in bridges[r]:continue
                        owner=i*self.tr+(k%micro)*self.ge+(r%self.ge)
                        sid=f'trajectory.r{rid}.d{r}.{t.trajectory_id}';src=f'trajectory.r{rid}.d{owner}.{t.trajectory_id}'
                        self.control(rid,'data_redistribution',f'allocate_samples_{r}_{t.trajectory_id}',(r,),alloc={r:{sid:size}})
                        self.control(rid,'data_redistribution',f'samples_read_{owner}_{r}_{t.trajectory_id}',(owner,),reads={owner:[src]},spec=KernelSpec('sample_transfer_read',read_bytes=size))
                        nodes={owner:self.blank(),r:self.blank()};nodes[owner]['reads']=[src]
                        self.append(Operation(0,f'r{rid}.samples.{r}.{t.trajectory_id}',rid,'data_redistribution','sample_redistribution','p2p',payload_bytes=size,
                            src_rank=owner,dst_rank=r,participant_ranks=(owner,r),metadata={'colocation_stage':'data_redistribution'}),nodes)
                        self.control(rid,'data_redistribution',f'samples_ready_{r}_{t.trajectory_id}',(r,),ready={r:[sid]},spec=KernelSpec('sample_transfer_endpoint',write_bytes=size));bridges[r].append(sid)
                self.control(rid,'switch_to_train',f'sleep_rollout_{i}',ps,
                    frees={r:[f'rollout.r{rid}.d{r}.{p.parameter_id}' for p in self.params] for r in ps},duration=1000,
                    meta={'training_shards_retained':True,'reverse_weight_copy_bytes':0})
            self.fence(rid,'train_layout_ready')
            for phase in ['reward','old_policy','reference','advantage']:
                for i,local in enumerate(locals):
                    mb=self.policy.logprob_micro_batch_trajectories if phase in ['old_policy','reference'] else len(local.workload.trajectories)
                    for start in range(0,len(local.workload.trajectories),mb):
                        part=replace(local,workload=replace(local.workload,trajectories=local.workload.trajectories[start:start+mb]))
                        self.stage(part,tuple(range(i*self.tr,(i+1)*self.tr)),{phase},f'r{rid}.{phase}{i}.{start}',bridges=bridges)
                self.fence(rid,phase+'_done')
            cleanups=[]
            for i,local in enumerate(locals):
                ps=tuple(range(i*self.tr,(i+1)*self.tr))
                mb=self.policy.train_micro_batch_trajectories
                for start in range(0,len(local.workload.trajectories),mb):
                    part=replace(local,workload=replace(local.workload,trajectories=local.workload.trajectories[start:start+mb]))
                    cleanups.extend(self.stage(part,ps,{'train'},f'r{rid}.training{i}.{start}',bridges=bridges,training=True,reset_gradient=start==0))
            self.fence(rid,'backward_done')
            for r in range(self.active):
                reads=self.roles(r,'gradient')+([f'd{r}.gradient.__padding'] if self.layout.padding_elements else [])
                self.control(rid,'dp_sync',f'pre_RS_scale_{r}',(r,),reads={r:reads},ready={r:reads},
                    spec=KernelSpec('pre_RS_gradient_scale',vector_ops=self.layout.padded_elements,
                        read_bytes=self.layout.padded_elements*4,write_bytes=self.layout.padded_elements*4),
                    meta={'scale_numerator':1,'scale_denominator':self.dp,'local_loss_numerator_scale':self.dp,'global_token_denominator':total_tokens})
            for shard in range(self.tr):
                ps=tuple(range(shard,self.active,self.tr))
                self.collective(rid,'dp_sync',f'do_gradient_RS_{shard}',ps,self.layout.padded_elements*4,'REDUCE_SCATTER',
                    reads={r:self.roles(r,'gradient') for r in ps},ready={r:self.roles(r,'gradient',owned=True) for r in ps},
                    source_bytes=self.layout.padded_elements*4,destination_bytes=self.layout.shard_elements*4,
                    meta={'distributed_optimizer':True,'reduction':'SUM','gradient_version':version,'main_grad_shard_is_view':True,'native_comm_size_semantics':'full_input_bytes_per_rank'})
                self.control(rid,'dp_sync',f'do_RS_local_sum_{shard}',ps,
                    spec=KernelSpec('RS_reduction',vector_ops=self.layout.shard_elements*(self.dp-1),read_bytes=2*self.layout.shard_elements*(self.dp-1)*4,write_bytes=self.layout.shard_elements*(self.dp-1)*4))
            self.fence(rid,'gradient_sum_done')
            norm_buffers={}
            for r in range(self.active):
                work=self.layout.work(r//self.tr,tp_rank=r%self.tr);sid=f'norm.r{rid}.d{r}';width=self.active*4;norm_buffers[r]=sid
                self.control(rid,'clip',f'do_gradient_square_sum_{r}',(r,),reads={r:self.roles(r,'gradient',owned=True,norm=True)},
                    alloc={r:{sid:width}},ready={r:[sid]},
                    spec=KernelSpec('unique_shard_gradient_norm',vector_ops=work['gradient_norm_unique_elements']*2+width//4,
                        read_bytes=work['gradient_norm_read_bytes'],write_bytes=width),meta={'padding_and_TP_duplicate_norm_excluded':True,'scalar_slot':0,'other_slots_zero_initialized':True,'owned_intersections':self.layout.intersections(r//self.tr,tp_rank=r%self.tr,norm_only=True)})
            ps=tuple(range(self.active));self.collective(rid,'clip','global_gradient_norm_SUM',ps,self.active*4,'ALL_REDUCE',
                reads={r:[norm_buffers[r]] for r in ps},ready={r:[norm_buffers[r]] for r in ps},
                meta={'scalar_payload_bytes':4,'other_padding_values_zero':True})
            for r in range(self.active):
                work=self.layout.work(r//self.tr,tp_rank=r%self.tr)
                grads=self.roles(r,'gradient',owned=True)
                self.control(rid,'clip',f'clip_owned_gradient_{r}',(r,),reads={r:grads+[norm_buffers[r]]},ready={r:grads},
                    spec=KernelSpec('clip_owned_gradient',vector_ops=work['gradient_clip_elements']+4,
                        read_bytes=work['gradient_clip_read_bytes']+4,write_bytes=work['gradient_clip_write_bytes']),meta={'max_grad_norm':self.policy.max_grad_norm,'read_global_norm_slot':0,'owned_intersections':self.layout.intersections(r//self.tr)})
                reads=grads+sum((self.roles(r,role,owned=True) for role in ['master','adam_m','adam_v']),[])
                writes=self.roles(r,'actor',owned=True)+sum((self.roles(r,role,owned=True) for role in ['master','adam_m','adam_v']),[])
                self.control(rid,'optimizer',f'Adam_owned_shard_{r}',(r,),reads={r:reads},ready={r:writes},
                    spec=KernelSpec('DO_AdamW',vector_ops=work['adam_vector_ops'],read_bytes=work['adam_read_bytes'],write_bytes=work['adam_write_bytes_including_BF16_actor_shard']),
                    meta={'actor_version_in':version,'actor_version_out':version+1,'padding_not_updated':True,'owned_valid_parameter_elements':work['adam_valid_elements'],'owned_intersections':self.layout.intersections(r//self.tr),'native_hook_scope':'parameter_storage_identity; interval semantics audited in metadata, not per-element native state'})
            for shard in range(self.tr):
                ps=tuple(range(shard,self.active,self.tr))
                self.collective(rid,'optimizer_sync',f'do_parameter_AG_{shard}',ps,self.layout.shard_elements*2,'ALL_GATHER',
                    reads={r:self.roles(r,'actor',owned=True) for r in ps},ready={r:self.roles(r,'actor') for r in ps},
                    source_bytes=self.layout.shard_elements*2,destination_bytes=self.layout.padded_elements*2,
                    meta={'parameter_version_in':version,'parameter_version_out':version+1,'in_place_parameter_buffer':True,'native_comm_size_semantics':'input_shard_bytes_per_rank','full_output_bytes_per_rank':self.layout.padded_elements*2})
            self.fence(rid,'parameters_ready')
            publication=validate_publication(actor_version=version,gradient_version=version,reference_version=self.case.workload.behavior_version,
                completed_parameter_all_gather_tp_groups=range(self.tr),tp_size=self.tr)
            self.control(rid,'publish','publish_actor_version',range(self.active),reads={r:self.roles(r,'actor') for r in range(self.active)},meta=publication)
            for op,nodes in cleanups:self.append(op,nodes)
            self.control(rid,'cleanup','release_cross_stage_outputs',range(self.active),frees={r:bridges[r]+[norm_buffers[r]] for r in range(self.active)})
            self.fence(rid,'update_published')
        index={t['storage_id']:t for t in self.tensors}
        for rank,c in self.contract['ranks'].items():
            for node,n in c['nodes'].items():
                for sid,size in n['allocations'].items():
                    if sid not in index:index[sid]=dict(storage_id=sid,tensor_id=sid,rank=int(rank),kind='runtime_buffer',shape=[size],dtype_bytes=1,nbytes=size,persistent=False,round_id=None,released_node=None)
                    index[sid].update(created_node=int(node),ready_node=int(node))
                for sid in n['ready_outputs']:
                    if sid in index and not index[sid]['persistent']:index[sid]['ready_node']=int(node)
                for sid in n['frees']:
                    if sid in index:index[sid]['released_node']=int(node)
        # External carriers are immutable between their producing callback and
        # the final cleanup. Omitting redundant active-reader declarations is
        # timing-preserving only under this checked lifetime and rank seriality.
        carrier_records=audit_immutable_carriers(self.contract,self.bridge_carriers,cleanup_nodes={o.node_id for o in self.ops if o.name=='release_cross_stage_outputs'})
        self.checks.append(CheckResult('aligned.external_inputs_immutable_read_pruning',True,
            'one allocation followed by one READY, no subsequent writes/realloc, one final free; each rank serial and first consumption checked',
            {'carriers':len(carrier_records),'read_permission_counts':self.bridge_read_counts},
            'only redundant external carrier read permissions omitted; original local read sets/cost/alloc/READY/free unchanged','IR_equivalence'))
        plan=TensorPlan(self.contract,list(index.values()),self.ops,preflight_contract(self.contract))
        graph=ColocatedGraph(plan,self.groups,self.placements,{'reference':'verl_selected_PP1_DO_no_overlap','operator_contract_id':'qwen2_PP1_GRPO_DO_last_row_sampling_full_teacher_logits_v1',
            'train_tp':self.tr,'rollout_tp':self.ge,'train_dp':self.dp,'rollout_dp':self.active//self.ge,'active_devices':self.active,
            'allocated_devices':self.world,'idle_devices':list(range(self.active,self.world)),'global_barriers':self.barriers,
            'policy_fingerprint':__import__('rl_xy.baseline_contract',fromlist=['policy_fingerprint']).policy_fingerprint(self.policy),
            'distributed_optimizer':self.layout.contract(),'global_loss_scaling':global_loss_scaling(self.dp,total_tokens),
            'reference_enabled':True,'hardware_calibrated':False,'full_verl_runtime_reproduced':False,
            'single_rank_kernel_fusion_and_registered_parameter_order_not_claimed':True,
            'generation_prefill_logits':'one selected last hidden row per independent request; explicit BF16 selection cost',
            'external_input_read_policy':'all_nodes' if self.conservative_bridge_reads else 'per_rank_stage_entry_and_projected_logical_consumers',
            'external_input_read_permission_counts':self.bridge_read_counts,
            'training_template_scope':'one real microbatch; gradient zero once per DP per update; global advantages/denominator retained',
            'component_template_count':len(self.component_cache),'component_invocation_count':len(self.component_expansions),
            'component_contraction_certificates':{x['label']:x['contraction'].certificate for x in self.component_expansions}},self.checks)
        graph.component_expansions=self.component_expansions
        return graph


def build_aligned_graph(case,allocated,policy,provider,log_prefix,*,active=48,compact_components=True,conservative_bridge_reads=False):
    return AlignedGraphBuilder(case,allocated,policy,provider,log_prefix,active=active,compact_components=compact_components,conservative_bridge_reads=conservative_bridge_reads).build()
