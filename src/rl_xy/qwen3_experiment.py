"""Frozen Qwen3 baseline orchestration: screen, native phases, paired evidence."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from functools import lru_cache
from collections import defaultdict
import gzip,hashlib,json,math,time
from pathlib import Path
from .qwen3_cost import Operators,Cost,Block
from .qwen3_native import Plan,execute,NetworkProfiles,generation_links_disjoint
from .qwen3_rollout import Rollout
from .model import enumerate_parameters


@dataclass(frozen=True)
class Policy:
    train_tp:int
    rollout_tp:int
    train_mb:int
    logprob_mb:int
    cap:int


def chunks(items,n):return [items[i:i+n] for i in range(0,len(items),n)]
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')


def training_groups(requests,tp):
    dp=48//tp;count=len(requests)//dp
    assert len(requests)%dp==0 and count%8==0
    return chunks(requests,count)


def teacher_lengths(xs):return tuple(x['prompt_length']+x['response_length']-1 for x in xs)


def weight_plan(op,p):
    plan=Plan(op);plan.local(tuple(range(48)),Cost(compute=1000),'switch_to_rollout');tr=p.train_tp;ge=p.rollout_tp;rows=enumerate_parameters(op.case_tp(tr));dest={x.parameter_id:x for x in enumerate_parameters(op.case_tp(ge))}
    # Every training replica holds the same parameter version. Per-tensor full
    # export allows each rank to load its target rollout shard even for ge>tr.
    for group in range(48//tr):
        ranks=tuple(range(group*tr,(group+1)*tr))
        for row in rows:
            if not row.replicated and tr>1:plan.comm(ranks,'ALL_GATHER',row.local_elements_per_die*2,'weight_export.'+row.parameter_id)
            plan.local(ranks,op.vec(0,row.unique_elements*2,dest[row.parameter_id].local_elements_per_die*2,'export_pack_load'),'pack.'+row.parameter_id)
    return plan


def samples_plan(op,p,requests):
    plan=Plan(op);plan.local(tuple(range(48)),Cost(compute=1000),'switch_to_training');owners={r['trajectory_id']:tuple(range((i%(48//p.rollout_tp))*p.rollout_tp,(i%(48//p.rollout_tp)+1)*p.rollout_tp)) for i,r in enumerate(requests)}
    pairs=defaultdict(int)
    for i,group in enumerate(training_groups(requests,p.train_tp)):
        for r in group:
            srcs=owners[r['trajectory_id']]
            for dst in range(i*p.train_tp,(i+1)*p.train_tp):
                if dst not in srcs:pairs[srcs[dst%len(srcs)],dst]+=(r['prompt_length']+r['response_length'])*4
    for (src,dst),size in sorted(pairs.items()):
        plan.local((src,),op.vec(0,size,0,'sample_read'),'sample_read')
        plan.append(__import__('rl_xy.types',fromlist=['Operation']).Operation(0,'samples',1,'samples','samples','p2p',payload_bytes=size,src_rank=src,dst_rank=dst,participant_ranks=(src,dst)))
        plan.local((dst,),op.vec(0,0,size,'sample_write'),'sample_write')
    return plan


def teacher_plan(op,p,requests,training=False):
    plan=Plan(op);tp=p.train_tp;mb=p.train_mb if training else p.logprob_mb
    layout=op.state(tp,p.rollout_tp)['layout']
    for i,group in enumerate(training_groups(requests,tp)):
        ranks=tuple(range(i*tp,(i+1)*tp))
        if training:plan.local(ranks,op.vec(0,0,layout.padded_elements*4,'gradient_zero'),'gradient_zero')
        for j,batch in enumerate(chunks(group,mb)):
            lengths=teacher_lengths(batch)
            blocks=op.training(lengths,tp,sum(x['response_length'] for x in batch)) if training else op.forward(lengths,tp)
            plan.blocks(ranks,blocks,f'{"train" if training else "logprob"}.dp{i}.mb{j}')
    return plan


def advantage_plan(op,p,requests):
    plan=Plan(op)
    for i,group in enumerate(training_groups(requests,p.train_tp)):
        n=len(group);g=n//8
        plan.local(tuple(range(i*p.train_tp,(i+1)*p.train_tp)),op.vec(5*n+4*g,8*n,4*n,'grpo_group_advantage'),'grpo_group_advantage')
    return plan


def optimizer_plan(op,p):
    plan=Plan(op);tp=p.train_tp;dp=48//tp;layout=op.state(tp,p.rollout_tp)['layout']
    # Q/K norm scales are replicated across TP, but their head contributions
    # are partitioned. Merge them once after all accumulated microbatches.
    if tp>1:
        qk_bytes=op.model.layers*2*op.model.head_dim*4
        for group in range(dp):plan.comm(tuple(range(group*tp,(group+1)*tp)),'ALL_REDUCE',qk_bytes,'qk_norm_parameter_grad_TP')
    for rank in range(48):
        plan.local((rank,),op.vec(layout.padded_elements,layout.padded_elements*4,layout.padded_elements*4,'gradient_scale'),'gradient_scale')
    for shard in range(tp):plan.comm(tuple(range(shard,48,tp)),'REDUCE_SCATTER',layout.padded_elements*4,'dp_gradient_RS')
    for rank in range(48):
        n=layout.shard_valid_elements(rank//tp,tp_rank=rank%tp,norm_only=True)
        plan.local((rank,),op.vec(n*2,n*4,4,'gradient_norm'),'gradient_norm')
    plan.comm(tuple(range(48)),'ALL_REDUCE',48*4,'gradient_global_norm')
    for rank in range(48):
        n=layout.shard_valid_elements(rank//tp)
        plan.local((rank,),op.vec(n*3,n*4,n*4,'clip'),'clip')
        plan.local((rank,),op.vec(n*12,n*16,n*14,'adamw'),'adamw')
    for shard in range(tp):plan.comm(tuple(range(shard,48,tp)),'ALL_GATHER',layout.shard_elements*2,'dp_parameter_AG')
    return plan


def simple_plan_time(plan,network):
    # Search-only stage estimate. Native backend resolves contention for finalists.
    ready={rank:0. for rank in range(plan.world)}
    for o in plan.operations:
        ranks=o.participant_ranks;start=max(ready[r] for r in ranks)
        if o.kind=='compute':duration=o.duration_ns
        elif o.kind=='collective':duration=network.time(len(ranks),o.collective,o.payload_bytes)
        else:
            hw=plan.op.hw;prof=plan.op.profile
            if prof:
                different=o.src_rank//8!=o.dst_rank//8
                bw=prof['node_uplink_bytes_per_ns_per_direction'] if different else prof['nvlink_bytes_per_ns_per_direction'];lat=prof['uplink_latency_ns'] if different else prof['nvlink_latency_ns']*2
            else:
                x,y=o.src_rank%6,o.src_rank//6;xx,yy=o.dst_rank%6,o.dst_rank//6
                bw=hw.d2d_bytes_per_ns_per_direction;lat=(abs(x-xx)+abs(y-yy))*hw.d2d_latency_ns
            duration=lat+o.payload_bytes/(bw*plan.op.network_scale)
        for r in ranks:ready[r]=start+duration
    return max(ready.values())


class Screen:
    def __init__(self,op,requests):
        self.op=op;self.requests=requests;self.net=NetworkProfiles(op,'unused',native=False);self.roll=Rollout(op,self.net)
    @lru_cache(maxsize=64)
    def generation(self,tp,cap):return self.roll.run(self.requests,tp,cap)['model_ns']
    @lru_cache(maxsize=64)
    def teacher(self,tp,mb,training):
        totals=[]
        for group in training_groups(self.requests,tp):
            total=0.
            for batch in chunks(group,mb):
                ls=teacher_lengths(batch);blocks=self.op.training(ls,tp,sum(x['response_length'] for x in batch)) if training else self.op.forward(ls,tp)
                total+=self.net.blocks(tp,blocks)
            totals.append(total)
        return max(totals)
    @lru_cache(maxsize=32)
    def fixed(self,tr,ge):
        p=Policy(tr,ge,1,1,4)
        return {'weight_sync':simple_plan_time(weight_plan(self.op,p),self.net),
                'samples':simple_plan_time(samples_plan(self.op,p,self.requests),self.net),
                'optimizer':simple_plan_time(optimizer_plan(self.op,p),self.net)}
    def run(self,output):
        out=Path(output);out.mkdir(parents=True,exist_ok=True);start=time.monotonic();rows=[];rejected=[]
        for tr in (1,2,4,8):
            local=len(self.requests)//(48//tr)
            for ge in (1,2,4,8):
                assigned=len(self.requests)//(48//ge)
                for mb in (1,2,4,8,16,32,64):
                    if mb>local:continue
                    for lb in (1,2,4,8,16,32,64):
                        if lb>local:continue
                        for cap in (4,8,16,32,64):
                            if cap>assigned:continue
                            p=Policy(tr,ge,mb,lb,cap);a=self.op.admission(self.requests,tr,ge,mb,lb,cap)
                            if not a['passed']:
                                rejected.append({'policy':asdict(p),'reasons':a['reasons']});continue
                            fixed=self.fixed(tr,ge)
                            parts={**fixed,'generation':self.generation(ge,cap),'old_policy':self.teacher(tr,lb,False),'reference':self.teacher(tr,lb,False),'training':self.teacher(tr,mb,True),'advantage':simple_plan_time(advantage_plan(self.op,p,self.requests),self.net),'publish':2.}
                            rows.append({'policy':asdict(p),'model_ns':sum(parts.values()),'stage_ns':parts,'memory_peak_bound_bytes':a['peak_bytes']})
                write(out/'progress.json',{'train_tp':tr,'rollout_tp':ge,'feasible':len(rows),'rejected':len(rejected),'wall_seconds':time.monotonic()-start})
                print(json.dumps({'screen':self.op.info['config']['model_type'],'tr':tr,'ge':ge,'feasible':len(rows),'wall_s':round(time.monotonic()-start,1)}),flush=True)
        rows.sort(key=lambda x:(x['model_ns'],x['memory_peak_bound_bytes'],tuple(x['policy'].values())))
        write(out/'candidates.json',{'status':'PASS','backend':'search-only analytical services; finalists require native ASTRA','feasible_count':len(rows),'rejected_count':len(rejected),'candidates':rows,'wall_seconds':time.monotonic()-start})
        write(out/'rejected.json',rejected)
        return rows


def run_candidate(op,requests,policy,output,*,records=True):
    from .qwen3_source import manifest,sha
    out=Path(output);out.mkdir(parents=True,exist_ok=False);write(out/'policy.json',asdict(policy))
    version=manifest(op.root);write(out/'source_manifest.json',version)
    input_hash=__import__('hashlib').sha256(json.dumps(requests,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    write(out/'run_inputs.json',{'source_tree_sha256':version['tree_sha256'],'trajectory_sha256':input_hash,'platform':op.platform,'compute_scale':op.compute_scale,'network_scale':op.network_scale,'model_config_sha256':op.info['config_sha256'],'global_prompt_cap':op.prompt_cap,'training_memory_policy':'actual fixed-order batch token counts after generation; no response-length oracle in generation admission or ordering'})
    admission=op.admission(requests,policy.train_tp,policy.rollout_tp,policy.train_mb,policy.logprob_mb,policy.cap);write(out/'admission.json',admission)
    if not admission['passed']:raise ValueError('capacity admission failed')
    from .qwen3_sequence import run_generation,teacher_program
    start=time.monotonic();stages={};proof={}
    r=execute(weight_plan(op,policy),out/'weight_sync');stages['weight_sync']=r['model_ns'];proof['weight_sync']=r
    # Immutable whole-generation proof reuse across tuning policies. Training
    # microbatch choices do not change generation once TP/cap/data are fixed.
    from .qwen3_source import sha
    import os
    from .qwen3_paths import native_binary,generation_cache
    binary=native_binary(op.root)
    cache_fields={'code':version['tree_sha256'],'binary':sha(binary),'config':op.info['config_sha256'],'platform':op.platform,'compute_scale':op.compute_scale,'network_scale':op.network_scale,'input':input_hash,'rollout_tp':policy.rollout_tp,'cap':policy.cap}
    cache_key=__import__('hashlib').sha256(json.dumps(cache_fields,sort_keys=True).encode()).hexdigest()
    cache_root=generation_cache(op.root)/cache_key;cache_root.mkdir(parents=True,exist_ok=True)
    complete=cache_root/'complete.json'
    reused=complete.exists()
    if reused:
        receipt=json.loads(complete.read_text());generation_path=cache_root/receipt['attempt']
        if receipt['fields']!=cache_fields:raise ValueError('generation cache identity mismatch')
        roll=json.loads((generation_path/'generation.json').read_text());r=json.loads((generation_path/'result.json').read_text())
        if sha(generation_path/'result.json')!=receipt['result_sha256'] or sha(generation_path/'generation.json')!=receipt['generation_sha256']:raise ValueError('generation cache evidence changed')
    else:
        attempt=1
        while (cache_root/f'attempt-{attempt}').exists():attempt+=1
        generation_path=cache_root/f'attempt-{attempt}'
        roll,r=run_generation(op,requests,policy.rollout_tp,policy.cap,generation_path)
        write(complete,{'fields':cache_fields,'attempt':generation_path.name,'result_sha256':sha(generation_path/'result.json'),'generation_sha256':sha(generation_path/'generation.json')})
    os.symlink(os.path.relpath(generation_path,out),out/'generation',target_is_directory=True)
    write(out/'generation_reuse.json',{'reused':reused,'cache_key':cache_key,'evidence_path':str(generation_path),'contract':'exact same source/binary/hardware/precision/data/order/TP/cap; no other phase overlaps generation'})
    stages['generation']=r['model_ns'];proof['generation']=r
    state=op.state(policy.train_tp,policy.rollout_tp)
    actual_gen_peak=state['training_base_max']+state['rollout_weights_per_rank']+max(x['actual_KV_peak_bytes'] for x in roll['replicas'])
    if actual_gen_peak>op.capacity:raise ValueError('runtime generation capacity violation')
    for name in ['samples','old_policy','reference','advantage','training','optimizer','publish']:
        if name=='samples':r=execute(samples_plan(op,policy,requests),out/name)
        elif name=='optimizer':r=execute(optimizer_plan(op,policy),out/name)
        elif name=='advantage':r=execute(advantage_plan(op,policy,requests),out/name)
        elif name=='publish':
            pp=Plan(op);pp.local(tuple(range(48)),Cost(compute=1),'publish_actor_v1_reference_v0');r=execute(pp,out/name)
        else:
            plan,program=teacher_program(op,policy,requests,training=name=='training')
            r=execute(plan,out/name,sequence=program)
        stages[name]=r['model_ns'];proof[name]=r
        write(out/'progress.json',{'stage_completed':name,'wall_seconds':time.monotonic()-start})
    duration=sum(stages.values());result={'status':'PASS','policy':asdict(policy),'platform':op.platform,'step_model_ns':duration,'stage_ns':stages,'admission':admission,'generation_mean_replica_wait_fraction':roll['mean_replica_barrier_wait_fraction'],'request_count':roll['request_count'],'wall_seconds':time.monotonic()-start,'execution':'native ASTRA per synchronous phase; compact repeated programs execute every collective/P2P on the native shared network; not one monolithic invocation','native_proof':{k:{kk:v[kk] for kk in ['model_ns','slowest_rank','native_nodes','native_rank_events','wall_seconds','binary_sha256','native_sequence_summaries','per_rank_costs']} for k,v in proof.items()},'source_tree_sha256':version['tree_sha256'],'trajectory_sha256':input_hash,'compute_scale':op.compute_scale,'network_scale':op.network_scale,'publication_status':'pending cross-platform consistency and sensitivity validation'}
    if manifest(op.root)['tree_sha256']!=version['tree_sha256']:raise ValueError('source changed during candidate run')
    if len({x['binary_sha256'] for x in proof.values()})!=1:raise ValueError('native binary changed during case')
    write(out/'result.json',result);return result
