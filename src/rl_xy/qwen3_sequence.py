"""Compact repeated programs; all child network events execute in native ASTRA."""
from __future__ import annotations
import json
from pathlib import Path
from .qwen3_cost import Cost
from .qwen3_native import Plan,execute,NetworkProfiles,ceildiv
from .qwen3_rollout import Rollout
from .communication_service import protocol_overhead_ns


def local(cost,delta=None,ranks=None):
    x={'kind':'local','compute':cost.compute,'memory':cost.memory}
    if delta:x.update(compute_delta=delta.compute,memory_delta=delta.memory)
    if ranks is not None:x['ranks']=list(ranks)
    return x


def compile_blocks(op,ranks,blocks,next_blocks=None):
    steps=[];n=len(ranks)
    for i,b in enumerate(blocks):
        delta=None
        if next_blocks:
            nb=next_blocks[i];delta=Cost(nb.cost.compute-b.cost.compute,nb.cost.memory-b.cost.memory)
            if (nb.collective,nb.payload)!=(b.collective,b.payload):raise ValueError('nonaffine communication')
        steps.append(local(b.cost,delta))
        if not b.collective or n==1:continue
        payload=max(4*n,ceildiv(int(b.payload),4*n)*4*n)
        overhead=protocol_overhead_ns(op,n,b.collective,payload)
        if overhead:steps.append(local(Cost(compute=overhead)))
        if b.collective in ('GATHER','BCAST'):
            pairs=[(r,ranks[0]) for r in ranks[1:]] if b.collective=='GATHER' else [(ranks[0],r) for r in ranks[1:]]
            for src,dst in pairs:
                steps.append(local(op.vec(0,payload,0,'network_read'),ranks=[src]))
                steps.append({'kind':'p2p','src':src,'dst':dst,'bytes':payload})
                steps.append(local(op.vec(0,0,payload,'network_write'),ranks=[dst]))
        elif b.collective=='ALL_REDUCE':
            steps.append(local(op.vec(0,payload,0,'network_read')))
            steps.append({'kind':'AR','bytes':payload})
            reduce=payload*(n-1)//n//4
            steps.append(local(op.vec(reduce,reduce*8,payload+reduce*4,'network_write_reduce')))
        else:raise ValueError('unsupported compact collective '+b.collective)
    return steps


def attach(plan,programs):
    seq={'schema':'rlxy.native_sequence/1','programs':{},'ranks':{}}
    for ranks,segments in programs:
        ranks=tuple(ranks)
        if ranks not in plan.groupids:
            gid=str(len(plan.groups)+1);plan.groups[gid]=list(ranks);plan.groupids[ranks]=gid
        gid=plan.groupids[ranks];key=gid;seq['programs'][key]={'group':int(gid),'segments':segments}
        for rank in ranks:seq['ranks'][str(rank)]=key
        plan.local(ranks,Cost(compute=1),'qwen3_sequence_'+gid)
    return seq


def generation_program(op,requests,tp,cap):
    # The FIFO program order depends only on assigned requests and recorded EOS.
    # Search services determine no scheduling decision; native clocks replace them.
    replay=Rollout(op,NetworkProfiles(op,'unused',native=False)).run(requests,tp,cap,records=True)
    byid={r['trajectory_id']:r for r in requests};programs=[]
    for i,r in enumerate(replay['replicas']):
        ranks=tuple(range(i*tp,(i+1)*tp));segments=[]
        for period in r['periods']:
            if period['kind']=='prefill':
                lengths=tuple(byid[k]['prompt_length'] for k in period['requests']);blocks=op.forward(lengths,tp,'generation');n=1;next_blocks=None
            else:
                n=period['iterations'];batch=period['batch'];q,extra=divmod(period['sum_context_start'],batch)
                ctx=tuple(q+(j<extra) for j in range(batch));blocks=op.forward((1,)*batch,tp,'generation',ctx)
                next_blocks=op.forward((1,)*batch,tp,'generation',tuple(x+1 for x in ctx))
            segments.append({'repeat':n,'steps':compile_blocks(op,ranks,blocks,next_blocks)})
        programs.append((ranks,segments))
    plan=Plan(op);seq=attach(plan,programs)
    return plan,seq,replay


def run_generation(op,requests,tp,cap,output):
    plan,seq,replay=generation_program(op,requests,tp,cap);r=execute(plan,output,sequence=seq)
    output=Path(output)
    for i,replica in enumerate(replay['replicas']):
        group=tuple(range(i*tp,(i+1)*tp));root=group[0]
        rows=[json.loads(x) for x in (output/f'sequence.{root}.jsonl').read_text().splitlines()]
        finish={}
        for j,p in enumerate(replica['periods']):
            p['native_end_ns']=rows[j]['end_tick']/1000
            for k in p['completed']:finish[k]=p['native_end_ns']
        if len(finish)!=len(replica['completion_ns']):raise ValueError('native request completion coverage')
        replica['search_estimate_ns']=replica['model_ns'];replica['completion_ns']=finish
        replica['model_ns']=max(r['per_rank_completion_ns'][str(rr)] if str(rr) in r['per_rank_completion_ns'] else r['per_rank_completion_ns'][rr] for rr in group)
        sums=[r['native_sequence_summaries'][rr] for rr in group]
        slow=max(sums,key=lambda x:x['end_tick']);replica['compute_ns']=slow['compute_ns'];replica['memory_ns']=slow['memory_ns'];replica['communication_service_ns']=slow['communication_ns']
    replay['model_ns']=r['model_ns'];replay['barrier_wait_ns_by_replica']=[r['model_ns']-x['model_ns'] for x in replay['replicas']]
    replay['mean_replica_barrier_wait_fraction']=sum(replay['barrier_wait_ns_by_replica'])/(len(replay['replicas'])*r['model_ns'])
    replay['execution']='native per-substep event/network execution from compact FIFO programs, including shared-route contention'
    (output/'generation.json').write_text(json.dumps(replay,indent=2)+'\n')
    return replay,r


def teacher_program(op,p,requests,training=False):
    from .qwen3_experiment import training_groups,chunks,teacher_lengths
    programs=[];tp=p.train_tp;layout=op.state(tp,p.rollout_tp)['layout']
    for i,group in enumerate(training_groups(requests,tp)):
        ranks=tuple(range(i*tp,(i+1)*tp));segments=[]
        if training:segments.append({'repeat':1,'steps':[local(op.vec(0,0,layout.padded_elements*4,'gradient_zero'))]})
        for batch in chunks(group,p.train_mb if training else p.logprob_mb):
            ls=teacher_lengths(batch);blocks=op.training(ls,tp,sum(x['response_length'] for x in batch)) if training else op.forward(ls,tp)
            segments.append({'repeat':1,'steps':compile_blocks(op,ranks,blocks)})
        programs.append((ranks,segments))
    plan=Plan(op);seq=attach(plan,programs);return plan,seq
