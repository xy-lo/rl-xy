#!/usr/bin/env python3
"""Attribute inclusive local service on the last backward executor.

GEMM FIFO service already contains computation and memory. These categories
are operator families, not physical compute-vs-bandwidth utilization estimates.
Every attributed body must match the immutable body actually issued natively.
"""
import argparse,gzip,hashlib,json,sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/idea1_layered/legacy'))
sys.path.append(str(ROOT/'src'))
from rl_xy.f2_runtime import make_operator
from rl_xy.qwen3_experiment import teacher_lengths
from rl_xy.qwen3_sequence import compile_blocks
from idea1_controller.history import digest


def load(p):return json.loads(p.read_text())
def normalized(steps):
    return [{k:v for k,v in s.items() if k!='group'} for s in steps]
def family(name):
    name=name.lower()
    if 'ffn' in name:return 'FFN'
    if 'attention' in name or 'attn' in name:return 'attention'
    if any(s in name for s in ('qkv','q_proj','k_proj','v_proj','o_proj')):return 'attention_projection'
    if any(s in name for s in ('logit','lm_head','vocab')):return 'output_head'
    if 'norm' in name:return 'normalization'
    return 'other_vector_or_embedding'


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fast-cost-validation',type=Path);p.add_argument('--cache-dir',type=Path)
    a=p.parse_args();folder=a.run.resolve();folder=folder/'native-run' if (folder/'native-run').is_dir() else folder
    result=load(folder/'result.json');model=result['model'].replace('Qwen3-','');cmd=load(folder/'command.json')
    trace=load(Path(cmd['trace_path']))['trajectories'];items={x['trajectory_id']:x for x in trace}
    events=load(folder/'training-events.json');executions=load(folder/'training-batch-executions.json')
    last=max(executions,key=lambda x:x['end_tick']);worker=last['worker']
    layout=load(folder/'resource-layout.json');regions={int(r):tuple(ws) for r,ws in layout['region_members'].items()}
    region=next(r for r,ws in regions.items() if worker in ws);tp=2*len(regions[region]);initial_training_tp=tp
    op=make_operator(model,'wafer',cache_dir=a.cache_dir or a.output.parent/'profile-cost-cache')
    if a.fast_cost_validation:
        from idea1_controller.cost_backend import install
        install(op,a.output.parent/'profile-cost-accelerator',load(a.fast_cost_validation))
    from gqa_cost import apply_generation_gqa
    apply_generation_gqa(op)
    compiled={int(f.stem.split('.')[0]):json.load(gzip.open(f,'rt')) for f in (folder/'compiled-regions').glob('*.json.gz')}
    jobs={j[3]['batch_id']:j for xs in compiled.values() for j in xs if j[0]=='training'}
    portable_file=folder/'portable-training-programs.json.gz';portable=json.load(gzip.open(portable_file,'rt')) if portable_file.exists() else {}
    totals=defaultdict(float);checked=[];collective_payload=defaultdict(int)
    def attribute(stage,batch,logical_tp,expected):
        blocks=op.training(teacher_lengths(batch),logical_tp,sum(x['response_length'] for x in batch)) if stage=='training' else op.forward(teacher_lengths(batch),logical_tp)
        body=compile_blocks(op,tuple(range(logical_tp)),blocks)
        assert normalized(body)==normalized(expected),'profile body differs from executed body'
        collective_payload[stage]+=sum(s['bytes'] for s in body if s['kind']=='AR')
        for block in blocks:totals[stage,family(block.name)]+=block.cost.ns
        extra=sum(s.get('compute',0)+s.get('memory',0) for s in body if s['kind']=='local')-sum(b.cost.ns for b in blocks)
        assert extra>=-1e-3
        totals[stage,'collective_endpoint_service']+=max(0,extra)
        checked.append(digest(dict(repeat=1,steps=normalized(body))))
    for stage,tids,steps,job in compiled[region]:
        if stage not in ('old_policy','reference'):continue
        expected=steps[2:-2] # synchronization begin/end wrappers
        attribute(stage,[items[t] for t in tids],tp,expected)
    for e in executions:
        if e['worker']==e['home'] and e['home']==region:
            job=jobs[e['batch_id']]
            attribute('training',[items[t] for t in e['tids']],e['training_tp'],job[3]['steps'])
        elif e['worker']==worker and e['home']!=region:
            job=portable[e['batch_id']];offset=0
            for sub in job['subbatches']:
                batch=[items[t] for t in sub]
                body=compile_blocks(op,(0,1),op.training(teacher_lengths(batch),2,sum(x['response_length'] for x in batch)))
                expected=job['steps'][offset:offset+len(body)];attribute('training',batch,2,expected);offset+=len(body)
            assert offset==len(job['steps'])
    rows=[dict(stage=s,operator_family=f,inclusive_local_service_s=ns/1e9) for (s,f),ns in sorted(totals.items())]
    sums=defaultdict(float)
    for r in rows:sums[r['stage']]+=r['inclusive_local_service_s']
    for r in rows:r['within_stage_fraction']=r['inclusive_local_service_s']/sums[r['stage']]
    core_rows=[]
    inf=load(folder/'inference-layout.json') if (folder/'inference-layout.json').exists() else None
    if inf:
        caps={}
        for r,ws in inf['groups'].items():
            tp=1 if inf.get('paired_tp1') else 2*len(ws);caps[tp]=max(caps.get(tp,0),inf['caps'][str(r)])
    else:caps={tp:layout['rollout_cap'] for tp in (2,4)}
    for logical_tp,cap in sorted(caps.items()):
        for batch in sorted({1,min(8,cap),min(16,cap),cap}):
            cost=op.attention((1,)*batch,(8192,)*batch,logical_tp)
            active=min(op.units,batch*(op.model.kv_heads//logical_tp))
            core_rows.append(dict(batch=batch,tp=logical_tp,context=8192,modeled_active_KV_cores_per_die=active,
                modeled_fraction=active/op.units,all_layers_attention_compute_ms=cost.compute*op.model.layers/1e6,
                all_layers_attention_memory_ms=cost.memory*op.model.layers/1e6,
                scope='conditional grouped-core operator formula; not observed hardware utilization'))
    report=dict(status='PASS_EXECUTED_BODY_PROFILE',model=model,window=result['window'],last_backward_worker=worker,
        initial_training_region=region,initial_tp=initial_training_tp,checked_bodies=len(checked),body_digest_set=sorted(set(checked)),rows=rows,
        decode_core_model=core_rows,
        collective_logical_payload_bytes_per_rank_by_stage=dict(collective_payload),
        scope='one physical TP2 pair on last backward executor; initial TP gang work counted once per rank; host oracle only, not extra device work',
        limitation='inclusive GEMM service contains FIFO memory; no pure arithmetic/bandwidth attribution or hardware claim')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('rows','body_digest_set','decode_core_model')}))


if __name__=='__main__':main()
