"""Retrospective critical-path bounds; deliberately not a new simulation result."""
import json, csv, statistics, gzip, argparse
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text())
all_data=load(OUT/'all-data.json')
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();DEST=args.output.resolve();DEST.mkdir(parents=True,exist_ok=False)
records=[]
for model in ('4B','8B','14B'):
    folder=OUT/model
    audit=load(folder/'audit.json'); result=load(folder/'result.json'); events=json.loads(gzip.open(folder/'raw/training-events.json.gz','rt').read())
    requests=load(ROOT/'experiments/2026-10-29-qwen3/inputs'/('Qwen3-'+model)/'test1.json')['trajectories']
    owner={x['trajectory_id']:i%27 for i,x in enumerate(requests)}
    eos={}; rows=[]
    for line in gzip.open(folder/'raw/occupancy.jsonl.gz','rt'):
        r=json.loads(line); rows.append(r)
        for t in r['new_done']:eos[t]=r['tick']/1e12
    release=[0.]*9
    gen_idle=0.
    origin=audit['prefix_until_generation_s']; end=audit['generation_end_s']
    for i,r in enumerate(rows[:-1]):
        at=r['tick']/1e12; till=rows[i+1]['tick']/1e12
        for d in range(9):
            if any(role in ('own','helper') for role in r['roles'][3*d:3*d+3]):release[d]=max(release[d],till)
        gen_idle+=max(0,min(till,end)-max(at,origin))*2*r['roles'].count('idle')
    workers=[]
    for w in range(27):
        es=[e for e in events if e['worker']==w]
        ids={t for e in es if e['phase']=='old_policy' for t in e['tids']}
        service=sum(e['end_tick']-e['start_tick'] for e in es)/1e12
        start=min(e['start_tick'] for e in es)/1e12; finish=max(e['end_tick'] for e in es)/1e12
        ready=max(release[w//3],max(eos[t] for t in ids))
        workers.append(dict(worker=w,samples=len(ids),domain_release_s=release[w//3],data_eos_s=max(eos[t] for t in ids),
            actual_prepare_s=start,native_batch_service_s=service,between_batch_wait_s=finish-start-service,
            earliest_ready_fixed_generation_s=ready,finish_without_delivery_wait_s=ready+service,actual_finish_s=finish))
    suffix=result['total_s']-max(w['actual_finish_s'] for w in workers)
    bound=max(w['finish_without_delivery_wait_s'] for w in workers)+suffix
    crit=max(workers,key=lambda w:w['actual_finish_s']); cfcrit=max(workers,key=lambda w:w['finish_without_delivery_wait_s'])
    base=next(r for r in all_data['comparison'] if r['model']==model)
    post_idle=audit['role_integral_die_s']['idle']-gen_idle
    records.append(dict(model=model,actual_step_s=result['total_s'],gpu_s=base['gpu_s'],
        fixed_service_ready_bound_s=bound,possible_reduction_under_bound_s=result['total_s']-bound,
        bound_still_above_gpu_s=bound-base['gpu_s'],critical_worker=crit['worker'],counterfactual_critical_worker=cfcrit['worker'],
        critical_data_delivery_wait_s=crit['actual_prepare_s']-crit['earliest_ready_fixed_generation_s'],
        native_batch_service_mean_s=statistics.mean(w['native_batch_service_s'] for w in workers),
        native_batch_service_max_s=max(w['native_batch_service_s'] for w in workers),
        native_batch_service_median_s=statistics.median(w['native_batch_service_s'] for w in workers),
        post_generation_idle_die_s=post_idle,post_generation_share_of_idle=post_idle/audit['role_integral_die_s']['idle'],workers=workers))
report=dict(status='PASS_RETROSPECTIVE_ANALYSIS',scope='actual full test1 native events; frozen generation and per-batch service costs, remove delivery/queue waits only; no new native replay',
    constraints=['not an achieved speedup','generation unchanged despite hypothetical sample service','native batch durations fixed despite possible contention changes','worker assignment and batches fixed','full GRPO group, actor versions and final update retained'],models=records)
(DEST/'bottleneck-analysis.json').write_text(json.dumps(report,indent=2)+'\n')
flat=[{k:v for k,v in r.items() if k!='workers'} for r in records]
with (DEST/'bottleneck-bounds.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
print(json.dumps(flat,indent=2))
