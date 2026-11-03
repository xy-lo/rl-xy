#!/usr/bin/env python3
"""Remote development acceptance: independent calibration, 4B TUNE only.

Old test1/test2 prompt histories form a declared calibration proxy. test3/test4
remain reserved. No current TUNE answer lengths enter forecasting or search.
"""
import argparse, gzip, hashlib, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from idea1_controller.workload import WorkloadSpec, Observation, DistributionSnapshot, forecasts, digest
from idea1_controller.offline_candidates import freeze_candidates, select_precomputed
from search import Environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',default='4B',choices=['4B','8B','14B'])
    p.add_argument('--mode',default='one_scene',choices=['one_scene','shortlist6','full'])
    p.add_argument('--no-history',action='store_true');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();start=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    inputs=ROOT/f'experiments/2026-10-29-qwen3/inputs/Qwen3-{a.model}'
    assets=ROOT/'experiments/2026-11-03-idea1-elastic/assets'
    hpath=assets/f'history-{a.model}.json.gz'
    h=json.loads(gzip.decompress(hpath.read_bytes()))
    canonical=json.loads((inputs/'prompt_lengths.json').read_text())
    def ids(window):
        return sorted({str(x['group_id']) for x in json.loads((inputs/(window+'.json')).read_text())['trajectories']})
    cal_raw=set(ids('test1')+ids('test2')); tune_ids=ids('tune'); held_raw=set(ids('test3')+ids('test4'))
    hashes=lambda group_ids:{canonical[g]['rendered_sha256'] for g in group_ids}
    tune_hashes=hashes(tune_ids)
    held_ids={g for g in held_raw if canonical[g]['rendered_sha256'] not in tune_hashes}
    held_hashes=hashes(held_ids)
    cal_ids={g for g in cal_raw if canonical[g]['rendered_sha256'] not in tune_hashes|held_hashes}
    cal_hashes=hashes(cal_ids)
    if cal_hashes&tune_hashes or cal_hashes&held_hashes or tune_hashes&held_hashes:
        raise ValueError('canonical prompt leakage across declared split')
    # Identity/lengths only; current response lengths never enter public.
    public=[dict(group_id=g,prompt_length=canonical[g]['prompt_length'],prompt_sha256=canonical[g]['rendered_sha256'],library=canonical[g]['library']) for g in tune_ids]
    contract=digest(dict(model_config=hashlib.sha256((inputs/'config.json').read_bytes()).hexdigest(),k=8,output_max=8192,generation='published offline ds1000-think-temp06 proxy; actor version unavailable'))
    spec=WorkloadSpec(contract)
    rows=[]
    excluded=0
    for g in sorted(cal_ids):
        bind=h['task_bindings'][g]
        for i,r in enumerate(h['records'].get(g,[])):
            samples=r['samples']
            if len(samples)!=8 or any(x['finish_reason'] not in ('stop','length') for x in samples):
                excluded+=1;continue
            rows.append(Observation(bind['prompt_sha256'],bind['prompt_length'],bind['library'],digest([h['source_sha256'],g,[x['sample'] for x in samples]]),tuple(x['response_length'] for x in samples),tuple(x['finish_reason']=='length' for x in samples)))
    snapshot=DistributionSnapshot.build(rows,spec,kind='offline-calibration-proxy',source_sha256=hashlib.sha256(hpath.read_bytes()).hexdigest())
    snapshot = None if a.no_history else snapshot
    forecast=forecasts(public,snapshot,spec)
    (a.output/'public.json').write_text(json.dumps(public))
    (a.output/'distribution.json').write_text(json.dumps(snapshot.to_dict() if snapshot else None))
    (a.output/'forecast.json').write_text(json.dumps(forecast))
    (a.output/'split-manifest.json').write_text(json.dumps(dict(calibration='historical test1/test2 prompt histories, offline proxy',tune='original tune',reserved='test3/test4; not scored',calibration_prompt_hashes=sorted(cal_hashes),tune_prompt_hashes=sorted(tune_hashes),reserved_prompt_hashes=sorted(held_hashes),excluded_calibration_groups=len(cal_raw)-len(cal_ids),excluded_reserved_groups=len(held_raw)-len(held_ids),groups=len(public),k=8,scenarios=16,excluded_incomplete=excluded,spec=spec.__dict__),indent=2))
    prep=time.perf_counter()-start
    env=Environment(a.output/'public.json',a.output/'mapper-cache',model=a.model,injected_forecasts=forecast)
    result=env.search('capacity_fallback' if a.no_history else a.mode)
    catalog_sha=hashlib.sha256((assets/f'inference-catalog-{a.model}.json').read_bytes()).hexdigest()
    artifact=freeze_candidates(public,snapshot,catalog_sha,result)
    (a.output/'offline-candidates.json').write_text(json.dumps(artifact,separators=(',',':')))
    # Actual current hard capacity check; do not treat an offline certificate as live proof.
    from idea1_controller.inference_groups import InferenceAdmission
    def admit(plan):
        home,inf=plan['training'],plan['inference']
        samples=len(public)*8
        maximum=max(sum(len(home['queues'][str(w)])*8 for w in ws) for ws in home['region_members'].values())
        reserve=2*(samples+max(64,maximum))*(max(env.op.prompt_cap,max(g['prompt_length'] for g in public))+8192)*4
        memory=InferenceAdmission(env.op,reserve)
        return all(memory.evaluate(1 if inf['paired_tp1'] else 2*len(ws),int(inf['caps'][str(r)]),public)['passed'] for r,ws in inf['groups'].items())
    before=time.perf_counter()
    selected,reason=select_precomputed(artifact,public,snapshot,catalog_sha,admit=admit,fallback=lambda:None)
    selection=time.perf_counter()-before
    if selected is None: raise ValueError('no verified deployment plan')
    summary=dict(status='PASS_DISTRIBUTION_PLANNING_PROXY',model=a.model,scope=result['score_scope'],mode=a.mode,groups=len(public),calibration_groups=len(cal_ids),calibration_vectors=len(rows),evidence_counts={k:sum(f['evidence_kind']==k for f in forecast.values()) for k in ('CLASS','GLOBAL','DEFAULT')},offline_preparation_s=prep,offline_environment_s=env.setup_s,offline_search_s=result['timings']['search_s'],online_artifact_check_and_capacity_s=selection,online_reason=reason,winner=result['winner'],counts=result['counts'],process_wall_s=time.perf_counter()-start,native_executed=False)
    (a.output/'result.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
