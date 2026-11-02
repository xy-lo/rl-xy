#!/usr/bin/env python3
"""Public inputs -> feasible resources -> history placement -> native execution."""
import argparse,json,sys,time,subprocess,os
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent/'legacy'));sys.path.append(str(ROOT/'src'))
from rl_xy.f2_runtime import INPUTS,make_operator,verify_assets,sha
from rl_xy.qwen3_experiment import Policy
from gqa_cost import apply_generation_gqa
from idea1_controller.system import SystemPlanner
from idea1_controller.catalog_views import ViewServiceCatalog
from idea1_controller.cost_catalog import validate_catalog
from idea1_controller.history import digest,validate_history

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=('4B','8B','14B'),required=True);p.add_argument('--window',choices=('tune','test1','test2','test3','test4'),default='test1')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--history',type=Path);p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--training-help-admission',choices=('global-eos','region-ready'))
    p.add_argument('--resource-family',choices=('regional','pooled-tp2'),default='regional')
    p.add_argument('--layout-from',type=Path);p.add_argument('--rollout-cap',type=int,default=0);p.add_argument('--native-root',type=Path)
    p.add_argument('--allow-offline-history-proxy',action='store_true');p.add_argument('--plan-only',action='store_true');p.add_argument('--budget-s',type=float,default=15)
    p.add_argument('--fast-cost-validation',type=Path)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    manifest=verify_assets();trace=INPUTS/('Qwen3-'+a.model)/(a.window+'.json');assert sha(trace)==manifest['input_hashes'][a.model][a.window+'.json']
    items=json.loads(trace.read_text())['trajectories'];prompt_manifest=json.loads((trace.parent/'prompt_lengths.json').read_text())
    public=[dict(group_id=str(x['group_id']),prompt_length=x['prompt_length'],prompt_sha256=prompt_manifest[str(x['group_id'])]['rendered_sha256'],library=prompt_manifest[str(x['group_id'])]['library']) for x in items[::8]]
    history=json.loads(a.history.read_text()) if a.history else dict(model=a.model,kind='cold-start',records={})
    if not a.history:history['snapshot_id']=digest(history)
    if a.history:validate_history(public,history,a.model,sha(trace.parent/'config.json'))
    if history.get('kind','').startswith('offline-') and not a.allow_offline_history_proxy:raise ValueError('explicit --allow-offline-history-proxy is required for this diagnostic history')
    hpath=out/'history-snapshot.json';hpath.write_text(json.dumps(history,separators=(',',':'))+'\n')
    frozen_layout=json.loads(a.layout_from.read_text()) if a.layout_from else {}
    policy={'4B':Policy(2,2,4,8,64),'8B':Policy(2,2,2,1,48),'14B':Policy(2,2,1,1,32)}[a.model]
    cap=a.rollout_cap or frozen_layout.get('rollout_cap',policy.cap)
    policy=Policy(policy.train_tp,policy.rollout_tp,policy.train_mb,policy.logprob_mb,cap)
    op=make_operator(a.model,'wafer',cache_dir=out.parent/'mapper-cache-fast')
    if a.fast_cost_validation:
        from idea1_controller.cost_backend import install
        install(op,out.parent/'cost-accelerator',json.loads(a.fast_cost_validation.read_text()))
    apply_generation_gqa(op);catalog=ViewServiceCatalog(json.loads(a.catalog.read_text()))
    validate_catalog(catalog.data,op,a.model,sha(trace.parent/'config.json'))
    if catalog.data['cap']<policy.cap:raise ValueError('service catalog does not cover requested concurrency')
    layout=frozen_layout.get('region_members')
    helper_domains=frozen_layout.get('helper_domains','search' if a.resource_family=='pooled-tp2' else None)
    setup_s=time.perf_counter()-start
    plan=SystemPlanner(op,policy,catalog).build(public,history,layout,a.budget_s,helper_domains=helper_domains)
    plan['scenario_search_wall_s']=plan['planning_wall_s']
    plan['planning_wall_s']=plan['total_planner_wall_s']
    plan.update(training_help_admission=a.training_help_admission or frozen_layout.get('training_help_admission','region-ready'),model=a.model,window=a.window,source_requests_digest=digest(items),trace_sha256=sha(trace),
        catalog_sha256=sha(a.catalog),policy=asdict(policy),setup_wall_s=setup_s,planner_source='idea1_controller.system.SystemPlanner -> ViewPlanSearch')
    plan_path=out/'selected-plan.json';plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(status='PLANNED',model=a.model,window=a.window,dp=plan['selected_dp'],cap=policy.cap,planning_s=plan['total_planner_wall_s'],fallback_groups=len(plan['fallback_group_ids']))),flush=True)
    if a.plan_only:return
    if not a.native_root:raise ValueError('--native-root required for execution')
    command=[sys.executable,str(Path(__file__).with_name('run_view_system_fast.py')),'--model',a.model,'--window',a.window,'--training-scope','wafer',
        '--assignment',str(plan_path),'--resource-layout',str(plan_path),'--history',str(hpath),'--rollout-cap',str(policy.cap),
        '--native-root',str(a.native_root.resolve()),'--output',str(out/'native-run')]
    if a.fast_cost_validation:command+=['--fast-cost-validation',str(a.fast_cost_validation.resolve())]
    (out/'command.json').write_text(json.dumps(dict(argv=command),indent=2)+'\n')
    begin=time.perf_counter()
    with (out/'native-run.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    status=dict(native_driver_exit_status=r.returncode,native_wall_s=time.perf_counter()-begin,setup_s=setup_s,planning_s=plan['total_planner_wall_s'],
        accepted=r.returncode==0 and (out/'native-run/result.json').exists())
    (out/'execution-status.json').write_text(json.dumps(status,indent=2)+'\n')
    if not status['accepted']:raise RuntimeError('native execution failed; inspect native-run.log and raw evidence')
    result=json.loads((out/'native-run/result.json').read_text());result.update(planning_s=plan['total_planner_wall_s'],setup_s=setup_s,selected_plan_sha256=sha(plan_path))
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status='PASS_COMPLETE_SYSTEM',model=a.model,total_s=result['total_s'],plan=str(plan_path))),flush=True)
if __name__=='__main__':main()
