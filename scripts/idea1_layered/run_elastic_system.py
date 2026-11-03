#!/usr/bin/env python3
"""History/public tasks -> TP bank search -> real variable-TP native run."""
import argparse,json,sys,time,subprocess
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent/'legacy'));sys.path.append(str(ROOT/'src'))
from rl_xy.f2_runtime import make_operator,verify_assets,INPUTS,sha
from rl_xy.qwen3_experiment import Policy
from gqa_cost import apply_generation_gqa
from idea1_controller.inference_catalog import InferenceServiceCatalog
from idea1_controller.elastic_planner import ElasticSystemPlanner
from idea1_controller.history import validate_history,digest
from idea1_controller.cost_catalog import validate_catalog


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=('4B','8B','14B'),required=True)
    p.add_argument('--window',choices=('tune','test1','test2','test3','test4'),default='test1');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--history',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--tp-bank',type=Path,required=True)
    p.add_argument('--native-root',type=Path);p.add_argument('--fast-cost-validation',type=Path);p.add_argument('--allow-offline-history-proxy',action='store_true')
    p.add_argument('--plan-only',action='store_true');p.add_argument('--force-profile');p.add_argument('--budget-s',type=float,default=15)
    p.add_argument('--subset-prompts',type=int,default=0,help='original prefix only for interface smoke; zero runs all192')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);setup=time.perf_counter();manifest=verify_assets()
    trace=INPUTS/('Qwen3-'+a.model)/(a.window+'.json');assert sha(trace)==manifest['input_hashes'][a.model][a.window+'.json']
    items=json.loads(trace.read_text())['trajectories'];prompts=json.loads((trace.parent/'prompt_lengths.json').read_text())
    if a.subset_prompts:
        if not 1<=a.subset_prompts<=192:raise ValueError('invalid original prompt subset')
        items=items[:a.subset_prompts*8]
    public=[dict(group_id=str(x['group_id']),prompt_length=x['prompt_length'],prompt_sha256=prompts[str(x['group_id'])]['rendered_sha256'],library=prompts[str(x['group_id'])]['library']) for x in items[::8]]
    history=json.loads(a.history.read_text());validate_history(public,history,a.model,sha(trace.parent/'config.json'))
    if history.get('kind','').startswith('offline-') and not a.allow_offline_history_proxy:raise ValueError('explicit --allow-offline-history-proxy is required')
    bank=json.loads(a.tp_bank.read_text());assert bank['model']==a.model
    policy=Policy(**bank['storage_policy']);op=make_operator(a.model,'wafer',cache_dir=out.parent/'mapper-cache-fast')
    if a.fast_cost_validation:
        from idea1_controller.cost_backend import install
        install(op,out.parent/'cost-accelerator',json.loads(a.fast_cost_validation.read_text()))
    apply_generation_gqa(op);catalog=InferenceServiceCatalog(json.loads(a.catalog.read_text()))
    validate_catalog(catalog.data,op,a.model,sha(trace.parent/'config.json'));setup_s=time.perf_counter()-setup
    plan=ElasticSystemPlanner(op,policy,catalog,bank).build(public,history,a.budget_s,a.force_profile)
    plan.update(model=a.model,window=a.window,source_requests_digest=digest(items),trace_sha256=sha(trace),
        policy=asdict(policy),setup_wall_s=setup_s,catalog_sha256=sha(a.catalog),tp_bank_sha256=sha(a.tp_bank),
        training_help_admission=bank.get('training_help_admission','region-ready'))
    (out/'selected-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (out/'inference-layout.json').write_text(json.dumps(plan['inference_layout'],indent=2)+'\n')
    (out/'history-snapshot.json').write_text(json.dumps(history,separators=(',',':'))+'\n')
    print(json.dumps(dict(status='PLANNED_ACTUAL_INFERENCE_TP',profile=plan['selected_inference_profile'],planning_s=plan['total_planner_wall_s'])),flush=True)
    if a.plan_only:return
    if not a.native_root:raise ValueError('--native-root required for execution')
    command=[sys.executable,str(Path(__file__).with_name('run_inference_system.py')),'--model',a.model,'--window',a.window,'--training-scope','wafer',
        '--assignment',str(out/'selected-plan.json'),'--resource-layout',str(out/'selected-plan.json'),'--inference-layout',str(out/'inference-layout.json'),
        '--history',str(out/'history-snapshot.json'),'--rollout-cap',str(policy.cap),'--native-root',str(a.native_root.resolve()),'--output',str(out/'native-run')]
    if a.fast_cost_validation:command+=['--fast-cost-validation',str(a.fast_cost_validation.resolve())]
    if a.subset_prompts:command+=['--subset-prompts',str(a.subset_prompts)]
    (out/'command.json').write_text(json.dumps(dict(argv=command),indent=2)+'\n');begin=time.perf_counter()
    with (out/'native-run.log').open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    status=dict(native_driver_exit_status=result.returncode,native_wall_s=time.perf_counter()-begin,planning_s=plan['total_planner_wall_s'],setup_s=setup_s)
    (out/'execution-status.json').write_text(json.dumps(status,indent=2)+'\n')
    if result.returncode:raise RuntimeError('native execution failed; inspect native-run.log')
    native=json.loads((out/'native-run/result.json').read_text());native.update(planning_s=plan['total_planner_wall_s'],setup_s=setup_s,
        selected_plan_sha256=sha(out/'selected-plan.json'),selected_inference_profile=plan['selected_inference_profile'])
    (out/'result.json').write_text(json.dumps(native,indent=2)+'\n');print(json.dumps(dict(status='PASS_VARIABLE_TP_SYSTEM',total_s=native['total_s'],profile=plan['selected_inference_profile'])))

if __name__=='__main__':main()
