#!/usr/bin/env python3
"""Docker-only Qwen2/group execution acceptance; no GPU or numeric RL training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',required=True)
    p.add_argument('--container',default='rl-xy-execution-final')
    p.add_argument('--image',default='rlxy-astrasim:0.4.0')
    p.add_argument('--reuse',type=Path,default=Path('runs/legacy_reuse/inputs-20261026-final/E1_gsm8k_qwen05b'))
    a=p.parse_args()
    if Path(a.run_id).name!=a.run_id or a.run_id in ('','.','..'):p.error('invalid run-id')
    out=ROOT/'acceptance'/a.run_id;out.mkdir(parents=True,exist_ok=False)
    report={'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),
        'image':a.image,'container':a.container,'commands':[],'checks':[],'bundles':{},'new_gpu_training':False}
    def save(): (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    def command(name,argv,expected=0):
        save()
        log=out/(name+'.log')
        with log.open('w') as stream:
            r=subprocess.run(argv,cwd=ROOT,env={**os.environ,'RLXY_CONTAINER':a.container},
                stdout=stream,stderr=subprocess.STDOUT,timeout=1200)
        report['commands'].append({'label':name,'argv':argv,'exit_status':r.returncode,'expected_exit':expected,
            'log':str(log.relative_to(ROOT)),'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()})
        save();print(f'{name}: {r.returncode} (expected {expected})',flush=True)
        if r.returncode!=expected:raise ValueError(name)
    def check(name,passed,evidence):
        report['checks'].append({'check':name,'passed':bool(passed),'evidence':evidence});save()
        if not passed:raise ValueError(name)
    def read(path):return json.loads((ROOT/path).read_text())
    try:
        command('build',['docker','build','-t',a.image,'-f','docker/Dockerfile.execution','.'])
        command('create',['bash','scripts/docker.sh','create-container',a.image])
        image=subprocess.check_output(['docker','inspect','--format','{{.Image}}',a.container],text=True).strip()
        report['image_id']=image
        docker=['docker','exec','-e',f'RLXY_IMAGE={a.image}','-e',f'RLXY_IMAGE_ID={image}',a.container,'python3']
        command('unit_tests',docker+['-m','unittest','discover','-s','tests','-v'])
        base=f'runs/qwen_execution/{a.run_id}'
        def run(label,case,rounds=1,expected=0,resource=True):
            target=base+'-'+label
            argv=docker+['-m','rl_xy.cli','run','--case',case,'--out',target,'--rounds',str(rounds)]
            if resource:argv+=['--resource-model']
            command(label,argv,expected)
            if not expected:report['bundles'][label]=target
            return target
        run('fixture','configs/qwen2_group_fixture/case.json')
        run('two_rounds','configs/qwen2_group_fixture/case.json',2)
        run('repeat','configs/qwen2_group_fixture/case.json')
        run('legacy','configs/toy_grpo_2die/case.json')
        run('no_resource_rejected','configs/qwen2_group_fixture/case.json',expected=2,resource=False)
        variants=out/'variants';variants.mkdir()
        slow=variants/'slow_dram';shutil.copytree(ROOT/'configs/qwen2_group_fixture',slow)
        raw=json.loads((slow/'case.json').read_text());raw['hardware']['dram_bytes_per_ns']=1
        (slow/'case.json').write_text(json.dumps(raw,indent=2)+'\n')
        run('slow_dram',str(slow.relative_to(ROOT)/'case.json'))
        # Real shapes use explicit nominal probe rates, not incomplete formal H.
        nominal=variants/'nominal_probe';shutil.copytree(ROOT/'configs/qwen2_group_fixture',nominal)
        raw=json.loads((nominal/'case.json').read_text())
        raw['hardware'].update(dram_bytes_per_die=64000000000,sram_bytes_per_core=1250000,
            gemm_flops_per_ns=1024,vector_ops_per_ns=256,dram_bytes_per_ns=2000,
            module_path_bytes_per_ns=2000,sram_bytes_per_ns=4000)
        (nominal/'case.json').write_text(json.dumps(raw,indent=2)+'\n')
        (nominal/'H_SCOPE.md').write_text('显式未校准TP2单逻辑核执行探针；不是用户54模块/256核正式H。所有rate是本验收声明的假设，不是芯片实测。D2D沿用8 B/ns、1ns教学链路。\n')
        model=a.reuse/'inputs/model_config.json'
        trace=a.reuse/'seed0/lengths.jsonl'
        # Preserve the old normalized input/hash; choose one COMPLETE group.
        normalized=read(a.reuse/'seed0/normalized.json')
        step=normalized['steps'][0]
        group=min(step['groups'],key=lambda g:sum(t['resp_len'] for t in g['trajectories']))['group_id']
        compiled=variants/'old_counterfactual'
        command('compile_old_lengths',docker+['scripts/make_execution_case.py','--template',str(nominal.relative_to(ROOT)),
            '--model-config',str(model),'--trace',str(trace),'--origin','statistics_constrained_synthetic',
            '--step',str(step['step']),'--groups',group,'--probe-response-cap','1','--out',str(compiled.relative_to(ROOT))])
        report['old_compilation']=str(compiled.relative_to(ROOT)/'compilation.json')
        run('actual_qwen_old_length_probe',str(compiled.relative_to(ROOT)/'case.json'))
        # Preserve actual Qwen shapes but exercise multiple groups/decode with tiny lengths.
        tiny=variants/'actual_qwen_multigroup';shutil.copytree(compiled,tiny)
        raw=json.loads((tiny/'case.json').read_text())
        fixture=json.loads((ROOT/'configs/qwen2_group_fixture/case.json').read_text())
        raw['workload']=fixture['workload'];raw['strategy']=fixture['strategy'];raw['case_id']='actual_qwen_tiny_synthetic_multigroup'
        (tiny/'case.json').write_text(json.dumps(raw,indent=2)+'\n')
        (tiny/'compilation.json').write_text(json.dumps({'scope':'actual model shapes plus explicit synthetic tiny multi-group/decode fixture; NOT old data lengths',
            'model_config_sha256':hashlib.sha256((tiny/'model_config.json').read_bytes()).hexdigest()},indent=2)+'\n')
        run('actual_qwen_multigroup',str(tiny.relative_to(ROOT)/'case.json'))
        command('native_fixtures',docker+['scripts/integration_fixtures.py','--output',f'runs/resource_integration/{a.run_id}'])
        report['integration']=f'runs/resource_integration/{a.run_id}'
        for label,bundle in report['bundles'].items():
            command('integrity_'+label,docker+['scripts/check_result.py',bundle])
        metrics={n:read(b+'/metrics.json') for n,b in report['bundles'].items()}
        check('old_toy_time_unchanged',metrics['legacy']['total_simulated_time_ns']==108832.27,metrics['legacy']['total_simulated_time_ns'])
        check('repeat_actual_events_identical',(ROOT/report['bundles']['fixture']/'events.jsonl').read_bytes()==(ROOT/report['bundles']['repeat']/'events.jsonl').read_bytes(),report['bundles']['repeat'])
        check('two_round_version_and_no_peak_growth',metrics['two_rounds']['actor_version_end']==2
            and metrics['two_rounds']['reference_version_end']==0
            and metrics['two_rounds']['memory_peak_bytes_by_location']==metrics['fixture']['memory_peak_bytes_by_location'],metrics['two_rounds']['memory_peak_bytes_by_location'])
        check('dram_rate_changes_actual_qwen_fixture_time',metrics['slow_dram']['total_simulated_time_ns']>metrics['fixture']['total_simulated_time_ns'],
            {k:metrics[k]['total_simulated_time_ns'] for k in ('fixture','slow_dram')})
        for label in ('actual_qwen_multigroup','actual_qwen_old_length_probe'):
            check(label+'_parameter_identity',metrics[label]['parameter_accounting']['unique_parameter_elements']==494032768,metrics[label]['parameter_accounting'])
        compilation=read(report['old_compilation'])
        check('old_probe_is_explicit_counterfactual',compilation['counterfactual_response_cap']==1 and compilation['response_lengths_modified']
            and not compilation['full_step_replayed'] and not compilation['gpu_times_used_as_chip_durations']
            and compilation['input_values']=={'token_ids':False,'reward':False,'logprob':False},compilation)
        native=read(report['integration']+'/report.json')
        check('native_resource_mesh_regressions',native['status']=='PASS' and all(c['passed'] for c in native['checks']),native['checks'])
        import sys
        sys.path.insert(0,str(ROOT/'src'))
        from rl_xy.reporting import deterministic_tree_hash
        tree=deterministic_tree_hash(ROOT)
        check('all_bundles_bound_to_source',all(read(b+'/manifest.json')['software']['rl_xy_source_tree_sha256']==tree for b in report['bundles'].values()) and native['source_tree_sha256']==tree,tree)
        report.update(status='PASS',source_tree_sha256=tree,metrics=metrics)
    except BaseException as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['finished_at_utc']=datetime.now(timezone.utc).isoformat();save()
    return 0


if __name__=='__main__':raise SystemExit(main())
