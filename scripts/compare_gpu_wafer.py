#!/usr/bin/env python3
"""Fail-closed matched-step comparison: GPU observation vs wafer scenario.

Even a matched comparison is an uncalibrated prediction, never a measured-chip
speedup, quality result, or equal-area/power/compute-budget comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rl_xy.config import load_case
from rl_xy.execution_contract import prompt_for,group_for
from rl_xy.verl_trace import load_verl_trace


def read(path):return json.loads(path.read_text())


def compare(gpu_dir,bundle):
    gpu=read(gpu_dir/'report.json');metrics=read(bundle/'metrics.json')
    compilation=read(bundle/'inputs/compilation.json');case=load_case(bundle/'inputs/case.json')
    blockers=[]
    def need(condition,message):
        if not condition:blockers.append(message)
    need(gpu['status']=='PASS' and gpu.get('exit_status')==0,'GPU process and capture did not pass')
    need((gpu_dir/'launch.log').is_file() and hashlib.sha256((gpu_dir/'launch.log').read_bytes()).hexdigest()==gpu.get('launch_log_sha256'),
         'GPU launch log missing or hash mismatch')
    captures=list(gpu_dir.glob('*.jsonl'))
    need(len(captures)==1,'exactly one GPU length capture required')
    if len(captures)!=1:return {'status':'NOT_COMPARABLE','blockers':blockers}
    capture=captures[0]
    trace=load_verl_trace(capture,origin='framework_capture')
    need(hashlib.sha256(capture.read_bytes()).hexdigest()==compilation['length_trace_sha256'],'different source length capture')
    need(gpu['model_config_sha256']==compilation['model_config_sha256'],'different model config')
    need(compilation['origin_declared']=='framework_capture','simulation input is not a real framework capture')
    need(compilation['full_step_replayed'] and compilation['counterfactual_response_cap'] is None
         and not compilation['response_lengths_modified'],'partial group selection or modified lengths')
    step=compilation['step'];observed=[s for s in trace['steps'] if s['step']==step]
    need(len(observed)==1,'selected GPU step not found')
    mapping={r['original']:r['execution'] for r in compilation['original_to_execution_ids']}
    by_id={t.trajectory_id:t for t in case.workload.trajectories}
    if observed:
        expected={}
        for g in observed[0]['groups']:
            for t in g['trajectories']:
                expected[t['traj_id']]=(g['group_id'],t['prompt_len'],t['resp_len'])
        actual={original:(group_for(case,by_id[execution]),len(prompt_for(case,by_id[execution])),len(by_id[execution].response_token_ids))
                for original,execution in mapping.items() if execution in by_id}
        need(actual==expected,'per-trajectory prompt/response/group mismatch')
    execution=read(gpu_dir/f'execution_step_{step}.json')
    cfg=execution['resolved_training_config'];actor=cfg['actor_rollout_ref']['actor']
    need(cfg['algorithm']['adv_estimator']=='grpo' and cfg['algorithm']['norm_adv_by_std_in_grpo'] is False,'GRPO advantage rule mismatch')
    need(actor['loss_agg_mode']=='token-mean' and actor['ppo_epochs']==1,'loss reduction or epoch mismatch')
    need(actor['ppo_mini_batch_size']==trace['run_meta']['algo']['batch_prompts'],
         'GPU optimizer minibatch differs from whole collected prompt batch')
    need(cfg['actor_rollout_ref']['model']['enable_gradient_checkpointing'] is False,'GPU checkpointing differs from full-tape model')
    need(cfg['actor_rollout_ref']['model']['use_remove_padding'] is True,'GPU padding policy differs from length replay')
    need(metrics['optimizer_updates']==1 and metrics['rounds']==1,'one captured GPU update must match one wafer update')
    need(read(bundle/'summary.json')['status']=='PASS','wafer bundle did not pass')
    raw=execution['timing_raw_s'];gpu_s=raw.get('step')
    need(isinstance(gpu_s,(int,float)) and gpu_s>0,'unrounded full-step GPU timer missing')
    wafer_s=metrics['total_simulated_time_ns']/1e9
    return {'status':'MATCHED_SCENARIO_PREDICTION' if not blockers else 'NOT_COMPARABLE',
        'blockers':blockers,'gpu_step_observed_s':gpu_s,'wafer_step_predicted_s':wafer_s,
        'scenario_time_ratio_NOT_validated_speedup':gpu_s/wafer_s if not blockers else None,
        'gpu_stage_timers_raw_s':raw,'model_config_sha256':gpu['model_config_sha256'],
        'length_capture_sha256':compilation['length_trace_sha256'],
        'wafer_resources':{k:metrics[k] for k in ('dies','cores_per_die','data_parallel','tensor_parallel')},
        'gpu_resources':trace['run_meta']['hardware'],
        'assumptions':metrics['assumptions'],'hardware_calibrated':False,'equal_resource_budget':False,
        'limitations':['single first-step smoke, not steady state or learning quality',
            'actual GPU vs analytic wafer prediction, not two measured devices',
            'different total compute/memory budgets; no equal-area/power claim',
            'wafer full-tape/reference kernels differ from optimized GPU kernels',
            'wafer host/reward/control costs remain explicit assumptions',
            'FP32/BF16/vector/SRAM/NoC/D2D caliber require sensitivity and calibration']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=Path,required=True)
    p.add_argument('--wafer',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    report=compare(a.gpu,a.wafer);a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'comparison.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    lines=['# GPU观测与晶圆场景预测对照','',f'状态：{report["status"]}。','',
        '**不是实测晶圆加速比，不是等算力/等面积/等功耗比较，不是学习质量或稳态吞吐结果。**','',
        '| 项目 | 数值 |','| --- | --- |',f'| GPU真实step秒 | {report.get("gpu_step_observed_s")} |',
        f'| 晶圆场景预测step秒 | {report.get("wafer_step_predicted_s")} |',
        f'| 场景时间比，非已验证speedup | {report.get("scenario_time_ratio_NOT_validated_speedup")} |','',
        '匹配失败原因：'+str(report['blockers']),'','完整硬件假设、输入hash与限制见comparison.json。']
    (a.out/'GPU与晶圆对照.md').write_text('\n'.join(lines)+'\n')
    print(report['status'])
    return 0 if report['status']=='MATCHED_SCENARIO_PREDICTION' else 2


if __name__=='__main__':raise SystemExit(main())
