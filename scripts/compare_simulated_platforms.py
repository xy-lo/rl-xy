#!/usr/bin/env python3
"""Near-budget GPU simulation vs frozen wafer simulation, no observed seconds."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rl_xy.config import load_case
from rl_xy.gpu_model import budget_audit


def read(path):return json.loads(path.read_text())


def compare(wafer,gpu):
    wm=read(wafer/'metrics.json');gm=read(gpu/'metrics.json')
    wc=load_case(wafer/'inputs/case.json');gc=load_case(gpu/'inputs/case.json')
    errors=[]
    def need(x,why):
        if not x:errors.append(why)
    for bundle in (wafer,gpu):
        need(read(bundle/'summary.json')['status']=='PASS','bundle did not pass')
        need(read(bundle/'manifest.json')['exit_status']==0,'native exit nonzero')
    need(gm.get('platform')=='GPU','GPU must be independently simulated, not observed step scaling')
    need(asdict(wc.model)==asdict(gc.model),'model structures differ')
    need(asdict(wc.workload)==asdict(gc.workload),'per-trajectory lengths/groups/values/version differ')
    need(asdict(wc.strategy)==asdict(gc.strategy),'logical strategy/minibatch/precision contract differs')
    need(wc.costs==gc.costs,'host/control assumptions differ')
    need((wm['rounds'],wm['optimizer_updates'])==(gm['rounds'],gm['optimizer_updates']),'update count differs')
    need(wm['tensor_parallel']==gm['tensor_parallel']==2,'controlled TP2 comparison required')
    for name in ('compilation.json','model_config.json','gpu_provenance.json'):
        need(hashlib.sha256((wafer/'inputs'/name).read_bytes()).digest()==hashlib.sha256((gpu/'inputs'/name).read_bytes()).digest(),
             'different source '+name)
    compilation=read(wafer/'inputs/compilation.json')
    need(compilation['full_step_replayed'] and not compilation['response_lengths_modified']
         and compilation['counterfactual_response_cap'] is None and compilation['origin_declared']=='framework_capture',
         'comparison requires one complete actual length capture, not modified/subset input')
    audit=budget_audit(wm,read(wafer/'resolved_config.json'),gm['gpu_profile'])
    need(audit['gpu_no_less_fp16_and_dram_capacity'],'GPU must not have less primary compute/capacity')
    need(all(abs(audit['gpu_minus_wafer_percent'][k])<=15 for k in ('bf16_peak_flops_per_ns','dram_capacity_bytes')),
         'near-budget primary resources differ by >15%')
    ws=wm['total_simulated_time_ns']/1e9;gs=gm['total_simulated_time_ns']/1e9
    need(ws>0 and gs>0,'invalid native time')
    return {'status':'NEAR_BUDGET_SIMULATION_COMPARISON' if not errors else 'NOT_COMPARABLE','blockers':errors,
        'wafer_predicted_s':ws,'gpu_predicted_s':gs,
        'gpu_over_wafer_time_ratio_SCENARIO_ONLY':gs/ws if not errors else None,
        'wafer_faster_in_this_scenario':gs>ws if not errors else None,
        'budget_audit':audit,'gpu_precision_mode':gm['gpu_profile']['precision_mode'],
        'observed_GPU_step_seconds_used_for_prediction':False,
        'wafer_source_tree':read(wafer/'manifest.json')['software']['rl_xy_source_tree_sha256'],
        'gpu_source_tree':read(gpu/'manifest.json')['software']['rl_xy_source_tree_sha256'],
        'gpu_assumptions':gm['assumptions'],'wafer_assumptions':wm['assumptions'],
        'hardware_calibrated':False,'feasibility_proven':False,
        'limitations':['same logical TP2 synchronous GRPO; not best GPU parallelism search or reproduced Splitwise/Megatron',
            'declared operator graph, not actual fused PyTorch/vLLM kernels',
            'GPU SM/CTA policy, launch/shared-memory/link latency require validation',
            'wafer FP32/vector/SRAM/NoC/D2D caliber include unvalidated assumptions',
            'GPU has more FP16 and DRAM capacity, not more SRAM or aggregate HBM bandwidth',
            'one short workload; no learning quality or equal power/area claim']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wafer',required=True,type=Path)
    p.add_argument('--gpu',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    r=compare(a.wafer,a.gpu);a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'comparison.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    rows=['# 48GPU仿真与54die晶圆仿真对照','',r['status'],'',
        '**两侧均为未校准仿真，不使用42.55秒GPU冷首步缩放，不预设晶圆胜出。**','',
        '| 平台 | 预测秒 |','| --- | ---: |',f'| 54die晶圆 | {r["wafer_predicted_s"]} |',
        f'| 48GPU | {r["gpu_predicted_s"]} |','',
        f'本场景晶圆更快：{r["wafer_faster_in_this_scenario"]}；时间比仅为场景结果：{r["gpu_over_wafer_time_ratio_SCENARIO_ONLY"]}。','',
        '| 总资源 | 晶圆 | GPU | GPU相对差异% |','| --- | ---: | ---: | ---: |']
    b=r['budget_audit']
    rows += [f'| {k} | {v} | {b["gpu"][k]} | {b["gpu_minus_wafer_percent"][k]:.6f} |' for k,v in b['wafer'].items()]
    rows+=['','FP32/向量/SRAM/带宽不是全部相等；晶圆部分rate仍是test_assumption。','',
        '结论只适用于当前算子/调度/参数场景，不能仅凭较快就标记方案已被硬件验证。','',str(r['blockers'])]
    (a.out/'仿真对照.md').write_text('\n'.join(rows)+'\n')
    print(r['status']);return 0 if not r['blockers'] else 2


if __name__=='__main__':raise SystemExit(main())
