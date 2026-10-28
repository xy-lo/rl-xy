#!/usr/bin/env python3
"""Compare native runs under the same explicitly restricted veRL protocol."""
import argparse,csv,hashlib,json,sys
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rl_xy.config import load_case
from rl_xy.gpu_model import budget_audit


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def compare(wafer,gpu):
    wm=read(wafer/'metrics.json');gm=read(gpu/'metrics.json');errors=[]
    def need(value,message):
        if not value:errors.append(message)
    for bundle in [wafer,gpu]:
        need(read(bundle/'summary.json')['status']=='PASS','run not PASS: '+bundle.name)
        need(read(bundle/'manifest.json')['exit_status']==0,'native exit nonzero')
        checks=list(csv.DictReader((bundle/'checks.csv').open()))
        need(any(c['check_id']=='colocation.global_stage_barriers' and str(c['passed']).lower()=='true' for c in checks),'global stage barrier not verified')
    wc=load_case(wafer/'inputs/case.json');gc=load_case(gpu/'inputs/case.json')
    for key in ['model','workload','strategy']:need(asdict(getattr(wc,key))==asdict(getattr(gc,key)),key+' differs')
    need(wc.costs==gc.costs,'control cost differs')
    need(gm.get('platform')=='GPU','GPU must be independent native simulation')
    for name in ['colocation.json','model_config.json','workload_provenance.json']:
        need(sha(wafer/'inputs'/name)==sha(gpu/'inputs'/name),name+' differs')
    ws=read(wafer/'manifest.json')['software']['rl_xy_source_tree_sha256'];gs=read(gpu/'manifest.json')['software']['rl_xy_source_tree_sha256']
    need(ws==gs,'source identity differs')
    for key in ['rounds','optimizer_updates','tensor_parallel','rollout_tensor_parallel','trajectory_count','response_tokens']:
        need(wm[key]==gm[key],key+' differs')
    w=wm['total_simulated_time_ns'];g=gm['total_simulated_time_ns']
    need(w>0 and g>0,'nonpositive clock')
    return dict(status='CONDITIONAL_PROTOCOL_SIMULATION_COMPARISON' if not errors else 'NOT_COMPARABLE',blockers=errors,
        wafer_model_ms=w/1e6,gpu_model_ms=g/1e6,gpu_over_wafer_model_time=g/w if not errors else None,
        wafer_faster=g>w if not errors else None,source_tree_sha256=ws,
        train_tp=wm['tensor_parallel'],rollout_tp=wm['rollout_tensor_parallel'],
        wafer_train_dp=wm['data_parallel'],gpu_train_dp=gm['data_parallel'],
        wafer_allocated=wm['dies'],wafer_active=wm['active_devices'],gpu_allocated=gm['dies'],gpu_active=gm['active_devices'],
        budget=budget_audit(wm,read(wafer/'resolved_config.json'),gm['gpu_profile']),
        workload=read(wafer/'inputs/workload_provenance.json'),
        stages={'wafer':read(wafer/'global_stage_times.json'),'gpu':read(gpu/'global_stage_times.json')},
        manifests={'wafer':sha(wafer/'manifest.json'),'gpu':sha(gpu/'manifest.json')},
        GPU_GEMM_measurement_profile_sha256=gm['gpu_profile'].get('gemm_profile_sha256'),
        hardware_calibrated=False,numerical_learning_validated=False,full_verl_runtime_reproduced=False,
        limitations=['PP1 dense Megatron protocol, full-batch update, microbatch one trajectory, no host offload',
          'common group-preserving LPT dispatch differs from default veRL trajectory balancing/padding',
          'packed generation linears, ragged unfused attention; no full vLLM kernel/fusion/admission reproduction',
          'naive gathered full-tensor export with distinct rollout storage; not paper zero-redundancy 3D-HybridEngine',
          'wafer execution rates, GPU analytic kernels or unmeasured vector/network/control costs need calibration',
          'deterministic model times are not statistical measurements; no learning quality/equal-power claim'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--wafer',type=Path,required=True);p.add_argument('--gpu',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=compare(a.wafer,a.gpu);a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'comparison.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(r['status']);return 2 if r['blockers'] else 0
if __name__=='__main__':raise SystemExit(main())
