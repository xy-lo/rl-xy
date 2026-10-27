#!/usr/bin/env python3
"""Check published summaries and inputs; does not replay absent raw logs."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'experiments/2026-10-27-v0.6'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    for name,wanted in read(DATA/'input_hashes.json').items():assert sha(DATA/'inputs'/name)==wanted,name
    for name,wanted in read(DATA/'core_runtime_hashes.json').items():assert sha(ROOT/name)==wanted,name
    c=read(DATA/'comparisons.json');m=read(DATA/'metrics_summary.json')
    pairs={'primary':('wafer_original','reference'),'gpu_favoring':('wafer_original','optimistic'),
        'wafer_rate_sensitivity':('wafer_rates','reference'),'joint_sensitivity':('wafer_rates','optimistic')}
    for name,(w,g) in pairs.items():
        r=c[name];ws=m[w]['total_simulated_time_ns']/1e9;gs=m[g]['total_simulated_time_ns']/1e9
        assert r['status']=='NEAR_BUDGET_SIMULATION_COMPARISON' and not r['blockers']
        assert r['wafer_predicted_s']==ws and r['gpu_predicted_s']==gs
        assert r['gpu_over_wafer_time_ratio_SCENARIO_ONLY']==gs/ws
        assert r['wafer_faster_in_this_scenario']==(gs>ws)
        assert r['observed_GPU_step_seconds_used_for_prediction'] is False
        assert r['hardware_calibrated'] is False and r['feasibility_proven'] is False
    accepted=read(DATA/'acceptance_summary.json');assert accepted['status']=='PASS'
    assert len(accepted['commands'])==20 and all(x['exit_status']==x['expected_exit'] for x in accepted['commands'])
    assert all(x['passed'] for x in accepted['checks'])
    assert sum(accepted['native_fixture_check_counts'].values())==21
    local=read(DATA/'local_verification_summary.json');assert local['status']=='PASS'
    assert sum(x['rank_events'] for x in local['independent_native_replay'].values())==1645238
    case=read(DATA/'inputs/case.json');tr=case['workload']['trajectories']
    assert len(tr)==64 and sum(t['response_length'] for t in tr)==256
    assert len({t['group_id'] for t in tr})==32
    assert all(m[k]['optimizer_updates']==1 for k in ('reference','optimistic','wafer_rates','wafer_original'))
    print(json.dumps({'status':'PASS','scope':'public summaries, input integrity and unchanged core; not raw-log replay',
        'comparison_count':len(c),'trajectory_count':len(tr),'original_replayed_events':1645238,
        'core_runtime_files':len(read(DATA/'core_runtime_hashes.json'))},indent=2))


if __name__=='__main__':main()
