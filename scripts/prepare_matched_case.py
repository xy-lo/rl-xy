#!/usr/bin/env python3
"""Compile every complete group from one actual GPU step, no response cap."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from make_execution_case import compile_case
from rl_xy.verl_trace import load_verl_trace


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',required=True,type=Path)
    p.add_argument('--template',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--model-config',required=True,type=Path);a=p.parse_args()
    r=json.loads((a.gpu/'report.json').read_text())
    if r['status']!='PASS' or r.get('exit_status')!=0:raise ValueError('real GPU step must have passed')
    paths=list(a.gpu.glob('*.jsonl'))
    if len(paths)!=1:raise ValueError('one unambiguous length capture required')
    trace=load_verl_trace(paths[0],origin='framework_capture')
    if len(trace['steps'])!=1:raise ValueError('matched smoke requires exactly one captured step')
    if hashlib.sha256(a.model_config.read_bytes()).hexdigest()!=r['model_config_sha256']:
        raise ValueError('GPU model shape identity mismatch')
    step=trace['steps'][0]
    result=compile_case(a.template,a.model_config,paths[0],'framework_capture',step['step'],
        [g['group_id'] for g in step['groups']],a.out)
    if not result['full_step_replayed'] or result['response_lengths_modified']:raise ValueError('refuse altered workload')
    (a.out/'gpu_provenance.json').write_text(json.dumps({'gpu_report_sha256':hashlib.sha256((a.gpu/'report.json').read_bytes()).hexdigest(),
        'gpu_run_id':trace['run_meta']['run_id'],'scope':'all recorded groups; exact lengths; no GPU seconds used as chip costs'},indent=2)+'\n')
    print(f'compiled {result["groups"]} groups / {result["trajectories"]} trajectories; full step unchanged')


if __name__=='__main__':main()
