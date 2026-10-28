#!/usr/bin/env python3
"""Native same-cost proof of external immutable carrier read pruning."""
import argparse,json,gzip
from pathlib import Path
from rl_xy.aligned_run import run_aligned
from rl_xy.evidence_io import load_json_auto
ROOT=Path(__file__).resolve().parents[1]
def lines(path):
 with gzip.open(path,'rt') as f:return [json.loads(x) for x in f]
def main():
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);x=a.parse_args();x.out.mkdir(parents=True,exist_ok=False);records=[]
 case=ROOT/'configs/aligned_fixture'
 for platform in ('wafer','gpu'):
  outs=[]
  for conservative in (True,False):
   out=x.out/(platform+('-all' if conservative else '-pruned'))
   run_aligned(case/'case.json',case/'wafer.json',case/'policy.json',out,gpu_profile_path=case/'gpu.json' if platform=='gpu' else None,active=4,conservative_bridge_reads=conservative)
   outs.append(out)
  old,new=outs
  for name in ['events.jsonl.gz','operations.jsonl.gz']:
   assert lines(old/name)==lines(new/name),(platform,name,'changed cost/dependency/event')
  a=load_json_auto(old/'resource_contract.json.gz');b=load_json_auto(new/'resource_contract.json.gz')
  for r,row in a['ranks'].items():
   from rl_xy.resource_encoding import expand_rank
   one,two=expand_rank(row),expand_rank(b['ranks'][r]);assert one['initial_objects']==two['initial_objects']
   for n,v in one['nodes'].items():
    w=two['nodes'][n];assert set(w['reads'])<=set(v['reads']);assert {k:z for k,z in w.items() if k!='reads'}=={k:z for k,z in v.items() if k!='reads'}
  ma=json.loads((old/'metrics.json').read_text());mb=json.loads((new/'metrics.json').read_text())
  assert ma['total_simulated_time_ns']==mb['total_simulated_time_ns']
  assert ma['memory_peak_bytes_by_location']==mb['memory_peak_bytes_by_location']
  records.append(dict(platform=platform,status='PASS',model_ns=mb['total_simulated_time_ns'],event_count=mb['operation_rank_event_count'],policy_sha256=mb['policy_fingerprint'],native_all_events_identical=True,all_costs_dependencies_and_nonread_resources_identical=True))
 (x.out/'comparison.json').write_text(json.dumps(dict(status='PASS',comparisons=records),indent=2)+'\n');print(json.dumps(records))
if __name__=='__main__':main()
