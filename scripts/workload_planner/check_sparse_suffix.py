#!/usr/bin/env python3
"""Compile/audit optional sparse suffix before any full native execution."""
import argparse,json,sys,copy,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from search import Environment
from idea1_controller.gradient_routing import sparse_optimizer_plan
from audit_idea1_gradient_route import audit_gradient_route
from rl_xy.qwen3_experiment import optimizer_plan

def main():
 p=argparse.ArgumentParser();p.add_argument('--offline-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
 env=Environment(a.offline_run/'public.json',a.output/'mapper-cache',injected_forecasts=json.loads((a.offline_run/'forecast.json').read_text()))
 result=json.loads((a.offline_run/'offline-candidates.json').read_text())['result'];regions={int(r):tuple(ws) for r,ws in result['training']['region_members'].items()}
 # Use independent admitted gradient restore bound from the standard protocol.
 # The inference admission exposes the same canonical gradient-restoration bound.
 from idea1_controller.inference_groups import InferenceAdmission
 cap=InferenceAdmission(env.op).evaluate(2,16,env.pub)
 base=cap['phase_peak_bytes']['gradient_restore']
 plan,proof=sparse_optimizer_plan(env.op,env.policy,regions,gradient_restore_peak_bytes=base,capacity_bytes=cap['capacity_bytes'],contributors_are_region_roots=True)
 assert proof['mode']=='sparse-root-rs'
 cap['phase_peak_bytes']['optimizer_rs']=proof['capacity']['required_bytes']
 (a.output/'gradient-route.json').write_text(json.dumps(proof,indent=2))
 rows=[dict(id=o.node_id,kind=o.kind,duration_ns=o.duration_ns,payload_bytes=o.payload_bytes,collective=o.collective,ranks=list(o.participant_ranks),src=o.src_rank,dst=o.dst_rank,name=o.op_id) for o in plan.operations]
 audit_gradient_route(a.output,rows,regions,cap)
 # Original optimizer compute bodies outside RS must remain byte-for-byte costs.
 old=optimizer_plan(env.op,env.policy)
 def body(plan):
  return [(plan.stats[o.node_id]) for o in plan.operations if o.kind=='compute' and not any(k in o.op_id for k in ('dp_gradient_RS','sparse_'))]
 assert body(plan)==body(old)
 for mutation in ('range','message','owner'):
  bad=copy.deepcopy(rows);saved=copy.deepcopy(proof)
  if mutation=='range':
   badproof=copy.deepcopy(proof);badproof['moves'][0]['global_end']+=1;(a.output/'gradient-route.json').write_text(json.dumps(badproof))
  elif mutation=='message':
   next(o for o in bad if o['kind']=='p2p')['payload_bytes']+=4
  else:
   next(o for o in bad if o['kind']=='collective' and 'dp_parameter_AG' in o['name'])['ranks'].reverse()
  try:audit_gradient_route(a.output,bad,regions,cap)
  except AssertionError:pass
  else:raise AssertionError('audit accepted '+mutation+' corruption')
  (a.output/'gradient-route.json').write_text(json.dumps(saved,indent=2))
 (a.output/'operations.json').write_text(json.dumps(rows))
 summary=dict(status='PASS_COMPILED_GRAPH_AND_NEGATIVE_AUDIT',native_executed=False,roots=len(regions),original_owners=27,operations=len(rows),extra_padding_bytes=proof['extra_allocation_bytes_per_contributor'],owner_route_payload_bytes=proof['logical_payload_bytes'],limitations='symbolic routing proof and toy numerical tests, not real Transformer numerical validation')
 (a.output/'result.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
if __name__=='__main__':main()
