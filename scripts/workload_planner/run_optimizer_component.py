#!/usr/bin/env python3
"""4B native suffix component ablation; no rollout/training or learning claim."""
import argparse,json,os,sys,hashlib,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from search import Environment
from idea1_controller.gradient_routing import sparse_optimizer_plan
from idea1_controller.inference_groups import InferenceAdmission
from audit_idea1_gradient_route import audit_gradient_route
from rl_xy.qwen3_experiment import optimizer_plan
from rl_xy.qwen3_native import execute
from rl_xy.qwen3_cost import Cost

def main():
 p=argparse.ArgumentParser();p.add_argument('--offline-run',type=Path,required=True);p.add_argument('--native-root',type=Path,required=True)
 p.add_argument('--mode',choices=['original','sparse-roots'],required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True);start=time.perf_counter()
 env=Environment(a.offline_run/'public.json',a.output/'mapper-cache',injected_forecasts=json.loads((a.offline_run/'forecast.json').read_text()))
 result=json.loads((a.offline_run/'offline-candidates.json').read_text())['result'];regions={int(r):tuple(ws) for r,ws in result['training']['region_members'].items()}
 cap=InferenceAdmission(env.op).evaluate(2,16,env.pub)
 if a.mode=='sparse-roots':
  plan,proof=sparse_optimizer_plan(env.op,env.policy,regions,gradient_restore_peak_bytes=cap['phase_peak_bytes']['gradient_restore'],capacity_bytes=cap['capacity_bytes'])
  assert proof['mode']=='sparse-root-rs'
  cap['phase_peak_bytes']['optimizer_rs']=proof['capacity']['required_bytes']
  (a.output/'gradient-route.json').write_text(json.dumps(proof,indent=2))
 else:plan=optimizer_plan(env.op,env.policy)
 plan.local(tuple(range(54)),Cost(compute=1),'publish_actor_v1')
 os.environ['ASTRA_SIM_ROOT']=str(a.native_root);os.environ['ASTRA_SIM_BIN']=str(a.native_root/'build-dynarl/bin/AstraSim_Analytical_Congestion_Aware')
 os.environ['RLXY_RESOURCE_BACKEND']='1';os.environ['RLXY_MESH_POLICY']='xy';os.environ['RLXY_MESH_TICKS_PER_NS']='1000'
 os.environ.pop('RLXY_MESH_TELEMETRY',None);os.environ['RLXY_MESH_SUMMARY']=str(a.output/'mesh-summary.json')
 sources=[Path(__file__),ROOT/'src/idea1_controller/gradient_routing.py',ROOT/'scripts/audit_idea1_gradient_route.py']
 sources+=list(Path(os.environ['RLXY_COST_ROOT']).joinpath('src').rglob('*.py'))
 hashes={str(s):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}
 archive={}
 for i,s in enumerate(sources):
  dst=a.output/'source-snapshot'/str(i)/s.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s,dst);archive[str(s)]=str(dst)
 (a.output/'source-hashes.json').write_text(json.dumps(hashes));(a.output/'source-archive-map.json').write_text(json.dumps(archive))
 native=execute(plan,a.output/'native',timeout=1800)
 if a.mode=='sparse-roots':
  audit_gradient_route(a.output,json.loads((a.output/'native/operations.json').read_text()),regions,cap)
 for s,h in hashes.items():assert hashlib.sha256(Path(s).read_bytes()).hexdigest()==h,'source changed '+s
 assert native['native_exit_status']==0 and native['native_coverage_and_dependencies_checked']
 summary=dict(status='PASS_NATIVE_OPTIMIZER_COMPONENT_AUDIT',mode=a.mode,model='4B',regional_contributors=len(regions),canonical_owners=27,total_s=native['model_ns']/1e9,
  native_exit_status=native['native_exit_status'],binary_sha256=native['binary_sha256'],wall_s=time.perf_counter()-start,
  scope='isolated original synchronous optimizer suffix or sparse-root RS variant, assumed completed canonical gradients; no rollout/training tensors or learning quality')
 (a.output/'capacity.json').write_text(json.dumps(cap,indent=2));(a.output/'command.json').write_text(json.dumps(dict(argv=sys.argv,source_commit=os.popen('git rev-parse HEAD').read().strip()),indent=2))
 (a.output/'result.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
if __name__=='__main__':main()
