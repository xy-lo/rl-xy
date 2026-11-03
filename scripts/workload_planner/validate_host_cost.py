#!/usr/bin/env python3
from pathlib import Path
import os,sys,json,time
ROOT=Path(__file__).resolve().parents[2];cost=Path(os.environ['RLXY_COST_ROOT'])
sys.path[:0]=[str(cost/'src'),str(ROOT/'src')]
from rl_xy.f2_runtime import make_operator
from rl_xy.operator_ir import Matmul
from idea1_controller.cost_backend import build,result_for

def main():
 out=Path(sys.argv[1]);op=make_operator('4B','wafer',cache_dir=out.parent/'validation-mapper-cache')
 import timeline_provider as provider
 fn,identity=build(out.parent/'validated-cost-accelerator')
 oldfile=json.loads((ROOT/'experiments/2026-11-03-idea1-elastic/assets/cost-backend-validation.json').read_text())
 problems=[Matmul(*c['shape']) for c in oldfile['cases']]
 problems += [Matmul(17,33,65),Matmul(1,1024,4096),Matmul(32,64,128,a_bytes=4,b_bytes=4,output_bytes=4)]
 cases=[]
 for p in problems:
  before=time.perf_counter();old=op.provider._compute(p,op.hw);python_s=time.perf_counter()-before
  parts=provider.capacity_partition(p,op.hw)
  new=result_for(fn,p,op.hw,parts,op.provider.load,op.provider.store)
  equal=old==new
  cases.append(dict(problem=p.__dict__,equal=equal,old=old,new=new,python_s=python_s))
  if not equal:
   out.write_text(json.dumps(dict(status='FAIL',identity=identity,cases=cases),indent=2));raise ValueError('host acceleration differs from Python oracle')
 out.write_text(json.dumps(dict(status='PASS_EXACT_ALL_FIELDS',identity=identity,cases=cases,scope='Linux compiler checked against frozen Python event model; not hardware calibration'),indent=2))
 print('PASS_EXACT_ALL_FIELDS',len(cases),flush=True)

if __name__=='__main__':main()
