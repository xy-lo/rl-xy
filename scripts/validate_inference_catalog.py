#!/usr/bin/env python3
"""Hold out non-node batches and ragged shapes from the TP planning surrogate."""
import argparse,json,sys,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/idea1_layered/legacy'));sys.path.append(str(ROOT/'src'))
from rl_xy.f2_runtime import make_operator,INPUTS,sha
from rl_xy.qwen3_native import NetworkProfiles
from gqa_cost import apply_generation_gqa
from idea1_controller.cost_catalog import validate_catalog
from idea1_controller.inference_catalog import InferenceServiceCatalog
from idea1_controller.cost_backend import install
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--fast-validation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
start=time.perf_counter();op=make_operator(a.model,'wafer',cache_dir=a.output.parent/'mapper-cache-fast');install(op,a.output.parent/'cost-accelerator',json.loads(a.fast_validation.read_text()));apply_generation_gqa(op)
data=json.loads(a.catalog.read_text());validate_catalog(data,op,a.model,sha(INPUTS/('Qwen3-'+a.model)/'config.json'));cat=InferenceServiceCatalog(data);net=NetworkProfiles(op,'unused',native=False);rows=[]
for tp in (1,2,4,8):
 nodes=cat._nodes['decode',tp];batches=sorted({(lo+hi)//2 for lo,hi in zip(nodes,nodes[1:]) if hi-lo>1})
 for b in batches:
  for stage in ('decode','prefill'):
   lengths=tuple(777+(i%3-1)*123 for i in range(b));contexts=tuple(4096+(i%3-1)*511 for i in range(b))
   blocks=op.forward((1,)*b,tp,'generation',contexts) if stage=='decode' else op.forward(lengths,tp,'generation')
   actual=net.blocks(tp,blocks);pred=cat.decode(tp,b,sum(contexts)/b) if stage=='decode' else cat.phase('prefill',tp,lengths)
   rows.append(dict(tp=tp,batch=b,phase=stage,relative_error=abs(pred-actual)/max(1,actual),operator_ns=actual,predicted_ns=pred))
errors=sorted(r['relative_error'] for r in rows);result=dict(status='PASS_BOUND_SIMULATOR_SHAPE_VALIDATION',model=a.model,catalog_sha256=sha(a.catalog),samples=len(rows),rows=rows,max_relative_error=max(errors),p95_relative_error=errors[math.ceil(.95*len(errors))-1],wall_s=time.perf_counter()-start,scope='held-out batch and ragged-length interpolation error against unchanged operator model; no physical hardware calibration')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
