#!/usr/bin/env python3
"""Query unchanged operator services at public shapes; never consume live R."""
import argparse,json,hashlib,sys,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/idea1_layered/legacy'));sys.path.append(str(ROOT/'src'))
from rl_xy.f2_runtime import make_operator,INPUTS
from rl_xy.qwen3_native import NetworkProfiles
from gqa_cost import apply_generation_gqa
from idea1_controller.cost_backend import install
from idea1_controller.cost_catalog import validate_catalog


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--base-catalog',type=Path,required=True)
    p.add_argument('--frontier',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--fast-validation',type=Path,required=True)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    import numpy as np
    op=make_operator(a.model,'wafer',cache_dir=a.output.parent/'mapper-cache-fast');backend=install(op,a.output.parent/'cost-accelerator',json.loads(a.fast_validation.read_text()));apply_generation_gqa(op)
    data=json.loads(a.base_catalog.read_text());validate_catalog(data,op,a.model,hashlib.sha256((INPUTS/('Qwen3-'+a.model)/'config.json').read_bytes()).hexdigest())
    table=data['table'];frontier=json.loads(a.frontier.read_text())['frontier'];net=NetworkProfiles(op,'unused',native=False);errors=[];added=0
    for tp in (1,2,4,8):
        cap=frontier[str(tp)];sizes=sorted(set([b for b in (1,2,4,8,16,24,32,48,64,96,128,192,256) if b<=cap]+[cap]))
        for batch in sizes:
            key=f'decode:{tp}:{batch}'
            if key not in table:
                values=[net.blocks(tp,op.forward((1,)*batch,tp,'generation',(c,)*batch)) for c in (1,11060)]
                slope=(values[1]-values[0])/11059;table[key]=[values[0]-slope,slope];added+=1
                for c in (2048,8192):
                    actual=net.blocks(tp,op.forward((1,)*batch,tp,'generation',(c,)*batch));pred=table[key][0]+table[key][1]*c
                    errors.append(abs(pred-actual)/max(1,actual))
            key=f'prefill:{tp}:{batch}'
            if key not in table:
                design=[];values=[]
                for length in (41,256,530,1024,2048,2869):
                    design.append([1,batch*length,batch*length*length]);values.append(net.blocks(tp,op.forward((length,)*batch,tp,'generation')))
                design=np.asarray(design,dtype=float);scale=np.maximum(np.max(abs(design),axis=0),1)
                co=np.linalg.lstsq(design/scale,np.asarray(values),rcond=None)[0]/scale;table[key]=co.tolist();added+=1
                for length in (128,768,1536):
                    actual=net.blocks(tp,op.forward((length,)*batch,tp,'generation'));pred=float(np.dot(co,[1,batch*length,batch*length*length]))
                    errors.append(abs(pred-actual)/max(1,actual))
            print(json.dumps(dict(tp=tp,batch=batch,added=added)),flush=True)
    data.update(inference_tp_supported=[1,2,4,8],inference_capacity_frontier=frontier,inference_extension_cpu_s=time.perf_counter()-start,
        inference_extension_new_nodes=added,inference_extension_max_shape_error=max(errors) if errors else 0,
        inference_extension_p95_shape_error=sorted(errors)[math.ceil(.95*len(errors))-1] if errors else 0,
        inference_host_backend=backend,scope='unchanged simulator services; sparse batch interpolation and alpha-beta network, not exact mesh or physical calibration')
    a.output.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(dict(status='PASS_BOUND_INFERENCE_SERVICE_CATALOG',model=a.model,added_nodes=added,wall_s=time.perf_counter()-start)))

if __name__=='__main__':main()
