#!/usr/bin/env python3
"""Enumerate exact BF16 cuBLAS signatures needed by selected model layouts."""
import argparse,json,hashlib,sys
from dataclasses import asdict,replace
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rl_xy.config import load_case
from rl_xy.colocation import StageBuilder


def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    shapes={};source={}
    for label,tps in [('qwen05b',[1,2]),('qwen3b',[1,2]),('qwen7b',[1,2,4])]:
        path=a.cases/label/'case.json';case=load_case(path);source[label]=hashlib.sha256(path.read_bytes()).hexdigest()
        for tp in tps:
            local=replace(case,hardware=replace(case.hardware,dies=tp),strategy=replace(case.strategy,tensor_parallel=tp))
            b=StageBuilder(local,{'input','rollout','reference','old_policy','train','clip','optimizer','publish','cleanup'});b.build_round(0)
            for op in b.operations:
                if op.kernel_spec:
                    for mat in op.kernel_spec.matmuls:
                        d=asdict(mat);key=hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                        shapes[key]=d
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps({'schema_version':'rlxy.gemm_shapes/1','source_cases':source,'shapes':shapes},indent=2)+'\n')
    print(len(shapes),'exact GEMM signatures')


if __name__=='__main__':main()
