#!/usr/bin/env python3
"""Execution wrapper for the unchanged hot-GEMM benchmark, capped at10% VRAM.

No GPU is accessed merely by importing this module. Run only after an explicit
fresh idle-GPU check and under the separate CPU/RAM/time/device container cap.
"""
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys


def capped_fraction(requested):
    if type(requested) not in (int,float) or not math.isfinite(requested) or not 0<requested<=1:
        raise ValueError('valid positive CUDA allocator fraction required')
    return min(float(requested),.1)


def main():
    args=sys.argv[1:]
    if '--out' not in args:
        raise ValueError('benchmark --out is required for wrapper provenance')
    target=Path(args[args.index('--out')+1])
    if target.exists():
        raise ValueError('preserve existing benchmark output')
    benchmark=Path(__file__).resolve().with_name('benchmark_gemm.py')
    import torch
    original=torch.cuda.set_per_process_memory_fraction;calls=[]
    def limited(fraction,device=None):
        actual=capped_fraction(fraction);calls.append({'requested':fraction,'applied':actual})
        return original(actual,device=device)
    torch.cuda.set_per_process_memory_fraction=limited
    previous=sys.argv
    try:
        sys.argv=[str(benchmark),*args]
        runpy.run_path(str(benchmark),run_name='__main__')
    finally:
        torch.cuda.set_per_process_memory_fraction=original;sys.argv=previous
        if target.exists():
            report=json.loads(target.read_text())
            report['allocator_cap_wrapper']={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'unchanged_benchmark_source_sha256':hashlib.sha256(benchmark.read_bytes()).hexdigest(),
                'fraction_calls':calls,'cache_regime':'repeated hot cuBLAS CUDA-graph GEMM',
                'whole_veRL_GPU_calibration_claimed':False}
            if torch.cuda.is_initialized():
                report['allocator_cap_wrapper'].update(peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_torch_reserved_bytes=torch.cuda.max_memory_reserved())
            target.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
