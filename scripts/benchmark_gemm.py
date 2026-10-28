#!/usr/bin/env python3
"""Bounded A100 cuBLAS BF16 microbenchmarks; never trains an RL/model step.

The main cluster comparison remains simulation. CUDA-graph hot-kernel labels
are shape-specific reference measurements, not a calibrated whole veRL step.
"""
import argparse
import ctypes as C
from datetime import datetime,timezone
import hashlib,json,statistics,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--shapes',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-seconds',type=int,default=900);p.add_argument('--graph-ms',type=int,default=5)
    p.add_argument('--seed-profile',type=Path,help='reuse matching individually verified labels; all new shapes still require measurement')
    a=p.parse_args();assert not a.out.exists()
    import torch
    from triton.testing import do_bench_cudagraph
    assert torch.cuda.is_available() and torch.cuda.device_count()==1,'isolate exactly one permitted GPU'
    torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(0.25)
    free,total=torch.cuda.mem_get_info();assert free>24*1024**3,'GPU does not have sufficient free memory'
    lib=C.CDLL('libcublas.so.12');handle=C.c_void_p()
    def checked(code):
        if code:raise RuntimeError('cuBLAS status '+str(code))
    lib.cublasCreate_v2.argtypes=[C.POINTER(C.c_void_p)];checked(lib.cublasCreate_v2(C.byref(handle)))
    lib.cublasSetStream_v2.argtypes=[C.c_void_p,C.c_void_p]
    fn=lib.cublasGemmStridedBatchedEx
    fn.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,C.c_void_p,
        C.c_void_p,C.c_int,C.c_int,C.c_longlong,C.c_void_p,C.c_int,C.c_int,C.c_longlong,
        C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_longlong,C.c_int,C.c_int,C.c_int]
    alpha=C.c_float(1);beta=C.c_float(0)
    def make(problem):
        assert problem['a_bytes']==problem['b_bytes']==2 and problem['output_bytes'] in (2,4)
        b,m,n,k=(problem[x] for x in ('batches','m','n','k'))
        needed=b*(m*k*2+k*n*2+m*n*problem['output_bytes'])
        if needed>4*1024**3:raise ValueError('bounded benchmark tensor allocation exceeds 4GiB')
        lhs=torch.randn((b,m,k),device='cuda',dtype=torch.bfloat16)
        rhs=torch.randn((b,k,n),device='cuda',dtype=torch.bfloat16)
        out=torch.empty((b,m,n),device='cuda',dtype=torch.bfloat16 if problem['output_bytes']==2 else torch.float32)
        def call():
            checked(lib.cublasSetStream_v2(handle,C.c_void_p(torch.cuda.current_stream().cuda_stream)))
            checked(fn(handle,0,0,n,m,k,C.byref(alpha),C.c_void_p(rhs.data_ptr()),14,n,k*n,
                C.c_void_p(lhs.data_ptr()),14,k,m*k,C.byref(beta),C.c_void_p(out.data_ptr()),
                14 if problem['output_bytes']==2 else 0,n,m*n,b,68,-1))
        return call,lhs,rhs,out
    numeric=[]
    for width,batch in [(2,1),(4,1),(4,3)]:
        call,lhs,rhs,out=make(dict(m=13,n=17,k=29,batches=batch,a_bytes=2,b_bytes=2,output_bytes=width))
        call();torch.cuda.synchronize();ref=torch.bmm(lhs.float(),rhs.float())
        torch.testing.assert_close(out.float(),ref,rtol=0.02 if width==2 else 1e-4,atol=0.02 if width==2 else 1e-4)
        numeric.append({'output_bytes':width,'batches':batch,'passed':True,'max_abs_error':float((out.float()-ref).abs().max())})
    source=json.loads(a.shapes.read_text());start=time.monotonic()
    report={'status':'running','schema_version':'rlxy.gemm_measurements/1','started_at_utc':datetime.now(timezone.utc).isoformat(),
        'device':torch.cuda.get_device_name(0),'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
        'source_shapes_sha256':hashlib.sha256(a.shapes.read_bytes()).hexdigest(),
        'benchmark_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'numeric_fixture_checks':numeric,'measurements':{},
        'scope':'hot CUDA graph cuBLAS BF16-input GEMM; includes device memory service; not CPU launch, vLLM fusion or whole-step calibration',
        'abi':'cublasGemmStridedBatchedEx CUDA_R_16BF=14 CUDA_R_32F=0 CUBLAS_COMPUTE_32F=68 DEFAULT=-1'}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if a.seed_profile:
        seed=json.loads(a.seed_profile.read_text())
        assert seed['schema_version']==report['schema_version'] and seed['device']==report['device']
        assert all(x['passed'] for x in seed['numeric_fixture_checks'])
        for key,row in seed['measurements'].items():
            if key in source['shapes'] and row['problem']==source['shapes'][key] and row['repeat_relative_difference']<=0.15:
                report['measurements'][key]=row
        report['seed_profile_sha256']=hashlib.sha256(a.seed_profile.read_bytes()).hexdigest()
        report['seeded_signature_count']=len(report['measurements'])
    def save():
        tmp=a.out.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(a.out)
    try:
        for i,(key,problem) in enumerate(sorted(source['shapes'].items())):
            if key in report['measurements']:continue
            if time.monotonic()-start>a.max_seconds:raise TimeoutError('bounded GEMM profiling timeout; partial labels not a complete profile')
            call,lhs,rhs,out=make(problem)
            call();call();torch.cuda.synchronize()
            first=float(do_bench_cudagraph(call,rep=a.graph_ms));second=float(do_bench_cudagraph(call,rep=a.graph_ms))
            if min(first,second)<=0:raise ValueError('nonpositive CUDA graph measurement')
            report['measurements'][key]={'problem':problem,'first_ms':first,'repeat_ms':second,
                'duration_ns':max(1,round(statistics.median([first,second])*1e6)),
                'repeat_relative_difference':abs(first-second)/max(first,second)}
            del call,lhs,rhs,out
            if (i+1)%50==0:save();print(i+1,'/',len(source['shapes']),flush=True)
        report.update(status='PASS',finished_at_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-start)
    except BaseException as exc:report.update(status='FAIL',error=type(exc).__name__+': '+str(exc));raise
    finally:
        save();lib.cublasDestroy_v2(handle)
    print('GEMM profile PASS',len(report['measurements']),flush=True)


if __name__=='__main__':main()
