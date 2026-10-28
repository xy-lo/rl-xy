#!/usr/bin/env python3
"""Bounded exact-shape cache-regime probe, never an RL/model training run.

Inputs are <=12 explicitly selected BF16 GEMM signatures and a bounded ordered
shape sequence. Every sequence occurrence has independent synthetic buffers.
CUDA-event intervals exclude128MiB pre-reads and the memory-neutral enqueue
 guard; sequence sums therefore are NOT an end-to-end latency measurement.
"""
import argparse
import ctypes as C
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

GIB = 1024**3
EVICTION_BYTES = 128 * 1024**2
MAX_SEQUENCE_TENSOR_BYTES = 6 * GIB


def signature(problem):
    return hashlib.sha256(json.dumps(problem,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def allocation_bytes(problem):
    b,m,n,k = (problem[x] for x in ('batches','m','n','k'))
    return b * (m*k*2 + k*n*2 + m*n*problem['output_bytes'])


def validate_selection(data):
    if data.get('schema_version') != 'rlxy.gemm_cache_selection/1':
        raise ValueError('require explicit rlxy.gemm_cache_selection/1, not the full4811 corpus')
    shapes = data.get('shapes',{})
    if not 1 <= len(shapes) <= 12:
        raise ValueError('select between1 and12 unique signatures')
    allowed = {'m','n','k','batches','a_bytes','b_bytes','output_bytes','accumulator_bytes','direction'}
    categories = set()
    for key,p in shapes.items():
        if set(p) != allowed or signature(p) != key:
            raise ValueError('exact signature schema/hash mismatch')
        if any(type(p[x]) is not int or not 0 < p[x] < 2**31 for x in ('m','n','k','batches')):
            raise ValueError('positive int32 GEMM dimensions required')
        if (p['a_bytes'],p['b_bytes'],p['accumulator_bytes']) != (2,2,4) or p['output_bytes'] not in (2,4):
            raise ValueError('selected ABI is BF16 input, FP32 accumulator, BF16/FP32 output')
        if p['direction'] not in ('forward','dinput','dweight','attention_backward'):
            raise ValueError('unknown direction')
        if p['direction']=='forward' and p['m']<=8:categories.add('decode_small_M')
        if p['direction']=='forward' and p['m']>=128:categories.add('prefill_or_teacher_forward')
        if p['direction']=='dweight':categories.add('dweight')
    sequence = data.get('ordered_sequence')
    if not isinstance(sequence,list) or not 1<=len(sequence)<=24 or any(k not in shapes for k in sequence):
        raise ValueError('explicit ordered sequence of1..24 selected keys required')
    if set(sequence)!=set(shapes):
        raise ValueError('sequence must cover every selected signature')
    if not isinstance(data.get('sequence_origin'),str) or not data['sequence_origin'].strip():
        raise ValueError('declare actual sequence source/order; do not imply a published runtime trace')
    # Repeated identical signatures are different logical operator instances,
    # and therefore use independent buffers to avoid fake cross-layer reuse.
    sequence_bytes = sum(allocation_bytes(shapes[key]) for key in sequence)
    if sequence_bytes > MAX_SEQUENCE_TENSOR_BYTES:
        raise ValueError('sequence resident A/B/C buffers exceed6GiB; reduce selected sequence')
    return {'selected_signature_count':len(shapes),'sequence_occurrences':len(sequence),
        'sequence_tensor_bytes':sequence_bytes,'eviction_buffer_bytes':EVICTION_BYTES,
        'conservative_sequence_allocation_with_context_margin_bytes':sequence_bytes+EVICTION_BYTES+512*1024**2,
        'categories_present':sorted(categories),
        'missing_requested_categories':sorted({'decode_small_M','prefill_or_teacher_forward','dweight'}-categories),
        'sequence_instance_policy':'independent synthetic A/B/C for every occurrence, including repeated signatures',
        'preflight_passed':True}


def summarize_samples(values):
    if not values or any(not math.isfinite(x) or x<=0 for x in values):
        raise ValueError('positive finite CUDA event durations required')
    ordered=sorted(values)
    def q(at):
        x=at*(len(ordered)-1);a=math.floor(x);b=math.ceil(x)
        return ordered[a] if a==b else ordered[a]*(b-x)+ordered[b]*(x-a)
    return {'samples':len(values),'median_ns':statistics.median(values),'p10_ns':q(.1),'p90_ns':q(.9),
        'min_ns':min(values),'max_ns':max(values),'raw_samples_ns':values}


def combine_repeats(repeats):
    if len(repeats)!=2:
        raise ValueError('this probe requires exactly two independent timing repetitions')
    medians=[x['median_ns'] for x in repeats]
    return {'repeats':repeats,'median_of_repeat_medians_ns':statistics.median(medians),
        'repeat_relative_difference':abs(medians[0]-medians[1])/max(medians),
        'not_a_statistical_generalization_error_bound':True}


def run_probe(selection,out,*,samples=20,deadline_seconds=150,enqueue_guard_cycles=1_000_000):
    preflight=validate_selection(selection)
    if not 2<=samples<=40 or not 1<=deadline_seconds<=170 or not 100_000<=enqueue_guard_cycles<=10_000_000:
        raise ValueError('bounded samples/deadline/guard configuration required')
    if preflight['missing_requested_categories']:
        raise ValueError('main probe must include decode small-M, prefill/teacher forward, and dweight')
    if out.exists():
        raise ValueError('preserve existing output; choose a new path')
    start=time.monotonic()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('isolate exactly one permitted GPU')
    torch.cuda.set_per_process_memory_fraction(.1)
    torch.backends.cuda.matmul.allow_tf32=False
    if not hasattr(torch.cuda,'_sleep'):
        raise ValueError('enqueue guard unavailable; do not compare biased event protocols')
    free,total=torch.cuda.mem_get_info()
    if free<12*GIB:
        raise ValueError('fresh isolation check requires at least12GiB free on the assigned GPU')
    device=torch.cuda.get_device_name(0)
    if device!='NVIDIA A100-SXM4-80GB':
        raise ValueError('this comparison is scoped to matching A100-SXM4-80GB')
    report={'schema_version':'rlxy.gemm_cache_regimes/1','status':'running',
        'started_at_utc':datetime.now(timezone.utc).isoformat(),'selection':selection,'preflight':preflight,
        'GPU_used':True,'device':device,'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
        'timing_protocol':'per-GEMM CUDA events after memory-neutral enqueue guard; guard/eviction outside event interval',
        'enqueue_guard_cycles':enqueue_guard_cycles,'samples_per_repeat':samples,'repeat_count':2,
        'memory_fraction':.1,'GPU_memory_budget_bytes':8*GIB,
        'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'eviction_buffer_dtype':'FP32','eviction_buffer_numel':EVICTION_BYTES//4,'signature_results':{},'sequence_results':{},
        'scope':'synthetic contiguous cuBLAS BF16 GEMM cache probe; not fused vLLM, actual model values, or whole-step validation',
        'GPU_cluster_model_remains_simulated':True,
        'absolute_times_not_directly_interchangeable_with_other_CUDA_graph_protocols':True,
        'clear_L2_claim':'128MiB independent buffer pre-read perturbation; no claim of measured complete L2 eviction',
        'numeric_checks':{},'sequence_dataflow':'independent synthetic A/B/C, shape order only; not wired model activations',
        'benchmark_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out.parent.mkdir(parents=True,exist_ok=True)
    def save():
        tmp=out.with_suffix(out.suffix+'.partial');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(out)
    def check():
        if time.monotonic()-start>deadline_seconds:raise TimeoutError('bounded probe deadline; partial results remain non-PASS')
        # Allocation preflight reserves512MiB for context/cuBLAS/temporary overhead.
        if torch.cuda.memory_reserved()+512*1024**2>8*GIB:
            raise MemoryError('conservative8GiB GPU allocation budget reached')
    lib=C.CDLL('libcublas.so.12');handle=C.c_void_p()
    def checked(code):
        if code:raise RuntimeError('cuBLAS status '+str(code))
    lib.cublasCreate_v2.argtypes=[C.POINTER(C.c_void_p)]
    lib.cublasDestroy_v2.argtypes=[C.c_void_p]
    lib.cublasSetStream_v2.argtypes=[C.c_void_p,C.c_void_p]
    fn=lib.cublasGemmStridedBatchedEx
    fn.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,C.c_void_p,
        C.c_void_p,C.c_int,C.c_int,C.c_longlong,C.c_void_p,C.c_int,C.c_int,C.c_longlong,
        C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_longlong,C.c_int,C.c_int,C.c_int]
    checked(lib.cublasCreate_v2(C.byref(handle)));alpha=C.c_float(1);beta=C.c_float(0)
    def make(p,seed):
        check();torch.manual_seed(seed)
        b,m,n,k=(p[x] for x in ('batches','m','n','k'))
        lhs=torch.randn((b,m,k),device='cuda',dtype=torch.bfloat16)
        rhs=torch.randn((b,k,n),device='cuda',dtype=torch.bfloat16)
        target=torch.empty((b,m,n),device='cuda',dtype=torch.bfloat16 if p['output_bytes']==2 else torch.float32)
        def call():
            checked(lib.cublasSetStream_v2(handle,C.c_void_p(torch.cuda.current_stream().cuda_stream)))
            checked(fn(handle,0,0,n,m,k,C.byref(alpha),C.c_void_p(rhs.data_ptr()),14,n,k*n,
                C.c_void_p(lhs.data_ptr()),14,k,m*k,C.byref(beta),C.c_void_p(target.data_ptr()),
                14 if p['output_bytes']==2 else 0,n,m*n,b,68,-1))
        check();return call,lhs,rhs,target
    def numeric(bundle,p):
        call,lhs,rhs,target=bundle;call();torch.cuda.synchronize()
        b,m,n,k=(p[x] for x in ('batches','m','n','k'));rows=[]
        for batch in sorted({0,b-1}):
            for i in sorted({0,m//2,m-1}):
                for j in sorted({0,n//2,n-1}):
                    # CPU float64 dot references use the actual BF16 operands;
                    # only sampled rows/columns transfer, never a huge FP64 GEMM.
                    left=lhs[batch,i,:].to(device='cpu',dtype=torch.float64)
                    right=rhs[batch,:,j].to(device='cpu',dtype=torch.float64)
                    ref=float((left*right).sum());got=float(target[batch,i,j].item())
                    tolerance=(.02+.02*abs(ref)) if p['output_bytes']==2 else (.002+1e-4*abs(ref))
                    if not math.isfinite(got) or abs(got-ref)>tolerance:
                        raise ValueError('sampled exact-shape GEMM numeric fixture failed')
                    rows.append({'batch':batch,'row':i,'column':j,'actual':got,'FP64_CPU_dot':ref,
                        'absolute_error':abs(got-ref),'declared_tolerance':tolerance})
        return {'passed':True,'scope':'sampled output entries, not every full output element','entries':rows}
    eviction=torch.ones(EVICTION_BYTES//4,device='cuda',dtype=torch.float32)
    def clear_read():
        # Eager sum reads the complete independent128MiB buffer. Its stream
        # work precedes the event pair and is not subtracted numerically.
        return eviction.sum(dtype=torch.float32)
    def measured(call,*,evict=False):
        check();sink=clear_read() if evict else None
        torch.cuda._sleep(enqueue_guard_cycles)
        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        begin.record();call();end.record();end.synchronize()
        elapsed=float(begin.elapsed_time(end))*1e6
        # Keep the pre-read result alive until the stream has passed the GEMM.
        if sink is not None:del sink
        return elapsed
    save()
    try:
        for index,(key,p) in enumerate(selection['shapes'].items()):
            bundle=make(p,1234+index);call=bundle[0]
            before=numeric(bundle,p)
            repeats={name:[] for name in ['hot_repeat','pre_read_128MiB']}
            for mode in repeats:
                for repeat in range(2):
                    check()
                    for _ in range(5):call()
                    torch.cuda.synchronize()
                    values=[measured(call,evict=mode=='pre_read_128MiB') for _ in range(samples)]
                    repeats[mode].append(summarize_samples(values))
            after=numeric(bundle,p)
            report['numeric_checks'][key]=[before,after]
            report['signature_results'][key]={'problem':p,
                **{mode:combine_repeats(values) for mode,values in repeats.items()}}
            report['signature_results'][key]['pre_read_over_hot']= (
                report['signature_results'][key]['pre_read_128MiB']['median_of_repeat_medians_ns']/
                report['signature_results'][key]['hot_repeat']['median_of_repeat_medians_ns'])
            del call,bundle;torch.cuda.empty_cache();save()
        sequence=selection['ordered_sequence']
        bundles=[make(selection['shapes'][key],9900+i) for i,key in enumerate(sequence)]
        sequence_repeats=[]
        for repeat in range(2):
            by_position=[[] for _ in sequence]
            for sample in range(samples):
                check();sink=clear_read()
                for position,bundle in enumerate(bundles):
                    by_position[position].append(measured(bundle[0]))
                del sink
            sequence_repeats.append([summarize_samples(xs) for xs in by_position]);save()
        for position,key in enumerate(sequence):
            report['sequence_results'][str(position)]={'signature_sha256':key,
                'independent_buffer_instance':position,
                **combine_repeats([r[position] for r in sequence_repeats])}
        for key in selection['shapes']:
            values=[row['median_of_repeat_medians_ns'] for row in report['sequence_results'].values() if row['signature_sha256']==key]
            report['signature_results'][key]['sequence_occurrence_median_ns']=statistics.median(values)
            report['signature_results'][key]['sequence_over_hot']=statistics.median(values)/report['signature_results'][key]['hot_repeat']['median_of_repeat_medians_ns']
        report['numeric_checks_sequence']=[numeric(bundle,selection['shapes'][key]) for bundle,key in zip(bundles,sequence)]
        report['repeat_difference_over15percent']=[{'signature':key,'regime':regime}
            for key,row in report['signature_results'].items()
            for regime in ['hot_repeat','pre_read_128MiB'] if row[regime]['repeat_relative_difference']>.15]
        report.update(status='PASS',finished_at_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_wall_seconds=time.monotonic()-start,
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),
            sum_sequence_timed_GEMM_medians_ns=sum(x['median_of_repeat_medians_ns'] for x in report['sequence_results'].values()),
            entire_sequence_wall_latency_claimed=False,full_veRL_cost_validated=False)
        check();save()
    except BaseException as error:
        report.update(status='FAIL',error=type(error).__name__+': '+str(error),elapsed_wall_seconds=time.monotonic()-start);save();raise
    finally:
        checked(lib.cublasDestroy_v2(handle))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--samples',type=int,default=20)
    p.add_argument('--deadline-seconds',type=int,default=150)
    p.add_argument('--enqueue-guard-cycles',type=int,default=1_000_000)
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args();selection=json.loads(a.selection.read_text())
    if a.preflight_only:
        result={'status':'PREFLIGHT_ONLY',**validate_selection(selection),'GPU_used':False}
        if a.out.exists():p.error('output exists')
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    else:
        result=run_probe(selection,a.out,samples=a.samples,deadline_seconds=a.deadline_seconds,
            enqueue_guard_cycles=a.enqueue_guard_cycles)
        result['selection_file_sha256']=hashlib.sha256(a.selection.read_bytes()).hexdigest()
        a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','GPU_used','elapsed_wall_seconds'] if k in result},indent=2))


if __name__=='__main__':main()
