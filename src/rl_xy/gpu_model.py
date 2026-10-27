"""Independent SM/CTA GPU analytic model; not a renamed Dojo OS mapper.

Static output CTAs round-robin over SMs; no split K, no cross-CTA L2 reuse,
one active CTA per SM. Shared memory is private per SM. All local compute,
HBM and shared-memory service are serial, matching the conservative wafer
timing contract. No CUDA fusion/occupancy calibration is implied.
"""
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path
from .cost_model import CostEstimate
from .mapping import ceildiv


@lru_cache(maxsize=4096)
def cta_mapping(problem,sms,capacity,tm,tn,tk):
    flops=[0]*sms;traffic=[0]*sms;workspace=[0]*sms
    reads=writes=blocks=0
    for _ in range(problem.batches):
        for m in range(0,problem.m,tm):
            mc=min(tm,problem.m-m)
            for n in range(0,problem.n,tn):
                nc=min(tn,problem.n-n);kc=min(tk,problem.k);sm=blocks%sms
                need=mc*kc*problem.a_bytes+kc*nc*problem.b_bytes+mc*nc*problem.accumulator_bytes
                if need>capacity:raise ValueError('GPU CTA exceeds private SM shared memory')
                r=mc*problem.k*problem.a_bytes+problem.k*nc*problem.b_bytes
                w=mc*nc*problem.output_bytes
                flops[sm]+=2*mc*nc*problem.k;traffic[sm]+=r+w
                workspace[sm]=max(workspace[sm],need)
                reads+=r;writes+=w;blocks+=1
    if sum(flops)!=problem.flops:raise ValueError('GPU CTA FLOP conservation failed')
    return tuple(flops),tuple(traffic),tuple(workspace),reads,writes,blocks


class GPUCostProvider:
    def __init__(self,profile):self.p=profile

    def estimate(self,spec,hardware):
        p=self.p;sms=p['sms_per_gpu'];workspace=[0]*sms;core=[0]*sms
        reads=writes=compute=memory=0;partitions=[]
        if spec.execution_domain!='device' or hardware.cores_per_die!=sms:
            raise ValueError('GPU device/SM contract mismatch')
        for problem in spec.matmuls:
            rate=p['bf16_flops_per_ns'] if max(problem.a_bytes,problem.b_bytes)<=2 else p['fp32_flops_per_ns']
            candidates=[]
            for tile in p.get('cta_candidates',[p['cta_tile']]):
                try:f,t,w,r,wr,blocks=cta_mapping(problem,sms,hardware.sram_bytes_per_core,*tile)
                except ValueError:continue
                work=[ceildiv(x*sms,rate) for x in f]
                service=max(ceildiv(r+wr,p['hbm_bytes_per_ns']),
                            max(ceildiv(x,p['shared_bytes_per_ns_per_sm']) for x in t))
                candidates.append((max(work)+service,r+wr,tuple(tile),f,w,r,wr,blocks,work,service))
            if not candidates:raise ValueError('no legal GPU CTA in private SM shared memory')
            _,_,tile,f,w,r,wr,blocks,work,service=min(candidates,key=lambda c:c[:3])
            core=[a+b for a,b in zip(core,work)]
            workspace=[max(a,b) for a,b in zip(workspace,w)]
            compute+=max(work)
            memory+=service
            reads+=r;writes+=wr
            partitions.append({'policy':'gpu_cta_output_tiles_round_robin_no_split_K',
                'selected_cta_tile':list(tile),'candidate_policy':'minimum_declared_compute_plus_local_service',
                'cta_count':blocks,'active_sms':min(blocks,sms),'flops_sum':sum(f),
                'read_bytes_sum':r,'write_bytes_sum':wr})
        active=min(sms,max(spec.vector_ops,spec.read_bytes+spec.write_bytes))
        if active:
            vector_ns=ceildiv(ceildiv(spec.vector_ops,active)*sms,p['vector_ops_per_ns'])
            compute+=vector_ns
            for sm in range(active):
                workspace[sm]=max(workspace[sm],16);core[sm]+=vector_ns
            traffic=spec.read_bytes+spec.write_bytes
            memory+=max(ceildiv(traffic,p['hbm_bytes_per_ns']),
                        ceildiv(ceildiv(traffic,active),p['shared_bytes_per_ns_per_sm']))
            reads+=spec.read_bytes;writes+=spec.write_bytes
        # Launch is a declared, separately documented scenario assumption.
        # It is included once in compute_ns so the additive timing contract
        # remains explicit and existing resource/statistics checks still apply.
        compute+=p['kernel_launch_ns']
        return CostEstimate(max(1,compute+memory),compute,memory,reads,writes,
            sum(workspace),(),source='analytic_gpu_cta_v1_'+p['precision_mode']+'_uncalibrated',
            core_workspace_bytes=tuple(workspace),core_compute_ns=tuple(core),partitions=tuple(partitions))


def load_gpu_profile(path):
    p=json.loads(Path(path).read_text())
    if p['schema_version']!='rlxy.gpu/1':raise ValueError('unsupported GPU profile')
    for key in ('gpu_count','gpus_per_node','sms_per_gpu','shared_bytes_per_sm','dram_bytes_per_gpu',
                'hbm_bytes_per_ns','bf16_flops_per_ns','fp32_flops_per_ns','vector_ops_per_ns',
                'shared_bytes_per_ns_per_sm','nvlink_bytes_per_ns_per_direction',
                'node_uplink_bytes_per_ns_per_direction','nvlink_latency_ns','uplink_latency_ns'):
        if type(p[key]) is not int or p[key]<=0:raise ValueError('positive integer GPU field required: '+key)
    if type(p['kernel_launch_ns']) is not int or p['kernel_launch_ns']<0:raise ValueError('invalid launch')
    if len(p['cta_tile'])!=3 or any(type(v) is not int or v<=0 for v in p['cta_tile']):raise ValueError('invalid CTA')
    if 'cta_candidates' in p and (not p['cta_candidates'] or any(len(t)!=3 or any(type(v) is not int or v<=0 for v in t) for t in p['cta_candidates'])):
        raise ValueError('invalid adaptive CTA candidates')
    if p['gpu_count']%p['gpus_per_node'] or p['gpu_count']%2:raise ValueError('GPU topology/TP2 mismatch')
    if p['precision_mode'] not in ('strict_declared_inputs','optimistic_tensorcore_sensitivity'):
        raise ValueError('GPU precision policy must be explicit')
    if p['precision_mode']=='optimistic_tensorcore_sensitivity' and p['physical_sku_prediction']:
        raise ValueError('optimistic FP32-to-TC rate is not a physical SKU prediction')
    return p


def effective_gpu_case(case,p):
    # Rates here are placeholders for old typed fields; only GPUCostProvider
    # supplies kernel costs, using aggregate peaks with exact integer division.
    return replace(case,hardware=replace(case.hardware,cores_per_die=p['sms_per_gpu'],
        sram_bytes_per_core=p['shared_bytes_per_sm'],dram_bytes_per_die=p['dram_bytes_per_gpu'],
        dram_bytes_per_ns=p['hbm_bytes_per_ns'],module_path_bytes_per_ns=p['hbm_bytes_per_ns'],
        sram_bytes_per_ns=p['shared_bytes_per_ns_per_sm'],
        gemm_flops_per_ns=ceildiv(p['bf16_flops_per_ns'],p['sms_per_gpu']),
        gemm_fp32_flops_per_ns=ceildiv(p['fp32_flops_per_ns'],p['sms_per_gpu']),
        vector_ops_per_ns=ceildiv(p['vector_ops_per_ns'],p['sms_per_gpu']),
        d2d_bytes_per_ns_per_direction=p['nvlink_bytes_per_ns_per_direction'],
        d2d_latency_ns=p['nvlink_latency_ns']))


def budget_audit(wafer_metrics,wafer_resolved,p):
    h=wafer_resolved['effective_hardware'];dies=wafer_metrics['dies'];cores=dies*h['cores_per_die']
    wafer={'bf16_peak_flops_per_ns':cores*h['gemm_flops_per_ns'],
        'fp32_peak_flops_per_ns':cores*h['gemm_fp32_flops_per_ns'],
        'vector_ops_per_ns':cores*h['vector_ops_per_ns'],
        'dram_capacity_bytes':dies*h['dram_bytes_per_die'],
        'dram_bandwidth_bytes_per_ns':dies*h['dram_bytes_per_ns'],
        'private_sram_capacity_bytes':cores*h['sram_bytes_per_core']}
    n=p['gpu_count'];gpu={'bf16_peak_flops_per_ns':n*p['bf16_flops_per_ns'],
        'fp32_peak_flops_per_ns':n*p['fp32_flops_per_ns'],'vector_ops_per_ns':n*p['vector_ops_per_ns'],
        'dram_capacity_bytes':n*p['dram_bytes_per_gpu'],'dram_bandwidth_bytes_per_ns':n*p['hbm_bytes_per_ns'],
        'private_sram_capacity_bytes':n*p['sms_per_gpu']*p['shared_bytes_per_sm']}
    deltas={k:100*(gpu[k]/wafer[k]-1) for k in wafer}
    return {'scope':'near_budget_FP16_and_DRAM_capacity_only_not_all_resources_equal',
        'wafer':wafer,'gpu':gpu,'gpu_minus_wafer_percent':deltas,
        'gpu_no_less_fp16_and_dram_capacity':all(gpu[k]>=wafer[k] for k in ('bf16_peak_flops_per_ns','dram_capacity_bytes')),
        'wafer_compute_rates_include_unvalidated_assumptions':True,
        'equal_area_power':False,'all_memory_levels_equal':False}
