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
import hashlib
from .cost_model import CostEstimate
from .mapping import ceildiv


def _ap_counts(start, step, count, bins):
    from math import gcd
    out=[0]*bins
    period=bins//gcd(step,bins);q,r=divmod(count,period)
    for i in range(period):out[(start+i*step)%bins]=q+(i<r)
    return out


@lru_cache(maxsize=65536)
def cta_mapping(problem,sms,capacity,tm,tn,tk):
    """Closed-form exact round-robin CTA ledger, including both edge tiles.

    Identical to enumerating every output CTA, without work proportional to
    large training token counts. This changes execution cost, not device cost.
    """
    mr=ceildiv(problem.m,tm);nc=ceildiv(problem.n,tn);b=problem.batches
    count=_ap_counts(0,1,b*mr*nc,sms)
    me=[0]*sms;ne=[0]*sms;corner=[0]*sms
    dm=problem.m-(mr-1)*tm-tm;dn=problem.n-(nc-1)*tn-tn
    if dm:
        for batch in range(b):
            x=_ap_counts(batch*mr*nc+(mr-1)*nc,1,nc,sms)
            me=[a+c for a,c in zip(me,x)]
    if dn:ne=_ap_counts(nc-1,nc,b*mr,sms)
    if dm and dn:corner=_ap_counts(mr*nc-1,mr*nc,b,sms)
    flops=[];traffic=[];workspace=[];reads=writes=0;kc=min(tk,problem.k)
    for sm in range(sms):
        sum_m=tm*count[sm]+dm*me[sm];sum_n=tn*count[sm]+dn*ne[sm]
        product=tm*tn*count[sm]+dm*tn*me[sm]+tm*dn*ne[sm]+dm*dn*corner[sm]
        r=problem.k*(sum_m*problem.a_bytes+sum_n*problem.b_bytes);w=product*problem.output_bytes
        sizes=[(tm,tn,count[sm]-me[sm]-ne[sm]+corner[sm]),
               (tm+dm,tn,me[sm]-corner[sm]),(tm,tn+dn,ne[sm]-corner[sm]),
               (tm+dm,tn+dn,corner[sm])]
        need=max((m*kc*problem.a_bytes+kc*n*problem.b_bytes+m*n*problem.accumulator_bytes
                  for m,n,c in sizes if c),default=0)
        if need>capacity:raise ValueError('GPU CTA exceeds private SM shared memory')
        flops.append(2*problem.k*product);traffic.append(r+w);workspace.append(need)
        reads+=r;writes+=w
    if sum(flops)!=problem.flops:raise ValueError('GPU CTA FLOP conservation failed')
    return tuple(flops),tuple(traffic),tuple(workspace),reads,writes,b*mr*nc


class GPUCostProvider:
    def __init__(self,profile):self.p=profile

    def estimate(self,spec,hardware):
        measured=self.p.get('gemm_measurements')
        if measured and spec.matmuls:
            from dataclasses import asdict
            keys=[hashlib.sha256(json.dumps(asdict(m),sort_keys=True,separators=(',',':')).encode()).hexdigest() for m in spec.matmuls]
            missing=[k for k in keys if k not in measured]
            if missing:raise ValueError('GEMM outside exact measured profile domain: '+str(missing))
            reference=GPUCostProvider({k:v for k,v in self.p.items() if k!='gemm_measurements'})
            quota=reference.estimate(spec,hardware)
            extra=reference.estimate(replace(spec,matmuls=()),hardware)
            # cuBLAS CUDA-event kernel duration already includes its device
            # memory service. Charge only additional vector/KV memory here.
            compute=sum(measured[k]['duration_ns'] for k in keys)+extra.compute_ns
            return replace(quota,duration_ns=max(1,compute+extra.local_memory_ns),compute_ns=compute,
                local_memory_ns=extra.local_memory_ns,core_compute_ns=tuple(0 for _ in quota.core_compute_ns),
                source='measured_hot_cublas_GEMM_plus_analytic_vector_v1',
                timing_contract='inclusive_measured_GEMM_no_second_HBM_charge; reference_CTA_workspace_not_measured_occupancy')
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
    if 'gemm_profile' in p:
        profile=(Path(path).resolve().parent/p['gemm_profile']).resolve()
        data=json.loads(profile.read_text())
        if data['schema_version']!='rlxy.gemm_measurements/1' or data['status']!='PASS':raise ValueError('incomplete GEMM profile')
        if not all(c['passed'] for c in data['numeric_fixture_checks']):raise ValueError('GPU GEMM numeric fixture failed')
        if data['device']!='NVIDIA A100-SXM4-80GB':raise ValueError('measured GPU model does not match profile')
        p.update(gemm_profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),gemm_measurements=data['measurements'],
            gemm_measurement_scope=data['scope'],gemm_measurement_source_shapes_sha256=data['source_shapes_sha256'])
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
