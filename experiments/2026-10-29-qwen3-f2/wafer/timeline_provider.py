"""Candidate common WSC GEMM service: exact reference adapter + SHA cache.

No fitted factor or interpolation. Mapping, traffic and private workspace are
the original OS policy; only cross-core natural load/compute overlap is timed.
The cached reference is conditional on peak rates/explicit resource policy and
is not a physical-hardware calibration. This file does not modify repo source.
"""
from pathlib import Path
from dataclasses import asdict,replace
from functools import lru_cache
import json,hashlib,math,fcntl,os
import independent_timeline as reference
from memory_scenarios import simulate_fluid
from rl_xy.capacity_mapping import CapacityAwareCostProvider,capacity_partition
from rl_xy.qwen3_cost import Operators,Cost
from rl_xy.operator_ir import KernelSpec
from rl_xy.mapping import choose_os_mapping
from rl_xy.qwen3_source import manifest

OUT=Path(__file__).resolve().parent
TIME_VERSION='fifo_os_singlebuffer_private_direction_sram_v2'

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def profile():return json.loads((OUT/'candidate-profile-v2.json').read_text())

def attention_memory_ns(reads,writes,units,hw,*,load_bytes_per_ns=400,store_bytes_per_ns=270):
    if units<=0 or min(load_bytes_per_ns,store_bytes_per_ns)<=0:raise ValueError('positive endpoint units/rates required')
    return max((reads+writes)/hw.dram_bytes_per_ns,(reads+writes)/hw.module_path_bytes_per_ns,
        reads/(units*load_bytes_per_ns)+writes/(units*store_bytes_per_ns))

class TimelineCostProvider:
    def __init__(self,cache_dir=None,*,load_bytes_per_ns=400,store_bytes_per_ns=270):
        self.base=CapacityAwareCostProvider();self.cache_dir=Path(cache_dir) if cache_dir else None
        self.load=load_bytes_per_ns;self.store=store_bytes_per_ns;self.hits=self.misses=0
        self.identity={'time_version':TIME_VERSION,'provider_sha256':sha(__file__),
            'source_tree_sha256':manifest(reference.REPO)['tree_sha256'],
            'reference_sha256':sha(OUT/'independent_timeline.py'),'resource_reference_sha256':sha(OUT/'memory_scenarios.py'),
            'mapper_evaluator_sha256':sha(OUT/'exact_mapper_cache.py'),'profile_sha256':sha(OUT/'candidate-profile-v2.json'),
            'production_mapper_sha256':sha(reference.REPO/'src/rl_xy/capacity_mapping.py'),
            'production_os_mapper_sha256':sha(reference.REPO/'src/rl_xy/mapping.py'),
            'sram_load_B_per_ns':self.load,'sram_store_B_per_ns':self.store,
            'buffer_count_per_core':1,'approximation':'none: exact event-reference lookup/cache'}
        if self.cache_dir:self.cache_dir.mkdir(parents=True,exist_ok=True)

    def _compute(self,p,hw):
        parts=capacity_partition(p,hw) if hw.cores_per_die>1 else ((0,(0,p.batches,0,p.m,0,p.n),choose_os_mapping(p,hw.sram_bytes_per_core)),)
        timing_hw=replace(hw,gemm_flops_per_ns=hw.gemm_fp32_flops_per_ns) if max(p.a_bytes,p.b_bytes)>=4 else hw
        result=simulate_fluid({c:reference.gemm_stream(p,ext,mp.tile) for c,ext,mp in parts},timing_hw,sram_load=self.load,sram_store=self.store)
        assert result['audit']['flops']==p.flops
        assert result['audit']['reads']==sum(mp.dram_read_bytes for _,_,mp in parts)
        assert result['audit']['writes']==sum(mp.dram_write_bytes for _,_,mp in parts)
        assert result['audit']['max_workspace']<=hw.sram_bytes_per_core
        return result

    def reference_result(self,p,hw):
        # Includes every hardware/mapping/timing identity, even nonbinding
        # fields. Concurrent workers share only immutable verified cache rows.
        request={'problem':asdict(p),'hardware':asdict(hw),'identity':self.identity}
        key=hashlib.sha256(canonical(request).encode()).hexdigest()
        if not self.cache_dir:self.misses+=1;return self._compute(p,hw)
        path=self.cache_dir/(key+'.json');lock=self.cache_dir/(key+'.lock')
        with lock.open('a+') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX)
            if path.exists():
                row=json.loads(path.read_text());assert row['request']==request
                assert row['result_sha256']==hashlib.sha256(canonical(row['result']).encode()).hexdigest()
                self.hits+=1;return row['result']
            value=self._compute(p,hw);self.misses+=1
            row={'request':request,'result':value,'result_sha256':hashlib.sha256(canonical(value).encode()).hexdigest()}
            temp=path.with_suffix('.'+str(os.getpid())+'.tmp');temp.write_text(json.dumps(row,sort_keys=True)+'\n');os.replace(temp,path)
            return value

    def estimate(self,spec,hardware):
        old=self.base.estimate(spec,hardware)
        if not spec.matmuls:return old
        references=[self.reference_result(p,hardware) for p in spec.matmuls]
        inclusive=sum(x['reference_ns'] for x in references)
        extra=self.base.estimate(replace(spec,matmuls=()),hardware) if (spec.vector_ops or spec.read_bytes or spec.write_bytes) else None
        if extra:inclusive+=extra.duration_ns
        ns=max(1,math.ceil(inclusive))
        diagnostics={'time_version':TIME_VERSION,'timing':'inclusive_reference_service_not_pure_compute',
            'original_compute_ns':old.compute_ns,'original_memory_ns':old.local_memory_ns,'original_serial_ns':old.duration_ns,
            'reference_ns_before_rounding':inclusive,'reference_memory_busy_ns':sum(x['reference_memory_busy_ns'] for x in references),
            'reference_peak_private_endpoint_flows':max(x['peak_concurrent_private_endpoint_transfers'] for x in references),
            'sram_load_B_per_ns':self.load,'sram_store_B_per_ns':self.store,'single_input_buffer':True}
        return replace(old,duration_ns=ns,compute_ns=ns,local_memory_ns=0,
            partitions=tuple({**p,'timing_diagnostic':diagnostics} for p in old.partitions),
            source='conditional_fifo_singlebuffer_GEMM_reference_exact_cache_v2',
            timing_contract='inclusive_local_service_contains_compute_and_memory; original_C_M_in_diagnostics')

def attention_geometry(op,queries,contexts,tp,backward):
    heads=op.model.query_heads//tp;hd=op.model.head_dim;valid=[]
    for q in (16,32,64,128,256):
        for k in (16,32,64,128,256):
            qq=min(q,max(queries));kk=min(k,max(contexts))
            workspace=(6*(qq+2*kk)*hd if backward else 2*(qq+2*kk)*hd+4*qq*hd)+4*qq*kk+16*qq
            if workspace<=op.hw.sram_bytes_per_core:valid.append((qq,kk))
    if not valid:raise ValueError('no legal private attention tile')
    qt,kt=max(valid);tasks=sum(heads if q==1 else heads*math.ceil(q/qt) for q in queries)
    return min(tasks,op.units)

class TimelineOperators(Operators):
    """Drop-in candidate for every WSC method; no full FA event substitution."""
    def __init__(self,root,config,platform='wafer',*,cache_dir=None,apply_d2d_candidate=True,**kwargs):
        if platform!='wafer':raise ValueError('TimelineOperators candidate is WSC-only')
        if kwargs.get('compute_scale',1.0)!=1.0:
            raise ValueError('inclusive GEMM service cannot use legacy compute_scale; set explicit arithmetic rates in hardware/reference pressure instead')
        super().__init__(root,config,platform,**kwargs)
        self.provider=TimelineCostProvider(cache_dir)
        if apply_d2d_candidate:
            self.hw=replace(self.hw,d2d_latency_ns=profile()['d2d_latency_ns_candidate'])
            self.case=replace(self.case,hardware=self.hw)
        self.local_timing_identity={**self.provider.identity,'attention':'original arithmetic + direction SRAM, decode-affine checked',
            'vector':'unchanged analytic streaming','d2d_latency_ns':self.hw.d2d_latency_ns,
            'network_semantics':'must be recorded separately; no implicit packetization change'}

    @lru_cache(maxsize=65536)
    def attention(self,queries,contexts,tp,backward=False):
        original=super().attention(queries,contexts,tp,backward)
        units=attention_geometry(self,queries,contexts,tp,backward)
        if not backward and max(queries)==1 and units in (6,7):
            raise ValueError('piecewise private/shared attention memory branch requires a non-affine decode engine; unsupported by this affine candidate')
        memory=attention_memory_ns(original.read,original.write,units,self.hw,
            load_bytes_per_ns=self.provider.load,store_bytes_per_ns=self.provider.store)
        return replace(original,memory=memory)
