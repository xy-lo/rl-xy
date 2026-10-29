#!/usr/bin/env python3
"""Independent FIFO tile/DRAM event model for a declared WSC abstraction.

Not a physical-hardware calibration. The reference consumes the selected legal
output partition but reconstructs every tail tile's memory/compute dependency.
DRAM+module form one shared bottleneck DMA service; private cores compute
independently. No fabricated array dimensions, bank timings or packet startup.
"""
from pathlib import Path
import os,sys,json,math,heapq,time,csv
from dataclasses import dataclass
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
REPO=Path(os.environ['RLXY_F2_REPO']).resolve() if os.environ.get('RLXY_F2_REPO') else ROOT/'scratch/rl-xy-release-20261029';BASE=ROOT/'outputs/rl-xy-qwen3-baselines-20261028'
sys.path.insert(0,str(REPO/'src'))
from rl_xy.qwen3_cost import Operators
from rl_xy.operator_ir import Matmul,KernelSpec
from rl_xy.capacity_mapping import capacity_partition
import exact_mapper_cache as fast
fast.install()

@dataclass
class Totals:
    reads:int=0
    writes:int=0
    flops:int=0
    vectors:int=0
    events:int=0
    tail_tiles:int=0
    max_workspace:int=0

def simulate(programs,hw,*,compute_factor=1.,dram_factor=1.,path_factor=1.):
    """Earliest-ready FIFO DMA, deterministic core-id ties, private compute.

    A tile DMA holds the common DRAM/path service until complete; SRAM endpoints
    are private. Different cores may compute while another core transfers. Each
    core uses one input tile buffer and has strict load->compute dependencies.
    """
    streams={core:iter(stream) for core,stream in programs.items()};queue=[];audit=Totals()
    ready={core:0. for core in streams};bus=0.;busy=0.;serial_compute={core:0. for core in streams}
    def advance(core,when):
        try:action=next(streams[core]);heapq.heappush(queue,(when,core,action))
        except StopIteration:ready[core]=when
    for core in streams:advance(core,0.)
    while queue:
        when,core,action=heapq.heappop(queue);kind,a,b,c=action;audit.events+=1
        if kind in ('R','W'):
            rate=min(hw.dram_bytes_per_ns*dram_factor,hw.module_path_bytes_per_ns*path_factor,hw.sram_bytes_per_ns)
            start=max(when,bus);end=start+a/rate;bus=end;busy+=a/rate
            if kind=='R':audit.reads+=a
            else:audit.writes+=a
        elif kind=='C':
            duration=a/(hw.gemm_flops_per_ns*compute_factor)+b/(hw.vector_ops_per_ns*compute_factor)
            end=when+duration;serial_compute[core]+=duration;audit.flops+=a;audit.vectors+=b
        elif kind=='A':
            audit.max_workspace=max(audit.max_workspace,a);audit.tail_tiles+=b
            assert a<=hw.sram_bytes_per_core;end=when
        else:raise ValueError(kind)
        advance(core,end)
    return {'reference_ns':max(ready.values()),'reference_memory_busy_ns':busy,
        'reference_per_core_compute_max_ns':max(serial_compute.values()),'active_cores':len(streams),
        'audit':audit.__dict__}

def gemm_stream(p,extent,tile):
    _,bc,_,mc,_,nc=extent;tm,tn,tk=tile
    for bi in range(bc):
        for mi in range(0,mc,tm):
            mm=min(tm,mc-mi)
            for ni in range(0,nc,tn):
                nn=min(tn,nc-ni)
                for ki in range(0,p.k,tk):
                    kk=min(tk,p.k-ki)
                    workspace=mm*kk*p.a_bytes+kk*nn*p.b_bytes+mm*nn*p.accumulator_bytes
                    yield ('A',workspace,int(mm<tm or nn<tn or kk<tk),0)
                    yield ('R',mm*kk*p.a_bytes+kk*nn*p.b_bytes,0,0)
                    yield ('C',2*mm*nn*kk,0,0)
                yield ('W',mm*nn*p.output_bytes,0,0)

def gemm_case(op,p,label,model):
    parts=capacity_partition(p,op.hw)
    programs={core:gemm_stream(p,ext,mp.tile) for core,ext,mp in parts}
    ref=simulate(programs,op.hw);backend=op.provider.estimate(KernelSpec(label,matmuls=(p,)),op.hw)
    assert ref['audit']['flops']==p.flops
    assert (ref['audit']['reads'],ref['audit']['writes'])==(backend.dram_read_bytes,backend.dram_write_bytes)
    return record(model,'GEMM',label,backend.duration_ns,backend.compute_ns,backend.local_memory_ns,ref,
        {'m':p.m,'n':p.n,'k':p.k,'direction':p.direction,'mapping':'same production private OS partition, independent tail-tile timeline'})

def attention_tiles(op,queries,contexts,heads,hd):
    valid=[]
    for q in (16,32,64,128,256):
        for k in (16,32,64,128,256):
            qq=min(q,max(queries));kk=min(k,max(contexts))
            ws=2*(qq+2*kk)*hd+4*qq*hd+4*qq*kk+16*qq
            if ws<=op.hw.sram_bytes_per_core:valid.append((qq,kk,ws))
    return max(valid,key=lambda x:(x[0],x[1]))

def attention_stream(tasks,hd,ktile,workspace):
    for qq,end in tasks:
        yield ('A',workspace,0,0)
        yield ('R',qq*hd*2,0,0)
        for k in range(0,end,ktile):
            kk=min(ktile,end-k)
            yield ('A',workspace,int(kk<ktile),0)
            yield ('R',kk*hd*4,0,0)
            yield ('C',qq*kk*hd*4,qq*kk*5,0)
        yield ('W',qq*hd*2+qq*8,0,0)

def attention_case(op,queries,contexts,tp,label,model):
    heads=op.model.query_heads//tp;hd=op.model.head_dim
    qt,kt,workspace=attention_tiles(op,queries,contexts,heads,hd)
    jobs=[]
    for q,c in zip(queries,contexts):
        if q==1:
            for _ in range(heads):jobs.append((1,c))
        else:
            assert q==c
            for start in range(0,q,qt):
                qq=min(qt,q-start)
                for _ in range(heads):jobs.append((qq,start+qq))
    cores=min(len(jobs),op.hw.cores_per_die);programs={i:attention_stream(jobs[i::cores],hd,kt,workspace) for i in range(cores)}
    ref=simulate(programs,op.hw);b=op.attention(tuple(queries),tuple(contexts),tp)
    assert (ref['audit']['reads'],ref['audit']['writes'],ref['audit']['flops'])==(b.read,b.write,b.flops)
    return record(model,'FA_forward',label,b.ns,b.compute,b.memory,ref,
        {'tp':tp,'batch':len(queries),'query_tokens':sum(queries),'context_tokens':sum(contexts),'q_tile':qt,'k_tile':kt,
         'assignment':'query-tile/head jobs in declared order, cyclic across private cores; no KV cache reuse across query heads'})

def vector_stream(items,ops,reads,writes):
    for count in items:
        workspace=count*(reads+writes)+16
        yield ('A',workspace,int(count<256),0)
        yield ('R',count*reads,0,0)
        yield ('C',0,count*ops,0)
        yield ('W',count*writes,0,0)

def vector_case(op,count,ops,reads,writes,label,model,row_size=None):
    if row_size:
        assert count%row_size==0
        jobs=[row_size]*(count//row_size)
    else:jobs=[min(256,count-i) for i in range(0,count,256)]
    cores=min(len(jobs),op.hw.cores_per_die)
    ref=simulate({i:vector_stream(jobs[i::cores],ops,reads,writes) for i in range(cores)},op.hw)
    b=op.vec(count*ops,count*reads,count*writes,label)
    assert (ref['audit']['reads'],ref['audit']['writes'],ref['audit']['vectors'])==(b.read,b.write,count*ops)
    return record(model,'vector',label,b.ns,b.compute,b.memory,ref,
        {'elements':count,'row_size':row_size,'assignment':'whole RMSNorm rows on one core; elementwise SwiGLU chunks of <=256 elements, cyclic private cores'})

def record(model,kind,label,total,compute,memory,ref,geometry):
    return {'model':model,'kind':kind,'label':label,'backend_ns':total,'backend_compute_ns':compute,'backend_memory_ns':memory,
        'backend_over_reference':total/ref['reference_ns'],'relative_discrepancy':(total-ref['reference_ns'])/ref['reference_ns'],
        **ref,'geometry':geometry,'scope':'conditional independent tile-FIFO reference; discrepancy is NOT physical-hardware error'}

def main():
    rows=[];start=time.monotonic()
    policies={'4B':(2,4,4,64),'8B':(4,4,4,64),'14B':(4,8,4,64)}
    for model,(tt,rt,mb,cap) in policies.items():
        op=Operators(REPO,BASE/f'inputs/Qwen3-{model}/config.json','wafer')
        req=json.loads((BASE/f'inputs/Qwen3-{model}/tune.json').read_text())['trajectories'];m=op.model
        train=tuple(x['prompt_length']+x['response_length']-1 for x in req[:mb]);t=sum(train)
        first=req[::48//rt][:cap];prefill=tuple(x['prompt_length'] for x in first)
        h=m.hidden_size;f=m.ffn_size//tt
        cases=[(Matmul(t,2*f,h),'train_ffn_up'),(Matmul(h,2*f,t,output_bytes=4,direction='dweight'),'train_ffn_up_dweight'),
            (Matmul(cap,(m.query_width+2*m.kv_heads*m.head_dim)//rt,h),'decode_qkv'),
            (Matmul(sum(prefill),h,m.ffn_size//rt),'prefill_ffn_down')]
        for p,label in cases:
            r=gemm_case(op,p,label,model);rows.append(r);print(model,label,round(r['backend_over_reference'],4),'events',r['audit']['events'],flush=True)
        contexts=tuple(x['prompt_length']+min(x['response_length']-1,2048) for x in first)
        for qs,cs,tp,label in [(train,train,tt,'teacher_attention'),((1,)*len(contexts),contexts,rt,'mixed_decode_attention')]:
            r=attention_case(op,qs,cs,tp,label,model);rows.append(r);print(model,label,round(r['backend_over_reference'],4),flush=True)
        for args in [(257*h,6,2,2,'rmsnorm_rows',h),(257*(m.ffn_size//rt),5,4,2,'swiglu_tail_chunks',None)]:
            r=vector_case(op,*args[:4],args[4],model,row_size=args[5]);rows.append(r);print(model,args[4],round(r['backend_over_reference'],4),flush=True)
    result={'scope':'Independent logical tile resource timeline against current C+M backend; not physical calibration',
        'reference_policy':'one input tile buffer per core, exact tail extents, FIFO atomic-tile DMA shared at min(DRAM,module-path,private SRAM), private peak compute; no array/bank/startup assumptions',
        'not_covered':['FA backward cross-query-head KV-gradient placement/reduction','DRAM banks/type/timings','matrix shape utilization/startup','physical NoC routing/banks'],
        'rows':rows,'wall_seconds':time.monotonic()-start}
    (OUT/'independent-timeline-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'independent-timeline-summary.csv').open('w') as f:
        fields=['model','kind','label','backend_ns','reference_ns','backend_over_reference','relative_discrepancy','active_cores','scope']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    print('COMPLETE',len(rows),'cases',result['wall_seconds'],'seconds',flush=True)
if __name__=='__main__':main()
