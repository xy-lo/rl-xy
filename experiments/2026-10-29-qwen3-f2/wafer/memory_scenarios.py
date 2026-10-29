#!/usr/bin/env python3
"""Private endpoint SRAM versus a shared die path, independent fluid events.

Uses the exact same one-buffer programs as independent_timeline.py. Multiple
private endpoints may transfer concurrently; SRAM400 is never misinterpreted as
the entire die's bandwidth. Source rates are proxy architecture scenarios, not a
claim the 256-core WSC is physical Dojo. No source backend is changed.
"""
from pathlib import Path
import json,math,heapq,time,csv
import independent_timeline as ref

def simulate_fluid(programs,hw,*,sram_load=None,sram_store=None,compute_factor=1.,dram_factor=1.,path_factor=1.):
    load=sram_load or hw.sram_bytes_per_ns;store=sram_store or hw.sram_bytes_per_ns
    shared=min(hw.dram_bytes_per_ns*dram_factor,hw.module_path_bytes_per_ns*path_factor)
    streams={c:iter(g) for c,g in programs.items()};events=[];flows=[];audit=ref.Totals()
    completion={c:0. for c in streams};work={c:0. for c in streams};clock=busy=0.;serial=0;peak_flows=0
    def next_action(core,when):
        try:heapq.heappush(events,(when,core,next(streams[core])))
        except StopIteration:completion[core]=when
    for core in streams:next_action(core,0.)
    while events or flows:
        while events and events[0][0]<=clock+1e-8:
            when,core,action=heapq.heappop(events);kind,a,b,c=action;audit.events+=1
            if kind=='A':
                audit.max_workspace=max(audit.max_workspace,a);audit.tail_tiles+=b
                assert a<=hw.sram_bytes_per_core;next_action(core,when)
            elif kind=='C':
                duration=a/(hw.gemm_flops_per_ns*compute_factor)+b/(hw.vector_ops_per_ns*compute_factor)
                audit.flops+=a;audit.vectors+=b;work[core]+=duration;next_action(core,when+duration)
            elif kind in ('R','W'):
                serial+=1;flow={'core':core,'remaining':float(a),'cap':load if kind=='R' else store,'arrival':when,'serial':serial}
                key=lambda f:(f['arrival'],f['core'],f['serial'])
                if not flows or key(flows[-1])<=key(flow):flows.append(flow)
                else:
                    import bisect
                    i=bisect.bisect_right([key(f) for f in flows],key(flow));flows.insert(i,flow)
                if kind=='R':audit.reads+=a
                else:audit.writes+=a
            else:raise ValueError(kind)
        available=shared;active_flows=[]
        for flow in flows:
            if available<=0:break
            rate=min(flow['cap'],available);active_flows.append((flow,rate));available-=rate
        active=len(active_flows);peak_flows=max(peak_flows,active)
        flow_dt=min((f['remaining']/r for f,r in active_flows),default=math.inf)
        event_time=events[0][0] if events else math.inf
        next_time=min(event_time,clock+flow_dt)
        if not math.isfinite(next_time):raise ValueError('no progress in resource model')
        dt=max(0.,next_time-clock)
        if active:busy+=dt
        for flow,rate in active_flows:flow['remaining']-=dt*rate
        clock=next_time
        done=[f for f,r in active_flows if f['remaining']<=1e-5]
        for f in done:flows.remove(f)
        for flow in done:next_action(flow['core'],clock)
    return {'reference_ns':max(completion.values()),'reference_memory_busy_ns':busy,
        'reference_per_core_compute_max_ns':max(work.values()),'active_cores':len(streams),'audit':audit.__dict__,
        'peak_concurrent_private_endpoint_transfers':peak_flows,'sram_load_B_per_ns':load,'sram_store_B_per_ns':store}

def run_catalog(options):
    rows=[]
    ref.simulate=lambda p,h:simulate_fluid(p,h,**options)
    policies={'4B':(2,4,4,64),'8B':(4,4,4,64),'14B':(4,8,4,64)}
    for model,(tt,rt,mb,cap) in policies.items():
        op=ref.Operators(ref.REPO,ref.BASE/f'inputs/Qwen3-{model}/config.json','wafer')
        req=json.loads((ref.BASE/f'inputs/Qwen3-{model}/tune.json').read_text())['trajectories'];m=op.model
        train=tuple(x['prompt_length']+x['response_length']-1 for x in req[:mb]);t=sum(train)
        first=req[::48//rt][:cap];prefill=tuple(x['prompt_length'] for x in first)
        h=m.hidden_size;f=m.ffn_size//tt
        cases=[(ref.Matmul(t,2*f,h),'train_ffn_up'),(ref.Matmul(h,2*f,t,output_bytes=4,direction='dweight'),'train_ffn_up_dweight'),
            (ref.Matmul(cap,(m.query_width+2*m.kv_heads*m.head_dim)//rt,h),'decode_qkv'),
            (ref.Matmul(sum(prefill),h,m.ffn_size//rt),'prefill_ffn_down')]
        for p,label in cases:rows.append(ref.gemm_case(op,p,label,model))
        contexts=tuple(x['prompt_length']+min(x['response_length']-1,2048) for x in first)
        for qs,cs,tp,label in [(train,train,tt,'teacher_attention'),((1,)*len(contexts),contexts,rt,'mixed_decode_attention')]:
            rows.append(ref.attention_case(op,qs,cs,tp,label,model))
        for args in [(257*h,6,2,2,'rmsnorm_rows',h),(257*(m.ffn_size//rt),5,4,2,'swiglu_tail_chunks',None)]:
            rows.append(ref.vector_case(op,*args[:4],args[4],model,row_size=args[5]))
    return rows

def validate_resource_fixtures():
    from types import SimpleNamespace
    h=SimpleNamespace(dram_bytes_per_ns=2000,module_path_bytes_per_ns=2000,sram_bytes_per_ns=4000,
        gemm_flops_per_ns=1020,vector_ops_per_ns=256,sram_bytes_per_core=1250000)
    def transfers(n,kind='R'):
        return {i:iter([(kind,4000,0,0)]) for i in range(n)}
    single=simulate_fluid(transfers(1),h,sram_load=400,sram_store=270)
    five=simulate_fluid(transfers(5),h,sram_load=400,sram_store=270)
    twenty=simulate_fluid(transfers(20),h,sram_load=400,sram_store=270)
    store=simulate_fluid(transfers(1,'W'),h,sram_load=400,sram_store=270)
    assert math.isclose(single['reference_ns'],4000/400)
    assert math.isclose(five['reference_ns'],20000/2000)
    assert math.isclose(twenty['reference_ns'],80000/2000)
    assert math.isclose(store['reference_ns'],4000/270)
    return {'status':'PASS','single_private_load_ns':single['reference_ns'],'five_concurrent_private_load_ns':five['reference_ns'],
        'twenty_load_shared_bus_ns':twenty['reference_ns'],'single_private_store_ns':store['reference_ns'],
        'meaning':'400 B/ns is per private endpoint, and enough endpoints saturate shared 2000 B/ns without multiplying shared bandwidth.'}

def main():
    start=time.monotonic();fixtures=validate_resource_fixtures()
    nominal=json.loads((ref.OUT/'independent-timeline-results.json').read_text())['rows'];old={(r['model'],r['label']):r for r in nominal}
    scenarios=[('declared_4000_symmetric',4000,4000),('primary_load_400_symmetric',400,400),
        ('primary_store_270_conservative_symmetric',270,270),('primary_separate_load400_store270',400,270)]
    results=[]
    for name,load,store in scenarios:
        rows=run_catalog({'sram_load':load,'sram_store':store})
        if load==4000:
            for r in rows:
                a=old[r['model'],r['label']]
                assert a['audit']==r['audit'],(a,r)
                r['two_reference_time_relative_difference']=(r['reference_ns']-a['reference_ns'])/a['reference_ns']
        for r in rows:r.update(scenario=name,reference_vs_nominal_ratio=r['reference_ns']/old[r['model'],r['label']]['reference_ns'])
        results.extend(rows);print(name,'cases',len(rows),'range reference/nominal',min(r['reference_vs_nominal_ratio'] for r in rows),max(r['reference_vs_nominal_ratio'] for r in rows),flush=True)
    result={'scope':'Primary-rate proxy scenarios in an abstract 256-core WSC; not a physical Dojo reproduction or hardware-error estimate',
        'reference':'independent FIFO fluid DMA sharing with private read/store endpoints and one SRAM tile buffer per core',
        'primary':'Tesla HotChips34 Node memory slide: 1.25MB/node, 400GB/s load and 270GB/s store; source archived by controller',
        'fixtures':fixtures,'nominal_reference_comparison':'All work/traffic/workspace events match; independent scheduler timestamp-tie discrepancies are reported per case, not claimed exact.',
        'rows':results,'wall_seconds':time.monotonic()-start}
    (ref.OUT/'sram-primary-scenarios.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('COMPLETE',len(results),'cases',result['wall_seconds'],'seconds')
if __name__=='__main__':main()
