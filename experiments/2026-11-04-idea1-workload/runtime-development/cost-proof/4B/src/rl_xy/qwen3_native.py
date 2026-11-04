"""Pinned ASTRA executions for coarse operator plans and isolated services.

No network fallback: failures preserve inputs and logs and raise. Mathematical
iteration compression is permitted only for groups with disjoint physical links.
"""
from __future__ import annotations
from dataclasses import replace,asdict
import hashlib,json,math,os,re,time
from pathlib import Path
from .types import Operation
from .qwen3_cost import Cost,Block
from .astra_adapter.generator import generate_traces
from .astra_adapter.runner import run_astrasim
from .astra_adapter.parser import EVENT_RE
from .qwen3_paths import native_root,native_binary
from .communication_service import protocol_overhead_ns


class Plan:
    def __init__(self,op):
        self.op=op;self.world=48 if op.platform=='gpu' else 54
        self.operations=[];self.groups={};self.groupids={};self.last={};self.stats={}
        self.local(tuple(range(self.world)),Cost(compute=1),'initialize')
    def append(self,operation):
        i=len(self.operations)+1;ps=operation.participant_ranks
        deps={r:() if r not in self.last else (self.last[r],) for r in ps}
        operation=replace(operation,node_id=i,op_id=f'n{i}.{operation.op_id}',dependencies_by_rank=deps)
        self.operations.append(operation)
        for r in ps:self.last[r]=i
        return operation
    def local(self,ranks,cost,name):
        if cost.ns<=0:return
        o=self.append(Operation(0,name,1,name,name,'compute',duration_ns=max(1,math.ceil(cost.ns)),participant_ranks=tuple(ranks)))
        self.stats[o.node_id]={'name':name,'compute_ns':cost.compute,'memory_ns':cost.memory,'flops':cost.flops,'read':cost.read,'write':cost.write}
    def comm(self,ranks,kind,payload,name):
        ps=tuple(ranks);n=len(ps)
        if n==1:return
        payload=max(4*n,ceildiv(int(payload),4*n)*4*n)
        overhead=protocol_overhead_ns(self.op,n,kind,payload)
        if overhead:self.local(ps,Cost(compute=overhead),name+'.protocol_overhead')
        if kind in ('GATHER','BCAST'):
            pairs=[(r,ps[0]) for r in ps[1:]] if kind=='GATHER' else [(ps[0],r) for r in ps[1:]]
            for src,dst in pairs:
                self.local((src,),self.op.vec(0,payload,0,'network_read'),name+'.read')
                self.append(Operation(0,name,1,name,name,'p2p',payload_bytes=payload,src_rank=src,dst_rank=dst,participant_ranks=(src,dst)))
                self.local((dst,),self.op.vec(0,0,payload,'network_write'),name+'.write')
            return
        input_bytes=payload;output_bytes=payload*n if kind=='ALL_GATHER' else payload//n if kind=='REDUCE_SCATTER' else payload
        self.local(ps,self.op.vec(0,input_bytes,0,'network_read'),name+'.read')
        if ps not in self.groupids:
            gid=str(len(self.groups)+1);self.groups[gid]=list(ps);self.groupids[ps]=gid
        gid=self.groupids[ps]
        self.append(Operation(0,name,1,name,name,'collective',payload_bytes=payload,collective=kind,participant_ranks=ps,metadata={'comm_group':gid}))
        reduction=0 if kind=='ALL_GATHER' else input_bytes*(n-1)//n//4
        self.local(ps,self.op.vec(reduction,2*reduction*4,output_bytes+reduction*4,'network_write_reduce'),name+'.write')
    def blocks(self,ranks,blocks,prefix):
        for i,b in enumerate(blocks):
            self.local(ranks,b.cost,prefix+'.'+str(i)+'.'+b.name)
            if b.collective:self.comm(ranks,b.collective,b.payload,prefix+'.'+str(i)+'.'+b.collective)


def ceildiv(a,b):return (a+b-1)//b


def execute(plan:Plan,folder,timeout=900,sequence=None):
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=False);op=plan.op
    system=json.loads((op.root/'configs/toy_grpo_2die/astra_system.json').read_text())
    for k in ['all-reduce-implementation','all-gather-implementation','reduce-scatter-implementation','all-to-all-implementation']:system[k]=['ring','ring']
    system['local-mem-bw']=1e12
    (folder/'system.json').write_text(json.dumps(system,indent=2)+'\n')
    (folder/'groups.json').write_text(json.dumps(plan.groups)+'\n')
    (folder/'remote.json').write_text('{"memory-type":"NO_MEMORY_EXPANSION"}\n')
    if op.profile:
        dims=[8,6];rates=[op.profile['nvlink_bytes_per_ns_per_direction'],op.profile['node_uplink_bytes_per_ns_per_direction']];lats=[op.profile['nvlink_latency_ns'],op.profile['uplink_latency_ns']]
    else:dims=[6,9];rates=[op.hw.d2d_bytes_per_ns_per_direction]*2;lats=[op.hw.d2d_latency_ns]*2
    bws=[r*op.network_scale/1000*1e9/2**30 for r in rates]
    (folder/'network.yml').write_text(f'topology: [ Ring, Ring ]\nnpus_count: [ {dims[0]}, {dims[1]} ]\nbandwidth: [ {bws[0]}, {bws[1]} ]\nlatency: [ {lats[0]*1000}, {lats[1]*1000} ]\n')
    generate_traces(plan.operations,folder/'workload',ranks=plan.world,astra_time_scale=1000,comm_groups=plan.groups)
    (folder/'operations.json').write_text(json.dumps([{'id':o.node_id,'kind':o.kind,'duration_ns':o.duration_ns,'payload_bytes':o.payload_bytes,'collective':o.collective,'ranks':o.participant_ranks,'src':o.src_rank,'dst':o.dst_rank,'name':o.op_id} for o in plan.operations],separators=(',',':'))+'\n')
    root=native_root(op.root);os.environ['ASTRA_SIM_ROOT']=str(root);os.environ['RLXY_RESOURCE_BACKEND']='1'
    old_sequence=os.environ.get('RLXY_SEQUENCE_PROGRAM');old_log=os.environ.get('RLXY_SEQUENCE_LOG_PREFIX')
    if sequence is not None:
        (folder/'sequence_program.json').write_text(json.dumps(sequence,separators=(',',':'))+'\n')
        os.environ['RLXY_SEQUENCE_PROGRAM']=str(folder/'sequence_program.json');os.environ['RLXY_SEQUENCE_LOG_PREFIX']=str(folder/'sequence')
    try:
        start=time.monotonic();r=run_astrasim(trace_prefix=folder/'workload',system_config=folder/'system.json',network_config=folder/'network.yml',remote_memory_config=folder/'remote.json',output_dir=folder,binary=native_binary(op.root),timeout_seconds=timeout,network_model='gpu_cluster' if op.platform=='gpu' else 'mesh2d',comm_groups=folder/'groups.json')
    finally:
        if old_sequence is None:os.environ.pop('RLXY_SEQUENCE_PROGRAM',None)
        else:os.environ['RLXY_SEQUENCE_PROGRAM']=old_sequence
        if old_log is None:os.environ.pop('RLXY_SEQUENCE_LOG_PREFIX',None)
        else:os.environ['RLXY_SEQUENCE_LOG_PREFIX']=old_log
    sequence_summaries={}
    if sequence is not None:
        for rank,key in sequence['ranks'].items():
            rows=[json.loads(line) for line in (folder/f'sequence.{rank}.jsonl').read_text().splitlines()]
            summary=rows[-1]
            if summary.get('status')!='PASS':raise ValueError('incomplete native sequence')
            program=sequence['programs'][key]['segments']
            expected_steps=sum(s['repeat']*len(s['steps']) for s in program)
            if summary['executed_steps']!=expected_steps or len(rows)!=len(program)+1:raise ValueError('sequence coverage mismatch')
            sequence_summaries[int(rank)]=summary
    issued={};ended={};perrank={i:{'compute_ns':0.,'memory_ns':0.,'communication_ns':0.} for i in range(plan.world)}
    log_paths=sorted(r.debug_log_path.parent.glob('log*.log'))
    for log_path in log_paths:
        for m in EVENT_RE.finditer(log_path.read_text()):
            key=(int(m['rank']),int(m['node']));t=int(m['tick'])/1000
            target=issued if m['action']=='issue' else ended
            if key in target:raise ValueError('duplicate native event')
            target[key]=t
    expected={(rank,o.node_id) for o in plan.operations for rank in o.participant_ranks}
    if set(issued)!=expected or set(ended)!=expected:raise ValueError('native issue/callback coverage mismatch')
    for o in plan.operations:
        for rank in o.participant_ranks:
            key=(rank,o.node_id);duration=ended[key]-issued[key]
            if duration<0:raise ValueError('native time reversal')
            for d in o.dependencies_by_rank[rank]:
                if ended[rank,d]>issued[key]+1e-6:raise ValueError('native dependency violation')
            if o.kind=='compute':
                if 'qwen3_sequence' in o.op_id:
                    summary=sequence_summaries[rank]
                    for field in ['compute_ns','memory_ns','communication_ns']:perrank[rank][field]+=summary[field]
                    residual=duration-sum(summary[f] for f in ['compute_ns','memory_ns','communication_ns'])
                    if residual < -.02 or residual>summary['executed_steps']+3:raise ValueError('sequence timing ledger mismatch')
                else:
                    s=plan.stats[o.node_id];perrank[rank]['compute_ns']+=s['compute_ns'];perrank[rank]['memory_ns']+=s['memory_ns']
                    if abs(duration-o.duration_ns)>0.1:raise ValueError('native local duration mismatch')
            else:perrank[rank]['communication_ns']+=duration
    ends={rank:max((t for (rr,n),t in ended.items() if rank==rr),default=0.) for rank in perrank}
    slow=max(ends,key=ends.get)
    result={'status':'PASS','native_exit_status':r.return_code,'model_ns':max(ends.values()),'slowest_rank':slow,'per_rank_completion_ns':ends,'per_rank_costs':perrank,'native_nodes':len(plan.operations),'native_rank_events':len(expected),'wall_seconds':time.monotonic()-start,'binary_sha256':r.binary_sha256,'astra_commit':r.astra_commit,'native_coverage_and_dependencies_checked':True,'debug_log_bytes':sum(p.stat().st_size for p in log_paths),'native_sequence_summaries':sequence_summaries,'storage_scope':'capacity admitted by explicit phase upper-bound ledger; private workspace checked in operator costs'}
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


class NetworkProfiles:
    def __init__(self,op,folder,native=False):self.op=op;self.folder=Path(folder);self.native=native;self.cache={}
    def time(self,tp,kind,payload):
        if tp==1:return 0.
        key=(tp,kind,int(payload))
        if key in self.cache:return self.cache[key]
        op=self.op
        if self.native:
            h=hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16];path=self.folder/h
            if (path/'result.json').exists():r=json.loads((path/'result.json').read_text())
            else:
                p=Plan(op);p.comm(tuple(range(tp)),kind,payload,'service');r=execute(p,path)
            value=r['model_ns']-1
        else:
            if op.profile:rate=op.profile['nvlink_bytes_per_ns_per_direction'];lat=op.profile['nvlink_latency_ns']*2
            else:rate=op.hw.d2d_bytes_per_ns_per_direction;lat=op.hw.d2d_latency_ns
            # Search-only alpha-beta service; detailed candidates use native services.
            steps=(2*(tp-1) if kind=='ALL_REDUCE' else tp-1)
            volume=(2*(tp-1)/tp*payload if kind=='ALL_REDUCE' else (tp-1)*payload)
            if kind=='REDUCE_SCATTER':volume=(tp-1)/tp*payload
            value=steps*lat+volume/(rate*op.network_scale)
            value+=op.vec(0,payload,payload*tp if kind=='ALL_GATHER' else payload,'network_endpoints').ns
            rounded=max(4*tp,ceildiv(int(payload),4*tp)*4*tp)
            value+=protocol_overhead_ns(op,tp,kind,rounded)
        self.cache[key]=value;return value
    def blocks(self,tp,blocks):return sum(b.cost.ns+(self.time(tp,b.collective,b.payload) if b.collective else 0.) for b in blocks)


def generation_links_disjoint(platform,tp):
    # GPU groups<=8 stay inside nodes and have distinct endpoint links.
    # Mesh TP2 contiguous pairs never wrap a row of width6 and own disjoint links.
    return tp==1 or (platform=='gpu' and tp<=8) or (platform=='wafer' and tp==2)
