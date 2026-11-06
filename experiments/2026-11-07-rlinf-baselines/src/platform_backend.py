"""Matched GPU/wafer replay using the repository's declared cost providers.

No fitted speedup. Mesh ring rounds reserve every traversed directed XY link.
This is an event projection, not a native ASTRA or hardware measurement.
"""
from functools import lru_cache
from pathlib import Path
from collections import defaultdict
import math
from sim import Backend, Sim, Trainer, NS
from rl_xy.qwen3_cost import Operators, Cost
from rl_xy.qwen3_native import NetworkProfiles, Plan
from rl_xy.model import enumerate_parameters
from rl_xy.distributed_optimizer import layout_for_case


class PlatformBackend(Backend):
    def __init__(self, root, model, policy, platform,active_world=None,cost_contract='f2',cost_cache_dir=None):
        self.cost_contract=cost_contract
        if cost_contract=='f2':
            from baseline_contract import make_f2_operator
            self.op=make_f2_operator(root,model,platform,cache_dir=cost_cache_dir)
        elif cost_contract=='legacy-analytic':
            self.op=Operators(root, Path(root)/f'experiments/2026-10-29-qwen3/inputs/Qwen3-{model}/config.json', platform)
        else:raise ValueError('unsupported cost contract')
        self.net=NetworkProfiles(self.op,'unused',native=False)
        self.p=policy; self.allocated_world=48 if platform=='gpu' else 54
        self.world=active_world if active_world is not None else self.allocated_world
        assert 0<self.world<=self.allocated_world
        assert self.world%policy.train_tp==self.world%policy.rollout_tp==0
        self.host_bw=25e9; self.params=self.op.info['unique_parameter_elements']
        self.kv=2*self.op.model.layers*(self.op.model.kv_heads//policy.rollout_tp)*self.op.model.head_dim*2
        rows=enumerate_parameters(self.op.case_tp(policy.train_tp))
        layout=layout_for_case(self.op.case_tp(policy.train_tp),dp_size=self.world//policy.train_tp)
        states=[layout.state_bytes(i)['total_allocated_bytes'] for i in range(self.world//policy.train_tp)]
        self.state={'training_base_per_rank':states,'training_base_max':max(states),'layout':layout,
            'rollout_weights_per_rank':sum(x.local_elements_per_die for x in enumerate_parameters(self.op.case_tp(policy.rollout_tp)))*2,
            'export_scratch_per_rank':max((x.unique_elements*2 for x in rows if not x.replicated),default=0) if policy.train_tp>1 else 0}
        self.capacity=self.op.capacity; self.max_param=max(r.local_elements_per_die for r in rows)
        self.history_response_bound=8192

    def route(self, src, dst):
        x,y=src%6,src//6; dx,dy=dst%6,dst//6; links=[]
        while x!=dx:
            nx=x+(1 if dx>x else -1); links.append((y*6+x,y*6+nx));x=nx
        while y!=dy:
            ny=y+(1 if dy>y else -1); links.append((y*6+x,ny*6+x));y=ny
        return links

    def fixed_plan(self, plan):
        if self.op.platform=='gpu':return self.gpu_ring_plan(plan)
        ready=[0.]*self.world; link_ready=defaultdict(float); hw=self.op.hw
        def round_service(pairs, payload, t):
            loads=defaultdict(float); paths=[]
            for src,dst in pairs:
                path=self.route(src,dst); paths.append(path)
                for link in path:loads[link]+=payload
            if not loads:return t
            start=max(t,max(link_ready[k] for k in loads))
            duration=max(loads.values())/(hw.d2d_bytes_per_ns_per_direction*NS)
            duration+=max(map(len,paths))*hw.d2d_latency_ns/NS
            end=start+duration
            for link in loads:link_ready[link]=end
            return end
        for o in plan.operations:
            ranks=o.participant_ranks; t=max(ready[r] for r in ranks)
            if o.kind=='compute':end=t+o.duration_ns/NS
            elif o.kind=='p2p':end=round_service([(o.src_rank,o.dst_rank)],o.payload_bytes,t)
            else:
                n=len(ranks); rounds=(2 if o.collective=='ALL_REDUCE' else 1)*(n-1)
                payload=o.payload_bytes if o.collective=='ALL_GATHER' else o.payload_bytes/n
                pairs=[(ranks[i],ranks[(i+1)%n]) for i in range(n)]
                end=t
                for _ in range(rounds):end=round_service(pairs,payload,end)
            for r in ranks:ready[r]=end
        return max(ready)

    def gpu_ring_plan(self,plan):
        """Explicit rank ring: AG payload is PER-RANK input, RS full input.

        The inherited coarse multi-node formula treated AG input like a full
        output buffer. Here every ring round charges actual node uplink bytes.
        This is a declared ring scenario, not a calibrated NCCL prediction.
        """
        ready=[0.]*self.world;links=defaultdict(float);p=self.op.profile
        f2=getattr(self,'cost_contract',None)=='f2'
        packet=getattr(self.op,'runtime_cost_identity',{}).get('numerical_packet_bytes',65536)
        def service(pairs,payload,t):
            loads=defaultdict(float);rates={};latency=0.
            for src,dst in pairs:
                if f2:
                    rate=p['nvlink_bytes_per_ns_per_direction']*NS
                    for key in [('gpu_tx',src),('gpu_rx',dst)]:loads[key]+=payload;rates[key]=rate
                    first=min(payload,packet)
                    path_latency=2*p['nvlink_latency_ns']/NS+first/rate
                    if src//8!=dst//8:
                        uplink=p['node_uplink_bytes_per_ns_per_direction']*NS
                        for key in [('tx',src//8),('rx',dst//8)]:loads[key]+=payload;rates[key]=uplink
                        path_latency+=2*p['uplink_latency_ns']/NS+2*first/uplink
                        path_latency+=first/rate-max(first/rate,first/uplink)
                    # Streaming path startup: sum(first-packet services) minus
                    # one bottleneck service already included in bulk transfer.
                    latency=max(latency,path_latency)
                elif src//8==dst//8:
                    key=('nvlink',src,dst);loads[key]+=payload;rates[key]=p['nvlink_bytes_per_ns_per_direction']*NS
                    latency=max(latency,p['nvlink_latency_ns']/NS)
                else:
                    for key in [('tx',src//8),('rx',dst//8)]:
                        loads[key]+=payload;rates[key]=p['node_uplink_bytes_per_ns_per_direction']*NS
                    latency=max(latency,p['uplink_latency_ns']/NS)
            start=max(t,max((links[k] for k in loads),default=t))
            end=start+latency+max((loads[k]/rates[k] for k in loads),default=0.)
            for k in loads:links[k]=end
            return end
        for o in plan.operations:
            ranks=o.participant_ranks;t=max(ready[r] for r in ranks)
            if o.kind=='compute':end=t+o.duration_ns/NS
            elif o.kind=='p2p':end=service([(o.src_rank,o.dst_rank)],o.payload_bytes,t)
            else:
                n=len(ranks);rounds=(2 if o.collective=='ALL_REDUCE' else 1)*(n-1)
                payload=o.payload_bytes if o.collective=='ALL_GATHER' else o.payload_bytes/n
                pairs=[(ranks[i],ranks[(i+1)%n]) for i in range(n)];end=t
                for _ in range(rounds):end=service(pairs,payload,end)
            for r in ranks:ready[r]=end
        return max(ready)

    @lru_cache(maxsize=1)
    def publish(self):
        p=self.p; plan=self.make_plan(); tr=p.train_tp
        rows=enumerate_parameters(self.op.case_tp(tr)); dest={x.parameter_id:x for x in enumerate_parameters(self.op.case_tp(p.rollout_tp))}
        plan.local(tuple(range(self.world)),Cost(compute=1000),'switch_to_rollout')
        for g in range(self.world//tr):
            ranks=tuple(range(g*tr,(g+1)*tr))
            for row in rows:
                if not row.replicated and tr>1:plan.comm(ranks,'ALL_GATHER',row.local_elements_per_die*2,'weight_export.'+row.parameter_id)
                plan.local(ranks,self.op.vec(0,row.unique_elements*2,dest[row.parameter_id].local_elements_per_die*2,'export_pack_load'),'pack.'+row.parameter_id)
        return self.fixed_plan(plan)

    @lru_cache(maxsize=1)
    def optimize(self):
        plan=self.make_plan();tp=self.p.train_tp;dp=self.world//tp;layout=self.state['layout']
        if tp>1:
            for g in range(dp):plan.comm(tuple(range(g*tp,(g+1)*tp)),'ALL_REDUCE',self.op.model.layers*2*self.op.model.head_dim*4,'qk_norm_parameter_grad_TP')
        for r in range(self.world):plan.local((r,),self.op.vec(layout.padded_elements,layout.padded_elements*4,layout.padded_elements*4,'gradient_scale'),'gradient_scale')
        for shard in range(tp):plan.comm(tuple(range(shard,self.world,tp)),'REDUCE_SCATTER',layout.padded_elements*4,'dp_gradient_RS')
        for r in range(self.world):
            n=layout.shard_valid_elements(r//tp,tp_rank=r%tp,norm_only=True)
            plan.local((r,),self.op.vec(n*2,n*4,4,'gradient_norm'),'gradient_norm')
        plan.comm(tuple(range(self.world)),'ALL_REDUCE',self.world*4,'gradient_global_norm')
        for r in range(self.world):
            n=layout.shard_valid_elements(r//tp)
            plan.local((r,),self.op.vec(n*3,n*4,n*4,'clip'),'clip')
            plan.local((r,),self.op.vec(n*12,n*16,n*14,'adamw'),'adamw')
        for shard in range(tp):plan.comm(tuple(range(shard,self.world,tp)),'ALL_GATHER',layout.shard_elements*2,'dp_parameter_AG')
        return self.fixed_plan(plan)

    def make_plan(self):
        # Upstream Plan hardcodes54 for wafer initialization. Construct its
        # exact fields for the declared ACTIVE world so idle dies get no work.
        plan=Plan.__new__(Plan);plan.op=self.op;plan.world=self.world
        plan.operations=[];plan.groups={};plan.groupids={};plan.last={};plan.stats={}
        plan.local(tuple(range(self.world)),Cost(compute=1),'initialize')
        return plan


class AlignedTrainer(Trainer):
    def enable(self,t,full=False):
        dp=self.s.be.world//self.s.be.p.train_tp
        first=self.s.retained_ranks//self.s.be.p.train_tp
        workers=range(dp) if full else range(first,dp)
        new=[w for w in workers if w not in self.ready]
        ready=self.s.prepare_training(t,new)
        for w in new:self.ready[w]=ready[w]
        self.started=True;self.drain(t,flush=full)


class PlatformSim(Sim):
    """Resident train states on both platforms; no fictitious CPU gradient copy."""
    def gradient_transfer(self):return 0.

    def decorate_summary(self,d):
        d=super().decorate_summary(d)
        allocated=getattr(self.be,'allocated_world',self.be.world)
        d['resource_budget']={'allocated':allocated,'active':self.be.world,'idle_ranks':list(range(self.be.world,allocated))}
        return d

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        unit=math.lcm(self.be.p.rollout_tp,self.be.p.train_tp)
        self.retained_ranks=(self.be.world//2//unit)*unit
        self.trainer=AlignedTrainer(self)

    def try_scale(self,t):
        # Source half split adapted to whole train/rollout groups: on 54 dies
        # with TP2 this is 26 rollout +28 training, never a cut DP group.
        if not 0<self.retained_ranks<self.be.world:
            self.log(t,'SCALE_REJECT',reason='no aligned whole-group partition');self.scaled=True;return
        remaining=sum(r.cancel_at is None and r.status not in ('finished','cancelled') for r in self.reqs.values())
        slots=min(self.cap,math.ceil(remaining/(self.retained_ranks//self.be.p.rollout_tp)))
        observed=max((r.generated for r in self.reqs.values() if r.status=='active'),default=0)
        length=max(self.be.history_response_bound,observed)
        reserve=slots*(self.be.op.prompt_cap+length)*self.be.kv
        available=self.rollout_available()
        if reserve>available:
            self.log(t,'SCALE_REJECT',reason='KV reservation',reserve=reserve,available=available);return
        self.scaling=True;self.log(t,'SCALE_REQUEST',reserve=reserve,available=available,retained_ranks=self.retained_ranks)
        for rep in self.replicas:rep.interrupt(t,halt=True)
        if all(rep.halted for rep in self.replicas):self.migrate(t)

    def migrate(self,t):
        if self.be.op.platform=='gpu':return super().migrate(t)
        # Parent migration uses a node uplink. Mesh migration routes tokens on
        # directed physical links and charges the same generated-prefix rebuild.
        self.scaling=False;self.scaled=True;self.scale_time=t
        half=self.retained_ranks//self.be.p.rollout_tp;receivers=self.replicas[:half];moved=[]
        for rep in self.replicas[half:]:
            moved.extend((r,rep.i*self.be.p.rollout_tp) for r in list(rep.active)+list(rep.queue));rep.active=[];rep.queue.clear()
        loads=defaultdict(int)
        for r,src in moved:
            if r.cancel_at is not None:
                if r.status not in ('finished','cancelled'):
                    r.status='cancelled';self.log(t,'CANCEL_ACK',request=r.rid,generated=r.generated,reason='migration-barrier')
                continue
            if r.status in ('finished','cancelled'):continue
            dst=min(receivers,key=lambda x:(len(x.active)+len(x.queue),x.i))
            r.migrations+=1;r.status='queued';r.replica=dst.i;dst.queue.append(r)
            for link in self.be.route(src,dst.i*self.be.p.rollout_tp):loads[link]+=(r.prompt+r.generated)*4
        delay=max(loads.values(),default=0)/(self.be.op.hw.d2d_bytes_per_ns_per_direction*NS)
        delay+=13*self.be.op.hw.d2d_latency_ns/NS if loads else 0.
        self.migration_seconds=delay;self.log(t,'MIGRATE',requests=len(moved),token_transfer_seconds=delay)
        for rep in receivers:
            rep.halted=False;rep.halt_requested=False;self.push(t+delay,'resume',rep.i)
        self.trainer.enable(t)
