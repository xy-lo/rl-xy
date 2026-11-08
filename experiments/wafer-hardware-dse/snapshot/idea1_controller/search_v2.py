"""Planner score follows packed PENDING, task history and adaptive TP training."""
from .search import NativePlanSearch
from .scenario_runtime import ScenarioRuntime
from .memory import RegionalMemory
import math

class PowerOfTwoMemory(RegionalMemory):
    def batches(self,items,stage,tp,maximum):
        cursor=0
        while cursor<len(items):
            limit=min(maximum,len(items)-cursor);n=2**int(math.log2(limit))
            while self.phase(stage,tp,items[cursor:cursor+n]).peak_bytes>self.op.capacity:
                if n==1:raise ValueError('single sample exceeds memory')
                n//=2
            yield items[cursor:cursor+n];cursor+=n

class AdaptivePlanSearch(NativePlanSearch):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.memory=PowerOfTwoMemory(self.op)
        self.simulator=ScenarioRuntime(self.catalog,self.op,self.policy.cap,interval=64,helper=self.helper)
    def _score(self,regions,queues,public,forecast,scenario,pressure=False):
        ends,counts=self.simulator.run(regions,queues,public,forecast,scenario,pressure);rows=[];max_end=0.
        from rl_xy.model import enumerate_parameters
        params=sum(x.unique_elements for x in enumerate_parameters(self.op.case_tp(2)))
        for root,ws in sorted(regions.items()):
            tp=2*len(ws);items=[]
            for w in ws:
                for g in queues[w]:
                    vec=(8192,)*8 if pressure else forecast[g].response_scenarios[scenario%16]
                    items.extend(dict(prompt_length=public[g]['prompt_length'],response_length=int(r)) for r in vec)
            teacher=training=0.
            logmax=self.policy.logprob_mb*len(ws)
            trainmax=self.policy.train_mb if tp==2 else min(8,self.policy.train_mb*tp)
            for batch in self.memory.batches(items,'teacher',tp,logmax):teacher+=2*self.catalog.phase('teacher',tp,[x['prompt_length']+x['response_length']-1 for x in batch])
            for batch in self.memory.batches(items,'training',tp,trainmax):training+=self.catalog.phase('training',tp,[x['prompt_length']+x['response_length']-1 for x in batch])
            size=sum((x['prompt_length']+x['response_length'])*4 for x in items)
            data=size/max(1,self.op.hw.d2d_bytes_per_ns_per_direction)*(len(ws)-1)*2
            restore=0. if tp==2 else params*4*(1-2/tp)/self.op.hw.d2d_bytes_per_ns_per_direction+params*16/self.op.hw.dram_bytes_per_ns
            finish=ends[root]+data+teacher+training+restore;max_end=max(max_end,finish)
            rows.append(dict(root=root,tp=tp,rollout_ns=ends[root],teacher_ns=teacher,training_ns=training,restore_ns=restore,finish_ns=finish,search_pending_moves=counts['pending'],search_helper_leases=counts['helper']))
        return max_end+self.common_ns,rows
