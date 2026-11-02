"""TP2 training replicas inside larger, physically compact cooperation domains.

A six-die cooperation domain contains three TP2 inference/training instances;
it is never described or executed as an unsupported TP6 training replica.
"""
import heapq
import time
from .search_views import ViewPlanSearch
from .history import predict_groups,digest
from .quotas import layout_quotas
from .search import RiskSelector


class PoolPlanSearch(ViewPlanSearch):
    def assignment_model(self,regions,public,forecast):
        # Fixed quotas keep per-instance data volumes bounded; runtime training
        # help handles residual imbalance. Region quotas concern cooperation.
        return None

    def _score(self,regions,queues,public,forecast,scenario,pressure=False):
        ends,counts=self.simulator.run(regions,queues,public,forecast,scenario,pressure)
        poolof={w:r for r,ws in regions.items() for w in ws};rows=[];work={};ready={}
        for w,gids in sorted(queues.items()):
            items=[]
            for g in gids:
                vec=(8192,)*8 if pressure else forecast[g].response_scenarios[scenario%16]
                items.extend(dict(prompt_length=public[g]['prompt_length'],response_length=int(r)) for r in vec)
            teacher=0
            for batch in self.memory.batches(items,'teacher',2,self.policy.logprob_mb):
                teacher+=2*self.catalog.phase('teacher',2,[x['prompt_length']+x['response_length']-1 for x in batch])
            jobs=[]
            for batch in self.memory.batches(items,'training',2,self.policy.train_mb):
                cost=self.catalog.phase('training',2,[x['prompt_length']+x['response_length']-1 for x in batch])
                size=sum(5*(x['prompt_length']+x['response_length'])+8*x['response_length']+68 for x in batch)
                jobs.append((cost,size))
            work[w]=jobs;ready[w]=ends[poolof[w]]+teacher
            rows.append(dict(root=w,tp=2,rollout_ns=ends[poolof[w]],teacher_ns=teacher,training_ns=sum(x[0] for x in jobs),restore_ns=0))
        # Safe-boundary, source-initiated whole-batch donation. This remains an
        # alpha-beta surrogate; real native mesh contention is measured later.
        global_eos=max(ends.values());events=[(t,w) for w,t in ready.items()];heapq.heapify(events);idle=set();finish={}
        while events:
            now,w=heapq.heappop(events)
            if work[w] and now>=global_eos and idle:
                dst=min(idle,key=lambda j:(abs((2*j)//6-(2*w)//6)+abs((2*j)%6-(2*w)%6),j));idle.remove(dst)
                job=max(work[w]);work[w].remove(job);cost,size=job
                hops=max(1,abs((2*dst)//6-(2*w)//6)+abs((2*dst)%6-(2*w)%6))
                move=size/self.op.hw.d2d_bytes_per_ns_per_direction*hops+2*size/self.op.hw.dram_bytes_per_ns
                heapq.heappush(events,(now+move+cost,dst));now+=move
            if work[w]:
                cost,size=work[w].pop(0);heapq.heappush(events,(now+cost,w))
            else:idle.add(w);finish[w]=now
        assert all(not q for q in work.values())
        return max(finish.values())+self.common_ns,rows

    def _manifest(self,winner,forecast,history,elapsed,final,audit,truncated,budget):
        result=super()._manifest(winner,forecast,history,elapsed,final,audit,truncated,budget)
        result.update(resource_family='pooled-tp2',helper_domains=winner['regions'],region_members={w:(w,) for w in range(27)},
            selected_dp=27,selected_helper_domains=len(winner['regions']),long_quota_scope='helper_domains',
            optimizer_owners='27 original TP2 distributed-optimizer replicas; unchanged owner ordinal',
            scope='cooperation-domain quotas + TP2 training; historical scenario surrogate, native validation required')
        return result

    def build(self,public_groups,history,deadline_s=10):
        # Horizontal legacy geometry and compact vertical 3x2-die rectangles.
        horizontal={3*r:tuple(range(3*r,3*r+3)) for r in range(9)}
        vertical={9*b+c:tuple(9*b+c+3*i for i in range(3)) for b in range(3) for c in range(3)}
        start=time.perf_counter();public={str(g['group_id']):dict(g) for g in public_groups};ids=list(public);forecast=predict_groups(public_groups,history);candidates=[]
        for geometry,regions in (('horizontal',horizontal),('vertical',vertical)):
            self.origin_quotas=layout_quotas(len(ids),regions)
            for adaptive,order in ((False,'input'),(True,'input'),(True,'history_long_first')):
                queues,longq=self._assign(ids,forecast,regions,adaptive,public,order)
                candidates.append(dict(id=digest(dict(regions=regions,queues=queues)),dp=27,regions=regions,queues=queues,long_quotas=longq,
                    adaptive=adaptive,queue_order=order,geometry=geometry,
                    scenario_ns=[self._score(regions,queues,public,forecast,s)[0] for s in range(16)],pressure_ns=self._score(regions,queues,public,forecast,0,True)[0]))
        elapsed=time.perf_counter()-start;winner,audit=RiskSelector().select(candidates[0],candidates[1:],elapsed+getattr(self,'control_overhead_s',0),uncertainty=self.catalog.data['p95_relative_validation_error'])
        return self._manifest(winner,forecast,history,elapsed,candidates,audit,False,deadline_s)
