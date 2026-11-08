"""Joint physical-region and history assignment scoring for actor-view execution."""
from .search_v2 import AdaptivePlanSearch
from .assignment_v2 import assign_groups
from .physical_layouts import guillotine_layouts,paired_region_candidates
from .history import predict_groups,digest,quantile
from .placement import largest_remainder
from .quotas import layout_quotas
from .search import RiskSelector,risk
from dataclasses import asdict
import statistics,time

class ViewPlanSearch(AdaptivePlanSearch):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from .lifecycle_memory import SystemMemory
        self.memory=SystemMemory(self.op)
        # Shared actor views remove the original weight export/pack prefix.
        from rl_xy.qwen3_native import NetworkProfiles
        from rl_xy.qwen3_experiment import weight_plan,simple_plan_time
        net=NetworkProfiles(self.op,'unused',native=False)
        self.common_ns-=simple_plan_time(weight_plan(self.op,self.policy),net)
        self.common_ns+=1000
        self.assignment_cache={}
    def assignment_model(self,regions,public,forecast):
        traincost={};gencost={}
        for g,f in forecast.items():
            v=sorted(f.response_scenarios,key=lambda v:(sum(v)+max(v),v))[len(f.response_scenarios)//2]
            items=[dict(prompt_length=public[g]['prompt_length'],response_length=r) for r in v]
            gencost[g]=f.score*self.catalog.decode(2,8,4096)/9
            for tp in {2*len(ws) for ws in regions.values()}:
                key=(g,tp)
                if key not in self.assignment_cache:
                    total=0;logmax=self.policy.logprob_mb*(tp//2);trainmax=self.policy.train_mb if tp==2 else min(8,self.policy.train_mb*tp)
                    for batch in self.memory.batches(items,'teacher',tp,logmax):total+=2*self.catalog.phase('teacher',tp,[x['prompt_length']+x['response_length']-1 for x in batch])
                    for batch in self.memory.batches(items,'training',tp,trainmax):total+=self.catalog.phase('training',tp,[x['prompt_length']+x['response_length']-1 for x in batch])
                    self.assignment_cache[key]=total
                traincost[key]=self.assignment_cache[key]
        def trial(r,queues,g):
            ws=regions[r];tp=len(ws)*2;loads={w:sum(gencost[x] for x in queues[w]) for w in ws}
            if forecast[g].label=='proxy-long':w=min((w for w in ws if len(queues[w])<self.origin_quotas[w]),key=lambda w:(not any(forecast[x].label=='proxy-long' for x in queues[w]),w))
            else:w=min((w for w in ws if len(queues[w])<self.origin_quotas[w]),key=lambda w:(any(forecast[x].label=='proxy-long' for x in queues[w]) if forecast[g].label=='proxy-short' else False,loads[w],w))
            loads[w]+=gencost[g]
            gen=max(loads.values())/max(1,self.policy.cap/8)
            return gen+sum(traincost[x,tp] for j in ws for x in queues[j])+traincost[g,tp]
        return trial
    def _assign(self,ids,forecast,regions,adaptive,public,queue_order='input'):
        queues,quotas=assign_groups(ids,forecast,regions,self.origin_quotas,adaptive,trial_score=self.assignment_model(regions,public,forecast) if adaptive else None)
        if queue_order=='history_long_first':
            order={g:i for i,g in enumerate(ids)}
            queues={w:sorted(q,key=lambda g:(-forecast[g].score,order[g])) for w,q in queues.items()}
        return queues,quotas
    def build_fixed(self,public_groups,history,region_members,deadline_s=5):
        start=time.perf_counter();public={str(g['group_id']):dict(g) for g in public_groups};ids=list(public);forecast=predict_groups(public_groups,history)
        self.assignment_cache.clear();regions={int(r):tuple(ws) for r,ws in region_members.items()};self.origin_quotas=layout_quotas(len(ids),regions);candidates=[]
        for adaptive,queue_order in ((False,'input'),(True,'input'),(True,'history_long_first')):
            queues,longq=self._assign(ids,forecast,regions,adaptive,public,queue_order)
            candidates.append(dict(id=digest(dict(regions=regions,queues=queues)),dp=len(regions),regions=regions,queues=queues,long_quotas=longq,adaptive=adaptive,queue_order=queue_order,
                scenario_ns=[self._score(regions,queues,public,forecast,s)[0] for s in range(16)],pressure_ns=self._score(regions,queues,public,forecast,0,True)[0]))
        elapsed=time.perf_counter()-start;winner,audit=RiskSelector().select(candidates[0],candidates[1:],elapsed+getattr(self,'control_overhead_s',0),uncertainty=self.catalog.data['p95_relative_validation_error'])
        return self._manifest(winner,forecast,history,elapsed,candidates,audit,False,deadline_s)
    def _manifest(self,winner,forecast,history,elapsed,final,audit,truncated,budget):
        return dict(schema='idea1.native-selected-plan/2',region_members=winner['regions'],rollout_tp=2,rollout_cap=self.policy.cap,queues=winner['queues'],
            queue_order=winner.get('queue_order','input'),history_snapshot_id=history['snapshot_id'],forecasts={g:asdict(f) for g,f in forecast.items()},selected_candidate=winner['id'],selected_dp=winner['dp'],
            long_quota_by_region=winner['long_quotas'],fully_scored_candidates=final,selection_audit=audit,planning_wall_s=elapsed,budget_s=budget,budget_truncated=truncated,
            scope='scenario service surrogate; native measured execution required',optimizer_owners='original TP2 distributed optimizer; immutable actor views',
            origin_group_quotas=layout_quotas(sum(map(len,winner['queues'].values())),winner['regions']),
            plan_id=digest(dict(regions=winner['regions'],queues=winner['queues'],history=history['snapshot_id'],cap=self.policy.cap)))
    def build(self,public_groups,history,deadline_s=10):
        start=time.perf_counter();public={str(g['group_id']):dict(g) for g in public_groups};ids=list(public);forecast=predict_groups(public_groups,history)
        self.assignment_cache.clear();mapping=getattr(self.op,'rlxy_pair_mapping',None)
        layouts=guillotine_layouts(2) if mapping is None else paired_region_candidates(mapping)
        baseline_count=27 if mapping is None else len(mapping.pairs)
        coarse=[]
        for dp,variants in layouts.items():
            for rects in variants:
                regions=({min(x//2 for x in r.dies(2)):tuple(sorted(set(x//2 for x in r.dies(2)))) for r in rects}
                         if mapping is None else rects)
                self.origin_quotas=layout_quotas(len(ids),regions)
                for adaptive,queue_order in ((False,'input'),(True,'input'),(True,'history_long_first')):
                    queues,longq=self._assign(ids,forecast,regions,adaptive,public,queue_order);value,rows=self._score(regions,queues,public,forecast,0)
                    coarse.append(dict(id=digest(dict(regions=regions,queues=queues)),dp=dp,regions=regions,queues=queues,long_quotas=longq,adaptive=adaptive,queue_order=queue_order,coarse_ns=value,coarse_regions=rows))
        counts=sorted(layouts);bins=min(4,len(counts));selected=[]
        for b in range(bins):
            ds={d for i,d in enumerate(counts) if bins*i//len(counts)==b}
            selected+=sorted((c for c in coarse if c['dp'] in ds),key=lambda c:(c['coarse_ns'],c['id']))[:2]
        baseline=min((c for c in coarse if c['dp']==baseline_count and not c['adaptive']),key=lambda c:c['id']);selected=[baseline]+[c for c in selected if c['id']!=baseline['id']]
        final=[];truncated=False
        for c in selected[:self.max_finalists]:
            if final and time.perf_counter()-start>deadline_s:truncated=True;break
            c=dict(c,scenario_ns=[self._score(c['regions'],c['queues'],public,forecast,s)[0] for s in range(16)],pressure_ns=self._score(c['regions'],c['queues'],public,forecast,0,True)[0]);final.append(c)
        elapsed=time.perf_counter()-start;winner,audit=RiskSelector().select(final[0],final[1:],elapsed+getattr(self,'control_overhead_s',0),uncertainty=self.catalog.data['p95_relative_validation_error'])
        result=self._manifest(winner,forecast,history,elapsed,final,audit,truncated,deadline_s);result['all_coarse_candidates']=coarse
        if mapping is not None:
            result['physical_mesh_shape']=[mapping.mesh.columns,mapping.mesh.rows]
            result['owner_pairs']=[list(pair) for pair in mapping.pairs]
            result['unpaired_dies']=list(mapping.unpaired)
            result['geometry_search_scope']='bounded connected pair-merge trajectories, not exhaustive'
        return result
