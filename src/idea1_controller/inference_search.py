"""Public-history inference placement and full-step candidate scoring.

This is a planning surrogate. Real mesh contention and runtime borrowing are
accepted only through native TUNE execution; no simulated device time is
replaced by this score.
"""
from collections import deque,defaultdict
import statistics
from .placement import largest_remainder
from .history import quantile
from .lifecycle_memory import SystemMemory


def balanced_execution(public,forecasts,inference,catalog):
    if any(set(g)-{'group_id','prompt_length','prompt_sha256','library'} for g in public):raise ValueError('live response/EOS is forbidden in inference placement')
    groups={int(r):tuple(ws) for r,ws in inference['groups'].items()};ids=[str(x['group_id']) for x in public]
    roots=sorted(groups);tp={r:1 if inference.get('paired_tp1') else 2*len(groups[r]) for r in roots}
    caps={r:int(inference['caps'][str(r)]) for r in roots}
    rate={r:caps[r]/catalog.decode(tp[r],caps[r],4096) for r in roots}
    minimum=min(rate.values());weights=[max(1,round(rate[r]/minimum*1e6)) for r in roots]
    quota=dict(zip(roots,largest_remainder(len(ids),weights)))
    longs=[g for g in ids if forecasts[g]['label']=='proxy-long'];shorts=[g for g in ids if forecasts[g]['label']=='proxy-short']
    longquota=dict(zip(roots,largest_remainder(len(longs),weights)))
    longquota={r:min(longquota[r],quota[r]) for r in roots}
    while sum(longquota.values())<len(longs):
        r=min((r for r in roots if longquota[r]<quota[r]),key=lambda r:((longquota[r]+1)/rate[r],r));longquota[r]+=1
    queues={r:[] for r in roots};work={r:0. for r in roots};tail={r:0. for r in roots};left=longquota.copy()
    means={g:statistics.mean(sum(v) for v in forecasts[g]['response_scenarios']) for g in ids}
    tails={g:quantile([max(v) for v in forecasts[g]['response_scenarios']],.9) for g in ids}
    shortcost={r:catalog.decode(tp[r],min(8,caps[r]),4096) for r in roots}
    def score(r,g):return (work[r]+means[g])/rate[r]+.5*max(tail[r],tails[g])*shortcost[r]
    def push(r,g):queues[r].append(g);work[r]+=means[g];tail[r]=max(tail[r],tails[g])
    for g in sorted(longs,key=lambda g:(-forecasts[g]['score'],ids.index(g))):
        r=min((r for r in roots if left[r]),key=lambda r:(score(r,g),r));push(r,g);left[r]-=1
    for g in sorted((g for g in ids if g not in longs),key=lambda g:(-forecasts[g]['score'],ids.index(g))):
        r=min((r for r in roots if len(queues[r])<quota[r]),key=lambda r:(score(r,g),r));push(r,g)
    for r in roots:queues[r].sort(key=lambda g:(-forecasts[g]['score'],ids.index(g)))
    assert sorted(g for gs in queues.values() for g in gs)==sorted(ids)
    assert all(sum(forecasts[g]['label']=='proxy-long' for g in queues[r])==longquota[r] for r in roots)
    return {str(r):gs for r,gs in queues.items()},dict(long_quota={str(r):q for r,q in longquota.items()},group_quota={str(r):q for r,q in quota.items()},scope='actual inference groups; independent of retained training-home quotas')


class InferenceScenarioScorer:
    def __init__(self,op,policy,catalog):
        self.op=op;self.policy=policy;self.catalog=catalog;self.memory=SystemMemory(op);self.replica_cache={}
        from rl_xy.model import enumerate_parameters
        self.parameter_count=sum(x.unique_elements for x in enumerate_parameters(op.case_tp(2)))
    def replica(self,gids,public,forecast,scenario,tp,cap):
        key=(tp,cap,tuple((g,public[g]['prompt_length'],tuple(forecast[g]['response_scenarios'][scenario%16])) for g in gids))
        if key in self.replica_cache:return self.replica_cache[key]
        queue=deque((public[g]['prompt_length'],int(r)) for g in gids for r in forecast[g]['response_scenarios'][scenario%16])
        active=[];elapsed=0.
        while queue or active:
            if queue and len(active)<cap:
                add=[]
                while queue and len(active)+len(add)<cap:add.append(queue.popleft())
                elapsed+=self.catalog.phase('prefill',tp,[p for p,r in add])
                active.extend([p+1,r-1] for p,r in add if r>1)
            else:
                steps=min(r for c,r in active);mean=sum(c for c,r in active)/len(active)
                a=self.catalog.decode(tp,len(active),mean);b=self.catalog.decode(tp,len(active),mean+1)
                elapsed+=steps*a+steps*(steps-1)/2*(b-a)
                active=[[c+steps,r-steps] for c,r in active if r>steps]
        self.replica_cache[key]=elapsed;return elapsed
    def training_services(self,home,public,forecast,scenario):
        rows={}
        for r,ws in home['region_members'].items():
            tp=2*len(ws);items=[]
            for w in ws:
                for g in home['queues'][str(w)]:items.extend(dict(prompt_length=public[g]['prompt_length'],response_length=int(v)) for v in forecast[g]['response_scenarios'][scenario%16])
            teacher=training=0.
            for batch in self.memory.batches(items,'teacher',tp,self.policy.logprob_mb*len(ws)):
                teacher+=2*self.catalog.phase('teacher',tp,[x['prompt_length']+x['response_length']-1 for x in batch])
            maximum=self.policy.train_mb if tp==2 else min(8,self.policy.train_mb*tp)
            for batch in self.memory.batches(items,'training',tp,maximum):
                training+=self.catalog.phase('training',tp,[x['prompt_length']+x['response_length']-1 for x in batch])
            data_bytes=sum(4*(x['prompt_length']+x['response_length']) for x in items)
            regional_copy=2*data_bytes*max(0,len(ws)-1)/self.op.hw.d2d_bytes_per_ns_per_direction
            restore=0 if tp==2 else self.parameter_count*4*(1-2/tp)/self.op.hw.d2d_bytes_per_ns_per_direction+self.parameter_count*16/self.op.hw.dram_bytes_per_ns
            rows[int(r)]=teacher+training+regional_copy+restore
        return rows
    def score(self,home,inference,public_groups,forecast,training_by_scenario=None):
        public={str(g['group_id']):g for g in public_groups};groups={int(r):tuple(ws) for r,ws in inference['groups'].items()}
        queues=inference.get('execution_queues') or {str(r):[g for w in ws for g in home['queues'][str(w)]] for r,ws in groups.items()}
        origin={g:int(r) for r,gs in queues.items() for g in gs};scores=[]
        for scenario in range(16):
            ends={}
            for r,ws in groups.items():
                tp=1 if inference.get('paired_tp1') else 2*len(ws);gids=queues[str(r)];cap=int(inference['caps'][str(r)])
                ends[r]=max(self.replica(gids[::2],public,forecast,scenario,1,cap),self.replica(gids[1::2],public,forecast,scenario,1,cap)) if tp==1 else self.replica(gids,public,forecast,scenario,tp,cap)
            services=training_by_scenario[scenario] if training_by_scenario is not None else self.training_services(home,public,forecast,scenario)
            finish=[]
            for root,ws in home['region_members'].items():
                gids=[g for w in ws for g in home['queues'][str(w)]]
                ready=max((ends[origin[g]] for g in gids),default=0.)
                finish.append(ready+services[int(root)])
            scores.append(max(finish,default=0.))
        mean=statistics.mean(scores);risk=.75*mean+.25*statistics.mean(sorted(scores)[-2:])
        return dict(mean_ns=mean,risk_ns=risk,scenario_ns=scores,limitations='no exact mesh queues or training stealing; TP1 pair synchronization approximated; fixed global prefix/update excluded from surrogate only; native TUNE validation required')


class InferenceHomePlacement:
    """Reapply a TUNE-locked placement strategy to current public/history tasks.

    The strategy is a hyperparameter, not a cached TUNE assignment. This avoids
    rescoring the same three old-TP2 placement strategies for every DP layout;
    full resource choices are subsequently compared at their actual inference TP.
    """
    def __init__(self,op,policy,catalog):
        from .lifecycle_memory import SystemMemory
        self.op=op;self.policy=policy;self.catalog=catalog;self.memory=SystemMemory(op);self.assignment_cache={}
    def build(self,public_groups,history,region_members,strategy):
        import time
        from dataclasses import asdict
        from .history import predict_groups,digest
        from .quotas import layout_quotas
        from .search_views import ViewPlanSearch
        from .coordinates import ParameterCoordinates
        if any(set(g)-{'group_id','prompt_length','prompt_sha256','library'} for g in public_groups):raise ValueError('live response/EOS fields forbidden')
        if set(strategy)!={'adaptive','queue_order'} or strategy['queue_order'] not in ('input','history_long_first'):raise ValueError('unrecognized TUNE home-placement strategy')
        start=time.perf_counter();regions={int(r):tuple(ws) for r,ws in region_members.items()};public={str(g['group_id']):g for g in public_groups};ids=list(public)
        self.origin_quotas=layout_quotas(len(ids),regions)
        missing=[g for g in ids if len(history.get('records',{}).get(g,()))<2]
        if missing:
            cursor=0;queues={}
            for w,n in self.origin_quotas.items():queues[w]=ids[cursor:cursor+n];cursor+=n
            forecast={};longq={r:0 for r in regions}
        else:
            forecast=predict_groups(public_groups,history)
            # Reuse the canonical placement algorithm without constructing an
            # obsolete TP2 resource scorer or its constant cost prefix.
            self.assignment_model=lambda rs,pub,fc:ViewPlanSearch.assignment_model(self,rs,pub,fc)
            queues,longq=ViewPlanSearch._assign(self,ids,forecast,regions,strategy['adaptive'],public,strategy['queue_order'])
            if not strategy['adaptive']:
                longq={r:sum(forecast[g].label=='proxy-long' for w in ws for g in queues[w]) for r,ws in regions.items()}
        return dict(schema='idea1.native-selected-plan/2',region_members=regions,queues=queues,rollout_tp=2,rollout_cap=self.policy.cap,
            selected_dp=len(regions),selected_candidate='TUNE-locked-home-strategy',home_assignment_strategy=strategy,queue_order=strategy['queue_order'],
            long_quota_by_region=longq,assignment_long_quotas=longq,forecasts={g:asdict(v) for g,v in forecast.items()},
            long_quota_kind='dispersed-history-quota' if strategy['adaptive'] else 'observed-baseline-count; no dispersion constraint',
            assignment_forecasts={g:asdict(v) for g,v in forecast.items()},history_snapshot_id=history['snapshot_id'],fallback_group_ids=ids if missing else [],
            parameter_schema_sha256=ParameterCoordinates(self.op.model).schema_digest,public_input_digest=digest(public_groups),
            origin_group_quotas=self.origin_quotas,history_kind=history.get('kind','unknown'),supported_training_tp=[2,4,8],
            planning_wall_s=time.perf_counter()-start,budget_truncated=False,plan_id=digest(dict(regions=regions,queues=queues,strategy=strategy,history=history['snapshot_id'])))
