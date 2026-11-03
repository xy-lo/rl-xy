"""Quota-preserving balanced queues with producer/home locality.

PB changes group allocation under fixed quotas. PS must retain PB queues and
training order, so it is a separate later operation. No deliberate short queue
or future output length is used to manufacture helpers.
"""
import math
from collections import Counter
from .placement import largest_remainder
from .workload import digest


def balanced_release_placement(public,forecast,regions,worker_quotas,inference_groups,*,training_queue_order='history_long_first',inference_queue_order='history_long_first',service_costs=None,inference_ns_per_score=None):
    if training_queue_order not in ('history_long_first','input'):raise ValueError('unknown training queue order')
    if inference_queue_order not in ('history_long_first','input'):raise ValueError('unknown inference queue order')
    ids=[str(g['group_id']) for g in public];index={g:i for i,g in enumerate(ids)}
    if len(set(ids))!=len(ids) or set(ids)!=set(forecast):raise ValueError('fixed group/forecast mismatch')
    regions={int(r):tuple(map(int,ws)) for r,ws in regions.items()}
    groups={int(r):tuple(map(int,ws)) for r,ws in inference_groups.items()}
    quotas={int(w):int(n) for w,n in worker_quotas.items()}
    if sum(quotas.values())!=len(ids) or any(n<0 for n in quotas.values()):raise ValueError('invalid frozen worker quotas')
    workers=sorted(quotas)
    if sorted(w for ws in regions.values() for w in ws)!=workers or sorted(w for ws in groups.values() for w in ws)!=workers:raise ValueError('region/inference partition mismatch')
    owner={w:r for r,ws in groups.items() for w in ws};region_of={w:r for r,ws in regions.items() for w in ws}
    service=service_costs is not None or inference_ns_per_score is not None
    if service:
        if service_costs is None or inference_ns_per_score is None:raise ValueError('both service and producer costs required')
        if set(service_costs)!=set(regions) or set(inference_ns_per_score)!=set(groups):raise ValueError('service cost region/cohort mismatch')
        values=list(inference_ns_per_score.values())
        for costs in service_costs.values():
            if set(costs)!=set(ids):raise ValueError('service cost group mismatch')
            values.extend(costs.values())
        if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or v<=0 for v in values):raise ValueError('invalid predicted service cost')
    physical_dependencies={r:{owner[w] for w in ws} for r,ws in regions.items()}
    capacity={r:sum(quotas[w] for w in ws) for r,ws in regions.items()}
    longs=[g for g in ids if forecast[g]['label']=='proxy-long'];longset=set(longs)
    roots=sorted(regions);longquota=dict(zip(roots,largest_remainder(len(longs),[capacity[r] for r in roots])))
    longleft=longquota.copy();remaining=capacity.copy();queues={w:[] for w in workers}
    work={w:0. for w in workers};producer_work={r:0. for r in groups};region_work={r:0. for r in regions}
    def score(g):
        value=forecast[g]['score']
        if not isinstance(value,(float,int)) or isinstance(value,bool) or not math.isfinite(value) or value<=0:raise ValueError('missing predicted workload')
        return float(value)
    def producer_ns(root,g=None):
        return (producer_work[root]+(score(g) if g is not None else 0))*inference_ns_per_score[root]
    def worker(r,g):
        candidates=[w for w in regions[r] if len(queues[w])<quotas[w]]
        if service:
            return min(candidates,key=lambda w:(producer_ns(owner[w],g),work[w]+service_costs[r][g],w))
        return min(candidates,key=lambda w:((producer_work[owner[w]]+score(g))/len(groups[owner[w]]),work[w]+score(g),w))
    def put(g,islong):
        candidates=[r for r in roots if remaining[r] and (not islong or longleft[r])]
        def key(r):
            w=worker(r,g)
            if service:
                ready=max(producer_ns(root,g if root==owner[w] else None) for root in physical_dependencies[r])
                return (ready+region_work[r]+service_costs[r][g],ready,work[w]+service_costs[r][g],r)
            return ((region_work[r]+score(g))/len(regions[r]),(producer_work[owner[w]]+score(g))/len(groups[owner[w]]),work[w]+score(g),r)
        r=min(candidates,key=key);w=worker(r,g)
        queues[w].append(g)
        amount=service_costs[r][g] if service else score(g)
        work[w]+=amount;producer_work[owner[w]]+=score(g);region_work[r]+=amount;remaining[r]-=1
        if islong:longleft[r]-=1
    for g in sorted(longs,key=lambda g:(-score(g),index[g])):put(g,True)
    for g in sorted((g for g in ids if g not in longset),key=lambda g:(-score(g),index[g])):put(g,False)
    for q in queues.values():
        q.sort(key=(lambda g:(-score(g),index[g])) if training_queue_order=='history_long_first' else index.get)
    inference_key=(lambda g:(-score(g),index[g])) if inference_queue_order=='history_long_first' else index.get
    inference={r:sorted((g for w in ws for g in queues[w]),key=inference_key) for r,ws in groups.items()}
    if Counter(g for q in queues.values() for g in q)!=Counter(ids) or Counter(g for q in inference.values() for g in q)!=Counter(ids):raise AssertionError('group conservation')
    if any(len(queues[w])!=quotas[w] for w in workers):raise AssertionError('worker quota changed')
    for r,ws in regions.items():
        if sum(g in longset for w in ws for g in queues[w])!=longquota[r]:raise AssertionError('regional long quota changed')
    producer={g:r for r,gs in inference.items() for g in gs}
    dependencies={str(r):sorted({owner[w] for w in ws}|{producer[g] for w in ws for g in queues[w]}) for r,ws in regions.items()}
    proof=dict(policy='PB-balanced-producer-local',training_queue_order=training_queue_order,inference_queue_order=inference_queue_order,region_group_quota={str(r):capacity[r] for r in roots},
        worker_group_quota={str(w):quotas[w] for w in workers},long_quota_by_region={str(r):longquota[r] for r in roots},
        region_dependencies=dependencies,all_groups_digest=digest(ids),group_to_home_digest=digest({g:region_of[w] for w,q in queues.items() for g in q}),
        scope='forecast load balancing and physical cohort locality; finite heuristic, no global optimum or guaranteed helper claim')
    if service:
        proof.update(policy='PB-service-producer-local',
            placement_objective='estimated producer release plus teacher/training service at actual region TP; no second division by region size',
            placement_cost_digest=digest([service_costs,inference_ns_per_score]),
            estimated_region_training_ns={str(r):v for r,v in region_work.items()})
    return dict(queues={str(w):q for w,q in queues.items()},execution_queues={str(r):q for r,q in inference.items()},proof=proof)
