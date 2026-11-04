"""Bounded whole-queue permutations for region release.

The supported backend borrows whole TP groups until EOS; it cannot shrink an
active TP. Therefore the forecast evaluator replays remaining decode to EOS,
rather than pretending to acquire/release a 16-token FFN-only lease.
No true current response lengths are accepted: all lengths come from forecasts.
"""
from dataclasses import dataclass
from itertools import combinations, permutations
import heapq,math,time
from .workload import digest


@dataclass(frozen=True)
class Slot:
    key: str
    region: str
    members: tuple
    tp: int
    cap: int


def _duration(catalog,tp,active,decoded,steps):
    if not steps:return 0.
    context=decoded+sum(offset for _,offset in active)/len(active)
    a,b=catalog.coefficients('decode',tp,len(active))
    start=max(1.,a+b*context);following=max(1.,a+b*(context+1))
    return steps*start+steps*(steps-1)/2*(following-start)


def _remaining(catalog,tp,active,decoded):
    active=list(active);heapq.heapify(active);elapsed=0.
    while active:
        finish=active[0][0];elapsed+=_duration(catalog,tp,active,decoded,finish-decoded);decoded=finish
        while active and active[0][0]==decoded:heapq.heappop(active)
    return elapsed


def _profile(queue,public,forecast,scene,tp,catalog,stress=False):
    items=[]
    for gid in queue:
        f=forecast[gid];vectors=f['response_scenarios']
        vector=max(vectors,key=lambda v:sum(v)+max(v)) if stress else vectors[scene]
        items.extend((public[gid]['prompt_length'],int(r)) for r in vector)
    prefill=catalog.phase('prefill',tp,[p for p,_ in items]) if items else 0.
    active=tuple((r-1,p+1) for p,r in items if r>1)
    return dict(prefill=prefill,active=active,end=prefill+_remaining(catalog,tp,active,0),items=items)


def _boundary(profile,catalog,tp,ready):
    active=list(profile['active']);heapq.heapify(active);decoded=0;tick=profile['prefill']
    while active:
        finish=active[0][0];count=finish-decoded;end=tick+_duration(catalog,tp,active,decoded,count)
        if ready<=end:
            low,high=0,count
            while low<high:
                mid=(low+high)//2
                if tick+_duration(catalog,tp,active,decoded,mid)>=ready:high=mid
                else:low=mid+1
            step=min(count,((low+15)//16)*16)
            tick+=_duration(catalog,tp,active,decoded,step);decoded+=step
            while active and active[0][0]<=decoded:heapq.heappop(active)
            if active:return dict(start_ns=tick,active=tuple(active),decoded=decoded,contexts=tuple(decoded+offset for _,offset in active))
        else:
            tick=end;decoded=finish
            while active and active[0][0]==decoded:heapq.heappop(active)
    return None


def optimize_positions(slots,queues,public,forecast,catalog,*,can_expand,migration,drain,training,margin_fraction=.02,max_maps=32,passes=2):
    """Rank PS and retain PB if no reliable supported gain.

    migration(source,helper,state) -> {duration_ns, edges}; drain(slot,profile)
    -> duration_ns. They must account for actual movement; neither defaults to
    zero. Two actions sharing a helper, source or directed migration edge cannot
    be combined. Ranking is a forecast proxy and requires native TUNE acceptance.
    """
    if not slots or set(queues)!={s.key for s in slots}:raise ValueError('slot/queue identity mismatch')
    if not 0<=margin_fraction<1 or max_maps<1 or passes not in (1,2):raise ValueError('invalid bounded-search contract')
    if set(g for q in queues.values() for g in q)!=set(public) or set(public)!=set(forecast):raise ValueError('fixed batch mismatch')
    if len([g for q in queues.values() for g in q])!=len(public):raise ValueError('duplicate group')
    bykey={s.key:s for s in slots};regions={}
    for slot in slots:regions.setdefault(slot.region,[]).append(slot.key)
    scenes=len(next(iter(forecast.values()))['response_scenarios'])
    if scenes<2 or any(len(f['response_scenarios'])!=scenes for f in forecast.values()):raise ValueError('inconsistent forecast scenarios')
    if any(not v or any(type(n) is not int or n<1 for n in v) for f in forecast.values() for v in f['response_scenarios']):raise ValueError('invalid forecast length vector')
    if any(sum(len(forecast[g]['response_scenarios'][0]) for g in queues[slot.key])>slot.cap for slot in slots):
        return dict(execution_queues={s:list(q) for s,q in queues.items()},logical_queue_at_slot={s:s for s in queues},baseline=None,candidate=None,stress=None,checked_maps=0,accepted_maps=0,events=[],host_wall_s=0.,forecast_digest=digest(forecast),scope='unsupported PENDING state: retain PB',reason='PB_RETAINED_PENDING_WORK_NOT_MODELED')
    profiles={};started=time.perf_counter();events=[]
    def profile(q,slot,scene):
        key=(q,slot.tp,scene)
        if key not in profiles:profiles[key]=_profile(queues[q],public,forecast,scene%scenes,slot.tp,catalog,scene==scenes)
        return profiles[key]
    def region_result(mapping,keys,scene):
        p={s:profile(mapping[s],bykey[s],scene) for s in keys}
        data={s:float(drain(bykey[s],p[s])) for s in keys}
        if any(not math.isfinite(v) or v<0 for v in data.values()):raise ValueError('invalid data/state drain cost')
        finish={s:p[s]['end']+data[s] for s in keys};actions=[]
        for source in keys:
            for helper in keys:
                if source==helper or not can_expand(bykey[source],bykey[helper]):continue
                # No PENDING work is manufactured or ignored by this model.
                if len(p[source]['items'])>bykey[source].cap or len(p[helper]['items'])>bykey[helper].cap:continue
                state=_boundary(p[source],catalog,bykey[source].tp,finish[helper])
                if state is None:continue
                cost=migration(bykey[source],bykey[helper],state)
                ns=float(cost['duration_ns'])
                if not math.isfinite(ns) or ns<0:raise ValueError('invalid migration cost')
                end=state['start_ns']+ns+_remaining(catalog,bykey[source].tp*2,state['active'],state['decoded'])+data[source]
                if end<finish[source]:actions.append(dict(source=source,helper=helper,end=end,start_ns=state['start_ns'],migration_ns=ns,edges=tuple(tuple(e) for e in cost['edges'])))
        base=max(finish.values());best=base;chosen=[]
        for action in actions:
            value=max(action['end'] if s==action['source'] else f for s,f in finish.items())
            if value<best:best=value;chosen=[action]
        for a,b in combinations(actions,2):
            if len({a['source'],a['helper'],b['source'],b['helper']})!=4:continue
            if set(a['edges'])&set(b['edges']):continue
            changed={a['source']:a['end'],b['source']:b['end']}
            value=max(changed.get(s,f) for s,f in finish.items())
            if value<best:best=value;chosen=[a,b]
        return best,dict(base_release_ns=base,release_ns=best,actions=chosen)
    def evaluate(mapping,scene_ids):
        values=[];details=[]
        for scene in scene_ids:
            rows={r:region_result(mapping,keys,scene) for r,keys in regions.items()}
            total=max(value[0]+float(training(r,scene)) for r,value in rows.items())
            values.append(total);details.append({r:x[1] for r,x in rows.items()})
        ordered=sorted(values);risk=.75*sum(values)/len(values)+.25*sum(ordered[-min(2,len(values)):])/min(2,len(values))
        return dict(values=values,risk=risk,details=details)
    current={s.key:s.key for s in slots};baseline=evaluate(current,range(scenes));stress_baseline=evaluate(current,[scenes])
    eligible=[]
    for region,keys in sorted(regions.items()):
        keys=sorted(keys)
        if not 2<=len(keys)<=4:continue
        types={(bykey[k].tp,bykey[k].cap) for k in keys}
        if len(types)!=1 or not any(can_expand(bykey[a],bykey[b]) for a in keys for b in keys if a!=b):continue
        if any(any(len(profile(current[k],bykey[k],s)['items'])>bykey[k].cap for s in range(scenes)) for k in keys):continue
        eligible.append((region,keys))
    ranked=sorted(range(scenes),key=lambda s:(baseline['values'][s],s));coarse=sorted(set([ranked[(scenes-1)//2],*ranked[-2:]]))
    accepted=baseline;checked=0;accepted_maps=0
    for iteration in range(passes):
        candidates=[];seen=set()
        # Round-robin across regions: global budget, never budget-per-region.
        generators=[(r,k,iter(permutations([current[s] for s in k]))) for r,k in eligible]
        active=list(generators);used=0
        while active and used<max_maps:
            remaining=[]
            for region,keys,generator in active:
                for permutation in generator:
                    mapping=dict(current);mapping.update(zip(keys,permutation));signature=digest(mapping)
                    if mapping==current or signature in seen:continue
                    seen.add(signature);used+=1;checked+=1
                    value=evaluate(mapping,coarse)
                    candidates.append((value['risk'],signature,mapping,region));break
                else:continue
                remaining.append((region,keys,generator))
                if used>=max_maps:break
            active=remaining
        if not candidates:break
        candidates.sort(key=lambda row:row[:2]);best=None
        # Only the leading coarse proposal receives complete scenarios+stress.
        _,signature,mapping,region=candidates[0];full=evaluate(mapping,range(scenes));stress=evaluate(mapping,[scenes])
        gain=accepted['risk']-full['risk'];threshold=margin_fraction*sum(accepted['values'])/scenes
        mean_ok=sum(full['values'])<=sum(accepted['values']);tail_ok=sum(sorted(full['values'])[-2:])<=sum(sorted(accepted['values'])[-2:])
        ok=gain>threshold and mean_ok and tail_ok and stress['values'][0]<=1.05*stress_baseline['values'][0]
        events.append(dict(iteration=iteration,region=region,checked_maps=used,signature=signature,predicted_gain_ns=gain,threshold_ns=threshold,adopted=ok))
        if not ok:break
        current=mapping;accepted=full;accepted_maps+=1
    result={s:list(queues[q]) for s,q in current.items()}
    if sorted(tuple(q) for q in result.values())!=sorted(tuple(q) for q in queues.values()):raise AssertionError('PS changed logical queues')
    return dict(execution_queues=result,logical_queue_at_slot=current,baseline=baseline,candidate=accepted,stress=stress_baseline,
        checked_maps=checked,accepted_maps=accepted_maps,events=events,host_wall_s=time.perf_counter()-started,
        forecast_digest=digest(forecast),scope='one wave of up to two disjoint whole-TP expansions; forecast ranking, native validation required',
        model_limits=['later sequential leases and further TP expansions are omitted in both PB and PS ranking','movement callbacks must declare costs; shared background traffic requires runtime recheck'],
        reason='PS_CANDIDATE_REQUIRES_NATIVE_VALIDATION' if accepted_maps else 'PB_RETAINED_NO_SUPPORTED_NET_POSITION_GAIN')
