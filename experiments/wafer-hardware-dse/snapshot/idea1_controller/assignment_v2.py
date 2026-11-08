"""Quota assignment with a TP- and training-aware prospective finish score."""
from .placement import largest_remainder

def assign_groups(group_ids, forecasts, region_instances, instance_quotas, adaptive=True, trial_score=None):
    """Quota + long dispersion + within-region complementary queues.

    Instance quotas are frozen integers, so 192/27 needs no batch alteration.
    """
    regions=sorted(region_instances);instances=sorted(instance_quotas)
    if sum(instance_quotas.values())!=len(group_ids):raise ValueError('quota batch mismatch')
    if sorted(j for r in regions for j in region_instances[r])!=instances:raise ValueError('instances must have unique region')
    capacity={r:sum(instance_quotas[j] for j in region_instances[r]) for r in regions}
    queues={j:[] for j in instances};load={j:0.0 for j in instances};owner={j:r for r in regions for j in region_instances[r]}
    def room(r):return sum(len(queues[j]) for j in region_instances[r])<capacity[r]
    def push(g,r,long=False,short=False):
        js=[j for j in region_instances[r] if len(queues[j])<instance_quotas[j]]
        if long:
            # Pack long groups into one origin; other origins can clear early.
            j=min(js,key=lambda j:(not any(forecasts[x].label=='proxy-long' for x in queues[j]),j))
        else:j=min(js,key=lambda j:(any(forecasts[x].label=='proxy-long' for x in queues[j]) if short else False,load[j],j))
        queues[j].append(g);load[j]+=forecasts[g].score
    if not adaptive:
        cursor=0
        for j in instances:queues[j]=list(group_ids[cursor:cursor+instance_quotas[j]]);cursor+=instance_quotas[j]
        return queues, {r:0 for r in regions}
    long=sorted((g for g in group_ids if forecasts[g].label=='proxy-long'),key=lambda g:(-forecasts[g].score,group_ids.index(g)))
    short=sorted((g for g in group_ids if forecasts[g].label=='proxy-short'),key=lambda g:(forecasts[g].score,group_ids.index(g)))
    quotas=dict(zip(regions,largest_remainder(len(long),[capacity[r] for r in regions])))
    left=quotas.copy()
    def estimate(r,g):
        if trial_score is not None:return trial_score(r,queues,g)
        # Coarse queue work + total training work; actual costs score finalists.
        return max(load[j] for j in region_instances[r])+sum(load[j] for j in region_instances[r])/max(1,len(region_instances[r]))+forecasts[g].score
    for g in long:
        r=min((r for r in regions if left[r]),key=lambda r:(estimate(r,g),r));push(g,r,long=True);left[r]-=1
    reserved=[r for r in regions if quotas[r] and len(region_instances[r])>1 and any(not any(forecasts[g].label=='proxy-long' for g in queues[j]) and len(queues[j])<instance_quotas[j] for j in region_instances[r])]
    reserved.sort(key=lambda r:(-max(load[j] for j in region_instances[r]),r))
    for r in reserved:
        if short:g=short.pop(0);push(g,r,short=True)
    sq=dict(zip(regions,largest_remainder(len(short),[capacity[r]-sum(len(queues[j]) for j in region_instances[r]) for r in regions]))) if short else {r:0 for r in regions}
    for g in short:
        r=min((r for r in regions if sq[r]),key=lambda r:(estimate(r,g),r));push(g,r,short=True);sq[r]-=1
    rest=sorted((g for g in group_ids if g not in long and forecasts[g].label!='proxy-short'),key=lambda g:(-forecasts[g].score,group_ids.index(g)))
    for g in rest:
        r=min((r for r in regions if room(r)),key=lambda r:(estimate(r,g),r));push(g,r)
    index={g:i for i,g in enumerate(group_ids)}
    for j in queues:queues[j].sort(key=index.get)
    assert all(len(queues[j])==instance_quotas[j] for j in instances)
    assert sorted(g for q in queues.values() for g in q)==sorted(group_ids)
    assert all(sum(forecasts[g].label=='proxy-long' for j in region_instances[r] for g in queues[j])==quotas[r] for r in regions)
    return queues,quotas
