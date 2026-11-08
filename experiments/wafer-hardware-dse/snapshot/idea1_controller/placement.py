"""Quota-conserving physical placement and history-aware whole-group assignment."""
from dataclasses import dataclass
from functools import lru_cache
import math


def largest_remainder(total,weights):
    if total<0 or not weights or sum(weights)<=0:raise ValueError('invalid quota')
    s=sum(weights);q=[total*w//s for w in weights]
    for i in sorted(range(len(weights)),key=lambda i:(-(total*weights[i]%s),i))[:total-sum(q)]:q[i]+=1
    assert sum(q)==total
    return q


@dataclass(frozen=True,order=True)
class Rectangle:
    row: int
    col: int
    height: int
    width: int
    def dies(self,rollout_tp=2,cols=6):
        # H0 native mesh uses 9 rows x 6 columns, row-major die IDs.
        # Reference 6x9 coordinates are transposed, preserving adjacency.
        return tuple((self.row+u)*cols+(self.col+z)*rollout_tp+h
                     for h in range(rollout_tp) for u in range(self.height) for z in range(self.width))
    def tp(self,r):return self.height*self.width*r


def guillotine_layouts(rollout_tp,supported_tp=(1,2,4,8),rows=9,cols=6,beam=2):
    """Exact reachable DP counts; at most beam representative layouts per count.

    Every die covered once. No silent truncation when TP does not divide 54.
    Backend rejects unavailable TP6 instead of pretending it is uniform TP.
    """
    if cols%rollout_tp:raise ValueError('rollout TP must tile native physical columns')
    atoms=cols//rollout_tp
    @lru_cache(None)
    def solve(r,c,h,w):
        out={}
        rect=Rectangle(r,c,h,w)
        if rect.tp(rollout_tp) in supported_tp:out[1]=[(rect,)]
        splits=[(solve(r,c,k,w),solve(r+k,c,h-k,w)) for k in range(1,h)]
        splits += [(solve(r,c,h,k),solve(r,c+k,h,w-k)) for k in range(1,w)]
        for a,b in splits:
            for da,la in a.items():
                for db,lb in b.items():
                    key=da+db;all_=set(out.get(key,()))
                    for x in la:
                        for y in lb:all_.add(tuple(sorted(x+y)))
                    # Geometry is tie-break/screen only, never performance truth.
                    out[key]=sorted(all_,key=lambda ls:(sum(t.height*t.width*(t.height+t.width) for t in ls),ls))[:beam]
        return out
    return solve(0,0,rows,atoms)


def assign_groups(group_ids, forecasts, region_instances, instance_quotas, adaptive=True):
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
