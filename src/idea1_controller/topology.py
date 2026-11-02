"""Optimize commutative AR cycles without changing logical parameter ranks.

The fixed DP ReduceScatter/AllGather owner groups are never reordered. Native
ASTRA custom RingTopology consumes the supplied group order, so these choices
change actual routed traffic, not only placement metadata.
"""
from functools import lru_cache
from itertools import permutations


def hops(a,b,cols=6):return abs(a//cols-b//cols)+abs(a%cols-b%cols)

@lru_cache(None)
def ring_order(members):
    members=tuple(sorted(members))
    if len(members) not in (4,8):return members
    start=members[0];best=None;winner=None
    for rest in permutations(members[1:]):
        # Fix reflection symmetry; reversal has the same physical edge set.
        if rest[0]>rest[-1]:continue
        ring=(start,)+rest;edges=[hops(a,b) for a,b in zip(ring,ring[1:]+ring[:1])]
        score=(max(edges),sum(edges),ring)
        if best is None or score<best:best=score;winner=ring
    return winner


def optimize_ar_groups(plan,group_ids):
    audit=[]
    for gid in sorted(set(map(str,group_ids)),key=int):
        for op in getattr(plan,'operations',()):
            if op.kind=='collective' and str(op.metadata.get('comm_group'))==gid and op.collective!='ALL_REDUCE':
                raise ValueError('cannot reorder a noncommutative owner/shard collective group')
        old=tuple(plan.groups[gid]);new=ring_order(tuple(sorted(old)))
        assert set(new)==set(old) and len(new)==len(old)
        old_edges=[hops(a,b) for a,b in zip(old,old[1:]+old[:1])];new_edges=[hops(a,b) for a,b in zip(new,new[1:]+new[:1])]
        plan.groups[gid]=list(new)
        audit.append(dict(group=gid,logical_parameter_order=old,physical_ring_order=new,old_hop_sum=sum(old_edges),new_hop_sum=sum(new_edges),old_max_hop=max(old_edges),new_max_hop=max(new_edges),collective='commutative SUM AllReduce only'))
    return audit
