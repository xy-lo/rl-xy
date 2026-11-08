"""Integer remainder quotas: regions first, then their inference instances.

For 192 groups the ideal per-die quota is fractional. Rounding globally by
instance ID can accidentally assign an extra whole group to a singleton TP2
region. Largest remainder at both levels preserves area-proportional regional
quotas without dropping or duplicating any group.
"""
from .placement import largest_remainder

def layout_quotas(total,regions):
    roots=sorted(regions);regional=largest_remainder(total,[len(regions[r]) for r in roots]);result={}
    for r,q in zip(roots,regional):
        ws=sorted(regions[r]);result.update(zip(ws,largest_remainder(q,[1]*len(ws))))
    assert sum(result.values())==total
    return result
