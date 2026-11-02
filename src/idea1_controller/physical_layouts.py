"""Guillotine coverage screened in physical die geometry, not atom aspect."""
from functools import lru_cache
from .placement import Rectangle

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
                    out[key]=sorted(all_,key=lambda ls:(sum(t.height*t.width*(t.height+rollout_tp*t.width) for t in ls),ls))[:beam]
        return out
    return solve(0,0,rows,atoms)

