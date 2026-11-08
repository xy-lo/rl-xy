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


def rectangular_candidates(rows, cols, supported_tp=(1, 2, 4, 8), beam=2):
    """Enumerate connected rectangular die-group layouts on an arbitrary mesh.

    Each rectangle's area is a supported TP and the full mesh is covered.
    This is a bounded geometry candidate generator, not a capacity proof or a
    native-compatibility assertion. In particular, it can return TP1 groups
    on odd-sized meshes that the legacy TP2-only training path cannot run.
    """
    from .mesh_geometry import MeshGeometry
    mesh=MeshGeometry(cols,rows)
    if not supported_tp or any(type(tp) is not int or tp <= 0 for tp in supported_tp):
        raise ValueError('supported TP values must be positive integers')
    if type(beam) is not int or beam <= 0:
        raise ValueError('beam must be a positive integer')
    layouts=guillotine_layouts(1,tuple(sorted(set(supported_tp))),rows,cols,beam)
    result={}
    for group_count,variants in layouts.items():
        groups=[]
        for variant in variants:
            physical=tuple(tuple(sorted(rect.dies(1,cols))) for rect in variant)
            mesh.validate_partition(physical)
            groups.append(physical)
        result[group_count]=tuple(groups)
    return result


def paired_region_candidates(pair_mapping, beam=4):
    """Bounded connected TP2/4/8 region candidates over actual owner pairs.

    Greedy merge trajectories deliberately form a finite candidate set, not
    an exhaustive optimum. Include all-singleton baseline and preserve every
    paired owner exactly once. A possible odd physical die is reported by
    pair_mapping.unpaired and is not silently treated as a TP2 owner.
    """
    if type(beam) is not int or beam<=0:raise ValueError('beam must be positive')
    n=len(pair_mapping.pairs)
    if n==0:raise ValueError('no paired owners')
    def connected(left,right):
        return any(pair_mapping.adjacent_workers(a,b) for a in left for b in right)
    def geometry(group):
        points=[pair_mapping.mesh.coordinate(d) for w in group for d in pair_mapping.pairs[w]]
        ys=[p[0] for p in points];xs=[p[1] for p in points]
        return min(ys),min(xs),max(ys)-min(ys)+1,max(xs)-min(xs)+1
    candidates={}
    for ordering in range(4):
        regions=[(w,) for w in range(n)]
        while True:
            normal=tuple(sorted(tuple(sorted(g)) for g in regions))
            candidates.setdefault(len(normal),set()).add(normal)
            choices=[]
            for i,a in enumerate(regions):
                for j in range(i+1,len(regions)):
                    b=regions[j]
                    if len(a)!=len(b) or len(a)+len(b)>4 or not connected(a,b):continue
                    union=tuple(sorted(a+b));y,x,h,w=geometry(union)
                    # Vary deterministic spatial priorities at equal sizes.
                    keys=((h*w,h+w,y,x),(h*w,h+w,x,y),(-y,x,h*w,h+w),(-x,y,h*w,h+w))
                    choices.append((len(union),keys[ordering],union,i,j))
            if not choices:break
            _,_,union,i,j=min(choices)
            regions=[g for index,g in enumerate(regions) if index not in (i,j)]+[union]
    output={}
    for count,variants in candidates.items():
        selected=sorted(variants)[:beam]
        output[count]=tuple({min(group):tuple(group) for group in groups} for groups in selected)
    return output
