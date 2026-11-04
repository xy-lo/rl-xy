"""Die-role snapshots reconstructed from immutable native runtime events.

Roles are scheduler occupancy, not FLOP utilization. Training-helper dies are
a subset of training, never added again to the 54-die conservation sum.
"""
from bisect import bisect_right
from collections import Counter
ROLE_MAP={'own':'own','helper':'generation_helper','prepare':'prepare',
          'training':'training','sample_transfer':'transfer','idle':'idle'}
CATEGORIES=('own','generation_helper','prepare','training','transfer','prefix','update','idle')

def snapshot(rows,base,executions,time_s):
    tick=round(float(time_s)*1e12)
    origin=round(base['dynamic_origin_s']*1e12);end=round(base['dynamic_end_s']*1e12)
    total=round(base['total_s']*1e12)
    if tick<0 or tick>=total:raise ValueError('sample within completed step interval required')
    if tick<origin:
        phase='prefix';roles={str(w):'prefix' for w in range(27)};stages={}
    elif tick>=end:
        phase='update';roles={str(w):'update' for w in range(27)};stages={}
    else:
        phase='runtime';index=bisect_right([r['tick'] for r in rows],tick)-1
        if index<0:raise ValueError('no observed runtime state at this timestamp')
        row=rows[index];raw_roles=row['roles'];raw_stages=row['stages']
        pairs=raw_roles.items() if isinstance(raw_roles,dict) else enumerate(raw_roles)
        roles={str(w):ROLE_MAP[v] for w,v in pairs}
        stage_pairs=raw_stages.items() if isinstance(raw_stages,dict) else enumerate(raw_stages)
        stages={str(w):v for w,v in stage_pairs}
    if set(map(int,roles))!=set(range(27)):raise ValueError('incomplete physical pair roles')
    helpers={d for e in executions if e.get('remote') and e['start_tick']<=tick<e['end_tick'] for d in e['physical_ranks']}
    counts=Counter();dies=[]
    for die in range(54):
        role=roles[str(die//2)];counts[role]+=1
        if die in helpers and role!='training':raise ValueError('training-helper interval disagrees with native role')
        dies.append(dict(die=die,row=die//6,col=die%6,worker=die//2,role=role,
            training_helper=die in helpers,stage=stages.get(str(die//2),phase)))
    values={key:counts[key] for key in CATEGORIES}
    if sum(values.values())!=54:raise ValueError('physical die conservation failed')
    return dict(time_s=tick/1e12,phase=phase,counts=values,training_helper_subset_dies=len(helpers),
        total_dies=54,per_die=dies,
        scope='last observed scheduler role; each physical TP2 pair exposes one role; no claim of hardware FLOP utilization')

def default_times(base):
    total=base['total_s'];points={0.,total-1e-6}
    points.update(total*f for f in (.1,.25,.5,.75,.9))
    points.update(float(t) for t in range(10,int(total)+1,10))
    points.update(base[k] for k in ('generation_end_s','first_training_s','dynamic_end_s'))
    return sorted(t for t in points if 0<=t<total)
