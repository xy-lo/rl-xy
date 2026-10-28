"""Small numeric oracle for the pinned GRPO objective, not model training.

Written from the equations; the independent verification harness evaluates the
same fixtures with functions extracted from the pinned upstream source. All
inputs here are explicit scalars; symbolic performance traces cannot invent
rewards, log probabilities, gradients, or learning quality through this module.
"""
import math
from collections import defaultdict


def group_advantages(rewards, group_ids, *, normalize_std=True, epsilon=1e-6):
    if len(rewards)!=len(group_ids) or not rewards:raise ValueError('aligned nonempty reward/group arrays required')
    if epsilon<=0:raise ValueError('positive std stabilizer required')
    groups=defaultdict(list)
    for r,g in zip(rewards,group_ids):
        if not math.isfinite(r):raise ValueError('finite supplied rewards required')
        groups[g].append(float(r))
    stats={}
    for g,rs in groups.items():
        mean=sum(rs)/len(rs) if len(rs)>1 else 0.0
        std=math.sqrt(sum((r-mean)**2 for r in rs)/(len(rs)-1)) if len(rs)>1 else 1.0
        stats[g]=(mean,std)
    return [(float(r)-stats[g][0])/(stats[g][1]+epsilon) if normalize_std else float(r)-stats[g][0]
            for r,g in zip(rewards,group_ids)]


def _rows(values,name):
    if not values or not all(isinstance(row,(list,tuple)) and row for row in values):raise ValueError(name+' requires nonempty rows')
    if not all(math.isfinite(x) for row in values for x in row):raise ValueError(name+' contains nonfinite values')


def aggregate(values, mask, mode='token-mean', *, dp_size=1, global_tokens=None, global_sequences=None):
    _rows(values,'values');_rows(mask,'mask')
    if len(values)!=len(mask) or any(len(a)!=len(b) for a,b in zip(values,mask)):raise ValueError('shape mismatch')
    if type(dp_size) is not int or dp_size<1 or any(m not in (0,1) for row in mask for m in row):raise ValueError('invalid DP/mask')
    counts=[sum(row) for row in mask];sums=[sum(v*m for v,m in zip(row,ms)) for row,ms in zip(values,mask)]
    if mode=='token-mean':
        if global_tokens is None:
            if dp_size!=1:raise ValueError('global token denominator required for distributed token-mean')
            global_tokens=sum(counts)
        if global_tokens<=0:raise ValueError('nonempty global token set required')
        return sum(sums)/global_tokens*dp_size
    if mode=='seq-mean-token-mean':
        if global_sequences is None:
            if dp_size!=1:raise ValueError('global sequence denominator required')
            global_sequences=sum(n>0 for n in counts)
        if global_sequences<=0:raise ValueError('nonempty sequence set required')
        return sum(s/(n+1e-8) for s,n in zip(sums,counts) if n)/global_sequences*dp_size
    raise ValueError('only explicitly validated loss aggregation modes are supported')


def policy_objective(current, old, advantages, mask, *, reference=None, clip_low=.2, clip_high=.2,
                     dual_clip=3.0, kl_coefficient=.001, kl_type='low_var_kl', loss_aggregation='token-mean',
                     dp_size=1, global_tokens=None, global_sequences=None):
    for name,x in [('current',current),('old',old),('advantages',advantages),('mask',mask)]:_rows(x,name)
    shapes=[list(map(len,x)) for x in [current,old,advantages,mask]]
    if any(s!=shapes[0] for s in shapes) or any(len(x)!=len(current) for x in [old,advantages,mask]):raise ValueError('objective shape mismatch')
    if not (0<=clip_low<1 and clip_high>=0 and dual_clip>1 and kl_coefficient>=0):raise ValueError('invalid clipping/KL parameters')
    if kl_coefficient:
        _rows(reference,'reference')
        if len(reference)!=len(current) or list(map(len,reference))!=shapes[0]:raise ValueError('reference shape mismatch')
        if kl_type!='low_var_kl':raise ValueError('only pinned k3 estimator validated')
    pg=[];kl=[];clipped=[];approx=[]
    for i,(cur,prev,adv) in enumerate(zip(current,old,advantages)):
        a=[];k=[];c=[];d=[]
        for j,(lp,op,av) in enumerate(zip(cur,prev,adv)):
            delta=max(-20.0,min(20.0,lp-op));ratio=math.exp(delta)
            raw=-av*ratio;bounded=-av*max(1-clip_low,min(1+clip_high,ratio))
            value=max(raw,bounded)
            if av<0:value=min(value,-av*dual_clip)
            a.append(value);c.append(float(bounded>raw));d.append(-delta)
            if kl_coefficient:
                diff=max(-20.0,min(20.0,reference[i][j]-lp))
                k.append(max(-10.0,min(10.0,math.exp(diff)-diff-1)))
            else:k.append(0.0)
        pg.append(a);kl.append(k);clipped.append(c);approx.append(d)
    kw=dict(dp_size=dp_size,global_tokens=global_tokens,global_sequences=global_sequences)
    pg_value=aggregate(pg,mask,loss_aggregation,**kw);kl_value=aggregate(kl,mask,loss_aggregation,**kw)
    return {'loss':pg_value+kl_coefficient*kl_value,'policy_loss':pg_value,'kl_loss':kl_value,
            'clip_fraction':aggregate(clipped,mask),'approx_policy_kl':aggregate(approx,mask),
            'per_token_policy_loss':pg,'per_token_kl':kl}
