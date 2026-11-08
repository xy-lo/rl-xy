"""Frozen same-task history, with explicit offline-proxy provenance.

No live response/EOS field is accepted by the predictor or assignment routines.
The trace importer separates history samples from every evaluation sample ID.
"""
from dataclasses import dataclass
from collections import defaultdict
import csv, gzip, hashlib, json, math, random, statistics
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def bind_history(history, inputs):
    """Bind an imported proxy to canonical task/model identities.

    Keep the previous explicit bootstrap seed so adding provenance does not
    change the statistical scenarios being compared in an ablation.
    """
    result=dict(history)
    result['scenario_seed']=history.get('scenario_seed',history['snapshot_id'])
    root=Path(inputs)
    prompts=json.loads((root/'prompt_lengths.json').read_text())
    result['model_config_sha256']=hashlib.sha256((root/'config.json').read_bytes()).hexdigest()
    result['unbound_group_ids']=sorted(set(history['records'])-set(prompts))
    result['records']={g:v for g,v in history['records'].items() if g in prompts}
    result['task_bindings']={g:dict(prompt_sha256=prompts[g]['rendered_sha256'],
        prompt_length=prompts[g]['prompt_length'],library=prompts[g]['library']) for g in result['records']}
    result.pop('snapshot_id',None)
    result['snapshot_id']=digest(result)
    return result


def validate_history(public_groups,history,model,config_sha256):
    """Fail closed for a mismatched snapshot; sparse valid tasks may fall back."""
    if history['model']!=model:raise ValueError('history model mismatch')
    raw={k:v for k,v in history.items() if k!='snapshot_id'}
    if digest(raw)!=history['snapshot_id']:raise ValueError('history snapshot digest mismatch')
    if history.get('model_config_sha256')!=config_sha256:raise ValueError('history model configuration mismatch')
    for g in public_groups:
        key=str(g['group_id']);binding=history.get('task_bindings',{}).get(key)
        if key not in history.get('records',{}):continue
        if not binding:raise ValueError('history lacks canonical task binding')
        if any(binding[k]!=g[k] for k in ('prompt_sha256','prompt_length','library')):
            raise ValueError('history canonical task identity mismatch')
    return True


def quantile(xs,q):
    xs=sorted(xs)
    if not xs: raise ValueError('empty quantile')
    return xs[max(0,math.ceil(q*len(xs))-1)]


@dataclass(frozen=True)
class GroupForecast:
    group_id: str
    prompt_length: int
    response_scenarios: tuple
    score: float
    q25: float
    q75: float
    label: str
    evidence_kind: str


def import_history(source, inputs, model, output):
    """Import eight disjoint K=8 pseudo-rounds; not chronological RL evidence."""
    excluded=defaultdict(set)
    for path in sorted(Path(inputs).glob('*.json')):
        value=json.loads(path.read_text())
        for x in value.get('trajectories',[]):excluded[str(x['group_id'])].add(int(x['sample']))
    source_rows=defaultdict(dict)
    with gzip.open(source,'rt') as stream:
        for x in csv.DictReader(stream):
            if x['model']!='Qwen3-'+model or 'ds1000-think-temp06' not in x['partition']:continue
            gid=str(x['group_id']);sample=int(x['sample'])
            if sample in excluded[gid]:continue
            if sample in source_rows[gid]:raise ValueError('duplicate history source identity')
            source_rows[gid][sample]=dict(sample=sample,response_length=int(x['response_length']),finish_reason=x['finish_reason'])
    records={}
    for gid,rows in sorted(source_rows.items()):
        samples=sorted(rows)
        random.Random('idea1-history-20261102:'+model+':'+gid).shuffle(samples)
        records[gid]=[dict(proxy_round=i,actor_update=None,samples=[rows[s] for s in samples[i*8:i*8+8]]) for i in range(min(8,len(samples)//8))]
        assert all(s['sample'] not in excluded[gid] for r in records[gid] for s in r['samples'])
    result=dict(schema='idea1.independent-history/1',model=model,kind='offline-independent-same-prompt-proxy',chronological=False,
                source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),excluded_sample_ids={g:sorted(v) for g,v in excluded.items()},
                assumptions=['independent answers are proxy appearances, not successful actor updates','no exact-task confidence across actor versions is asserted'],records=records)
    result['snapshot_id']=digest(result)
    result=bind_history(result,inputs)
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return result


def predict_groups(public_groups, history, *, proxy_adaptation=True):
    """public_groups has group identity and prompt length only; never live R."""
    forecasts={};scores=[]
    for g in public_groups:
        gid=str(g['group_id']);rows=history['records'].get(gid,[])
        if len(rows)<2:
            raise ValueError('history too sparse: explicit baseline fallback required')
        vectors=[tuple(s['response_length'] for s in r['samples']) for r in rows]
        # Stable bootstrap preserves each historical K-dimensional joint vector.
        rng=random.Random(history.get('scenario_seed',history['snapshot_id'])+':'+gid)
        scenarios=tuple(vectors[rng.randrange(len(vectors))] for _ in range(16))
        # Token service proxy only, replaced by operator service during full scoring.
        work=[sum(v)+max(v) for v in scenarios]
        med=statistics.median(work);u=med+0.25*(quantile(work,.9)-quantile(work,.5));scores.append(u)
        forecasts[gid]=GroupForecast(gid,int(g['prompt_length']),scenarios,u,quantile(work,.25),quantile(work,.75),'unknown',history['kind'])
    low,high=quantile(scores,1/3),quantile(scores,2/3)
    result={}
    for gid,f in forecasts.items():
        # Proxy labels are explicit experimental assumptions, not calibrated truth.
        label=('proxy-long' if f.q25>high else 'proxy-short' if f.q75<low else 'ordinary') if proxy_adaptation else 'unknown'
        result[gid]=GroupForecast(**(f.__dict__|{'label':label}))
    return result
