"""Explicit offline same-prompt proxy, separate from committed online history.

This retains the old experiment's information set for controlled comparisons,
while novel tasks still use independent calibration. No chronology is invented.
"""
from dataclasses import dataclass,asdict
import random,statistics,math
from .workload import digest,forecasts
from .history import validate_history,predict_groups,quantile


@dataclass(frozen=True)
class ControlledProxySnapshot:
    version: str
    contract: str
    source_snapshot_id: str
    global_snapshot_id: str
    source_file_sha256: str
    exact_weight: float
    summary_mode: str = 'legacy'
    kind: str = 'offline-controlled-same-prompt-proxy'

    def to_dict(self):
        value=asdict(self)
        if self.summary_mode=='legacy':value.pop('summary_mode')
        value.update(chronological=False,causality='offline calibration source available before planning; no online actor sequence asserted',
                     source_sample_policy='canonical binding and excluded sample IDs checked per exact prompt')
        return value


def controlled_forecasts(public,source,global_snapshot,spec,*,model,config_sha256,source_file_sha256,allow_controlled_proxy=False,exact_weight=.75,summary_mode='pooled'):
    if not allow_controlled_proxy:raise ValueError('controlled proxy requires explicit opt-in')
    if not math.isfinite(exact_weight) or not 0<=exact_weight<=1:raise ValueError('invalid mixture weight')
    if spec.scenarios!=16:raise ValueError('legacy exact proxy has 16 scenarios; other contracts not yet supported')
    if source.get('kind')!='offline-independent-same-prompt-proxy' or source.get('chronological') is not False:raise ValueError('not the declared offline proxy source')
    if 'excluded_sample_ids' not in source:raise ValueError('exact proxy lacks evaluation-sample exclusion proof')
    validate_history(public,source,model,config_sha256)
    global_predictions=forecasts(public,global_snapshot,spec,summary_mode=summary_mode)
    identity=[spec.__dict__,source['snapshot_id'],global_snapshot.version if global_snapshot else None,source_file_sha256,exact_weight,'controlled-proxy-not-online']
    if summary_mode!='legacy':identity.append('pooled-point-summary/2')
    snapshot_id=digest(identity)
    snapshot=ControlledProxySnapshot(snapshot_id,spec.contract,source['snapshot_id'],global_snapshot.version if global_snapshot else None,source_file_sha256,float(exact_weight),summary_mode=summary_mode)
    result={}
    for g in public:
        gid=str(g['group_id']);base=global_predictions[gid];rows=source['records'].get(gid,[])
        if exact_weight==0 or len(rows)<2:
            result[gid]=dict(base,source_snapshot=snapshot.version,controlled_proxy_exact=False);continue
        if gid not in source['excluded_sample_ids']:
            raise ValueError('exact prompt lacks evaluation-sample exclusion proof')
        seen=set();excluded=set(source['excluded_sample_ids'][gid])
        usable=[]
        for row in rows:
            samples=row['samples']
            if len(samples)!=spec.k or any(s.get('finish_reason') not in ('stop','length') for s in samples):continue
            for sample in samples:
                uid=sample['sample'];length=sample['response_length']
                if uid in excluded or uid in seen:raise ValueError('exact source overlaps evaluation or duplicates a sample')
                if type(length) is not int or not 0<length<=spec.output_max:raise ValueError('exact source violates output contract')
                seen.add(uid)
            usable.append(row)
        if len(usable)<2:
            result[gid]=dict(base,source_snapshot=snapshot.version,controlled_proxy_exact=False);continue
        # Importer identity and seed are preserved so weight=1 reproduces old
        # proxy vectors exactly; this is a controlled information-set comparison.
        if len(usable)!=len(rows):
            raise ValueError('incomplete exact source requires rebuilding its audited snapshot')
        exact=predict_groups([g],source,proxy_adaptation=False)[gid]
        rng=random.Random(digest([snapshot.version,gid,spec.seed]))
        vectors=tuple(exact.response_scenarios[s] if exact_weight==1 or rng.random()<exact_weight else base['response_scenarios'][s] for s in range(spec.scenarios))
        work=[sum(v)+max(v) for v in vectors]
        value=dict(base,response_scenarios=vectors,score=statistics.median(work)+.25*(quantile(work,.9)-quantile(work,.5)),q25=quantile(work,.25),q75=quantile(work,.75),
                   evidence_kind='OFFLINE_EXACT_PROXY_PLUS_GLOBAL',controlled_proxy_exact=True,chronological=False,exact_independent_prompts=1,exact_joint_vectors=len(usable),
                   source_snapshot=snapshot.version,forecast_version=digest([snapshot.version,gid,vectors]),exact_mixture_weight=exact_weight,
                   provenance='disjoint offline same-prompt draws; not chronological RL history')
        if summary_mode=='pooled':value['score_basis']='independent same-prompt proxy vectors plus explicitly sampled global mixture'
        if exact_weight==1:
            value['conditional_lengths']=tuple(s['response_length'] for row in usable for s in row['samples'])
            value['conditional_support_kind']='raw disjoint offline samples, K-correlated groups'
        result[gid]=value
    scores=[f['score'] for f in result.values()];lo,hi=quantile(scores,1/3),quantile(scores,2/3)
    for f in result.values():
        if f['evidence_kind']!='DEFAULT':f['label']='proxy-long' if f['q25']>hi else 'proxy-short' if f['q75']<lo else 'ordinary'
    return snapshot,result
