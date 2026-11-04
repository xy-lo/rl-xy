"""Auditable workload distributions. No current response/EOS is an input.

An offline calibration split is deliberately not a chronological RL history.
The service catalogue is a separate object from this immutable snapshot.
"""
from dataclasses import dataclass
import hashlib, json, math, random, statistics


def digest(value):
    return hashlib.sha256(json.dumps(json.loads(json.dumps(value)), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class WorkloadSpec:
    contract: str
    k: int = 8
    scenarios: int = 16
    prompt_max: int = 16384
    output_max: int = 8192
    min_groups: int = 2
    bucket_width: int = 1024
    seed: int = 162

    def __post_init__(self):
        if not self.contract or min(self.k, self.scenarios, self.prompt_max, self.output_max, self.min_groups, self.bucket_width) < 1:
            raise ValueError("invalid workload contract")


@dataclass(frozen=True)
class Observation:
    prompt_hash: str
    prompt_length: int
    category: str
    source_id: str
    lengths: tuple
    truncated: tuple
    observed_seq: object = None


@dataclass(frozen=True)
class DistributionSnapshot:
    version: str
    contract: str
    kind: str
    records: tuple
    source_sha256: str

    @classmethod
    def build(cls, records, spec, *, kind, source_sha256):
        if kind not in ("offline-calibration-proxy", "committed-history"):
            raise ValueError("unknown provenance")
        records = tuple(records)
        if len({r.source_id for r in records}) != len(records):
            raise ValueError("duplicate source identity")
        for r in records:
            if not r.prompt_hash or not r.source_id or not 0 < r.prompt_length <= spec.prompt_max:
                raise ValueError("invalid identity or prompt limit")
            if len(r.lengths) != spec.k or len(r.truncated) != spec.k:
                raise ValueError("incomplete K-vector")
            if any(type(x) is not int or not 0 < x <= spec.output_max for x in r.lengths):
                raise ValueError("invalid output length")
            if any(type(x) is not bool for x in r.truncated):
                raise ValueError("missing truncation status")
            if kind == "committed-history" and (type(r.observed_seq) is not int or r.observed_seq < 0):
                raise ValueError("missing real commit sequence")
            if kind == "offline-calibration-proxy" and r.observed_seq is not None:
                raise ValueError("offline records cannot invent chronological sequence")
        payload = dict(contract=spec.contract, kind=kind, source_sha256=source_sha256,
                       records=[r.__dict__ for r in records])
        return cls(digest(payload), spec.contract, kind, records, source_sha256)

    def to_dict(self):
        return dict(version=self.version, contract=self.contract, kind=self.kind,
                    source_sha256=self.source_sha256, records=[r.__dict__ for r in self.records])

    @classmethod
    def from_dict(cls, data, spec):
        if data['contract'] != spec.contract:
            raise ValueError("snapshot contract mismatch")
        rows = [Observation(**dict(r, lengths=tuple(r['lengths']), truncated=tuple(r['truncated']))) for r in data['records']]
        value = cls.build(rows, spec, kind=data['kind'], source_sha256=data['source_sha256'])
        if value.version != data['version']:
            raise ValueError("snapshot digest mismatch")
        return value


def forecasts(public, snapshot, spec, *, decision_seq=None, exact=False, summary_mode='pooled'):
    if summary_mode not in ('pooled','legacy'):raise ValueError('unknown forecast summary contract')
    allowed = {'group_id', 'prompt_length', 'prompt_sha256', 'library'}
    if not public or any(set(g) != allowed for g in public):
        raise ValueError("public fields only; batch must be nonempty")
    ids = [str(g['group_id']) for g in public]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate group")
    if any(type(g['prompt_length']) is not int or not 0 < g['prompt_length'] <= spec.prompt_max for g in public):
        raise ValueError("public prompt exceeds hard limit")
    if snapshot and snapshot.contract != spec.contract:
        raise ValueError("model/template/generation contract mismatch")
    if exact and (snapshot is None or snapshot.kind != 'committed-history'):
        raise ValueError("exact enhancement requires real committed history")
    rows = snapshot.records if snapshot else ()
    if snapshot and snapshot.kind == 'committed-history':
        if type(decision_seq) is not int or any(r.observed_seq >= decision_seq for r in rows):
            raise ValueError("future or uncommitted observation")
    target_hashes = {g['prompt_sha256'] for g in public}
    if snapshot and snapshot.kind == 'offline-calibration-proxy' and any(r.prompt_hash in target_hashes for r in rows):
        raise ValueError("calibration and evaluation prompt overlap")
    result = {}; summary_cache = {}
    for g in public:
        ph = g['prompt_sha256']
        # Repeated tasks are not included in the independent fallback pool.
        global_pool = [r for r in rows if r.prompt_hash not in target_hashes]
        class_pool = [r for r in global_pool if r.category == g['library'] and
                      (r.prompt_length - 1)//spec.bucket_width == (g['prompt_length'] - 1)//spec.bucket_width]
        exact_pool = [r for r in rows if r.prompt_hash == ph and r.prompt_length == g['prompt_length']]
        pool, evidence = [], 'DEFAULT'
        if exact and len(exact_pool) >= 2:
            # Retain a global component to avoid treating repeated prompts as drift-free.
            pool, evidence = exact_pool + global_pool, 'EXACT_PLUS_GLOBAL'
        elif len({r.prompt_hash for r in class_pool}) >= spec.min_groups:
            pool, evidence = class_pool, 'CLASS'
        elif len({r.prompt_hash for r in global_pool}) >= spec.min_groups:
            pool, evidence = global_pool, 'GLOBAL'
        key = digest([spec.__dict__, snapshot.version if snapshot else None, str(g['group_id']), ph])
        rng = random.Random(key)
        vectors = tuple(pool[rng.randrange(len(pool))].lengths for _ in range(spec.scenarios)) if pool else ((spec.output_max,)*spec.k,)*spec.scenarios
        if summary_mode=='legacy':
            work=sorted(sum(v)+max(v) for v in vectors)
        else:
            # Task-specific samples are independent scenario draws, NOT
            # evidence that otherwise identical unknown tasks differ in mean.
            pool_key=(evidence,ph,g['prompt_length']) if evidence=='EXACT_PLUS_GLOBAL' else (evidence,g['library'],(g['prompt_length']-1)//spec.bucket_width) if evidence=='CLASS' else (evidence,)
            if pool_key not in summary_cache:
                summary_cache[pool_key]=sorted(sum(r.lengths)+max(r.lengths) for r in pool) if pool else [spec.output_max*(spec.k+1)]
            work=summary_cache[pool_key]
        q = lambda x: work[max(0, math.ceil(x*len(work))-1)]
        result[str(g['group_id'])] = dict(group_id=str(g['group_id']), prompt_length=g['prompt_length'],
            response_scenarios=vectors, score=statistics.median(work)+.25*(q(.9)-q(.5)),
            q25=q(.25), q75=q(.75), label='unknown', evidence_kind=evidence,
            independent_groups=len({r.prompt_hash for r in pool}),
            truncated_support=any(any(r.truncated) for r in pool),
            forecast_version=key if summary_mode=='legacy' else digest([key,'pooled-point-summary/2']), source_snapshot=snapshot.version if snapshot else None)
        if summary_mode=='pooled':result[str(g['group_id'])]['score_basis']='pooled conditional distribution; task-specific draws only for scenarios'
    scores = sorted(f['score'] for f in result.values())
    lo, hi = scores[max(0, math.ceil(len(scores)/3)-1)], scores[max(0, math.ceil(2*len(scores)/3)-1)]
    for f in result.values():
        if f['evidence_kind'] != 'DEFAULT':
            f['label'] = 'proxy-long' if f['q25'] > hi else 'proxy-short' if f['q75'] < lo else 'ordinary'
    return result
