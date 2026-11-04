"""Independent predictor provenance, candidate identity and native binding.

Recorded candidate scores are ranked again; the cost search itself is not
re-executed by this audit. It does not turn an offline proxy into RL history.
"""
import gzip,json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from idea1_controller.workload import WorkloadSpec,Observation,DistributionSnapshot,forecasts,digest
from idea1_controller.controlled_forecast import controlled_forecasts
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def audit_calibration_source(offline,public,spec,model,raw):
    """Rebuild independent calibration from original identities and source rows."""
    if raw is None:return
    inputs=ROOT/f'experiments/2026-10-29-qwen3/inputs/Qwen3-{model}'
    canonical=load(inputs/'prompt_lengths.json')
    def ids(window):return {str(x['group_id']) for x in load(inputs/(window+'.json'))['trajectories']}
    def hashes(gs):return {canonical[g]['rendered_sha256'] for g in gs}
    tune_hashes=hashes(ids('tune'));reserved=ids('test3')|ids('test4')
    reserved={g for g in reserved if canonical[g]['rendered_sha256'] not in tune_hashes}
    target={g['prompt_sha256'] for g in public}
    allowed={g for g in ids('test1')|ids('test2') if canonical[g]['rendered_sha256'] not in tune_hashes|hashes(reserved)|target}
    split=load(Path(offline)/'split-manifest.json')
    assert set(split['calibration_prompt_hashes'])==hashes(allowed),'calibration prompt split differs from canonical source'
    source=ROOT/f'experiments/2026-11-03-idea1-elastic/assets/history-{model}.json.gz'
    history=json.loads(gzip.decompress(source.read_bytes()))
    from idea1_controller.history import validate_history
    used=[dict(group_id=g,prompt_length=canonical[g]['prompt_length'],prompt_sha256=canonical[g]['rendered_sha256'],library=canonical[g]['library'])
        for g in sorted(allowed) if history['records'].get(g)]
    validate_history(used,history,model,sha(inputs/'config.json'))
    rows=[]
    for g in sorted(allowed):
        records=history['records'].get(g,[])
        if not records:continue
        binding=history['task_bindings'][g]
        for record in records:
            samples=record['samples']
            if len(samples)!=spec.k or any(x['finish_reason'] not in ('stop','length') for x in samples):continue
            rows.append(Observation(binding['prompt_sha256'],binding['prompt_length'],binding['library'],
                digest([history['source_sha256'],g,[x['sample'] for x in samples]]),
                tuple(x['response_length'] for x in samples),tuple(x['finish_reason']=='length' for x in samples)))
    expected=DistributionSnapshot.build(rows,spec,kind='offline-calibration-proxy',source_sha256=sha(source))
    assert digest(expected.to_dict())==digest(raw),'calibration snapshot cannot be reconstructed from declared permitted source samples'


def audit_planning(native,offline):
    native=Path(native);offline=Path(offline);command=load(native/'command.json');model=command['model']
    assert model in ('4B','8B','14B')
    inputs=ROOT/f'experiments/2026-10-29-qwen3/inputs/Qwen3-{model}'
    result=load(native/'result.json');trace=load(inputs/(result['window']+'.json'))['trajectories']
    if command['subset_prompts']:trace=trace[:command['subset_prompts']*8]
    public=load(offline/'public.json');split=load(offline/'split-manifest.json');spec=WorkloadSpec(**split['spec'])
    prompts=load(inputs/'prompt_lengths.json');gids={str(x['group_id']) for x in trace}
    assert {str(g['group_id']) for g in public}==gids
    assert all(set(g)=={'group_id','prompt_length','prompt_sha256','library'} for g in public)
    for g in public:
        canonical=prompts[str(g['group_id'])]
        assert (g['prompt_length'],g['prompt_sha256'],g['library'])==(canonical['prompt_length'],canonical['rendered_sha256'],canonical['library'])
    summary_mode=split.get('forecast_summary','legacy')
    global_raw=load(offline/'global-distribution.json')
    audit_calibration_source(offline,public,spec,model,global_raw)
    global_snapshot=DistributionSnapshot.from_dict(global_raw,spec) if global_raw else None
    snapshot=load(offline/'distribution.json');saved=load(offline/'forecast.json')
    if snapshot and snapshot['kind']=='offline-controlled-same-prompt-proxy':
        hp=ROOT/f'experiments/2026-11-03-idea1-elastic/assets/history-{model}.json.gz'
        history=json.loads(gzip.decompress(hp.read_bytes()))
        recomputed,pred=controlled_forecasts(public,history,global_snapshot,spec,model=model,
            config_sha256=sha(inputs/'config.json'),source_file_sha256=sha(hp),
            allow_controlled_proxy=True,exact_weight=snapshot['exact_weight'],summary_mode=summary_mode)
        assert recomputed.to_dict()==snapshot and snapshot['chronological'] is False
        used={g:{s['sample'] for row in history['records'].get(g,[]) for s in row['samples']} for g in gids}
        assert all(x['sample'] not in used[str(x['group_id'])] for x in trace)
        version=recomputed.version;information='OFFLINE_CONTROLLED_SAME_PROMPT_PROXY'
    else:
        obj=DistributionSnapshot.from_dict(snapshot,spec) if snapshot else None
        pred=forecasts(public,obj,spec,summary_mode=summary_mode);version=obj.version if obj else None
        information='INDEPENDENT_CALIBRATION' if obj else 'HARD_LIMIT_FALLBACK'
    assert digest(pred)==digest(saved),'forecast cannot be reconstructed from declared independent inputs'
    artifact=load(offline/'offline-candidates.json')
    assert digest({k:v for k,v in artifact.items() if k!='artifact_digest'})==artifact['artifact_digest']
    assert artifact['public_digest']==digest(public) and artifact['distribution_version']==version
    assert artifact.get('forecast_summary','legacy')==summary_mode
    catalog=ROOT/f'experiments/2026-11-03-idea1-elastic/assets/inference-catalog-{model}.json'
    assert artifact['catalog_sha256']==sha(catalog)
    selected=artifact['result'];winner=selected['winner'];mode=selected['mode']
    stage={'one_scene':'coarse','full':'full','shortlist6':'fine','shortlist12':'fine',
        'capacity_fallback':'capacity-only-fallback-diagnostic'}[mode]
    candidates=[c for c in selected['events'] if c['passed'] and c['stage']==stage]
    assert candidates and min(candidates,key=lambda c:(c['risk_s'],c['signature']))['signature']==winner['signature']
    constraint=selected.get('candidate_constraint')
    if constraint is not None:
        assert constraint and len(set(constraint))==len(constraint)
        assert all(c['inf_name']+':'+c['training_name'] in constraint for c in selected['events'])
        assert winner['inf_name']+':'+winner['training_name'] in constraint
    home=selected['training'];native_home=load(native/'resource-layout.json');inf=selected['inference'];native_inf=load(native/'inference-layout.json')
    assert native_home['workload_offline_artifact']==artifact['artifact_digest']
    assert native_home['workload_distribution_version']==version
    assert native_home.get('workload_forecast_summary','legacy')==summary_mode
    assert native_home['workload_forecast_digest']==digest(saved)
    assert digest(native_home['forecasts'])==digest(saved)
    for name in ('queues','region_members'):assert digest(home[name])==digest(native_home[name])
    for name in ('groups','caps','paired_tp1'):assert digest(inf[name])==digest(native_inf[name])
    mapping=native_inf.get('logical_queue_at_slot',{k:k for k in inf['groups']})
    assert set(mapping)==set(inf['groups'])==set(mapping.values())
    region={w:r for r,ws in home['region_members'].items() for w in ws}
    for slot,queue in mapping.items():
        assert native_inf['execution_queues'][slot]==inf['execution_queues'][queue]
        if slot!=queue:
            a,b=inf['groups'][slot],inf['groups'][queue]
            assert len(a)==len(b) and inf['caps'][slot]==inf['caps'][queue]
            assert len({region[w] for w in a+b})==1,'unsupported cross-region PS permutation'
    return dict(status='PASS_PREDICTION_AND_DEPLOYMENT_BINDING',model=model,window=result['window'],information=information,
        groups=len(public),k=spec.k,scenarios=spec.scenarios,forecast_recomputed=True,evaluation_samples_excluded=True,
        recorded_score_ranking=True,native_assignment_bound=True,
        scope='declared predictor provenance and recorded finite search binding; no independent replay of every candidate, global optimum, numerical learning or chronology claim')
