#!/usr/bin/env python3
"""Verify portable final evidence without a simulator or tensor runtime."""
import gzip
import hashlib
import json
import math
import tarfile
import tempfile
from pathlib import Path

from audit_idea1_system import audit
from audit_idea1_layered import audit as audit_legacy

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'experiments/2026-11-03-idea1-complete'
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def extract(path,dst,identities):
    with tarfile.open(path,'r:gz') as tar:
        members=tar.getmembers()
        assert set(x.name for x in members)==set(identities)
        for x in members:
            assert x.isfile() and not Path(x.name).is_absolute() and '..' not in Path(x.name).parts
            content=tar.extractfile(x).read()
            assert hashlib.sha256(content).hexdigest()==identities[x.name]['published_sha256']
            if 'source-snapshot' in Path(x.name).parts:
                assert identities[x.name]['original_sha256']==identities[x.name]['published_sha256']
            target=dst/x.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)

def main():
    manifest=load(DATA/'publication-manifest.json')['files']
    actual={str(p.relative_to(DATA)) for p in DATA.rglob('*') if p.is_file() and p.name!='publication-manifest.json'}
    assert actual==set(manifest)
    for name,digest in manifest.items():assert sha(DATA/name)==digest,name
    values=load(DATA/'all-data.json');rows=values['comparison'];assert len(rows)==6
    assert len(values['die_snapshots'])==180 and len(values['per_die'])==9720
    for r in values['die_snapshots']:
        assert r['total_dies']==54 and r['training_helper_subset_dies']<=r['training_dies']
        assert sum(r[k] for k in ('own_dies','generation_helper_dies','prepare_dies','training_dies','transfer_dies','prefix_dies','update_dies','idle_dies'))==54
    identities=load(DATA/'historical-identities.json')
    locked=load(DATA/'locked-profiles.json')['profiles']
    import sys
    sys.path.insert(0,str(ROOT/'src'))
    from idea1_controller.history import validate_history
    for model in ('4B','8B','14B'):
        inputs=ROOT/'experiments/2026-10-29-qwen3/inputs'/('Qwen3-'+model)
        history=json.loads(gzip.decompress((DATA/f'assets/history-{model}.json.gz').read_bytes()))
        prompts=load(inputs/'prompt_lengths.json')
        public=[dict(group_id=g,prompt_length=p['prompt_length'],prompt_sha256=p['rendered_sha256'],library=p['library']) for g,p in prompts.items()]
        validate_history(public,history,model,sha(inputs/'config.json'))
        excluded={}
        for p in inputs.glob('*.json'):
            for t in load(p).get('trajectories',[]):excluded.setdefault(str(t['group_id']),set()).add(t['sample'])
        for g,rounds in history['records'].items():
            samples=[s['sample'] for r in rounds for s in r['samples']]
            assert len(samples)==len(set(samples)) and not set(samples)&excluded.get(g,set())
        assert not history['chronological']
    with tempfile.TemporaryDirectory() as temp:
        tmp=Path(temp)
        for row in rows:
            model,window=row['model'],row['window'];key=model+'-'+window;dst=tmp/key
            extract(DATA/'evidence'/(key+'.tar.gz'),dst,identities[key])
            report=audit(dst/'native-run');result=load(dst/'native-run/result.json')
            assert report['status']=='PASS_SIMULATION_EVIDENCE_AUDIT'
            assert abs(result['total_s']-row['total_s'])<1e-9
            assert abs(row['device_s']+row['planning_proxy_s']-row['total_s'])<1e-9
            assert math.isclose(row['speedup_vs_gpu'],row['gpu_s']/row['total_s'])
            assert math.isclose(row['reduction_vs_previous'],1-row['total_s']/row['previous_innovation3_s'])
            plan=load(dst/'native-run/resource-layout.json')
            assert plan['region_members']==locked[model]['region_members']
            assert row['rollout_cap']==locked[model]['rollout_cap']
        for model in ('4B','8B','14B'):
            key='old-innovation3-'+model+'-test2';dst=tmp/key
            extract(DATA/'baselines'/f'innovation3-{model}-test2.tar.gz',dst,identities[key])
            audit_legacy(dst)
            row=next(r for r in rows if r['model']==model and r['window']=='test2')
            assert abs(load(dst/'result.json')['total_s']-row['previous_innovation3_s'])<1e-9
            gpu=load(DATA/'baselines'/f'GPU-{model}-test2.json')
            assert abs(gpu['result']['step_model_ns']/1e9-row['gpu_s'])<1e-9
    print(json.dumps(dict(status='PASS_COMPLETE_SYSTEM_PUBLIC_EVIDENCE',final_native_runs=6,legacy_test2_runs=3,
        files=len(manifest),samples_per_run=1536,die_snapshots=180,physical_or_learning_claim=False)))

if __name__=='__main__':main()
