#!/usr/bin/env python3
"""Verify published compressed native evidence; never promote hardware accuracy."""
import gzip, hashlib, json, math, sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'experiments/2026-11-02-idea1-joint'
sys.path.insert(0,str(ROOT/'src'))
from rl_xy.astra_adapter.parser import EVENT_RE

load=lambda p:json.loads(p.read_text())
digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def raw(model,name):
    with gzip.open(DATA/model/'raw'/(name.replace('/','__')+'.gz'),'rt') as f:return f.read()

def main():
    manifest=load(DATA/'publication-manifest.json')
    actual={str(p.relative_to(DATA)) for p in DATA.rglob('*') if p.is_file() and p.name!='publication-manifest.json'}
    assert actual==set(manifest['files'])
    for name,expected in manifest['files'].items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==expected,name
    historical=load(DATA/'historical-identities.json'); table=load(DATA/'all-data.json')
    for model in ('4B','8B','14B'):
        folder=DATA/model;result=load(folder/'result.json');audit=load(folder/'audit.json')
        assert result['native_exit_status']==0 and result['requests']==result['completed']==result['trained_samples']==1536
        assert audit['status']=='PASS_EVIDENCE_AUDIT' and max(load(folder/'capacity.json')['phase_peak_bytes'].values())<=64000000000
        row=next(x for x in table['comparison'] if x['model']==model)
        assert row['idea1_joint_s']==result['total_s']==audit['total_s']
        for name,entry in historical[model].items():
            with gzip.open(DATA/entry['path'],'rb') as f:value=f.read()
            assert hashlib.sha256(value).hexdigest()==entry['published_uncompressed_sha256']
        trace=load(ROOT/'experiments/2026-10-29-qwen3/inputs'/('Qwen3-'+model)/'test1.json')['trajectories']
        ids={x['trajectory_id'] for x in trace}; assert set(Counter(x['group_id'] for x in trace).values())=={8}
        command=load(folder/'command.json');assert digest(trace)==command['canonical_requests_sha256']
        assert command['policy']==load(folder/'baseline-metadata.json')['result_fields']['policy']
        eos={};occupation=[json.loads(x) for x in raw(model,'occupancy.jsonl').splitlines()]
        idle=0.
        for i,obs in enumerate(occupation):
            assert len(obs['roles'])==27
            for tid in obs['new_done']:
                assert tid in ids and tid not in eos;eos[tid]=obs['tick']
            if i+1<len(occupation):idle+=(occupation[i+1]['tick']-obs['tick'])/1e12*2*obs['roles'].count('idle')
        assert set(eos)==ids and math.isclose(idle,audit['role_integral_die_s']['idle'],abs_tol=1e-6)
        events=json.loads(raw(model,'training-events.json'));moves=json.loads(raw(model,'sample-transfers.json'));delivered={}
        owners={x['trajectory_id']:i%27 for i,x in enumerate(trace)}
        for move in moves:
            for tid in move['tids']:
                assert tid not in delivered and eos[tid]<=move['start_tick'];delivered[tid]=move['end_tick']
        for phase in ('old_policy','reference','training'):
            es=[e for e in events if e['phase']==phase];used=[t for e in es for t in e['tids']]
            assert set(used)==ids and len(used)==len(ids)
            for event in es:
                for tid in event['tids']:
                    assert eos[tid]<=event['start_tick']
                    if owners[tid]!=event['worker']:assert delivered[tid]<=event['start_tick']
        issued=load(folder/'issued-programs.json')
        for phase in ('old_policy','reference','training'):
            with gzip.open(folder/(phase+'-program.json.gz'),'rt') as f:program=json.load(f)
            for w in range(27):
                body=program['programs'][program['ranks'][str(2*w)]]['segments']
                assert digest(body)==issued[str(w)][phase]['sha256']
        native=json.loads(raw(model,'native/result.json')); assert native['native_exit_status']==0
        operations=json.loads(raw(model,'native/operations.json'))
        expected={(r,o['id']) for o in operations for r in o['ranks']}
        observed={'issue':{},'callback':{}}
        for match in EVENT_RE.finditer(raw(model,'native/astra_logs/log.log')):
            key=int(match['rank']),int(match['node']);action=match['action']
            assert key not in observed[action];observed[action][key]=int(match['tick'])
        assert set(observed['issue'])==set(observed['callback'])==expected
        assert math.isclose(max(observed['callback'].values())/1e12,result['total_s'],abs_tol=1e-9)
        assert len(load(folder/'raw/native-sequence-summaries.json'))==54
    scale=load(DATA/'trace-scale-audit.json')
    assert scale['around_30b_matches']==[] and scale['same_as_frozen_input']
    print(json.dumps(dict(status='PASS',models=3,requests_each=1536,raw_native_event_coverage=True,
        teacher_training_program_parity=True,hardware_accuracy_promoted=False)))

if __name__=='__main__':main()
