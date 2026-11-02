#!/usr/bin/env python3
"""Audit portable layered native evidence and exact reference parity."""
import gzip,hashlib,json,math,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'experiments/2026-11-02-idea1-layered'
sys.path.insert(0,str(ROOT/'src'))
from rl_xy.astra_adapter.parser import EVENT_RE
load=lambda p:json.loads(p.read_text())
digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def raw(folder,name):
    with gzip.open(folder/'raw'/(name.replace('/','__')+'.gz'),'rt') as f:return f.read()
def main():
    manifest=load(DATA/'publication-manifest.json')
    actual={str(p.relative_to(DATA)) for p in DATA.rglob('*') if p.is_file() and p.name!='publication-manifest.json'}
    assert actual==set(manifest['files'])
    for name,value in manifest['files'].items():assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==value,name
    table=load(DATA/'all-data.json');historical=load(DATA/'historical-identities.json')
    for model in ('4B','8B','14B'):
        folder=DATA/model;result=load(folder/'result.json');command=load(folder/'command.json')
        assert result['native_exit_status']==0 and result['requests']==result['completed']==result['trained_samples']==1536
        assert load(folder/'framework-summary.json')['legacy_top_level_step_enabled'] is False
        assert load(folder/'framework-summary.json')['controller_phase']=='FINISHED'
        assert max(load(folder/'capacity.json')['phase_peak_bytes'].values())<=64000000000
        row=next(r for r in table['comparison'] if r['model']==model)
        assert row['layered_idea_s']==row['reference_full_idea_s']==result['total_s'] and row['exact_delta_s']==0
        trace=load(ROOT/command['trace_path'])['trajectories'];ids={x['trajectory_id'] for x in trace}
        assert digest(trace)==command['canonical_requests_sha256'] and command['gradient_normalization']['global_denominator']==1536
        assert set(Counter(x['group_id'] for x in trace).values())=={8}
        for name,entry in historical[model].items():
            p=DATA/entry['path']
            if p.suffix=='.gz':
                with gzip.open(p,'rb') as f:value=f.read()
            else:value=p.read_bytes()
            assert hashlib.sha256(value).hexdigest()==entry['published_uncompressed_sha256']
        eos={}
        for line in raw(folder,'occupancy.jsonl').splitlines():
            obs=json.loads(line);assert len(obs['roles'])==27
            for tid in obs['new_done']:assert tid not in eos;eos[tid]=obs['tick']
        assert set(eos)==ids and max(eos.values())==result['generation_end_tick']
        events=json.loads(raw(folder,'training-events.json'));manifest_batches=json.loads(raw(folder,'training-batches.json'))
        executed=json.loads(raw(folder,'training-batch-executions.json'));moves=json.loads(raw(folder,'training-transfers.json'))
        for phase in ('old_policy','reference','training'):
            used=[t for e in events if e['phase']==phase for t in e['tids']]
            assert len(used)==len(set(used))==1536 and set(used)==ids
            for e in events:
                if e['phase']==phase:assert all(eos[t]<=e['start_tick'] for t in e['tids'])
        assert len({e['batch_id'] for e in executed})==len(executed)==len(manifest_batches)
        assert sum(e['home']!=e['worker'] for e in executed)==len(moves)==result['training_transfers']
        byid={e['batch_id']:e for e in executed}
        for move in moves:
            e=byid[move['batch_id']];assert e['worker']==move['dst'] and e['home']==move['src']
            assert move['end_tick']<=e['start_tick'] and move['tids']==e['tids']
        for phase in ('old_policy','reference','training'):
            with gzip.open(ROOT/'experiments/2026-11-02-idea1-joint'/model/(phase+'-program.json.gz'),'rt') as f:p=json.load(f)
            issued=load(folder/'issued-programs.json')
            for w in range(27):
                segments=p['programs'][p['ranks'][str(w*2)]]['segments']
                if phase=='training':
                    assert load(folder/'gradient-zero-programs.json')[str(w)]==digest(segments[0])
                    for i,segment in enumerate(segments[1:]):assert manifest_batches[f'{w}:{i}']['program_sha256']==digest(segment)
                else:assert issued[str(w)][phase]['sha256']==digest(segments)
        operations=json.loads(raw(folder,'native/operations.json'));expected={(r,o['id']) for o in operations for r in o['ranks']};seen={'issue':{},'callback':{}}
        for m in EVENT_RE.finditer(raw(folder,'native/astra_logs/log.log')):
            key=int(m['rank']),int(m['node']);assert key not in seen[m['action']];seen[m['action']][key]=int(m['tick'])
        assert set(seen['issue'])==set(seen['callback'])==expected
        assert math.isclose(max(seen['callback'].values())/1e12,result['total_s'],abs_tol=1e-9)
        ref=folder/'reference'
        assert json.loads(raw(ref,'controller.jsonl').splitlines()[0])==json.loads(raw(folder,'controller.jsonl').splitlines()[0])
        assert raw(ref,'controller.jsonl')==raw(folder,'controller.jsonl')
        assert load(ref/'native-sequence-summaries.json')==load(folder/'raw/native-sequence-summaries.json')
        for name in ('training-events.json','sample-transfers.json','training-batches.json','training-batch-executions.json','training-transfers.json','gradient-zero-programs.json','issued-programs.json','generation-events.json','capacity.json','native/operations.json','native/sequence_program.json','native/groups.json'):
            current=json.loads(raw(folder,name)) if (folder/'raw'/(name.replace('/','__')+'.gz')).exists() else load(folder/name)
            assert current==json.loads(raw(ref,name)),(model,name)
    print(json.dumps(dict(status='PASS_LAYERED_PUBLIC_EVIDENCE',models=3,exact_native_parity=True,requests_each=1536,hardware_accuracy_promoted=False)))
if __name__=='__main__':main()
