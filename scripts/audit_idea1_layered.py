"""Independent native evidence audit; never imports the scheduler under test."""
import csv
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'experiments/2026-11-02-idea1-joint'

def audit(folder):
    folder=Path(folder).resolve(); result=load(folder/'result.json'); cmd=load(folder/'command.json')
    full=not result['subset_prompts']; model=cmd['model']; trace=Path(cmd['trace_path'])
    items0=load(trace)['trajectories']; items0=items0 if full else items0[:result['subset_prompts']*8]
    items={x['trajectory_id']:x for x in items0}; ids=set(items)
    assert len(ids)==len(items0)==result['completed']==result['trained_samples']==result['requests']
    assert cmd['gradient_normalization']['global_denominator']==len(ids)
    assert cmd['gradient_normalization']['accumulation']=='sum of all assigned sample contributions'
    assert set(Counter(x['group_id'] for x in items0).values())=={8}
    assert sha(trace)==cmd['trace_sha256'] and digest(items0)==cmd['canonical_requests_sha256']
    assert result['native_exit_status']==0 and result['trained_workers']==27
    archive=load(folder/'source-archive-map.json')
    assert all(sha(archive[p])==h for p,h in load(folder/'source_hashes.json').items())
    cap=load(folder/'capacity.json'); assert cap['passed'] and max(cap['phase_peak_bytes'].values())<=64000000000
    # Reproduce complete-GRPO-group allocation, independently of runtime imports.
    ngr=len(items0)//8; q,r=divmod(ngr,27); homegroups=[]; cursor=0
    for w in range(27):
        n=(q+(w<r))*8; homegroups.append(items0[cursor:cursor+n]);cursor+=n
    homes={x['trajectory_id']:w for w,xs in enumerate(homegroups) for x in xs}
    owners={x['trajectory_id']:i%27 for i,x in enumerate(items0)}
    rows=[json.loads(line) for line in (folder/'occupancy.jsonl').open()]
    eos={}; last=-1
    for row in rows:
        assert row['tick']>=last;last=row['tick'];assert len(row['roles'])==27
        for tid in row['new_done']:
            assert tid in ids and tid not in eos;eos[tid]=row['tick']
    assert set(eos)==ids
    gen_end=max(eos.values()); assert gen_end==result['generation_end_tick']
    assert not any(set(row['roles']) & {'own','helper'} for row in rows if row['tick']>=gen_end)
    moves=load(folder/'sample-transfers.json'); delivered={}
    for m in moves:
        assert m['start_tick']<=m['end_tick']
        assert m['bytes_per_lane']==sum(4*(items[t]['prompt_length']+items[t]['response_length']) for t in m['tids'])
        for tid in m['tids']:
            assert owners[tid]==m['src'] and homes[tid]==m['dst'] and eos[tid]<=m['start_tick']
            assert tid not in delivered;delivered[tid]=m['end_tick']
    assert set(delivered)=={t for t in ids if owners[t]!=homes[t]}
    stages=load(folder/'training-events.json'); byphase=defaultdict(list)
    for e in stages:
        assert e['start_tick']<=e['end_tick'];byphase[e['phase']].append(e)
    for phase in ('old_policy','reference','training'):
        seen=[t for e in byphase[phase] for t in e['tids']]
        assert len(seen)==len(set(seen)) and set(seen)==ids
    for phase in ('old_policy','reference'):
        for e in byphase[phase]:
            for tid in e['tids']:
                assert homes[tid]==e['worker'] and eos[tid]<=e['start_tick']
                if owners[tid]!=e['worker']:assert delivered[tid]<=e['start_tick']
    advantages={}; zeros={}
    for w in range(27):
        own=[e for e in stages if e['worker']==w]
        zero=[e for e in own if e['phase']=='gradient_zero'];adv=[e for e in own if e['phase']=='advantage']
        assert len(zero)==len(adv)==1;zeros[w]=zero[0];advantages[w]=adv[0]
        assert all(e['end_tick']<=adv[0]['start_tick'] for e in own if e['phase'] in ('reference','old_policy'))
        assert adv[0]['end_tick']<=zero[0]['start_tick']
        teacher=[e for e in own if e['phase']=='old_policy']
        if teacher:
            begin=min(e['start_tick'] for e in teacher)
            assert all(eos[t]<=begin for t in ids if owners[t]//3==w//3)
    manifest=load(folder/'training-batches.json'); executed=load(folder/'training-batch-executions.json')
    transfer=load(folder/'training-transfers.json'); moved={m['batch_id']:m for m in transfer}
    assert len(moved)==len(transfer)
    actual={e['batch_id']:e for e in executed};assert len(actual)==len(executed) and set(actual)==set(manifest)
    expected={}; mb=cmd['policy']['train_mb']
    for w,xs in enumerate(homegroups):
        for i,start in enumerate(range(0,len(xs),mb)):
            expected[f'{w}:{i}']=[x['trajectory_id'] for x in xs[start:start+mb]]
    assert set(manifest)==set(expected)
    eventkeys=Counter((e['worker'],tuple(e['tids']),e['start_tick'],e['end_tick']) for e in byphase['training'])
    for bid,m in manifest.items():
        e=actual[bid];home=m['home'];w=e['worker']
        assert m['tids']==e['tids']==expected[bid] and e['home']==home
        assert e['program_sha256']==m['program_sha256']
        assert advantages[home]['end_tick']==m['ready_tick']<=e['start_tick']
        assert zeros[w]['end_tick']<=e['start_tick']<=e['end_tick']
        assert eventkeys[(w,tuple(e['tids']),e['start_tick'],e['end_tick'])]==1
        payload=sum(5*(items[t]['prompt_length']+items[t]['response_length'])+8*items[t]['response_length']+4+64 for t in e['tids'])
        assert payload==m['payload_bytes_per_lane']
        if w!=home:
            tx=moved[bid]
            assert tx['src']==home and tx['dst']==w and tx['tids']==e['tids'] and tx['bytes_per_lane']==payload
            assert max(gen_end,m['ready_tick'])<=tx['start_tick']<=tx['end_tick']<=e['start_tick']
            assert result['training_scope']=='wafer' or w//3==home//3
            # Receiver completed its entire remaining home queue first.
            assert all(x['end_tick']<=tx['start_tick'] for x in executed if x['home']==w and x['worker']==w)
            assert not any(x['start_tick']<tx['start_tick']<x['end_tick'] for x in executed if x['worker']==home)
        else:assert bid not in moved
    assert len(moved)==result['training_transfers']==result['remotely_executed_batches']
    counts=Counter(e['worker'] for e in executed for t in e['tids'])
    if 'worker_trained_samples' in result:assert all(result['worker_trained_samples'][str(w)]==counts[w] for w in range(27))
    if full:
        programs=load(folder/'issued-programs.json');zeroproofs=load(folder/'gradient-zero-programs.json'); frozen_jobs={}; frozen_zeros={}
        for phase in ('old_policy','reference','training'):
            with gzip.open(EVIDENCE/model/(phase+'-program.json.gz'),'rt') as f:frozen=json.load(f)
            for w in range(27):
                segments=frozen['programs'][frozen['ranks'][str(2*w)]]['segments']
                if phase!='training':
                    assert programs[str(w)][phase]==dict(segments=len(segments),sha256=digest(segments))
                else:
                    assert zeroproofs[str(w)]==digest(segments[0]);frozen_zeros[w]=segments[0]
                    assert len(segments)-1==sum(m['home']==w for m in manifest.values())
                    for i,s in enumerate(segments[1:]):
                        frozen_jobs[f'{w}:{i}']=s
                        assert manifest[f'{w}:{i}']['program_sha256']==digest(s)
                        assert all(x['kind'] in ('local','AR') and not any(k in x for k in ('src','dst','ranks','group')) for x in s['steps'])
        for w in range(27):
            ordered=[(zeros[w]['start_tick'],frozen_zeros[w])]+[(e['start_tick'],frozen_jobs[e['batch_id']]) for e in executed if e['worker']==w]
            actual_program=[s for tick,s in sorted(ordered,key=lambda x:x[0])]
            assert programs[str(w)]['training']==dict(segments=len(actual_program),sha256=digest(actual_program))
        assert cmd['policy']==load(EVIDENCE/model/'baseline-metadata.json')['result_fields']['policy']
    native=load(folder/'native/result.json');assert native['native_exit_status']==0 and native['native_coverage_and_dependencies_checked']
    assert len(native['native_sequence_summaries'])==54
    ops=load(folder/'native/operations.json')
    assert len([o for o in ops if 'publish_actor_v1' in o['name']])==1
    assert len([o for o in ops if o['kind']=='collective' and 'dp_gradient_RS' in o['name']])==2
    assert len([o for o in ops if o['kind']=='collective' and 'early_restore_reference' in o['name']])==(2 if model=='14B' else 0)
    generation=load(folder/'generation-events.json');config=load(trace.parent/'config.json')
    kh=config['num_key_value_heads'];qh=config['num_attention_heads']; hd=config.get('head_dim',config['hidden_size']//qh)
    factor=config['num_hidden_layers']*2*(kh//4)*hd*2;prior={}
    assert {x['tx'] for x in generation['migrations']}=={x['tx'] for x in generation['releases']}
    for m in generation['migrations']:
        assert m['source']//3==m['helper']//3
        assert sorted(h for lane in m['head_layout'] for h in lane['kv_heads'])==list(range(kh))
        assert sorted(h for lane in m['head_layout'] for h in lane['query_heads'])==list(range(qh))
        assert m['kv_bytes_per_lane']==sum(m['initial_actual_contexts'])*factor
        prior[m['tx']]=dict(zip(m['active_ids'],m['initial_actual_contexts']))
    for s in generation['helper_runs']:
        assert all(prior[s['tx']][t]==c for t,c in zip(s['active_ids'],s['actual_contexts']))
        assert s['kv_append_bytes_per_rank_per_step']==len(s['active_ids'])*factor
        prior[s['tx']]={t:c+s['repeat'] for t,c in zip(s['active_ids'],s['actual_contexts'])}
    # Reconstruct post-generation roles from native stage intervals. Legacy raw
    # roles can remain 'training' after a helper finished; do not trust that label.
    end=max(e['end_tick'] for e in byphase['training']); intervals=[]
    for e in stages:
        role='training' if e['phase'] in ('training','gradient_zero') else 'prepare'
        intervals.append((e['start_tick'],e['end_tick'],e['worker'],role))
    for e in moves+transfer:
        intervals.extend((e['start_tick'],e['end_tick'],w,'sample_transfer') for w in (e['src'],e['dst']))
    changes=defaultdict(list)
    for a,b,w,role in intervals:
        if a<b:
            changes[a].append((1,w,role));changes[b].append((-1,w,role))
    state=[Counter() for _ in range(27)];timeline=[];integrals=Counter();times=sorted(set(changes)|{gen_end,end})
    for i,t in enumerate(times):
        for sign,w,role in changes[t]:
            state[w][role]+=sign;assert state[w][role]>=0
        if not gen_end<=t<=end:continue
        roles=[]
        for w in range(27):
            live=[role for role,n in state[w].items() if n]
            assert len(live)<=1,(w,t,live)
            roles.append(live[0] if live else 'idle')
        c=Counter(roles); row=dict(step_s=t/1e12,**{r+'_dies':2*c[r] for r in ('prepare','training','sample_transfer','idle')})
        timeline.append(row)
        if i+1<len(times):
            dt=(min(times[i+1],end)-t)/1e12
            for role,n in c.items():integrals[role]+=2*n*dt
    with (folder/'die-timeline-after-generation.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(timeline[0]));wr.writeheader();wr.writerows(timeline)
    assert abs(sum(integrals.values())-54*(end-gen_end)/1e12)<1e-5
    critical=max(executed,key=lambda e:e['end_tick'])
    critical_detail=dict(batch_id=critical['batch_id'],home=critical['home'],worker=critical['worker'],ready_s=manifest[critical['batch_id']]['ready_tick']/1e12,start_s=critical['start_tick']/1e12,end_s=critical['end_tick']/1e12,duration_s=(critical['end_tick']-critical['start_tick'])/1e12,response_tokens=sum(items[t]['response_length'] for t in critical['tids']))
    report=dict(status='PASS_INDEPENDENT_EVIDENCE_AUDIT',model=model,full_test1=full,requests=len(ids),
        frozen_teacher_parity=full,frozen_training_batch_multiset_parity=full,
        trained_once=True,gradient_zero_once_per_worker=True,prepared_transfer_causality=True,
        generation_end_s=gen_end/1e12,first_training_s=min(e['start_tick'] for e in byphase['training'])/1e12,
        training_end_s=end/1e12,total_s=result['total_s'],training_tail_s=(end-gen_end)/1e12,
        training_batches=len(manifest),remotely_executed_batches=len(moved),critical_last_batch=critical_detail,latest_advantage_ready_s=max(e['end_tick'] for e in advantages.values())/1e12,
        transferred_bytes_all_lanes=2*sum(m['bytes_per_lane'] for m in transfer),
        transfer_duration_sum_s=sum(m['end_tick']-m['start_tick'] for m in transfer)/1e12,
        worker_trained_samples={str(w):counts[w] for w in range(27)},
        peak_bytes=max(cap['phase_peak_bytes'].values()),post_generation_integrals_die_s=dict(integrals),
        average_idle_dies_after_generation=integrals['idle']/((end-gen_end)/1e12),
        role_note='Post-generation roles reconstructed from native start/end events; stale raw helper role labels excluded. Includes local memory/network waits, not arithmetic utilization.',
        limitations=['conditional single-window cost simulation','no numerical transformer gradients or learning quality','no physical wafer calibration','GPU same-mechanism comparison pending'])
    (folder/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':audit(sys.argv[1])
