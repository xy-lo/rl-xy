"""Independent TP/EOS/KV and actual native reply-accounting checks."""
import argparse,gzip,json,math,hashlib
from pathlib import Path
from collections import Counter,defaultdict
from audit_idea1_system import audit,digest,load


def audit_training_packing(choices,layout,eos,train_mb,regional_items=None):
    """Reconstruct the local-EOS information boundary and full sample coverage."""
    cap=layout.get('adaptive_training_batch_max',0)
    assert type(cap) is int and cap in (0,1,2,4,8)
    if not cap:
        assert layout.get('training_tail_fine_samples',0)==0
        assert not any('adaptive_training_batch_max' in x for x in choices)
        return None
    assert type(train_mb) is int and train_mb>0
    tail=layout.get('training_tail_fine_samples',0)
    assert type(tail) is int and tail in (0,8,16)
    if tail:assert regional_items is not None,'fine tail requires independently reconstructed home order'
    tailsets={r:set(ids[-tail:]) for r,ids in regional_items.items()} if tail else {}
    tids=[];grown=0;training=[x for x in choices if x['stage']=='training']
    for x in training:
        assert x['adaptive_training_batch_max']==cap
        ids=x['packing_tids'];assert len(ids)==x['actual'] and len(set(ids))==len(ids)
        assert all(t in eos and eos[t]<=x['decision_tick'] for t in ids),'packing precedes observed EOS'
        maximum=cap
        if x.get('late_region_cap_eligible'):maximum=min(maximum,layout['late_region_batch_cap_per_pair']*(x['tp']//2))
        if tail:
            region=x['region'];assert all(t in regional_items[region] for t in ids)
            expected=all(t in tailsets[region] for t in ids)
            assert expected or all(t not in tailsets[region] for t in ids),'batch crosses bulk/tail boundary'
            assert x['training_tail_fine_samples']==tail and type(x['tail_fine']) is bool and x['tail_fine']==expected
            if expected:maximum=min(maximum,x['tp']//2)
        assert x['actual']<=x['maximum']==maximum
        assert x['tp'] in (2,4,8)
        expected_baseline=train_mb if x['tp']==2 else min(8,train_mb*x['tp'])
        assert type(x['baseline_maximum']) is int and x['baseline_maximum']==expected_baseline
        grown+=x['actual']>x['baseline_maximum'];tids.extend(ids)
    assert Counter(tids)==Counter(eos.keys()),'packing drops or duplicates samples'
    assert all('adaptive_training_batch_max' not in x for x in choices if x['stage']!='training')
    proof=dict(adaptive_training_batch_max=cap,adaptive_training_batches=len(training),larger_than_baseline_batches=grown,packing_uses_observed_EOS=True)
    if tail:proof.update(training_tail_fine_samples=tail,tail_fine_batches=sum(x['tail_fine'] for x in training),tail_membership_reconstructed=True)
    return proof

def check(folder):
    base=audit(folder);cmd=load(folder/'command.json');trace=Path(cmd['trace_path'])
    if not trace.is_absolute():trace=Path(__file__).resolve().parents[1]/trace
    items=load(trace)['trajectories'];items=items[:cmd['subset_prompts']*8] if cmd['subset_prompts'] else items
    byid={x['trajectory_id']:x for x in items};config=load(trace.parent/'config.json')
    layers=config['num_hidden_layers'];heads=config['num_key_value_heads'];dim=config.get('head_dim',config['hidden_size']//config['num_attention_heads'])
    layout=load(folder/'inference-layout.json');owners={w:int(r) for r,ws in layout['groups'].items() for w in ws}
    assert sorted(owners)==list(range(27))
    initial={int(r):tuple(ws) for r,ws in layout['groups'].items()};groupmap=load(folder/'native/groups.json')
    events=load(folder/'inference-program-events.json');byevent={(e['group'],e['epoch']):e for e in events}
    assert len(byevent)==len(events)
    active=defaultdict(dict);prefilled=set();done=set();eos={};cohort_time=defaultdict(list);awaiting_copy={}
    for e in events:
        assert e['tp'] in (1,2,4,8) and len(e['ranks'])==len(set(e['ranks']))==(2 if e['tp']==1 else e['tp'])
        assert e['tick']<=e['end_tick'];r=e['group']
        for w in e['members']:cohort_time[w].append((e['tick'],e['end_tick']))
        if e['tp']==1:
            assert layout['paired_tp1']
            if e['kind']=='paired_copy':
                for copy in e['copies']:
                    assert copy['src']!=copy['dst'] and {copy['src'],copy['dst']}==set(e['ranks'])
                    assert copy['bytes']==sum(4*(byid[t]['prompt_length']+byid[t]['response_length']) for t in copy['tids'])
                    for t in copy['tids']:
                        assert awaiting_copy.pop(t)==copy['src'];done.add(t);eos[t]=e['end_tick']
            else:
                assert e['kind']=='paired_body' and {x['rank'] for x in e['lanes']}==set(e['ranks'])
                for lane in e['lanes']:
                    rank=lane['rank']
                    if lane['kind']=='prefill':
                        assert not set(lane['pending_ids'])&prefilled;prefilled.update(lane['pending_ids'])
                        for t in lane['pending_ids']:
                            if byid[t]['response_length']==1:awaiting_copy[t]=rank
                            else:active[rank][t]=byid[t]['prompt_length']+1
                    elif lane['kind']=='decode':
                        assert set(lane['active_ids'])==set(active[rank])
                        assert [active[rank][t] for t in lane['active_ids']]==lane['actual_contexts']
                        for t in lane['active_ids']:
                            active[rank][t]+=e['repeat'];generated=active[rank][t]-byid[t]['prompt_length']
                            assert generated<=byid[t]['response_length']
                            if generated==byid[t]['response_length']:del active[rank][t];awaiting_copy[t]=rank
                    else:assert lane['kind']=='empty' and not active[rank]
            continue
        if e['kind']=='prefill':
            assert not set(e['pending_ids'])&prefilled;prefilled.update(e['pending_ids'])
            for t in e['pending_ids']:
                assert t not in done
                if byid[t]['response_length']==1:done.add(t);eos[t]=e['end_tick']
                else:active[r][t]=byid[t]['prompt_length']+1
        else:
            assert e['kind']=='decode' and set(e['active_ids'])==set(active[r])
            assert [active[r][t] for t in e['active_ids']]==e['actual_contexts']
            for t in e['active_ids']:
                active[r][t]+=e['repeat'];generated=active[r][t]-byid[t]['prompt_length']
                assert generated<=byid[t]['response_length']
                if generated==byid[t]['response_length']:del active[r][t];done.add(t);eos[t]=e['end_tick']
    assert done==prefilled==set(byid) and not any(active.values()) and not awaiting_copy
    if layout.get('paired_tp1'):
        storage=load(folder/'inference-storage.json');assert storage['tp']==1 and storage['canonical_actor_bytes_per_die']==storage['extra_actor_bytes_per_die']>0
        restores=[o for o in load(folder/'native/operations.json') if 'TP1_restore_full_actor_' in o['name'] and o['kind']=='collective']
        assert len(restores)==27 and sorted(r for o in restores for r in o['ranks'])==list(range(54))
    observed={t:x['tick'] for line in (folder/'occupancy.jsonl').open() for x in [json.loads(line)] for t in x['new_done']}
    assert observed==eos
    for e in load(folder/'training-events.json'):
        if not e['tids']:continue
        assert all(e['end_tick']<=s or t<=e['start_tick'] for s,t in cohort_time[e['worker']])
    resize_by={(e['root'],e['tick']):e for e in load(folder/'inference-resizes.json')}
    migration_count=0;broadcasted=set()
    pending=load(folder/'pending-transfers.json');pending_by_id={t:m for m in pending if m.get('cohort_pending') for t in m['tids']}
    for tx in pending:
        if not tx.get('cohort_pending'):continue
        leased=set(initial[tx['src']])|set(tx['dst_members'])
        for move in resize_by.values():
            if tx['start_tick']<=move['tick']<=tx['end_tick']:
                assert not (set(move['new_members'])-set(move['old_members']))&leased,'TP growth overlaps pending cohort reservation'
        for e in load(folder/'training-events.json'):
            if e['worker'] in leased and e['tids']:
                assert e['end_tick']<=tx['start_tick'] or tx['end_tick']<=e['start_tick'],'training overlaps pending cohort reservation'
    with gzip.open(folder/'inference-programs.jsonl.gz','rt') as stream:
        for line in stream:
            entry=json.loads(line);e=entry['event'];assert (e['group'],e['epoch']) in byevent
            assert digest(dict(steps=entry['steps'],repeat=e['repeat']))==e['program_sha256']
            assert not any(s['kind']=='p2p' for s in entry['steps'])
            for s in entry['steps']:
                if s['kind']=='AR':assert set(groupmap[str(s['group'])])==set(e['ranks'])
                if s['kind']=='p2p_direct':assert {s['src'],s['dst']}<=set(e['ranks'])
            broadcast_ids=e.get('input_broadcast_ids',[]);broadcast=entry.get('input_broadcast',[])
            if broadcast_ids:
                assert not set(broadcast_ids)&broadcasted
                for t in broadcast_ids:
                    tx=pending_by_id[t];assert tx['dst']==e['group'] and tx['end_tick']<=e['tick']
                size=sum(byid[t]['prompt_length']*4+64 for t in broadcast_ids)
                root=e['group'];expected_pairs={(2*root+h,2*w+h) for w in e['members'] if w!=root for h in (0,1)}
                copies=[s for s in broadcast if s['kind']=='p2p_direct']
                assert {(s['src'],s['dst']) for s in copies}==expected_pairs and len(copies)==len(expected_pairs)
                assert all(s['bytes']==size for s in copies)
                broadcasted.update(broadcast_ids)
            else:assert not broadcast
            if entry['migration']:
                migration_count+=1;move=resize_by[e['group'],e['tick']]
                assert digest(entry['migration'])==move['migration_program_sha256']
                assert set(move['old_members'])<set(move['new_members']) and move['new_tp']==2*move['old_tp']
                expected=2*layers*sum(move['contexts'])*dim*2*heads
                assert sum(m['bytes'] for m in move['moves'])==expected
                perhead=Counter(h for m in move['moves'] for h in range(m['head_start'],m['head_end']))
                assert perhead==Counter(range(heads))
                old_ranks={2*w+h for w in move['old_members'] for h in (0,1)}
                assert move.get('storage_protocol'),'old migration lacks native in-place retirement protocol'
                for i,m in enumerate(move['moves']):
                    later=move['moves'][i+1:]
                    if m['src']==m['dst']:assert not any(n['src']==m['src'] and n['dst']!=n['src'] for n in later)
                    elif m['dst'] in old_ranks:assert not any(n['src']==m['dst'] for n in later)
                gates={s['key']:s for s in entry['migration'] if s['kind']=='gate_signal'}
                for rank in old_ranks:
                    key=f"KV-reshape-{e['group']}-{e['epoch']}-retired-{rank}"
                    assert gates[key]['rank']==rank
                for m in move['moves']:
                    if m['dst'] in old_ranks and m['dst']!=m['src']:
                        key=f"KV-reshape-{e['group']}-{e['epoch']}-retired-{m['dst']}"
                        assert any(s['kind']=='gate_wait' and s['key']==key for s in entry['migration'])
                assert sum(s['bytes'] for s in entry['migration'] if s['kind']=='p2p_direct')==sum(m['bytes'] for m in move['moves'] if m['src']!=m['dst'])+move['metadata_bytes_per_new_die']*(move['new_tp']-move['old_tp'])
    assert migration_count==len(resize_by)
    assert broadcasted==set(pending_by_id)
    # Reconstruct every local service and communication issue from the exact
    # replies actually delivered to the two physical ranks of each owner.
    expected={r:dict(compute_ns=0.,memory_ns=0.,communication_steps=0,executed_steps=0) for r in range(54)}
    with gzip.open(folder/'native-replies.jsonl.gz','rt') as stream:
        for line in stream:
            event=json.loads(line);reply=event['reply'];repeat=reply.get('repeat',1)
            for rank in (2*event['worker'],2*event['worker']+1):
                d=expected[rank];d['executed_steps']+=len(reply['steps'])*repeat+(event['phase']=='arrive')
                for s in reply['steps']:
                    if s['kind']=='local' and ('ranks' not in s or rank in s['ranks']):
                        for key in ('compute','memory'):d[key+'_ns']+=repeat*s.get(key,0)+repeat*(repeat-1)/2*s.get(key+'_delta',0)
                    if s['kind'] in ('AR','AG','RS') and ('ranks' not in s or rank in s['ranks']):d['communication_steps']+=repeat
                    if s['kind']=='p2p_direct' and rank in (s['src'],s['dst']):d['communication_steps']+=repeat
    actual=load(folder/'native/result.json')['native_sequence_summaries']
    for rank,values in expected.items():
        for key,value in values.items():
            got=actual[str(rank)][key]
            assert math.isclose(got,value,rel_tol=2e-11,abs_tol=.1),(rank,key,got,value)
    training_layout=load(folder/'resource-layout.json')
    ready_order=training_layout.get('training_ready_order','input')
    assert ready_order in ('input','ready_lpt')
    if ready_order=='ready_lpt':
        manifest=load(folder/'training-batches.json');tickets=load(folder/'training-ticket-events.json')
        homes={int(x['home']) for x in manifest.values()}
        for region in homes:
            observed=[x['batch_id'] for x in tickets if x['region']==region]
            expected_ids=sorted((bid for bid,x in manifest.items() if x['home']==region),
                key=lambda bid:(-manifest[bid]['estimate_local_ns'],manifest[bid]['index']))
            assert observed==expected_ids,('ready LPT ticket order',region)
    late_cap=training_layout.get('late_region_batch_cap_per_pair',0)
    assert type(late_cap) is int and late_cap in (0,1,2,4)
    applied=0
    if late_cap:
        generation_end=load(folder/'result.json')['generation_end_tick']
        choices=load(folder/'microbatch-choices.json')
        for choice in choices:
            if choice['stage']!='training':continue
            assert 'late_region_cap_applied' in choice
            if choice['late_region_cap_eligible']:
                assert choice['global_rollout_complete'] and choice['decision_tick']>=generation_end
                assert choice['actual']<=choice['maximum']<=late_cap*(choice['tp']//2)
            if choice['late_region_cap_applied']:
                applied+=1
                assert choice['late_region_cap_eligible'] and choice['maximum']<choice['original_maximum']
    choices=load(folder/'microbatch-choices.json') if (folder/'microbatch-choices.json').exists() else []
    regional_items=None
    if training_layout.get('training_tail_fine_samples',0):
        bygroup=defaultdict(list)
        for x in items:bygroup[str(x['group_id'])].append(x['trajectory_id'])
        regional_items={int(r):[t for w in ws for gid in training_layout['queues'][str(w)] for t in bygroup[str(gid)]]
            for r,ws in training_layout['region_members'].items()}
    packing=audit_training_packing(choices,training_layout,eos,cmd.get('policy',{}).get('train_mb'),regional_items)

    if packing:
        recorded=Counter(tuple(x['packing_tids']) for x in choices if x['stage']=='training')
        manifest=load(folder/'training-batches.json')
        assert recorded==Counter(tuple(x['tids']) for x in manifest.values()),'packing choices differ from executable batches'
    report=dict(status='PASS_ACTUAL_TP_INFERENCE_NATIVE_AUDIT',base=base,inference_programs=len(events),
        initial_inference_instances=54 if layout.get('paired_tp1') else len(initial),initial_tp_counts={1:54} if layout.get('paired_tp1') else dict(Counter(len(ws)*2 for ws in initial.values())),
        late_region_batch_cap=late_cap,late_region_capped_batches=applied,training_ready_order=ready_order,native_resizes=len(resize_by),generation_reply_local_and_comm_accounting='all54ranks',
        kv_storage_hazard_checked=True,
        pending_input_fanout_checked=True,
        pending_resize_exclusion_checked=True,
        EOS_and_KV_conservation=True,scope='cost/native timing; not numeric Transformer or physical hardware')
    if packing:report.update(packing)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=check(a.folder.resolve());a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='base'}))
