"""Independently replay C++ admission logs against actual Astra issue/callback."""
import json
from pathlib import Path
from .types import CheckResult, MemorySnapshot


def validate_resources(plan, case, events, *, gradient_visits=None):
    checks=[];snapshots=[];totals={};errors=[]
    expected={(e.rank,e.node_id):(e.issue_tick,e.callback_tick) for e in events}
    for rank_text,r in plan.contract['ranks'].items():
        rank=int(rank_text)
        path=Path(plan.contract['log_prefix']+f'.{rank}.jsonl')
        if not path.is_file():raise ValueError('required real Astra resource log missing')
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        live=dict(r['initial_objects']);ready=set(live);active={};memory_owner=0;reserved=0
        pairs={};peak=sum(live.values());sram_peak=0
        cores=r.get('cores_per_die',1);core_used=[0]*cores;core_peaks=[0]*cores
        for i,row in enumerate(rows):
            node=row['node_id'];kind=row['event'];tick=row['tick']
            if kind=='initial':
                if i or node or tick:errors.append('invalid initial resource event')
            else:
                n=r['nodes'][str(node)]
                if kind=='issue':
                    if node in active or set(n['reads'])-ready:errors.append(f'{rank}/{node}: unsafe read/issue')
                    if n['local_memory'] and memory_owner:errors.append(f'{rank}/{node}: local memory overlap')
                    if set(n['allocations']) & set(live):errors.append('duplicate storage')
                    live.update(n['allocations']);ready.difference_update(n['ready_outputs'])
                    reserved+=n['reserve_before'];active[node]=n['workspace_bytes']
                    if cores>1:
                        values=n['core_workspace_bytes']
                        if len(values)!=cores or sum(values)!=n['workspace_bytes']:errors.append('invalid per-core workspace')
                        core_used=[a+b for a,b in zip(core_used,values)]
                    if n['local_memory']:memory_owner=node
                    pairs.setdefault(node,{})['issue']=tick
                elif kind=='callback':
                    if node not in active:errors.append(f'{rank}/{node}: callback before issue')
                    active.pop(node,None)
                    if cores>1:core_used=[a-b for a,b in zip(core_used,n['core_workspace_bytes'])]
                    if set(n['ready_outputs'])-set(live):errors.append('READY without storage')
                    ready.update(n['ready_outputs'])
                    for sid in n['frees']:
                        if sid not in live:errors.append('double free')
                        live.pop(sid,None);ready.discard(sid)
                    reserved+=n['reserve_after']
                    if n['local_memory']:
                        if memory_owner!=node:errors.append('wrong memory owner')
                        memory_owner=0
                    pairs.setdefault(node,{})['callback']=tick
                else:errors.append('unknown resource event')
            dram=sum(live.values());sram=sum(active.values());peak=max(peak,dram);sram_peak=max(sram_peak,sram)
            if (row['dram_bytes'],row['sram_bytes'],row['local_memory_owner'],row['reserved_bytes'])!=(dram,sram,memory_owner,reserved):
                errors.append(f'{rank}/{node}: C++ snapshot differs from Python replay')
            if reserved<0 or dram+reserved>r['dram_capacity'] or (sram>case.hardware.sram_bytes_per_core and cores==1):
                errors.append(f'{rank}/{node}: capacity/reservation exceeded')
            if cores>1:
                vector_ok=(row['core_sram_bytes']==core_used if 'core_sram_bytes' in row else
                           row.get('core_sram_nonzero')==[[i,v] for i,v in enumerate(core_used) if v])
                if not vector_ok or sum(core_used)!=sram or any(
                    x<0 or x>r['sram_bytes_per_core'] for x in core_used):errors.append('private SRAM snapshot/capacity mismatch')
                core_peaks=[max(a,b) for a,b in zip(core_peaks,core_used)]
            for location,used,capacity,budget in (
                (f'd{rank}.dram',dram,r['dram_capacity'],reserved),
                (f'd{rank}.sram_workspace_sum' if cores>1 else f'd{rank}.c0.sram',sram,
                 cores*case.hardware.sram_bytes_per_core,0)):
                snapshots.append(MemorySnapshot(len(snapshots),'runtime',location,used,used,budget,capacity,
                    f'astra_tick={tick};node={node};{kind}'))
        wanted={node for rnk,node in expected if rnk==rank}
        if set(pairs)!=wanted:errors.append(f'{rank}: incomplete node resource coverage')
        for node,pair in pairs.items():
            if (pair.get('issue'),pair.get('callback'))!=expected.get((rank,node)):
                errors.append(f'{rank}/{node}: not on the same Astra event clock')
        if live!=r['initial_objects'] or active or reserved or memory_owner:
            errors.append(f'{rank}: resource/tensor/reservation leak')
        totals[f'd{rank}.dram']=peak
        if cores>1:
            totals[f'd{rank}.sram_workspace_sum']=sram_peak
            totals.update({f'd{rank}.c{core}.sram':value for core,value in enumerate(core_peaks)})
        else:totals[f'd{rank}.c0.sram']=sram_peak
    checks.append(CheckResult('resources.actual_clock_capacity_ready_lifetime',not errors,
        'all participating nodes admitted/retired on actual Astra ticks; valid live/READY/capacity/ownership',
        errors,'astra_logs/resources.<rank>.jsonl independently replayed against resource_contract.json and events.jsonl',
        'synthetic_resource_contract'))
    gradient_errors=[]
    expected_bytes=sum(t['nbytes'] for t in plan.tensors if t['persistent'] and t['rank']==0 and t['kind']=='gradient')
    # Tied weights have two independently computed contributions, merged into
    # one persistent gradient storage before accumulation and one Adam update.
    contribution_bytes=expected_bytes
    if case.model.tie_word_embeddings:
        contribution_bytes+=case.model.vocab_size*case.model.hidden_size//case.strategy.tensor_parallel*4
    from .training_schedule import training_visits
    if gradient_visits is None:
        gradient_visits=[(rank,rid,phase+'_backward') for rank in range(case.hardware.dies)
                         for rid,_,_,_,_,phase in training_visits(case)]
    for rank,round_id,prefix in gradient_visits:
        count=sum(t['nbytes'] for t in plan.tensors if t['kind']=='parameter_gradient' and t['rank']==rank
                  and t['round_id']==round_id and prefix in t['tensor_id'])
        if count!=contribution_bytes:gradient_errors.append([rank,round_id,prefix,count,contribution_bytes])
    checks.append(CheckResult('tensors.backward_parameter_gradient_shapes',not gradient_errors,
        contribution_bytes,gradient_errors,'typed dweight/norm/bias/embedding contributions; tied head and embedding merge into one persistent gradient',
        'synthetic_shape_contract'))
    return checks,snapshots,totals
