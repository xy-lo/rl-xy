"""Independently replay C++ admission logs against actual Astra issue/callback."""
import json
from pathlib import Path
from .types import CheckResult, MemorySnapshot


def _resource_rows(path):
    """Read each original event once without retaining the entire JSONL file."""
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


class _CoreVector:
    __slots__=('dense','sparse','total','within_capacity','peak_seen')

    def __init__(self,values,total,capacity):
        # Addition to the original zero vector converts bool to int too.
        self.dense=[0+x for x in values]
        self.sparse=[[i,x] for i,x in enumerate(self.dense) if x]
        self.total=total
        self.within_capacity=not any(x<0 or x>capacity for x in self.dense)
        self.peak_seen=False


class _CoreVectorCache:
    """Reuse integer vector derivations, never source-object identities.

    A bounded rank-local cache also handles overlapping legacy schedules. A
    cache eviction only repeats derivation/peak work; it cannot skip a check.
    """
    _MAX_ENTRIES=512

    def __init__(self,cores,capacity):
        self.cores=cores;self.capacity=capacity;self.entries={}
        self.zero_key=(cores,capacity,(0,)*cores)
        self.zero=_CoreVector(self.zero_key[2],0,capacity)
        self.entries[self.zero_key]=self.zero

    def get(self,values,total=None):
        if len(values)!=self.cores:return None
        if total is None:total=sum(values)
        # Float and int tuples compare/hash equal. Preserve the original
        # floating-point additions, cancellation and peak types instead of
        # aliasing a float vector to an integer cached vector. Bool sums are
        # integers, and 0+bool above exactly matches the old replay arithmetic.
        if type(total) is not int:return None
        key=(self.cores,self.capacity,tuple(values))
        vector=self.entries.get(key)
        if vector is None:
            vector=_CoreVector(key[2],total,self.capacity)
            if len(self.entries)>=self._MAX_ENTRIES:
                self.entries.clear();self.entries[self.zero_key]=self.zero
            self.entries[key]=vector
        return vector


def validate_resources(plan, case, events, *, gradient_visits=None):
    checks=[];snapshots=[];totals={};errors=[]
    # Preserve last-record behavior for duplicate keys, while indexing by rank
    # once instead of rescanning every rank event for each physical device.
    expected={}
    for e in events:
        expected.setdefault(e.rank,{})[e.node_id]=(e.issue_tick,e.callback_tick)
    for rank_text,r in plan.contract['ranks'].items():
        rank=int(rank_text)
        path=Path(plan.contract['log_prefix']+f'.{rank}.jsonl')
        if not path.is_file():raise ValueError('required real Astra resource log missing')
        rows=_resource_rows(path);rank_expected=expected.get(rank,{})
        live=dict(r['initial_objects']);ready=set(live);active={};memory_owner=0;reserved=0
        dram=sum(live.values());sram=0
        pairs={};peak=dram;sram_peak=0
        cores=r.get('cores_per_die',1);core_used=[0]*cores;core_peaks=[0]*cores
        vectors=_CoreVectorCache(cores,r['sram_bytes_per_core']) if cores>1 else None
        current_vector=vectors.zero if vectors is not None else None
        if vectors is not None:core_used=current_vector.dense
        for i,row in enumerate(rows):
            if vectors is not None and vectors.capacity!=r['sram_bytes_per_core']:
                vectors=_CoreVectorCache(cores,r['sram_bytes_per_core'])
                current_vector=vectors.get(core_used)
                if current_vector is not None:core_used=current_vector.dense
            node=row['node_id'];kind=row['event'];tick=row['tick']
            if kind=='initial':
                if i or node or tick:errors.append('invalid initial resource event')
            else:
                n=r['nodes'][str(node)]
                if kind=='issue':
                    idle_vector=(not active and sram==0 and vectors is not None
                                 and core_used is vectors.zero.dense)
                    if node in active or not ready.issuperset(n['reads']):errors.append(f'{rank}/{node}: unsafe read/issue')
                    if n['local_memory'] and memory_owner:errors.append(f'{rank}/{node}: local memory overlap')
                    if any(sid in live for sid in n['allocations']):errors.append('duplicate storage')
                    # Count the actual dictionary update even for invalid
                    # replacement attempts; errors above still reject them.
                    for sid,size in n['allocations'].items():
                        dram+=size-live.get(sid,0);live[sid]=size
                    ready.difference_update(n['ready_outputs'])
                    reserved+=n['reserve_before']
                    sram+=n['workspace_bytes']-active.get(node,0);active[node]=n['workspace_bytes']
                    if cores>1:
                        values=n['core_workspace_bytes']
                        total=sum(values) if len(values)==cores else None
                        if len(values)!=cores or total!=n['workspace_bytes']:errors.append('invalid per-core workspace')
                        vector=vectors.get(values,total) if total is not None else None
                        if idle_vector and vector is not None:
                            core_used=vector.dense;current_vector=vector
                        else:
                            core_used=[a+b for a,b in zip(core_used,values)]
                            current_vector=vectors.get(core_used)
                            if current_vector is not None:core_used=current_vector.dense
                    if n['local_memory']:memory_owner=node
                    pairs.setdefault(node,{})['issue']=tick
                elif kind=='callback':
                    sole_active=(len(active)==1 and node in active)
                    if node not in active:errors.append(f'{rank}/{node}: callback before issue')
                    sram-=active.pop(node,0)
                    if cores>1:
                        values=n['core_workspace_bytes'];vector=vectors.get(values)
                        if sole_active and vector is not None and core_used is vector.dense:
                            core_used=vectors.zero.dense;current_vector=vectors.zero
                        else:
                            core_used=[a-b for a,b in zip(core_used,values)]
                            current_vector=vectors.get(core_used)
                            if current_vector is not None:core_used=current_vector.dense
                    if any(sid not in live for sid in n['ready_outputs']):errors.append('READY without storage')
                    ready.update(n['ready_outputs'])
                    for sid in n['frees']:
                        if sid not in live:errors.append('double free')
                        size=live.pop(sid,None)
                        if size is not None:dram-=size
                        ready.discard(sid)
                    reserved+=n['reserve_after']
                    if n['local_memory']:
                        if memory_owner!=node:errors.append('wrong memory owner')
                        memory_owner=0
                    pairs.setdefault(node,{})['callback']=tick
                else:errors.append('unknown resource event')
            peak=max(peak,dram);sram_peak=max(sram_peak,sram)
            if (row['dram_bytes'],row['sram_bytes'],row['local_memory_owner'],row['reserved_bytes'])!=(dram,sram,memory_owner,reserved):
                errors.append(f'{rank}/{node}: C++ snapshot differs from Python replay')
            if reserved<0 or dram+reserved>r['dram_capacity'] or (sram>case.hardware.sram_bytes_per_core and cores==1):
                errors.append(f'{rank}/{node}: capacity/reservation exceeded')
            if cores>1:
                vector_ok=(row['core_sram_bytes']==core_used if 'core_sram_bytes' in row else
                           row.get('core_sram_nonzero')==(current_vector.sparse if current_vector is not None else
                                                        [[i,v] for i,v in enumerate(core_used) if v]))
                private_ok=(current_vector.total==sram and current_vector.within_capacity
                            if current_vector is not None else
                            sum(core_used)==sram and not any(x<0 or x>r['sram_bytes_per_core'] for x in core_used))
                if not vector_ok or not private_ok:errors.append('private SRAM snapshot/capacity mismatch')
                if current_vector is None or not current_vector.peak_seen:
                    core_peaks=[max(a,b) for a,b in zip(core_peaks,core_used)]
                    if current_vector is not None:current_vector.peak_seen=True
            for location,used,capacity,budget in (
                (f'd{rank}.dram',dram,r['dram_capacity'],reserved),
                (f'd{rank}.sram_workspace_sum' if cores>1 else f'd{rank}.c0.sram',sram,
                 cores*case.hardware.sram_bytes_per_core,0)):
                snapshots.append(MemorySnapshot(len(snapshots),'runtime',location,used,used,budget,capacity,
                    f'astra_tick={tick};node={node};{kind}'))
        wanted=set(rank_expected)
        if set(pairs)!=wanted:errors.append(f'{rank}: incomplete node resource coverage')
        for node,pair in pairs.items():
            if (pair.get('issue'),pair.get('callback'))!=rank_expected.get(node):
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
