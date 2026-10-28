"""CPU-only replay equivalence against the frozen 0.9.4-a8 implementation.

The embedded module is the exact archive member, not a second implementation of
this optimization. Its SHA binds the oracle without depending on a local archive.
"""
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from rl_xy import resource_validation

_FROZEN_A8_SHA256='b4a1ba3f491df14840c383f60023563ccb658d07519c1339ddbb0c3c44e732d9'
_FROZEN_A8_SOURCE='''"""Independently replay C++ admission logs against actual Astra issue/callback."""
import json
from pathlib import Path
from .types import CheckResult, MemorySnapshot


def _resource_rows(path):
    """Read each original event once without retaining the entire JSONL file."""
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


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
        for i,row in enumerate(rows):
            node=row['node_id'];kind=row['event'];tick=row['tick']
            if kind=='initial':
                if i or node or tick:errors.append('invalid initial resource event')
            else:
                n=r['nodes'][str(node)]
                if kind=='issue':
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
                        if len(values)!=cores or sum(values)!=n['workspace_bytes']:errors.append('invalid per-core workspace')
                        core_used=[a+b for a,b in zip(core_used,values)]
                    if n['local_memory']:memory_owner=node
                    pairs.setdefault(node,{})['issue']=tick
                elif kind=='callback':
                    if node not in active:errors.append(f'{rank}/{node}: callback before issue')
                    sram-=active.pop(node,0)
                    if cores>1:core_used=[a-b for a,b in zip(core_used,n['core_workspace_bytes'])]
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
'''


def _frozen_reference():
    assert hashlib.sha256(_FROZEN_A8_SOURCE.encode()).hexdigest()==_FROZEN_A8_SHA256
    namespace={'__name__':'rl_xy._frozen_a8_resource_validation','__package__':'rl_xy'}
    exec(compile(_FROZEN_A8_SOURCE,'<frozen-0.9.4-a8-resource-validation>','exec'),namespace)
    return namespace


def _case(cores):
    return SimpleNamespace(hardware=SimpleNamespace(dies=1,sram_bytes_per_core=64),
        model=SimpleNamespace(tie_word_embeddings=False),strategy=SimpleNamespace(tensor_parallel=1))


def _node(vector,workspace=None,**updates):
    value={'reads':['persistent'],'allocations':{},'ready_outputs':[],'frees':[],
        'reserve_before':0,'reserve_after':0,'workspace_bytes':sum(vector) if workspace is None else workspace,
        'core_workspace_bytes':vector,'local_memory':False}
    value.update(updates)
    return value


def _row(node,kind,tick,vector,sram=None,*,sparse=False,dram=4,owner=0,reserved=0):
    row={'node_id':node,'event':kind,'tick':tick,'dram_bytes':dram,
        'sram_bytes':sum(vector) if sram is None else sram,'local_memory_owner':owner,'reserved_bytes':reserved}
    row['core_sram_nonzero' if sparse else 'core_sram_bytes']=(
        [[i,x] for i,x in enumerate(vector) if x] if sparse else list(vector))
    return row


def _fixture(vectors,*,cores=None,capacity=64):
    cores=len(vectors[0]) if cores is None else cores
    nodes={str(i):_node(vector) for i,vector in enumerate(vectors,1)}
    rank={'initial_objects':{'persistent':4},'dram_capacity':1024,
        'cores_per_die':cores,'sram_bytes_per_core':capacity,'nodes':nodes}
    rows=[_row(0,'initial',0,[0]*cores)]
    events=[]
    for node,vector in enumerate(vectors,1):
        issue=node*2;callback=issue+1
        rows.extend([_row(node,'issue',issue,vector,sparse=bool(node%2)),
                     _row(node,'callback',callback,[0]*len(vector),sparse=not bool(node%2))])
        events.append(SimpleNamespace(rank=0,node_id=node,issue_tick=issue,callback_tick=callback))
    return rank,rows,events


def _serialized(result):
    checks,snapshots,peaks=result
    # Dataclass/list equality treats True == 1 == 1.0; this also checks the
    # serialized numeric types of every byte value, snapshot and peak.
    return json.dumps([[asdict(x) for x in checks],[asdict(x) for x in snapshots],peaks],sort_keys=True)


class ResourceReplayCacheTests(unittest.TestCase):
    def _compare(self,rank,rows,events,*,expected_pass=True,mutate=None,other_rank=None):
        with tempfile.TemporaryDirectory() as directory:
            prefix=str(Path(directory)/'resources')
            ranks={'0':rank}
            if other_rank is not None:ranks['1']=other_rank
            for rank_id in ranks:
                Path(prefix+f'.{rank_id}.jsonl').write_text(
                    ''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
            plan=SimpleNamespace(contract={'ranks':ranks,'log_prefix':prefix},tensors=[])
            reference=_frozen_reference();case=_case(rank.get('cores_per_die',1))

            def run(validate,namespace=None):
                local_plan=copy.deepcopy(plan)
                if mutate is None:return validate(local_plan,case,events,gradient_visits=[])
                def stream(path):
                    local_rank=local_plan.contract['ranks'][path.name.split('.')[-2]]
                    for index,row in enumerate(copy.deepcopy(rows)):
                        mutate(index,local_rank)
                        yield row
                if namespace is not None:
                    with patch.dict(namespace,{'_resource_rows':stream}):
                        return validate(local_plan,case,events,gradient_visits=[])
                with patch.object(resource_validation,'_resource_rows',stream):
                    return validate(local_plan,case,events,gradient_visits=[])

            expected=run(reference['validate_resources'],reference)
            actual=run(resource_validation.validate_resources)
            self.assertEqual(actual,expected)
            self.assertEqual(_serialized(actual),_serialized(expected))
            self.assertEqual(actual[0][0].passed,expected_pass,actual[0][0].actual)
            self.assertEqual(len(actual[1]),len(rows)*2*len(ranks))
            return actual

    def test_serial_repeated_108_and_256_core_vectors_keep_every_snapshot(self):
        for cores in (108,256):
            vector=[0]*cores;vector[0]=3;vector[-1]=7
            rank,rows,events=_fixture([vector]*40)
            for node in range(1,41):
                name=f'temporary.{node}';n=rank['nodes'][str(node)]
                n.update(allocations={name:node},ready_outputs=[name],frees=[name],
                    reserve_before=2,reserve_after=-2,local_memory=True)
                rows[node*2-1].update(dram_bytes=4+node,local_memory_owner=node,reserved_bytes=2)
            checks,snapshots,peaks=self._compare(rank,rows,events)
            self.assertEqual(peaks['d0.dram'],44)
            self.assertEqual(peaks['d0.sram_workspace_sum'],10)
            self.assertEqual(peaks[f'd0.c{cores-1}.sram'],7)
            self.assertEqual(len(snapshots),162)

    def test_legacy_legal_overlap_and_return_to_serial(self):
        rank,_,events=_fixture([[2,0,1,0],[0,3,1,0],[2,0,1,0]])
        rows=[_row(0,'initial',0,[0]*4),_row(1,'issue',2,[2,0,1,0]),
              _row(2,'issue',4,[2,3,2,0],sparse=True),_row(1,'callback',5,[0,3,1,0]),
              _row(2,'callback',6,[0]*4,sparse=True),_row(3,'issue',8,[2,0,1,0]),
              _row(3,'callback',9,[0]*4)]
        events[0].callback_tick=5;events[1].callback_tick=6
        events[2].issue_tick=8;events[2].callback_tick=9
        _,_,peaks=self._compare(rank,rows,events)
        self.assertEqual([peaks[f'd0.c{i}.sram'] for i in range(4)],[2,3,2,0])

    def test_bad_workspace_is_checked_for_each_node_despite_equal_vector(self):
        rank,rows,events=_fixture([[2,0,1,0]]*3)
        rank['nodes']['2']['workspace_bytes']=4
        self._compare(rank,rows,events,expected_pass=False)

    def test_shape_negative_private_capacity_and_fake_native_snapshots(self):
        for vector,cores,capacity in [([2,1],4,64),([-1,2,0,0],4,64),([2,0,0,0],4,1)]:
            with self.subTest(vector=vector,cores=cores,capacity=capacity):
                rank,rows,events=_fixture([vector],cores=cores,capacity=capacity)
                self._compare(rank,rows,events,expected_pass=False)
        for sparse in (False,True):
            rank,rows,events=_fixture([[2,0,1,0]]*3)
            rows[3]=_row(2,'issue',4,[2,0,1,0],sparse=sparse)
            if sparse:rows[3]['core_sram_nonzero']=[[0,2],[0,2],[2,1]]
            else:rows[3]['core_sram_bytes'][0]+=1
            self._compare(rank,rows,events,expected_pass=False)

    def test_missing_duplicate_and_illegal_events_do_not_reset_corruption(self):
        for action in ('missing_callback','missing_node','callback_first','duplicate_issue',
                       'duplicate_callback','wrong_tick','extra_initial','unknown_event'):
            rank,rows,events=_fixture([[2,0,1,0]]*2)
            if action=='missing_callback':rows.pop()
            elif action=='missing_node':rows=rows[:3]
            elif action=='callback_first':rows.insert(1,_row(1,'callback',1,[-2,0,-1,0],sram=0))
            elif action=='duplicate_issue':rows.insert(2,copy.deepcopy(rows[1]))
            elif action=='duplicate_callback':rows.insert(3,copy.deepcopy(rows[2]))
            elif action=='wrong_tick':rows[2]['tick']+=10
            elif action=='extra_initial':rows.insert(1,copy.deepcopy(rows[0]))
            else:rows.insert(2,_row(1,'other',2,[2,0,1,0]))
            with self.subTest(action=action):self._compare(rank,rows,events,expected_pass=False)

    def test_float_bool_aliases_and_cancellation_preserve_old_numeric_types(self):
        vectors=[[True,False,0,0],[1,0,0,0],[1.0,0,0,0],[0.0,-0.0,0,0],
                 [2**53,1,0,0],[float(2**53),1.0,0,0],[1,0,0,0]]
        rank,rows,events=_fixture(vectors,capacity=2**54)
        # Prior float workspace arithmetic leaves a float zero; the next large
        # integer workspace then rounds. Preserve the old rejection as well.
        self._compare(rank,rows,events,expected_pass=False)

    def test_capacity_and_core_count_are_rank_local(self):
        rank,rows,events=_fixture([[2,0,1,0]])
        other=copy.deepcopy(rank);other['sram_bytes_per_core']=1
        second_events=[SimpleNamespace(rank=1,node_id=e.node_id,issue_tick=e.issue_tick,
                                       callback_tick=e.callback_tick) for e in events]
        self._compare(rank,rows,events+second_events,other_rank=other,expected_pass=False)

    def test_source_mutation_is_observed_on_issue_and_callback(self):
        shared=[2,0,1,0];rank,rows,events=_fixture([shared]*2)
        rows[3]=_row(2,'issue',4,[5,0,0,0])
        def change_before_second_issue(index,local_rank):
            if index==3:
                local_rank['nodes']['2']['core_workspace_bytes'][:]=[5,0,0,0]
                local_rank['nodes']['2']['workspace_bytes']=5
        self._compare(rank,rows,events,mutate=change_before_second_issue)
        rank,rows,events=_fixture([[2,0,1,0]])
        rows[2]=_row(1,'callback',3,[-3,0,1,0],sram=0)
        def change_before_callback(index,local_rank):
            if index==2:local_rank['nodes']['1']['core_workspace_bytes'][:]=[5,0,0,0]
        self._compare(rank,rows,events,expected_pass=False,mutate=change_before_callback)
        rank,rows,events=_fixture([[2,0,1,0]]*2)
        def change_capacity(index,local_rank):
            if index==3:local_rank['sram_bytes_per_core']=1
        self._compare(rank,rows,events,expected_pass=False,mutate=change_capacity)

    def test_bounded_cache_eviction_does_not_drop_checks_or_peak_updates(self):
        # Force eviction with a small cache during the comparison; the oracle
        # stays unchanged and never uses this class.
        rank,rows,events=_fixture([[i%9,(i*7)%11,0,0] for i in range(80)])
        with patch.object(resource_validation._CoreVectorCache,'_MAX_ENTRIES',4):
            self._compare(rank,rows,events)

    def test_single_core_legacy_path_is_unchanged(self):
        rank,rows,events=_fixture([[4],[7],[4]])
        self._compare(rank,rows,events)


if __name__=='__main__':unittest.main()
