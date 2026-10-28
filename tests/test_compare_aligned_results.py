"""Tiny constructed metadata/gzip fixtures; no native or GPU experiment."""
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=Path(__file__).with_name('compare_aligned_results.py')
if not SCRIPT.is_file():SCRIPT=ROOT/'scripts/compare_aligned_results.py'
spec=importlib.util.spec_from_file_location('compare_aligned_results',SCRIPT)
pair=importlib.util.module_from_spec(spec);spec.loader.exec_module(pair)


def write(path,obj):path.write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')


def seal(root, verification):
    m=json.loads((root/'manifest.json').read_text())
    m['artifact_sha256']={x.relative_to(root).as_posix():pair.sha(x) for x in root.rglob('*') if x.is_file() and x.name!='manifest.json'}
    m['metrics_sha256']=pair.sha(root/'metrics.json');write(root/'manifest.json',m)
    write(verification,dict(schema_version='rlxy.aligned_result_check/1',status='PASS',manifest_sha256=pair.sha(root/'manifest.json'),
        checker_source_sha256='b'*64,checks=[dict(check='constructed_MVP_receipt_not_native',status='PASS')],paper_readiness_promoted_by_checker=False))


def rewrite_table(root,logical,rows):
    path=root/(logical+'.gz');raw=b''.join((json.dumps(row,sort_keys=True)+'\n').encode() for row in rows)
    with gzip.open(path,'wb') as stream:stream.write(raw)
    index=json.loads((root/'table_format.json').read_text()) if (root/'table_format.json').exists() else dict(schema_version='rlxy.aligned_tables/1',tables={})
    index['tables'][logical]=dict(path=path.name,encoding='gzip_jsonl',row_count=len(rows),stored_bytes=path.stat().st_size,
        stored_sha256=pair.sha(path),uncompressed_sha256=hashlib.sha256(raw).hexdigest(),uncompressed_bytes=len(raw))
    write(root/'table_format.json',index)


def read_table(root,logical):
    return [json.loads(line) for line in gzip.open(root/(logical+'.gz'),'rt')]


def rewrite_idle_rank(root,rank,data):
    path=root/'inputs'/f'resources.{rank}.json';raw=pair.canonical(data).encode();path.write_bytes(raw);digest=pair.sha(path)
    index=json.loads((root/'resource_contract_format.json').read_text())
    index['native_rank_contracts'][str(rank)]=dict(path=path.relative_to(root).as_posix(),encoding='minified_json',
        stored_sha256=digest,stored_bytes=len(raw),canonical_semantic_sha256=digest,
        uncompressed_canonical_sha256=digest,uncompressed_canonical_bytes=len(raw))
    write(root/'resource_contract_format.json',index)
    if (root/'manifest.json').exists():
        m=json.loads((root/'manifest.json').read_text());m['inputs']['resources.'+str(rank)]=dict(path=path.relative_to(root).as_posix(),sha256=digest)
        write(root/'manifest.json',m)


def idle_resources(root,markers):
    write(root/'resource_contract_format.json',dict(schema_version='rlxy.resource_contract_format/1',native_rank_contracts={}))
    index=dict(schema_version='1.0',rank_files={},log_prefix='astra_logs/resources');inputs={}
    if markers:(root/'astra_logs').mkdir()
    vector=[0]*256;key=hashlib.sha256(pair.canonical(vector).encode()).hexdigest()
    for rank in range(48,54) if markers else ():
        nodes={};base=0
        for op in markers:
            nodes[str(op['node_id'])]=dict(allocations={},ready_outputs=[],frees=[],read_delta=dict(base_node=base,add=[],remove=[]),
                workspace_bytes=0,reserve_before=0,reserve_after=0,local_memory=False,sram_capacity=16,core_workspace_profile=key)
            base=op['node_id']
        data=dict(initial_objects={},nodes=nodes,dram_capacity=64000000000,cores_per_die=256,sram_bytes_per_core=16,
            read_id_symbols=[],core_workspace_profiles={key:vector})
        rewrite_idle_rank(root,rank,data);path='inputs/resources.'+str(rank)+'.json'
        inputs['resources.'+str(rank)]=dict(path=path,sha256=pair.sha(root/path));index['rank_files'][str(rank)]=path
        rows=[dict(event='initial',node_id=0,tick=0,dram_bytes=0,sram_bytes=0,reserved_bytes=0,local_memory_owner=0,core_sram_nonzero=[])]
        for offset,op in enumerate(markers):
            for kind,tick in [('issue',offset*1000),('callback',(offset+1)*1000)]:
                rows.append(dict(event=kind,node_id=op['node_id'],tick=tick,dram_bytes=0,sram_bytes=0,reserved_bytes=0,local_memory_owner=0,core_sram_nonzero=[]))
        (root/'astra_logs'/f'resources.{rank}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    write(root/'inputs/resource_index.json',index)
    inputs['resource_index']=dict(path='inputs/resource_index.json',sha256=pair.sha(root/'inputs/resource_index.json'))
    return inputs


def actual_maps(case,policy,platform):
    ids=[t['trajectory_id'] for t in case['workload']['trajectories']];tr=policy['train_tp'];ge=policy['rollout_tp'];dp=48//tr
    group={t['trajectory_id']:t['group_id'] for t in case['workload']['trajectories']};count=len(ids)//dp
    placement=[]
    for i in range(dp):
        local=ids[i*count:(i+1)*count]
        placement.append(dict(replica=i,tp_ranks=list(range(i*tr,(i+1)*tr)),groups=list(dict.fromkeys(group[t] for t in local)),
            trajectories=local,rollout_tp_groups=[list(range(i*tr+j*ge,i*tr+(j+1)*ge)) for j in range(tr//ge)],
            global_loss_denominator=12288,distributed_optimizer_shard=[i*4,(i+1)*4],
            hardware={'cores_per_die':108 if platform=='GPU' else 256}))
    templates={};keys={}
    for width in (1,2):
        cert=dict(schema_version='rlxy.local_chain_contraction/1',logical_operation_count=2,native_operation_count=2,
            strict_per_rank_serialization_proved=True,hardware_workspace_marker=108 if platform=='GPU' else 256,
            blocks=[dict(native_node_id=n,logical_node_ids=[n],participants=list(range(width)),
                duration_prefix_model_ns=[0,1 if platform=='GPU' else 2]) for n in (1,2)])
        key=hashlib.sha256(pair.canonical(cert).encode()).hexdigest();templates[key]=cert;keys[width]=key
    invocations=[];cursor=10 if platform=='GPU' else 11
    def add(label,physical,source):
        nonlocal cursor
        invocations.append(dict(label=label,physical=physical,source_to_canonical_trajectories={uid:'T'+str(i) for i,uid in enumerate(source)},
            certificate_sha256=keys[len(physical)],global_to_local={str(cursor):1,str(cursor+1):2}))
        cursor+=3
    for rid in range(1,case['rounds']+1):
        for i,row in enumerate(placement):
            for j,ps in enumerate(row['rollout_tp_groups']):add(f'r{rid}.generation{i}.{j}',ps,row['trajectories'][j::tr//ge])
        for phase in ('reward','old_policy','reference','advantage','training'):
            for i,row in enumerate(placement):
                mb=policy['logprob_micro_batch_trajectories'] if phase in ('old_policy','reference') else policy['train_micro_batch_trajectories'] if phase=='training' else count
                for start in range(0,count,mb):add(f'r{rid}.{phase}{i}.{start}',row['tp_ranks'],row['trajectories'][start:start+mb])
    return placement,dict(component_templates=templates,component_invocations=invocations)


def fixture(root,platform,verification,rounds=1):
    root.mkdir();(root/'inputs').mkdir();allocated=48 if platform=='GPU' else 54
    policy=dict(train_tp=2,rollout_tp=1,prompt_batch_size=48,responses_per_prompt=2,ppo_epochs=1,
        train_micro_batch_trajectories=2,logprob_micro_batch_trajectories=2)
    fp=hashlib.sha256(pair.canonical(policy).encode()).hexdigest()
    case=dict(case_id='tiny-pair-checker-fixture',rounds=rounds,model=dict(layers=1),
        workload=dict(round_id=1,behavior_version=0,group_id='global',trajectories=[dict(trajectory_id='t'+str(i),
            group_id='g'+str(i//2),response_token_ids=[1]*128) for i in range(96)]),strategy=dict(schedule='strict_synchronous'))
    audit=dict(operator_contract_id=pair.OPERATOR_CONTRACT,policy_fingerprint=fp,
        distributed_optimizer=dict(dp_size=24,tp_size=2,shard_elements=4,padded_elements=96))
    metrics=dict(platform=platform,active_devices=48,dies=allocated,inactive_devices=list(range(48,allocated)),policy=policy,
        policy_fingerprint=fp,aligned_protocol=audit,trajectory_count=96,response_tokens=12288,rounds=rounds,optimizer_updates=rounds,
        actor_version_end=rounds,reference_version_end=0,tensor_parallel=2,data_parallel=24,rollout_tensor_parallel=1,rollout_data_parallel=48,
        total_simulated_time_ns=200 if platform=='GPU' else 100,memory_peak_bytes_by_location={'d0.dram':50,'d0.c0.sram':5},memory_peak_scope='constructed fixture')
    for flag in ['paper_performance_ready','hardware_calibrated','numerical_training','full_verl_runtime_reproduced']:metrics[flag]=False
    write(root/'metrics.json',metrics);write(root/'resolved_config.json',dict(case=case,policy=policy,active_devices=48,allocated_devices=allocated))
    placement,mapping=actual_maps(case,policy,platform)
    write(root/'placement.json',placement);write(root/'aligned_contract.json',dict(audit=audit,**mapping))
    stages={f'r{rid}.generation':dict(span_model_ns=metrics['total_simulated_time_ns']) for rid in range(1,rounds+1)}
    shape=dict(m=2,n=3,k=4,batches=1,a_bytes=2,b_bytes=2,output_bytes=2,accumulator_bytes=4,direction='forward')
    key=hashlib.sha256(pair.canonical(shape).encode()).hexdigest()
    write(root/'actual_gemm_census.json',dict(policy_sha256=fp,shapes={key:shape},signatures={key:dict(signature_sha256=key,problem=shape,occurrences=48,logical_matmul_occurrences=48,weighted_flops=2304)},
        totals=dict(unique_signatures=1,rank_local_matmul_occurrences=48,logical_matmul_occurrences=48,weighted_flops=2304)))
    ops=[dict(kind='compute',metadata={},name='hardware-dependent-compute',duration_ns=metrics['total_simulated_time_ns'],participant_ranks=[0]),
        dict(round_id=1,phase='dp_sync',kind='collective',participant_ranks=list(range(0,48,2)),src_rank=None,dst_rank=None,collective='REDUCE_SCATTER',payload_bytes=96,metadata={'transport_payload_bytes':96}),
        dict(round_id=1,phase='barrier',kind='p2p',participant_ranks=[0,1],src_rank=0,dst_rank=1,collective=None,payload_bytes=4,metadata={})]
    profile=dict(duration_ns=1,source='explicit_control_assumption');profile_key=hashlib.sha256(pair.canonical(profile).encode()).hexdigest()
    write(root/'cost_profiles.json',{profile_key:profile});markers=[]
    if platform=='WAFER':
        for offset in range(rounds):
            rid=offset+1;nid=offset+1;deps=[] if offset==0 else [offset]
            markers.append(dict(node_id=nid,op_id=f'r{rid}.inactive_rank_marker',round_id=rid,phase='standby',name='inactive_rank_marker',
                kind='compute',duration_ns=1,flops=0,vector_ops=0,tensor_bytes=0,payload_bytes=0,kernel_spec=None,
                src_rank=None,dst_rank=None,collective=None,participant_ranks=list(range(48,54)),dependencies=deps,
                dependencies_by_rank={str(rank):deps for rank in range(48,54)},metadata=dict(idle_rank_marker=True,
                    colocation_stage='standby',execution_domain='device',cost_profile_sha256=profile_key)))
            stages[f'r{rid}.standby']=dict(start_tick=offset*1000,end_tick=(offset+1)*1000,span_model_ns=1.0)
    write(root/'global_stage_times.json',stages);rewrite_table(root,'operations.jsonl',markers+ops)
    active_end=metrics['total_simulated_time_ns']*1000
    events=[dict(rank=0,node_id=99,op_id='active',round_id=rounds,phase='generation',node_type=4,
        issue_tick=0,callback_tick=active_end,duration_ticks=active_end,issue_ns=0,callback_ns=active_end/1000,duration_ns=active_end/1000)]
    for offset,op in enumerate(markers):
        for rank in range(48,54):events.append(dict(rank=rank,node_id=op['node_id'],op_id=op['op_id'],round_id=op['round_id'],
            phase='standby',node_type=4,issue_tick=offset*1000,callback_tick=(offset+1)*1000,duration_ticks=1000,
            issue_ns=float(offset),callback_ns=float(offset+1),duration_ns=1.0))
    rewrite_table(root,'events.jsonl',events)
    native_inputs=idle_resources(root,markers)
    inputs={}
    for name,obj in [('case.json',case),('model_config.json',case['model']),('policy.json',policy),('workload_provenance.json',{'kind':'constructed'})]:
        write(root/'inputs'/name,obj);inputs[name]=dict(path='inputs/'+name,sha256=pair.sha(root/'inputs'/name))
    inputs.update(native_inputs)
    write(root/'manifest.json',dict(status='success',exit_status=0,checks_failed=0,case_id=case['case_id'],inputs=inputs,
        software=dict(rl_xy_version='0.9.2',rl_xy_source_tree_sha256='a'*64,container_image_id='sha256:'+'c'*64,astrasim_binary_sha256='d'*64,astrasim_commit='e'*40)))
    seal(root,verification)


class PairCheckerTests(unittest.TestCase):
    def compare_fixture(self,mutate=None,mutate_both=None,rounds=1):
        with tempfile.TemporaryDirectory()as name:
            p=Path(name);g,w=p/'GPU',p/'WAFER';gv,wv=p/'g-verified.json',p/'w-verified.json'
            fixture(g,'GPU',gv,rounds=rounds);fixture(w,'WAFER',wv,rounds=rounds)
            if mutate:mutate(w,wv)
            if mutate_both:mutate_both(g,gv);mutate_both(w,wv)
            return pair.compare(g,w,gv,wv)

    def assert_rejected(self,result):
        self.assertEqual(result['status'],'REJECTED_PAIR',result);self.assertFalse(result['conditional_ratio_released'])
        self.assertIsNone(result['conditional_gpu_time_over_wafer_time']);self.assertFalse(result['paper_performance_ready'])

    def test_valid_gzip_pair_ignores_compute_time_and_idle_marker(self):
        result=self.compare_fixture();self.assertEqual(result['status'],'PASS_CONDITIONAL_PAIR',result)
        self.assertEqual(result['conditional_gpu_time_over_wafer_time'],2);self.assertFalse(result['paper_performance_ready'])

    def test_nonempty_actual_dispatch_local_blocks_and_software_placement_are_certified(self):
        result=self.compare_fixture();self.assertEqual(result['status'],'PASS_CONDITIONAL_PAIR',result)
        by_name={r['check']:r for r in result['checks']}
        detail=by_name['identical_actual_ordered_source_dispatch_and_local_block_coverage']['detail']
        self.assertEqual((detail['component_invocations'],detail['full_source_UIDs']),(240,96))
        self.assertEqual(detail['each_source_visits_per_phase_per_round'],1)
        self.assertFalse(detail['global_node_keys_cross_platform_compared'])
        self.assertTrue(detail['local_block_IDs_and_order_compared'])
        self.assertEqual(by_name['identical_actual_software_placement']['detail']['replicas'],24)

    def test_missing_mapping_or_empty_legacy_structures_cannot_fallback_skip(self):
        def missing(p,v):
            d=json.loads((p/'aligned_contract.json').read_text());d['component_invocations'][0].pop('source_to_canonical_trajectories')
            write(p/'aligned_contract.json',d);seal(p,v)
        def empty(p,v):
            d=json.loads((p/'aligned_contract.json').read_text());d['component_invocations']=[]
            write(p/'aligned_contract.json',d);seal(p,v)
        def no_placement(p,v):
            (p/'placement.json').unlink();seal(p,v)
        for change in (missing,empty,no_placement):
            with self.subTest(change=change.__name__):self.assert_rejected(self.compare_fixture(change))

    def test_single_known_UID_change_with_unchanged_counts_is_rejected(self):
        def mutate(p,v):
            d=json.loads((p/'aligned_contract.json').read_text());mapping=d['component_invocations'][0]['source_to_canonical_trajectories']
            mapping['t1']=mapping.pop('t0');write(p/'aligned_contract.json',d);seal(p,v)
        self.assert_rejected(self.compare_fixture(mutate))
        # Two equally wrong bundles still violate the declared source dispatch.
        self.assert_rejected(self.compare_fixture(mutate_both=mutate))

    def test_one_physical_rank_or_invocation_order_change_is_rejected(self):
        def rank(p,v):
            d=json.loads((p/'aligned_contract.json').read_text());d['component_invocations'][0]['physical']=[1]
            write(p/'aligned_contract.json',d);seal(p,v)
        def order(p,v):
            d=json.loads((p/'aligned_contract.json').read_text());d['component_invocations'][:2]=reversed(d['component_invocations'][:2])
            write(p/'aligned_contract.json',d);seal(p,v)
        for change in (rank,order):
            with self.subTest(change=change.__name__):self.assert_rejected(self.compare_fixture(change))

    def test_placement_group_UID_rollout_and_DO_interval_drift_is_rejected(self):
        for field,replacement in [('groups',['wrong']),('trajectories',['t1','t0','t2','t3']),
                ('rollout_tp_groups',[[1],[0]]),('distributed_optimizer_shard',[0,5])]:
            def mutate(p,v):
                rows=json.loads((p/'placement.json').read_text());rows[0][field]=replacement
                write(p/'placement.json',rows);seal(p,v)
            with self.subTest(field=field):self.assert_rejected(self.compare_fixture(mutate))

    def test_hardware_area_annotation_is_explicitly_separate_from_software_mapping(self):
        def mutate(p,v):
            rows=json.loads((p/'placement.json').read_text());rows[0]['hardware']={'core_area_mm2':999.5,'cores_per_die':256}
            write(p/'placement.json',rows);seal(p,v)
        result=self.compare_fixture(mutate);self.assertEqual(result['status'],'PASS_CONDITIONAL_PAIR',result)
        def unknown(p,v):
            rows=json.loads((p/'placement.json').read_text());rows[0]['unreviewed_software_dispatch_mode']='other'
            write(p/'placement.json',rows);seal(p,v)
        self.assert_rejected(self.compare_fixture(unknown))

    def test_local_block_value_coverage_and_order_cannot_be_hidden_by_global_key_offset(self):
        def missing(p,v):
            d=json.loads((p/'aligned_contract.json').read_text());m=d['component_invocations'][0]['global_to_local'];m.pop(next(iter(m)))
            write(p/'aligned_contract.json',d);seal(p,v)
        def order(p,v):
            d=json.loads((p/'aligned_contract.json').read_text());m=d['component_invocations'][0]['global_to_local'];keys=sorted(m,key=int)
            m[keys[0]],m[keys[1]]=m[keys[1]],m[keys[0]];write(p/'aligned_contract.json',d);seal(p,v)
        def duplicate(p,v):
            d=json.loads((p/'aligned_contract.json').read_text());entries=d['component_invocations'];entries[1]['global_to_local']=entries[0]['global_to_local']
            write(p/'aligned_contract.json',d);seal(p,v)
        for change in (missing,order,duplicate):
            with self.subTest(change=change.__name__):self.assert_rejected(self.compare_fixture(change))

    def test_duplicate_canonical_UID_and_source_count_inconsistency_are_rejected(self):
        def duplicate(p,v):
            d=json.loads((p/'aligned_contract.json').read_text());m=d['component_invocations'][0]['source_to_canonical_trajectories']
            m['t2']='T0';write(p/'aligned_contract.json',d);seal(p,v)
        def count(p,v):
            d=json.loads((p/'metrics.json').read_text());d['trajectory_count']=95;write(p/'metrics.json',d);seal(p,v)
        for change in (duplicate,count):
            with self.subTest(change=change.__name__):self.assert_rejected(self.compare_fixture(change))

    def test_verified_standby_is_separate_and_total_model_MAX_is_unchanged(self):
        result=self.compare_fixture();self.assertEqual(result['status'],'PASS_CONDITIONAL_PAIR',result)
        self.assertEqual(result['verified_standby_envelopes'],{'GPU':{},'WAFER':{'r1.standby':{'start_tick':0,'end_tick':1000,'span_model_ns':1.}}})
        self.assertNotIn('r1.standby',result['stage_spans'])
        self.assertEqual(result['model_time_ns'],{'GPU':200,'WAFER':100})
        self.assertEqual(result['conditional_gpu_time_over_wafer_time'],2)
        check=next(c for c in result['checks'] if c['check']=='only_declared_six_idle_wafer_standby_exception')
        proof=check['detail']['WAFER'];self.assertEqual(proof['idle_ranks'],list(range(48,54)))
        self.assertEqual((proof['marker_count'],proof['idle_event_count']),(1,6))
        self.assertTrue(all(r['resource_rows']==3 for r in proof['resources'].values()))
        self.assertTrue(result['total_model_time_includes_original_allocated_rank_MAX'])

    def test_one_serial_idle_marker_per_round_is_supported_without_phase_work(self):
        result=self.compare_fixture(rounds=2);self.assertEqual(result['status'],'PASS_CONDITIONAL_PAIR',result)
        self.assertEqual(set(result['verified_standby_envelopes']['WAFER']),{'r1.standby','r2.standby'})
        self.assertEqual(result['verified_standby_envelopes']['WAFER']['r2.standby'],dict(start_tick=1000,end_tick=2000,span_model_ns=1.))
        self.assertEqual(set(result['stage_spans']),{'r1.generation','r2.generation'})

    def test_active_rank_hidden_work_non1ns_and_missing_marker_are_rejected(self):
        mutations=[lambda op:op.update(participant_ranks=[47,49,50,51,52,53]),
            lambda op:op.update(duration_ns=2),lambda op:op['metadata'].update(idle_rank_marker=False),
            lambda op:op.update(flops=1),lambda op:op.update(vector_ops=1),lambda op:op.update(payload_bytes=4),
            lambda op:op.update(read_bytes=1),lambda op:op.update(kernel_spec={'matmuls':[{'m':1}]}),
            lambda op:op.update(name='hidden_compute')]
        for index,change in enumerate(mutations):
            def mutate(p,v):
                ops=read_table(p,'operations.jsonl');change(ops[0]);rewrite_table(p,'operations.jsonl',ops);seal(p,v)
            with self.subTest(mutation=index):self.assert_rejected(self.compare_fixture(mutate))
        def missing(p,v):
            ops=read_table(p,'operations.jsonl');rewrite_table(p,'operations.jsonl',ops[1:]);seal(p,v)
        self.assert_rejected(self.compare_fixture(missing))

    def test_wrong_idle_event_clock_type_missing_duplicate_or_rank_is_rejected(self):
        def shift(event):event.update(issue_tick=1,callback_tick=1001,issue_ns=.001,callback_ns=1.001)
        for change in [shift,lambda e:e.update(node_type=7),lambda e:e.update(rank=47),lambda e:e.update(duration_ticks=999)]:
            def mutate(p,v):
                events=read_table(p,'events.jsonl');change(events[1]);rewrite_table(p,'events.jsonl',events);seal(p,v)
            with self.subTest(change=change):self.assert_rejected(self.compare_fixture(mutate))
        def active(p,v):
            events=read_table(p,'events.jsonl');events[0]['phase']='standby';rewrite_table(p,'events.jsonl',events);seal(p,v)
        self.assert_rejected(self.compare_fixture(active))
        for duplicate in (False,True):
            def mutate(p,v):
                events=read_table(p,'events.jsonl')
                if duplicate:events.append(copy.deepcopy(events[1]))
                else:events.pop(1)
                rewrite_table(p,'events.jsonl',events);seal(p,v)
            with self.subTest(duplicate=duplicate):self.assert_rejected(self.compare_fixture(mutate))

    def test_nonempty_idle_resource_state_or_hidden_node_is_rejected_after_resealing(self):
        changes=[lambda d:d['initial_objects'].update(hidden=4),lambda d:d['nodes']['1']['allocations'].update(hidden=4),
            lambda d:d['nodes']['1'].update(ready_outputs=['hidden']),lambda d:d['nodes']['1'].update(frees=['hidden']),
            lambda d:d['nodes']['1'].update(workspace_bytes=1),lambda d:d['nodes']['1']['read_delta'].update(add=[0]),
            lambda d:d['nodes'].update({'2':copy.deepcopy(d['nodes']['1'])})]
        for index,change in enumerate(changes):
            def mutate(p,v):
                data=json.loads((p/'inputs/resources.48.json').read_text());change(data);rewrite_idle_rank(p,48,data);seal(p,v)
            with self.subTest(change=index):self.assert_rejected(self.compare_fixture(mutate))
        def workspace(p,v):
            data=json.loads((p/'inputs/resources.48.json').read_text());vector=[0]*256;vector[0]=1
            key=hashlib.sha256(pair.canonical(vector).encode()).hexdigest();data['core_workspace_profiles']={key:vector}
            data['nodes']['1']['core_workspace_profile']=key;rewrite_idle_rank(p,48,data);seal(p,v)
        self.assert_rejected(self.compare_fixture(workspace))

    def test_nonzero_or_extra_actual_idle_resource_log_is_rejected(self):
        for extra in (False,True):
            def mutate(p,v):
                path=p/'astra_logs/resources.48.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
                if extra:rows.append({**rows[-1],'node_id':99})
                else:rows[1]['dram_bytes']=1
                path.write_text(''.join(json.dumps(row)+'\n' for row in rows));seal(p,v)
            with self.subTest(extra=extra):self.assert_rejected(self.compare_fixture(mutate))

    def test_unknown_extra_stages_or_standby_without_exact_envelope_are_not_skipped(self):
        for key in ('r1.mystery','r9.standby'):
            def mutate(p,v):
                stages=json.loads((p/'global_stage_times.json').read_text());stages[key]=dict(start_tick=0,end_tick=1000,span_model_ns=1.)
                write(p/'global_stage_times.json',stages);seal(p,v)
            with self.subTest(stage=key):self.assert_rejected(self.compare_fixture(mutate))
        def missing(p,v):
            stages=json.loads((p/'global_stage_times.json').read_text());stages.pop('r1.standby');write(p/'global_stage_times.json',stages);seal(p,v)
        self.assert_rejected(self.compare_fixture(missing))
        def both(p,v):
            stages=json.loads((p/'global_stage_times.json').read_text());stages['r1.standby']=dict(start_tick=0,end_tick=1000,span_model_ns=1.)
            write(p/'global_stage_times.json',stages);seal(p,v)
        self.assert_rejected(self.compare_fixture(mutate_both=both))

    def test_policy_drift_rejected(self):
        def mutate(p,v):
            d=json.loads((p/'metrics.json').read_text());d['policy']['train_tp']=1;write(p/'metrics.json',d);seal(p,v)
        self.assert_rejected(self.compare_fixture(mutate))

    def test_workload_or_input_SHA_drift_rejected(self):
        def mutate(p,v):
            d=json.loads((p/'resolved_config.json').read_text());d['case']['workload']['trajectories'][0]['response_token_ids']=[2];write(p/'resolved_config.json',d);seal(p,v)
        self.assert_rejected(self.compare_fixture(mutate))

    def test_GEMM_or_network_semantic_drift_rejected(self):
        def census(p,v):
            d=json.loads((p/'actual_gemm_census.json').read_text());next(iter(d['signatures'].values()))['occurrences']+=1;write(p/'actual_gemm_census.json',d);seal(p,v)
        def network(p,v):
            path=p/'operations.jsonl.gz';ops=[json.loads(x)for x in gzip.open(path,'rt')];next(op for op in ops if op['kind']=='collective')['metadata']['transport_payload_bytes']=192
            raw=b''.join((json.dumps(o,sort_keys=True)+'\n').encode()for o in ops)
            with gzip.open(path,'wb')as f:f.write(raw)
            d=json.loads((p/'table_format.json').read_text());row=d['tables']['operations.jsonl'];row.update(stored_sha256=pair.sha(path),stored_bytes=path.stat().st_size,uncompressed_sha256=hashlib.sha256(raw).hexdigest(),uncompressed_bytes=len(raw));write(p/'table_format.json',d);seal(p,v)
        for change in [census,network]:
            with self.subTest(change=change.__name__):self.assert_rejected(self.compare_fixture(change))

    def test_unverified_stale_or_legacy_bundle_rejected(self):
        def no_pass(p,v):
            d=json.loads(v.read_text());d['status']='FAIL';write(v,d)
        def stale(p,v):
            d=json.loads(v.read_text());d['manifest_sha256']='f'*64;write(v,d)
        def legacy(p,v):
            d=json.loads((p/'manifest.json').read_text());d['software']['rl_xy_version']='0.9.1';write(p/'manifest.json',d);seal(p,v)
        for change in [no_pass,stale,legacy]:
            with self.subTest(change=change.__name__):self.assert_rejected(self.compare_fixture(change))


if __name__=='__main__':unittest.main()
