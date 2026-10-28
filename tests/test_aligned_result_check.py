"""Constructed checker fixtures; no native simulation or performance claims."""
from dataclasses import asdict
import importlib.util
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('check_aligned_result',ROOT/'scripts/check_aligned_result.py')
checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
from rl_xy.evidence_io import write_gzip_json,write_minified_json,write_aligned_tables
from rl_xy.baseline_contract import BaselinePolicy
from rl_xy.types import Operation,AstraEvent,AstraRankSummary,CheckResult


def write(p,x):p.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')


def reseal(root):
    m=json.loads((root/'manifest.json').read_text())
    m['artifact_sha256']={p.relative_to(root).as_posix():checker.sha(p)for p in root.rglob('*')if p.is_file()and p.name!='manifest.json'}
    m['metrics_sha256']=checker.sha(root/'metrics.json');write(root/'manifest.json',m)


def bundle(root):
    (root/'inputs').mkdir();(root/'astra_logs').mkdir()
    policy=asdict(BaselinePolicy(prompt_batch_size=2,ppo_mini_batch_prompts=2,train_tp=1,
        rollout_tp=1,use_kl_loss=True,kl_coefficient=.001));fp=checker.semantic(policy,True)
    case={'rounds':1,'workload':{'round_id':1,'behavior_version':0,
        'trajectories':[{'trajectory_id':f'T{i}','prompt_length':4,'response_length':3}for i in range(4)]}}
    ops=[];events=[];last={};stages=['weight_reshard','generation','reward','old_policy','reference','advantage','train','dp_sync','clip','optimizer','optimizer_sync','publish']
    for stage in stages:
        for ps in ([(0,),(1,)]if stage=='optimizer'else[(0,1)]):
            n=len(ops)+1;name='Adam_owned_shard_'+str(ps[0])if stage=='optimizer'else'publish_actor_version'if stage=='publish'else stage
            meta={'colocation_stage':stage}
            if stage in ['optimizer','publish']:meta.update(actor_version_in=0,actor_version_out=1)
            if stage=='publish':meta.update(reference_version_in=0,reference_version_out=0)
            kind='collective'if stage in ['dp_sync','optimizer_sync']else'compute'
            deps={r:(last[r],)if r in last else()for r in ps};opid='comp.local'if stage=='generation'else'r1.'+name
            op=Operation(n,opid,1,stage,name,kind,duration_ns=1 if kind=='compute'else 0,
                payload_bytes=16 if kind=='collective'else 0,collective='REDUCE_SCATTER'if stage=='dp_sync'else'ALL_GATHER'if kind=='collective'else None,
                participant_ranks=ps,dependencies_by_rank=deps,dependencies=tuple(sorted({d for ds in deps.values()for d in ds})),metadata=meta)
            ops.append(op)
            for rank in ps:
                events.append(AstraEvent(rank,n,opid,1,stage,4 if kind=='compute'else 7,(n-1)*1000,n*1000,1000,n-1,n,1));last[rank]=n
    events.sort(key=lambda e:(e.rank,e.node_id))
    checks=[CheckResult('constructed_fixture',True,'all-pass fixture',True,'not a real simulator result','unit')]
    summaries=[AstraRankSummary(r,last[r]*1000,0,last[r]*1000,1,1)for r in range(2)]
    write_aligned_tables(root,operations=ops,events=events,summaries=summaries,objects=[],snapshots=[],transfers=[],checks=checks)
    ranks={}
    for rank in range(2):
        previous=0;nodes={}
        for op in ops:
            if rank in op.participant_ranks:
                nodes[str(op.node_id)]={'read_delta':{'base_node':previous,'add':[],'remove':[]},'allocations':{},'ready_outputs':[],'frees':[],'workspace_bytes':0};previous=op.node_id
        ranks[str(rank)]={'nodes':nodes,'initial_objects':{},'dram_capacity':1000}
    compact={'schema_version':'1.0','log_prefix':'/remote/result/astra_logs/resources','ranks':ranks}
    native={}
    for rank,value in ranks.items():
        p=root/'inputs'/f'resources.{rank}.json';native[rank]={'path':p.relative_to(root).as_posix(),**write_minified_json(p,value)}
    audit=write_gzip_json(root/'resource_contract.json.gz',compact)
    write(root/'resource_contract_format.json',{'schema_version':'rlxy.resource_contract_format/1','audit_contract':{'path':'resource_contract.json.gz',**audit},'native_rank_contracts':native,'native_index_path':'inputs/resource_index.json'})
    write(root/'inputs/resource_index.json',{'rank_files':{rank:f'/remote/result/inputs/resources.{rank}.json'for rank in ranks},'log_prefix':compact['log_prefix']})
    cert={'blocks':[{'native_node_id':1,'logical_node_ids':[1],'participants':[0,1],'duration_prefix_model_ns':[0,1]}],
        'logical_operations_sha256':'a'*64,'logical_resources':{'original_contract_sha256':'b'*64}}
    ck=checker.semantic(cert,True);sig={'certificate_sha256':ck,'blocks_sha256':checker.semantic(cert['blocks'],True),
        'logical_operations_sha256':'a'*64,'logical_contract_sha256':'b'*64,'native_operations_sha256':'c'*64,'native_contract_sha256':'d'*64};sk=checker.semantic(sig,True)
    audit={'active_devices':2,'allocated_devices':2,'idle_devices':[],'policy_fingerprint':fp}
    pre={'status':'VALID_CONFIGURATION_RUNTIME_ALIGNED_COST_PENDING','runtime_alignment_ready':True,'policy':policy,'policy_sha256':fp}
    write(root/'policy_preflight.json',pre);write(root/'actual_gemm_census.json',{'policy_sha256':fp})
    write(root/'aligned_contract.json',{'audit':audit,'policy_preflight':pre,'component_templates':{ck:cert},'component_invocations':[{'label':'comp','certificate_sha256':ck,'physical':[0,1],'global_to_local':{'2':1}}]})
    proof={'status':'PASS','template_sha256':sk,'physical_ranks':[0,1],'logical_operations_sha256':'a'*64,'logical_events_represented':2,'expanded_logical_events':2,'mode':'full_template_replay','clock_bound_to_expanded_native_events':True}
    for flag in ['all_native_macro_clocks_validated','network_issue_callback_ticks_preserved','exact_compute_durations_validated','strict_per_rank_serialization_validated','local_dependencies_validated']:proof[flag]=True
    write(root/'component_logical_proofs.json',{'comp':proof})
    write(root/'component_template_signature_guard.json',{'status':'PASS','every_invocation_native_macro_clocks_validated':True,'unique_templates':1,'full_template_replays':1,'reused_serial_template_invocations':0,'invocations':1,'template_signatures':{sk:sig}})
    metrics={'paper_performance_ready':False,'numerical_training':False,'hardware_calibrated':False,'full_verl_runtime_reproduced':False,
        'policy':policy,'policy_fingerprint':fp,'aligned_protocol':audit,'rounds':1,'optimizer_updates':1,'actor_version_end':1,'reference_version_end':0,
        'trajectory_count':4,'response_tokens':12,'active_devices':2,'dies':2,'data_parallel':2,'rollout_data_parallel':2,'tensor_parallel':1,'rollout_tensor_parallel':1,'inactive_devices':[],
        'operation_count':len(ops),'operation_rank_event_count':len(events),'total_simulated_time_ns':max(e.callback_tick for e in events)/1000,
        'component_invocation_proof_count':1,'component_logical_replay_count':1,'component_reused_serial_proof_count':0,'logical_component_event_count':2,'materialized_logical_component_event_count':2}
    write(root/'metrics.json',metrics);write(root/'summary.json',{'status':'PASS','checks_passed':1,'checks_failed':0,'metrics':metrics})
    write(root/'status.json',{'status':'success'});write(root/'checks.json',{'checks':[asdict(x)for x in checks]})
    write(root/'resolved_config.json',{'policy':policy,'case':case,'active_devices':2,'allocated_devices':2})
    write(root/'inputs/policy.json',policy)
    times={'started_at_utc':'2026-10-28T00:00:00Z','finished_at_utc':'2026-10-28T00:00:01Z'}
    write(root/'astra_command.json',{'argv':['constructed-fixture'],'exit_status':0,**times})
    for name in ['astra_stdout.log','astra_stderr.log','astra_logs/log.log']:(root/name).write_text('constructed checker fixture; not native execution\n'if 'stderr'not in name else'')
    write(root/'manifest.json',{'status':'success','exit_status':0,'checks_passed':1,'checks_failed':0,'command':['constructed-fixture'],
        'started_at_utc':times['started_at_utc'],'simulator_finished_at_utc':times['finished_at_utc'],
        'software':{'astrasim_commit':checker.PINNED_ASTRA,'container_image_tag':'checker-fixture','container_image_id':'sha256:constructed'},'execution_host':{'inside_docker':True},
        'inputs':{'policy':{'path':'/remote/result/inputs/policy.json','sha256':checker.sha(root/'inputs/policy.json')}},
        'headline_metrics':{'operation_count':len(ops),'total_simulated_time_ns':metrics['total_simulated_time_ns']}})
    reseal(root);return ops,events,checks,summaries


def two_instances_same_semantic_signature(root):
    aligned=json.loads((root/'aligned_contract.json').read_text())
    original=aligned['component_invocations'][0]
    aligned['component_invocations'].append({**original,'label':'comp2','global_to_local':{'3':1}})
    write(root/'aligned_contract.json',aligned)
    proofs=json.loads((root/'component_logical_proofs.json').read_text())
    proofs['comp2']=dict(proofs['comp']);write(root/'component_logical_proofs.json',proofs)
    guard=json.loads((root/'component_template_signature_guard.json').read_text())
    guard.update(unique_templates=2,full_template_replays=2,invocations=2)
    write(root/'component_template_signature_guard.json',guard)
    metrics=json.loads((root/'metrics.json').read_text())
    metrics.update(component_invocation_proof_count=2,component_logical_replay_count=2,
        logical_component_event_count=4,materialized_logical_component_event_count=4)
    write(root/'metrics.json',metrics)
    summary=json.loads((root/'summary.json').read_text());summary['metrics']=metrics;write(root/'summary.json',summary)
    reseal(root)


class AlignedCheckerTests(unittest.TestCase):
    def run_fixture(self,mutate=None):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);values=bundle(root)
            if mutate:mutate(root,values)
            return checker.verify_bundle(root)
    def test_complete_compressed_fixture_passes_without_promoting_paper(self):
        r=self.run_fixture();self.assertEqual(r['status'],'PASS',r);self.assertFalse(r['paper_performance_ready_from_input']);self.assertFalse(r['paper_readiness_promoted_by_checker']);self.assertTrue(r['input_path_remaps'])
    def test_missing_artifact_SHA_fails(self):
        def f(p,_):
            m=json.loads((p/'manifest.json').read_text());del m['artifact_sha256']['events.jsonl.gz'];write(p/'manifest.json',m)
        self.assertEqual(self.run_fixture(f)['status'],'FAIL')
    def test_gzip_tamper_fails_even_after_outer_manifest_reseal(self):
        def f(p,_):
            q=p/'events.jsonl.gz';q.write_bytes(q.read_bytes()+b'bad');reseal(p)
        self.assertEqual(self.run_fixture(f)['status'],'FAIL')
    def test_missing_event_fails_even_with_resealed_table_hashes(self):
        def f(p,v):
            ops,events,checks,summaries=v;write_aligned_tables(p,operations=ops,events=events[:-1],summaries=summaries,objects=[],snapshots=[],transfers=[],checks=checks);reseal(p)
        self.assertEqual(self.run_fixture(f)['status'],'FAIL')
    def test_native_null_exit_is_not_success(self):
        def f(p,_):
            x=json.loads((p/'astra_command.json').read_text());x['exit_status']=None;write(p/'astra_command.json',x);reseal(p)
        self.assertEqual(self.run_fixture(f)['status'],'FAIL')
    def test_duplicate_event_fails(self):
        def f(p,v):
            ops,events,checks,summaries=v;write_aligned_tables(p,operations=ops,events=events+[events[0]],summaries=summaries,objects=[],snapshots=[],transfers=[],checks=checks);reseal(p)
        self.assertEqual(self.run_fixture(f)['status'],'FAIL')
    def test_report_binds_actual_manifest_bytes_and_checker_source(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);bundle(root);manifest=root/'manifest.json'
            first=checker.verify_bundle(root)
            self.assertEqual(first['status'],'PASS',first)
            self.assertEqual(first['manifest_sha256'],checker.sha(manifest))
            self.assertEqual(first['checker_source_sha256'],checker.sha(ROOT/'scripts/check_aligned_result.py'))
            # Even a semantically neutral byte change makes the old PASS
            # binding stale; only a fresh report can bind the new manifest.
            manifest.write_bytes(manifest.read_bytes()+b'\n')
            self.assertNotEqual(first['manifest_sha256'],checker.sha(manifest))
            second=checker.verify_bundle(root)
            self.assertEqual(second['status'],'PASS',second)
            self.assertEqual(second['manifest_sha256'],checker.sha(manifest))
            self.assertNotEqual(first['manifest_sha256'],second['manifest_sha256'])
            changed=json.loads(manifest.read_text());changed['exit_status']=3;write(manifest,changed)
            rejected=checker.verify_bundle(root)
            self.assertEqual(rejected['status'],'FAIL',rejected)
            self.assertEqual(rejected['manifest_sha256'],checker.sha(manifest))
    def test_old_tiny_wrapper_only_fills_three_pinned_false_defaults(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);bundle(root);path=root/'inputs/policy.json'
            policy=json.loads(path.read_text())
            for key in checker.PINNED_INPUT_POLICY_DEFAULTS:del policy[key]
            def update():
                write(path,{'schema_version':'rlxy.aligned_policy/1','policy':policy})
                manifest=json.loads((root/'manifest.json').read_text())
                manifest['inputs']['policy']['sha256']=checker.sha(path)
                write(root/'manifest.json',manifest);reseal(root)
            update();accepted=checker.verify_bundle(root)
            self.assertEqual(accepted['status'],'PASS',accepted)
            self.assertEqual(accepted['pinned_input_policy_defaults_applied'],checker.PINNED_INPUT_POLICY_DEFAULTS)
            del policy['clip_ratio'];update();rejected=checker.verify_bundle(root)
            self.assertEqual(rejected['status'],'FAIL',rejected)
    def test_two_instances_can_share_one_semantic_signature(self):
        def f(p,_):two_instances_same_semantic_signature(p)
        result=self.run_fixture(f);self.assertEqual(result['status'],'PASS',result)
        details=next(c['detail']for c in result['checks']if c['check']=='template_signature_guard_and_every_invocation_clock_flags')
        self.assertEqual(details['contraction_instances'],2)
        self.assertEqual(details['distinct_semantic_signatures'],1)
        self.assertEqual(details['full'],2)
    def test_missing_semantic_signature_rejected(self):
        def f(p,_):
            two_instances_same_semantic_signature(p)
            guard=json.loads((p/'component_template_signature_guard.json').read_text())
            guard['template_signatures']={};write(p/'component_template_signature_guard.json',guard);reseal(p)
        self.assertEqual(self.run_fixture(f)['status'],'FAIL')
    def test_wrong_instance_full_replay_count_rejected(self):
        def f(p,_):
            two_instances_same_semantic_signature(p)
            guard=json.loads((p/'component_template_signature_guard.json').read_text())
            guard['full_template_replays']=1;write(p/'component_template_signature_guard.json',guard);reseal(p)
        self.assertEqual(self.run_fixture(f)['status'],'FAIL')


def resource_fixture(root,rank_count=3,payload_chars=40000):
    """Standalone resource layer, so a low cap need not fit unrelated metrics."""
    (root/'inputs').mkdir();ranks={};native={}
    for rank in range(rank_count):
        key=str(rank);obj={'dram_capacity':1000,'initial_objects':{},'nodes':{'1':{}},'unknown_rank_extension':'x'*payload_chars}
        ranks[key]=obj;p=root/'inputs'/f'resources.{key}.json'
        native[key]={'path':p.relative_to(root).as_posix(),**write_minified_json(p,obj)}
    obj={'aux':{'bool':False,'unicode':'晶圆🙂'},'log_prefix':'/remote/resource','ranks':ranks,'schema_version':'1.0','zz_extra':[1e-7,None,True]}
    rf={'audit_contract':{'path':'resource_contract.json.gz',**write_gzip_json(root/'resource_contract.json.gz',obj)},'native_rank_contracts':native}
    index={'rank_files':{str(r):f'/remote/inputs/resources.{r}.json'for r in range(rank_count)},'log_prefix':obj['log_prefix']}
    return obj,rf,index


def replace_unified_raw(root,rf,raw):
    path=root/'resource_contract.json.gz'
    with gzip.open(path,'wb')as f:f.write(raw)
    h=hashlib.sha256(raw).hexdigest()
    rf['audit_contract'].update(stored_sha256=checker.sha(path),stored_bytes=path.stat().st_size,
        uncompressed_canonical_sha256=h,uncompressed_canonical_bytes=len(raw),canonical_semantic_sha256=h)


class StreamedResourceCheckerTests(unittest.TestCase):
    def test_whole_exceeds_cap_but_each_rank_and_unknown_top_values_fit(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);obj,rf,index=resource_fixture(root);c=checker.Checker(root,65536)
            self.assertGreater(rf['audit_contract']['uncompressed_canonical_bytes'],c.limit)
            with self.assertRaisesRegex(ValueError,'expanded JSON over'):c.record(rf['audit_contract'],True)
            nodes,detail=c.stream_resource_contract(rf,index,3,chunk_chars=997)
            self.assertEqual(nodes,{0:{1},1:{1},2:{1}})
            self.assertEqual(detail['canonical_semantic_sha256'],checker.semantic(obj))
            self.assertLess(detail['largest_decoded_value_bytes'],c.limit)
            self.assertEqual(detail['top_level_keys'],sorted(obj))

    def test_48rank_large_fixture_never_loads_the_whole_unified_JSON(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);obj,rf,index=resource_fixture(root,48,256*1024);del obj
            c=checker.Checker(root,512*1024);original=c.load
            def per_rank_only(name):
                self.assertNotEqual(name,'resource_contract.json.gz')
                return original(name)
            with mock.patch.object(c,'load',side_effect=per_rank_only):nodes,detail=c.stream_resource_contract(rf,index,48)
            self.assertEqual(len(nodes),48)
            self.assertGreater(detail['unified_expanded_bytes'],24*c.limit)
            self.assertLess(detail['largest_decoded_value_bytes'],c.limit)

    def test_duplicate_rank_top_and_nested_keys_rejected(self):
        for kind in ['rank','top','nested']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory()as directory:
                root=Path(directory);obj,rf,index=resource_fixture(root,2,1)
                raw=json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'))
                if kind=='rank':
                    rank=json.dumps(obj['ranks']['0'],ensure_ascii=False,sort_keys=True,separators=(',',':'))
                    raw=raw.replace('"ranks":{','"ranks":{"0":'+rank+',',1)
                elif kind=='top':raw=raw.replace('"log_prefix":','"aux":{},"log_prefix":',1)
                else:raw=raw.replace('"initial_objects":{}','"initial_objects":{},"initial_objects":{}',1)
                replace_unified_raw(root,rf,raw.encode())
                with self.assertRaisesRegex(ValueError,'duplicate|noncanonical'):checker.Checker(root,65536).stream_resource_contract(rf,index,2,chunk_chars=17)

    def test_missing_extra_or_index_rank_rejected(self):
        for kind in ['missing','extra','index']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory()as directory:
                root=Path(directory);obj,rf,index=resource_fixture(root,2,1)
                if kind=='missing':del obj['ranks']['1']
                elif kind=='extra':obj['ranks']['2']=obj['ranks']['0']
                else:del index['rank_files']['1']
                replace_unified_raw(root,rf,json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())
                with self.assertRaisesRegex(ValueError,'rank'):checker.Checker(root,65536).stream_resource_contract(rf,index,2)

    def test_corrupt_semantic_even_resealed_byte_SHA_rejected(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);obj,rf,index=resource_fixture(root,2,1);obj['ranks']['0']['dram_capacity']=999
            replace_unified_raw(root,rf,json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())
            with self.assertRaisesRegex(ValueError,'unified/native rank semantic'):checker.Checker(root,65536).stream_resource_contract(rf,index,2)

    def test_newline_whitespace_keyorder_and_numeric_noncanonical_rejected(self):
        for kind in ['newline','space','keyorder','numeric']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory()as directory:
                root=Path(directory);obj,rf,index=resource_fixture(root,2,1)
                raw=json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'))
                if kind=='newline':raw+='\n'
                elif kind=='space':raw=raw.replace('{','{ ',1)
                elif kind=='numeric':raw=raw.replace('1000','1e3',1)
                else:raw=json.dumps({k:obj[k]for k in reversed(sorted(obj))},ensure_ascii=False,separators=(',',':'))
                replace_unified_raw(root,rf,raw.encode())
                with self.assertRaises(ValueError):checker.Checker(root,65536).stream_resource_contract(rf,index,2,chunk_chars=11)

    def test_nonfinite_trailing_invalid_UTF8_and_oversize_value_rejected(self):
        for kind in ['NaN','overflow','trailing','utf8','oversize']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory()as directory:
                root=Path(directory);obj,rf,index=resource_fixture(root,2,70000 if kind=='oversize'else 1)
                raw=json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
                if kind=='NaN':raw=raw.replace(b'1000',b'NaN',1)
                elif kind=='overflow':raw=raw.replace(b'1000',b'1e999',1)
                elif kind=='trailing':raw+=b'{}'
                elif kind=='utf8':raw=raw.replace('晶'.encode(),b'\xff',1)
                replace_unified_raw(root,rf,raw)
                with self.assertRaises((ValueError,UnicodeDecodeError)):checker.Checker(root,65536).stream_resource_contract(rf,index,2,chunk_chars=101)

    def test_UTF8_and_numeric_tokens_across_one_byte_reads(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);obj,rf,index=resource_fixture(root,2,1)
            nodes,detail=checker.Checker(root,65536).stream_resource_contract(rf,index,2,chunk_chars=1)
            self.assertEqual(nodes,{0:{1},1:{1}});self.assertEqual(detail['canonical_semantic_sha256'],checker.semantic(obj))


if __name__=='__main__':unittest.main()
