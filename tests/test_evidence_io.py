"""Encoding changes preserve complete data; deadlines restore process state."""
import gzip
import hashlib
import json
from pathlib import Path
import signal
import tempfile
import time
import unittest
from unittest.mock import patch
from dataclasses import asdict,dataclass
from copy import deepcopy
import io

from rl_xy.evidence_io import (canonical_semantic_sha256, write_minified_json,
    write_gzip_json, load_json_auto, phase_deadline, EvidencePhaseTimeout, write_aligned_tables)
from rl_xy.reporting import write_json, write_jsonl, write_csv, write_tables
from rl_xy import evidence_io as evidence
from rl_xy.resource_encoding import compact_rank, expand_rank
from rl_xy.operator_ir import KernelSpec, Matmul
from rl_xy.types import (Operation,AstraEvent,AstraRankSummary,ObjectRecord,
    MemorySnapshot,TransferRecord,CheckResult,ParameterRow)


def old_canonical_chunks(value,buffer_bytes=65536):
    encoder=json.JSONEncoder(ensure_ascii=False,sort_keys=True,separators=(',',':'),
        allow_nan=False,default=evidence._default)
    pending=[];size=0
    for part in encoder.iterencode(value):
        piece=part.encode('utf-8');pending.append(piece);size+=len(piece)
        if size>=buffer_bytes:
            yield b''.join(pending);pending=[];size=0
    if pending:yield b''.join(pending)


@dataclass(frozen=True)
class NestedValue:
    name:str
    scores:tuple


@dataclass(frozen=True)
class NestedRow:
    payload:object
    label:str


class EvidenceIOTests(unittest.TestCase):
    def contract(self):
        rank=dict(initial_objects={'权重':16},dram_capacity=128,cores_per_die=2,sram_bytes_per_core=16,
            nodes={'1':dict(reads=['权重'],allocations={'临时':8},ready_outputs=['临时'],frees=[],
                workspace_bytes=4,core_workspace_bytes=[4,0],sram_capacity=16,local_memory=True,reserve_before=0,reserve_after=0),
            '2':dict(reads=['权重','临时'],allocations={},ready_outputs=[],frees=['临时'],
                workspace_bytes=8,core_workspace_bytes=[4,4],sram_capacity=16,local_memory=True,reserve_before=0,reserve_after=0)})
        return dict(schema_version='1.0',log_prefix='/fixture/not-executed',ranks={'0':compact_rank(rank)})

    def test_hybrid_canonical_bytes_equal_original_python_encoder(self):
        class CustomDict(dict):pass
        class CustomList(list):pass
        values=[self.contract(),{'unicode':'中文😀\b\f\n\r\t\x00\\\"','floats':[-0.,1e-7,1e20,1.2345678901234567]},
            {'tuple':({'嵌套':(1,False,None)},[.125]),'path':Path('/tmp/证据')},
            {3:'三',1:'一',2:'二'},[-1.,{2.5:'a',-1:'b'}],{None:'null'},
            {'outer':{'ranks':{'0':{4:'int-key-at-leaf',2:'sorted-numerically'}}}},
            {'custom':CustomDict(a=CustomList([1,2,3]))},NestedValue('嵌套',(1,.25)),
            {'nested':{'a':{'b':[NestedValue('deep',(2,3)),Path('/tmp/p')]}}}]
        for value in values:
            with self.subTest(value=type(value).__name__):
                expected=b''.join(old_canonical_chunks(value))
                self.assertEqual(b''.join(evidence._canonical_chunks(value)),expected)
                self.assertEqual(evidence.canonical_semantic_sha256(value),hashlib.sha256(expected).hexdigest())

    def test_manual_prefix_never_C_encodes_global_or_all_ranks_and_fallbacks_remain(self):
        nodes0={'1':{'read_delta':{'add':['a','b'],'remove':[]}}};nodes1={'2':{'x':(1,2)}}
        rank0={'nodes':nodes0};rank1={'nodes':nodes1};ranks={'0':rank0,'1':rank1};value={'ranks':ranks}
        original=json.JSONEncoder.iterencode;calls=[]
        def trace(encoder,item,_one_shot=False):
            calls.append((item,_one_shot));return original(encoder,item,_one_shot=_one_shot)
        with patch.object(json.JSONEncoder,'iterencode',trace):
            result=b''.join(evidence._canonical_chunks(value))
        self.assertEqual(result,b''.join(old_canonical_chunks(value)))
        for item in (value,ranks,rank0,rank1):self.assertFalse(any(x is item for x,_ in calls))
        for item in (nodes0,nodes1):self.assertTrue(any(x is item and fast for x,fast in calls))
        integer_keys={2:'二',1:'一'};path=Path('/tmp/路径');calls.clear()
        with patch.object(json.JSONEncoder,'iterencode',trace):
            b''.join(evidence._canonical_chunks({'int':integer_keys,'custom':path}))
        self.assertTrue(any(x is integer_keys and not fast for x,fast in calls))
        self.assertTrue(any(x is path and not fast for x,fast in calls))

    def test_hybrid_gzip_stream_preserves_exact_compressed_bytes_and_sha(self):
        nodes={str(i):{'weights':['张量'+str(j) for j in range(16)],'ratio':.125,'tuple':(i,-0.)} for i in range(1600)}
        value={'schema_version':'1.0','ranks':{'0':{'nodes':nodes},'1':{'nodes':nodes}}}
        previous=io.BytesIO()
        with gzip.GzipFile(filename='',mode='wb',fileobj=previous,compresslevel=6,mtime=0) as stream:
            for chunk in old_canonical_chunks(value):stream.write(chunk)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'hybrid.json.gz';record=write_gzip_json(path,value)
            self.assertEqual(path.read_bytes(),previous.getvalue())
            self.assertEqual(record['stored_sha256'],hashlib.sha256(previous.getvalue()).hexdigest())

    def test_hybrid_nonfinite_cycles_mixed_keys_fail_like_original_and_atomically(self):
        cyclic={};cyclic['self']=cyclic
        for value,error in [({'a':{'b':{'c':[float('nan')]}}},ValueError),
                ({'a':[float('inf')]},ValueError),({1:'a','2':'b'},TypeError),(cyclic,ValueError)]:
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(error):b''.join(old_canonical_chunks(value))
                with self.assertRaises(error):b''.join(evidence._canonical_chunks(value))
                with tempfile.TemporaryDirectory() as tmp:
                    path=Path(tmp)/'old.json.gz';path.write_bytes(b'existing')
                    with self.assertRaises(error):write_gzip_json(path,value)
                    self.assertEqual(path.read_bytes(),b'existing')
                    self.assertFalse(path.with_name('.'+path.name+'.tmp').exists())

    def test_shallow_dataclass_rows_match_legacy_and_do_not_copy_or_mutate_large_metadata(self):
        data=self.table_rows();op=data['operations'][0]
        op.metadata['large_readonly']=[{'ids':list(range(500)),'nested':NestedValue('值',(1,2))} for _ in range(4)]
        before=deepcopy(data);metadata=op.metadata;payload=metadata['large_readonly'];inner=payload[0]['ids']
        with tempfile.TemporaryDirectory() as tmp:
            old=Path(tmp)/'old';new=Path(tmp)/'new';old.mkdir();new.mkdir();write_tables(old,**data)
            with patch('rl_xy.evidence_io.asdict',wraps=asdict) as conversion:
                records=write_aligned_tables(new,**data)
            top=(Operation,AstraEvent,ObjectRecord,MemorySnapshot,TransferRecord)
            self.assertFalse(any(isinstance(call.args[0],top) for call in conversion.call_args_list))
            self.assertTrue(any(isinstance(call.args[0],KernelSpec) for call in conversion.call_args_list))
            for logical,record in records['tables'].items():
                path=new/record['path'];actual=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()
                self.assertEqual(actual,(old/logical).read_bytes())
        self.assertEqual(data,before);self.assertIs(op.metadata,metadata);self.assertIs(metadata['large_readonly'],payload)
        self.assertIs(payload[0]['ids'],inner)

    def test_nested_csv_dataclasses_and_original_dict_row_rules_are_preserved(self):
        child=NestedValue('嵌套',(1,2));rows=[NestedRow(child,'direct'),
            NestedRow({'层':[child]},'dict'),NestedRow((child,child),'tuple'),
            {'payload':child,'label':'dict-row-direct-repr'}]
        before=deepcopy(rows)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_csv(root/'old.csv',rows,['payload','label'])
            record=evidence._write_gzip_csv(root/'new.csv.gz',rows,['payload','label'])
            self.assertEqual(gzip.decompress((root/'new.csv.gz').read_bytes()),(root/'old.csv').read_bytes())
            self.assertEqual(record['row_count'],4)
            # Dict input rows did not run asdict in legacy: nested dataclasses
            # inside JSON containers therefore remain an unsupported input.
            unsupported=[{'payload':{'child':child},'label':'must-fail'}]
            with self.assertRaises(TypeError):write_csv(root/'bad-old.csv',unsupported,['payload','label'])
            with self.assertRaises(TypeError):evidence._write_gzip_csv(root/'bad-new.csv.gz',unsupported,['payload','label'])
        self.assertEqual(rows,before)

    def test_full_gzip_audit_and_native_rank_roundtrip_semantics_and_replay_contract(self):
        original=self.contract()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plain=root/'old.json';compact=root/'rank.json';audit=root/'resource_contract.json.gz'
            write_json(plain,original)
            a=write_gzip_json(audit,original);b=write_minified_json(compact,original['ranks']['0'])
            self.assertEqual(load_json_auto(audit),load_json_auto(plain))
            self.assertEqual(load_json_auto(compact),load_json_auto(audit)['ranks']['0'])
            self.assertEqual(expand_rank(load_json_auto(compact)),expand_rank(original['ranks']['0']))
            self.assertEqual(a['canonical_semantic_sha256'],canonical_semantic_sha256(load_json_auto(plain)))
            self.assertEqual(b['stored_sha256'],b['canonical_semantic_sha256'])
            self.assertEqual(hashlib.sha256(gzip.decompress(audit.read_bytes())).hexdigest(),a['uncompressed_canonical_sha256'])
            self.assertEqual(audit.read_bytes()[:2],b'\x1f\x8b')
            self.assertFalse((root/'resource_contract.json').exists())
            self.assertEqual(a['stored_sha256'],hashlib.sha256(audit.read_bytes()).hexdigest())
            self.assertEqual(a['uncompressed_canonical_bytes'],len(gzip.decompress(audit.read_bytes())))

    def test_streaming_does_not_use_dumps_and_is_deterministic(self):
        data={'节点':[{'read':'参数','n':i} for i in range(4000)]}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch('rl_xy.evidence_io.json.dumps',side_effect=AssertionError('whole JSON allocation forbidden')):
                a=write_gzip_json(root/'one.json.gz',data);b=write_gzip_json(root/'two.json.gz',data)
                p=write_minified_json(root/'min.json',data)
            self.assertEqual(a,b);self.assertEqual(load_json_auto(root/'one.json.gz'),data)
            self.assertLess(a['stored_bytes'],p['stored_bytes'])

    def test_failed_encoding_is_atomic_and_nan_or_mislabelled_gzip_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'existing.json.gz';path.write_bytes(b'previous evidence')
            with self.assertRaises(ValueError):write_gzip_json(path,{'bad':float('nan')})
            self.assertEqual(path.read_bytes(),b'previous evidence')
            self.assertFalse(path.with_name('.'+path.name+'.tmp').exists())
            with self.assertRaisesRegex(ValueError,'.gz suffix'):write_gzip_json(Path(tmp)/'fake.json',{})

    def test_decoded_tables_are_byte_identical_across_json_encodings(self):
        rows=[{'rank':0,'node':1,'bytes':16,'ids':['权重']},{'rank':0,'node':2,'bytes':24,'ids':['临时','权重']}]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_json(root/'pretty.json',rows);write_gzip_json(root/'rows.json.gz',rows)
            write_csv(root/'old.csv',load_json_auto(root/'pretty.json'))
            write_csv(root/'new.csv',load_json_auto(root/'rows.json.gz'))
            self.assertEqual((root/'old.csv').read_bytes(),(root/'new.csv').read_bytes())

    def table_rows(self):
        return dict(operations=[Operation(1,'r0.算子',0,'train','引用,\"换行\n','compute',duration_ns=3,
            dependencies=(4,5),participant_ranks=(0,1),dependencies_by_rank={0:(4,),1:(5,)},
            metadata={'数组':[1,None,True],'路径':Path('/tmp/证据'),'浮点':.125},
            kernel_spec=KernelSpec('linear',matmuls=(Matmul(2,3,4),)))],
            events=[AstraEvent(0,1,'r0.算子',0,'train',4,1000,4000,3000,1.,4.,3.)],
            summaries=[AstraRankSummary(0,4000,0,4000,3000,0)],
            objects=[ObjectRecord('权重,\n','张量','actor','d0.dram',48,'BF16',None,'initial','initial','r0.算子',None,True,0)],
            snapshots=[MemorySnapshot(1,'train','d0.dram',48,40,8,100,'包含,引号\"与\r\n换行')],
            transfers=[TransferRecord(0,'r0.copy','train','p2p','d0','d1','route,\n',16,2)],
            checks=[CheckResult('check',True,{'张量':[1,None]},(1,2),'引用,\"换行\n','scope')],
            parameters=[ParameterRow('layer0.qkv',24,12,False,'column')])

    def test_all_aligned_tables_decompress_to_exact_legacy_bytes(self):
        data=self.table_rows()
        for empty in (False,True):
            with self.subTest(empty=empty),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);old=root/'old';new=root/'new';old.mkdir();new.mkdir()
                inputs={key:[] if empty else value for key,value in data.items()}
                write_tables(old,**inputs)
                # All large iterables may be one-shot generators.
                report=write_aligned_tables(new,**{key:iter(value) for key,value in inputs.items()})
                self.assertEqual(json.loads((new/'table_format.json').read_text()),report)
                self.assertEqual(set(report['tables']),{p.name for p in old.iterdir()})
                for logical,record in report['tables'].items():
                    path=new/record['path'];expected=(old/logical).read_bytes()
                    if record['encoding'].startswith('gzip'):
                        self.assertTrue(path.name.endswith('.gz'));self.assertFalse((new/logical).exists())
                        with gzip.open(path,'rb') as stream:actual=stream.read()
                    else:actual=path.read_bytes()
                    self.assertEqual(actual,expected,logical)
                    self.assertEqual(record['uncompressed_sha256'],hashlib.sha256(expected).hexdigest())
                    self.assertEqual(record['stored_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
                    self.assertEqual(record['uncompressed_bytes'],len(expected))
                    self.assertEqual(record['stored_bytes'],path.stat().st_size)
                # Small schema files remain plain even for empty inputs.
                self.assertEqual((new/'parameters.csv').read_bytes(),(old/'parameters.csv').read_bytes())

    def test_table_generator_failure_is_atomic_and_never_touches_native_logs(self):
        data=self.table_rows()
        def broken():
            yield data['operations'][0]
            raise RuntimeError('source row producer failed')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'operations.jsonl.gz';previous=gzip.compress(b'previous complete row\n')
            target.write_bytes(previous);native=root/'astra_logs';native.mkdir()
            log=native/'resources.0.jsonl';log.write_bytes(b'raw native evidence\n')
            with self.assertRaisesRegex(RuntimeError,'producer failed'):
                write_aligned_tables(root,**{**data,'operations':broken()})
            self.assertEqual(target.read_bytes(),previous)
            self.assertEqual(log.read_bytes(),b'raw native evidence\n')
            self.assertFalse((root/'table_format.json').exists())
            self.assertFalse(target.with_name('.'+target.name+'.tmp').exists())

    def test_unknown_csv_field_cannot_be_silently_discarded(self):
        data=self.table_rows();bad=asdict(data['objects'][0]);bad['unexpected_field']='must reject'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';new=root/'new';old.mkdir();new.mkdir()
            with self.assertRaises(ValueError):write_tables(old,**{**data,'objects':[bad]})
            previous=gzip.compress(b'previous CSV evidence\r\n');target=new/'objects.csv.gz';target.write_bytes(previous)
            with self.assertRaises(ValueError):write_aligned_tables(new,**{**data,'objects':[bad]})
            self.assertEqual(target.read_bytes(),previous)
            self.assertFalse((new/'table_format.json').exists())

    @unittest.skipUnless(hasattr(signal,'setitimer'),'POSIX interval timer required')
    def test_timeout_and_normal_or_exception_exit_restore_handler_and_cancel_timer(self):
        previous=signal.getsignal(signal.SIGALRM)
        def sentinel(_signum,_frame):pass
        signal.signal(signal.SIGALRM,sentinel)
        try:
            with self.assertRaisesRegex(EvidencePhaseTimeout,'planner.*hard deadline'):
                with phase_deadline('planner',.015,enabled=True):time.sleep(.04)
            self.assertIs(signal.getsignal(signal.SIGALRM),sentinel)
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL),(0.0,0.0))
            with phase_deadline('encode',1,enabled=True):pass
            self.assertIs(signal.getsignal(signal.SIGALRM),sentinel)
            with self.assertRaisesRegex(RuntimeError,'inside'):
                with phase_deadline('post',1,enabled=True):raise RuntimeError('inside')
            self.assertIs(signal.getsignal(signal.SIGALRM),sentinel)
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL),(0.0,0.0))
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)

    @unittest.skipUnless(hasattr(signal,'setitimer'),'POSIX interval timer required')
    def test_existing_timer_is_not_overridden_and_disabled_phase_is_inert(self):
        previous=signal.getsignal(signal.SIGALRM)
        def sentinel(_signum,_frame):pass
        signal.signal(signal.SIGALRM,sentinel);signal.setitimer(signal.ITIMER_REAL,10)
        try:
            with self.assertRaisesRegex(ValueError,'preexisting real timer'):
                with phase_deadline('post',1,enabled=True):pass
            remaining=signal.getitimer(signal.ITIMER_REAL)[0]
            self.assertGreater(remaining,9)
            with phase_deadline('notLinuxDocker',.001,enabled=False):time.sleep(.003)
            self.assertIs(signal.getsignal(signal.SIGALRM),sentinel)
            self.assertGreater(signal.getitimer(signal.ITIMER_REAL)[0],9)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        with self.assertRaisesRegex(ValueError,'positive finite'):
            with phase_deadline('post',0,enabled=False):pass


if __name__=='__main__':unittest.main()
