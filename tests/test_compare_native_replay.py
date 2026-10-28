"""Constructed two-rank serial fixtures; no native or GPU execution."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('compare_native_replay',ROOT/'scripts/compare_native_replay.py')
replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)


def write(path,obj):path.write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')


def line(action,rank,node,tick):
    return f'[2026-10-28 00:00:00.000] [workload] [debug] {action},sys->id={rank}, tick={tick}, node->id={node}, node->name=r{rank}.node{node}, node->type=4\n'


def fixture(root,reverse_interleave=False):
    root.mkdir();(root/'inputs').mkdir();(root/'astra_logs').mkdir()
    declared='/workspace/'+root.name
    args=['/opt/astra/AstraSim_Analytical_Congestion_Aware',
        f'--workload-configuration={declared}/inputs/workload',
        f'--system-configuration={declared}/inputs/astra_system.json',
        f'--network-configuration={declared}/inputs/astra_network.yml',
        f'--remote-memory-configuration={declared}/inputs/astra_remote_memory.json',
        f'--logging-folder={declared}/astra_logs',
        f'--comm-group-configuration={declared}/inputs/groups.json']
    write(root/'astra_command.json',dict(argv=args,exit_status=0,started_at_utc='2026-10-28T00:00:00Z',finished_at_utc='2026-10-28T00:00:01Z'))
    write(root.parent/'report.json',dict(status='FAIL',exit_status=2,OOMKilled=False))
    for name,obj in [('astra_system.json',{'endpoint-delay':0,'trace-enabled':1}),('astra_remote_memory.json',{'memory-type':'NO_MEMORY_EXPANSION'}),
        ('groups.json',{'0':[0,1]}),('case.json',{'case_id':'tiny-serial','workload':{'P':2,'R':1}}),
        ('model_config.json',{'hidden_size':8}),('policy.json',{'train_tp':2,'overlap':False}),
        ('workload_provenance.json',{'kind':'constructed_fixture'}),
        ('gpu_profile.json',{'gpu_count':2,'gemm_profile':declared+'/original/labels.json'}),
        ('gemm_measurements.json',{'labels':{'shapeA':{'nanoseconds':10}}})]:write(root/'inputs'/name,obj)
    (root/'inputs/astra_network.yml').write_text('topology: [Ring]\nnpus_count: [2]\nbandwidth: [300.0]\nlatency: [250]\n')
    write(root/'actual_gemm_census.json',dict(shapes={'x':{'m':1,'n':8,'k':8}},totals={'rank_local_matmul_occurrences':2}))
    write(root/'inputs/resource_index.json',dict(schema_version='1.0',log_prefix=declared+'/astra_logs/resources',rank_files={str(r):f'{declared}/inputs/resources.{r}.json'for r in range(2)}))
    for rank in range(2):
        write(root/f'inputs/resources.{rank}.json',{'rank':rank,'nodes':{'1':{'name':'first'},'3':{'name':'second'}}})
        (root/f'astra_logs/resources.{rank}.jsonl').write_text('{"tick":0,"bytes":8}\n{"tick":20,"bytes":0}\n')
        (root/f'inputs/workload.{rank}.et').write_bytes(b'constructed protobuf-map-order '+root.name.encode())
    ranks=[1,0]if reverse_interleave else[0,1]
    (root/'astra_logs/log.1.log').write_text(''.join(line('issue',r,1,0)for r in ranks)+''.join(line('callback',r,1,10)for r in ranks))
    (root/'astra_logs/log.log').write_text(''.join(line('issue',r,3,10)for r in reversed(ranks))+''.join(line('callback',r,3,20)for r in ranks))
    (root/'astra_stdout.log').write_text(''.join(f'sys[{r}] finished, 20 cycles, exposed communication 0 cycles.\n'for r in ranks))
    (root/'astra_stderr.log').write_text('')


class ReplayCheckerTests(unittest.TestCase):
    def check(self,mutate=None):
        with tempfile.TemporaryDirectory()as name:
            left,right=Path(name)/'left',Path(name)/'right';fixture(left);fixture(right,True)
            if mutate:mutate(right)
            return replay.compare(left,right,ranks=2,expected_rank_events=4)

    def rejected(self,result,contains=None):
        self.assertEqual(result['status'],'FAIL_NOT_EQUIVALENT_OR_INCOMPLETE',result)
        self.assertFalse(result['paper_performance_ready']);self.assertFalse(result['performance_ratio_published'])
        if contains:self.assertIn(contains,result['checks'][-1]['error'])

    def test_interleave_path_and_ET_byte_differences_allowed_with_explicit_limits(self):
        result=self.check();self.assertEqual(result['status'],'PASS_OBSERVED_NATIVE_REPLAY_EQUIVALENCE',result)
        self.assertEqual(result['native_logs']['left']['raw_event_count'],8)
        self.assertEqual(result['model_max_callback_tick'],20)
        self.assertEqual(result['ET_bytes']['differing_ranks'],['0','1'])
        self.assertFalse(result['ET_bytes']['semantic_equality_independently_checked'])
        self.assertFalse(result['wrapper_success_implied']);self.assertFalse(result['paper_performance_ready'])

    def test_resource_raw_byte_change_rejected_even_whitespace(self):
        def change(p):
            with(p/'astra_logs/resources.0.jsonl').open('a')as f:f.write('\n')
        self.rejected(self.check(change),'resource raw bytes')

    def test_single_tick_change_rejected(self):
        def change(p):
            path=p/'astra_logs/log.1.log';path.write_text(path.read_text().replace('callback,sys->id=0, tick=10','callback,sys->id=0, tick=9'))
        self.rejected(self.check(change),'event/tick/name/type')

    def test_duplicate_node_and_callback_rejected(self):
        def callback(p):
            path=p/'astra_logs/log.log';path.write_text(path.read_text()+line('callback',0,3,20))
        def reused_node(p):
            path=p/'astra_logs/log.log';path.write_text(path.read_text().replace('node->id=3','node->id=1').replace('node3','node1'))
        for change in(callback,reused_node):
            with self.subTest(change=change.__name__):self.rejected(self.check(change))

    def test_rotation_gap_rejected(self):
        def change(p):(p/'astra_logs/log.1.log').rename(p/'astra_logs/log.2.log')
        self.rejected(self.check(change),'rotation gap')

    def test_native_exit_missing_or_nonzero_rejected(self):
        for value in(None,1):
            def change(p):
                path=p/'astra_command.json';d=json.loads(path.read_text());d.pop('exit_status')
                if value is not None:d['exit_status']=value
                write(path,d)
            with self.subTest(value=value):self.rejected(self.check(change),'actual native exit0')

    def test_input_semantic_change_and_duplicate_JSON_keys_rejected(self):
        def input_change(p):write(p/'inputs/policy.json',{'train_tp':1,'overlap':False})
        def duplicate_key(p):(p/'inputs/policy.json').write_text('{"train_tp":2,"train_tp":2}')
        for change in(input_change,duplicate_key):
            with self.subTest(change=change.__name__):self.rejected(self.check(change))

    def test_missing_invalid_callback_and_fixed_matrix_rejected(self):
        def missing(p):
            path=p/'astra_logs/log.log';path.write_text(path.read_text().replace(line('callback',0,3,20),''))
        def invalid(p):
            path=p/'astra_logs/log.log';path.write_text(path.read_text().replace('node->type=4','node->type=9',1))
        for change in(missing,invalid):
            with self.subTest(change=change.__name__):self.rejected(self.check(change))
        with tempfile.TemporaryDirectory()as name:
            left,right=Path(name)/'left',Path(name)/'right';fixture(left);fixture(right)
            self.rejected(replay.compare(left,right,ranks=2,expected_rank_events=5),'fixed expected matrix')

    def test_native_stderr_completion_and_JSON_nonfinite_rejected(self):
        def stderr(p):(p/'astra_stderr.log').write_text('error')
        def completion(p):(p/'astra_stdout.log').write_text('sys[0] finished, 20 cycles, exposed communication 0 cycles.\n')
        def nonfinite(p):(p/'inputs/policy.json').write_text('{"number":1e999}')
        for change in(stderr,completion,nonfinite):
            with self.subTest(change=change.__name__):self.rejected(self.check(change))


if __name__=='__main__':unittest.main()
