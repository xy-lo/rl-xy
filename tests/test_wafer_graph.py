import copy
from dataclasses import replace
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import ast
import unittest

from rl_xy.config import load_case
from rl_xy.cost_model import AnalyticalCostProvider
from rl_xy.hardware_profile import load_hardware_profile,inspect_hardware
from rl_xy.multicore import partition_matmul
from rl_xy.operator_ir import Matmul,KernelSpec
from rl_xy.tensor_plan import preflight_contract
from rl_xy.wafer_graph import build_wafer_graph
from rl_xy.wafer_run import effective_case
from rl_xy.astra_adapter.generator import _validate_operations,TraceGenerationError

ROOT=Path(__file__).resolve().parents[1]


class WaferGraphTest(unittest.TestCase):
    def test_completion_summary_precedes_manifest_and_early_integrity(self):
        # Structural regression plus actual native fixture integrity in the
        # Docker acceptance; this is not a substitute for that native check.
        tree=ast.parse((ROOT/'src/rl_xy/wafer_run.py').read_text())
        calls={n.func.id:n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        self.assertLess(calls['_write_summary_json'],calls['write_manifest'])
        acceptance=(ROOT/'scripts/wafer_acceptance.py').read_text()
        self.assertLess(acceptance.index("command('early_integrity_'"),acceptance.index("command('compile_full_gpu_step'"))

    def test_large_native_logs_have_explicit_provenance(self):
        dockerfile=(ROOT/'docker/Dockerfile.wafer').read_text()
        self.assertIn('1024 * 1024 * 10, 1000)',dockerfile)
        self.assertIn('astra-sim/common/Logging.cc > /opt/astra-sim/.rlxy-multicore-patch',dockerfile)

    def test_external_artifact_root_requires_exact_owned_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project';(root/'scripts').mkdir(parents=True)
            target=Path(tmp)/'owned-runs';target.mkdir()
            other=Path(tmp)/'wrong-runs';other.mkdir()
            (root/'runs').symlink_to(target,target_is_directory=True)
            script=root/'scripts/docker.sh';shutil.copy2(ROOT/'scripts/docker.sh',script)
            for value,expected in ((str(target),0),(str(other),4),('relative-runs',4)):
                env={**os.environ,'RLXY_RUNS_ROOT':value}
                result=subprocess.run(['bash',str(script),'help'],env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,expected,result.stderr)

    def test_private_partition_ragged_shapes_conserve_outputs(self):
        p=Matmul(5,7,11,batches=3)
        parts=partition_matmul(p,16,4096)
        outputs=set()
        for core,(b,bc,m,mc,n,nc),mapping in parts:
            self.assertLessEqual(mapping.workspace_bytes,4096)
            for bb in range(b,b+bc):
                for mm in range(m,m+mc):
                    for nn in range(n,n+nc):
                        key=(bb,mm,nn)
                        self.assertNotIn(key,outputs);outputs.add(key)
        self.assertEqual(len(outputs),5*7*3)
        self.assertEqual(sum(x[2].flops for x in parts),p.flops)

    def test_small_output_does_not_use_all_256_cores(self):
        parts=partition_matmul(Matmul(1,2,128),256,4096)
        self.assertEqual(len(parts),2)

    def test_shared_dram_not_multiplied_and_fp32_rate_consumed(self):
        c=load_case(ROOT/'configs/qwen2_group_fixture/case.json')
        h=replace(c.hardware,cores_per_die=4,dram_bytes_per_ns=1,gemm_flops_per_ns=1000,gemm_fp32_flops_per_ns=1)
        spec=KernelSpec('x',matmuls=(Matmul(16,16,16),))
        e=AnalyticalCostProvider().estimate(spec,h)
        self.assertGreaterEqual(e.local_memory_ns,e.dram_read_bytes+e.dram_write_bytes)
        self.assertEqual(sum(e.core_workspace_bytes),e.sram_workspace_bytes)
        fp=AnalyticalCostProvider().estimate(KernelSpec('fp',matmuls=(Matmul(16,16,16,a_bytes=4,b_bytes=4),)),h)
        self.assertGreater(fp.compute_ns,e.compute_ns)

    def test_dp_unequal_groups_normalize_globally_and_bind_native_groups(self):
        c=load_case(ROOT/'configs/qwen2_group_fixture/case.json')
        g=build_wafer_graph(c,[2,2],2,'/tmp/no-native')
        self.assertEqual([p['local_response_tokens'] for p in g.placements],[4,5])
        self.assertEqual([p['global_loss_denominator'] for p in g.placements],[9,9])
        losses=[o for o in g.plan.operations if o.name.endswith('_loss')]
        self.assertEqual([o.metadata['global_denominator'] for o in losses],[9]*4)
        self.assertEqual(g.groups,{'1':[0,1],'2':[2,3],'3':[0,2],'4':[1,3]})
        _validate_operations(g.plan.operations,4,g.groups)
        dp=[o for o in g.plan.operations if o.name=='dp_gradient_sum']
        self.assertEqual(len(dp),2)
        self.assertTrue(all(o.metadata['sum_not_mean'] for o in dp))
        for i in (0,1):
            acc=[o.node_id for o in g.plan.operations if o.metadata.get('replica_id')==i and o.name.endswith('.gradient_accumulate_rmw')]
            clip=[o.node_id for o in g.plan.operations if o.metadata.get('replica_id')==i and o.name=='local_gradient_square_sum']
            self.assertLess(max(acc),min(o.node_id for o in dp))
            self.assertLess(max(o.node_id for o in dp),min(clip))

    def test_communicator_mismatch_and_missing_groups_rejected(self):
        c=load_case(ROOT/'configs/qwen2_group_fixture/case.json')
        g=build_wafer_graph(c,[2,2],2,'/tmp/no-native')
        with self.assertRaises(TraceGenerationError):_validate_operations(g.plan.operations,4)
        bad=copy.deepcopy(g.groups);bad['1']=[0,2]
        with self.assertRaises(TraceGenerationError):_validate_operations(g.plan.operations,4,bad)

    def test_per_core_overflow_rejected_even_when_module_sum_fits(self):
        c=load_case(ROOT/'configs/qwen2_group_fixture/case.json')
        c=replace(c,hardware=replace(c.hardware,cores_per_die=4))
        g=build_wafer_graph(c,[2,2],2,'/tmp/no-native')
        bad=copy.deepcopy(g.plan.contract)
        r=bad['ranks']['0'];n=next(n for n in r['nodes'].values() if n['workspace_bytes'])
        n['core_workspace_bytes']=[4097,0,0,0];n['workspace_bytes']=4097
        self.assertLess(n['workspace_bytes'],4096*4)
        with self.assertRaisesRegex(ValueError,'private SRAM'):preflight_contract(bad)

    def test_case3_template_geometry_is_not_a_fixed_placement(self):
        h=load_hardware_profile(ROOT/'configs/hardware/xy_h0_case3_performance.json')
        r=inspect_hardware(h)
        self.assertEqual(r['aggregates']['modules'],54)
        self.assertFalse(r['geometry_errors'])
        self.assertFalse(r['physical_geometry_validated'])
        self.assertEqual(r['geometry_status'],'reference_template_only_case3_placement_unknown')
        self.assertEqual(len(r['geometry_details']['template_only_conflicts_not_case3_proof']),3)

    def test_resource_encoding_is_lossless_and_bad_delta_rejected(self):
        from rl_xy.resource_encoding import compact_rank,expand_rank
        c=load_case(ROOT/'configs/qwen2_group_fixture/case.json')
        g=build_wafer_graph(c,[2,2],2,'/tmp/no-native')
        for rank in g.plan.contract['ranks'].values():
            compact=compact_rank(rank)
            self.assertEqual(expand_rank(compact),rank)
            first=next(iter(compact['nodes'].values()))
            first['read_delta']['base_node']=999
            with self.assertRaises(ValueError):expand_rank(compact)

    def test_full_54_source_binding_and_placement(self):
        c,s,_=effective_case(load_case(ROOT/'configs/wafer54_fixture/case.json'),ROOT/'configs/wafer54_fixture/wafer.json')
        self.assertEqual(c.hardware.cores_per_die,256)
        self.assertEqual(c.hardware.dram_bytes_per_die,64000000000)
        self.assertEqual(c.hardware.sram_bytes_per_core,1250000)
        self.assertEqual(c.hardware.d2d_bytes_per_ns_per_direction,2000)
        self.assertEqual(s['data_parallel'],27)
        # Source H values and all synthetic rate assumptions remain distinct.
        self.assertTrue(all(v['status']=='test_assumption' for v in s['assumptions'].values()))

    def test_dp_requires_complete_groups_and_full_batch_update(self):
        c=load_case(ROOT/'configs/qwen2_group_fixture/case.json')
        with self.assertRaises(ValueError):build_wafer_graph(c,[2,3],3,'/tmp/no-native')
        with self.assertRaises(ValueError):build_wafer_graph(replace(c,strategy=replace(c.strategy,optimizer_minibatch_trajectories=2)),[2,2],2,'/tmp/no-native')
