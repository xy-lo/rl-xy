import json
import tempfile
import unittest
from pathlib import Path
from architecture import contract, h0_memory_check, load_profiles, mesh_view_for, tp2_pairs, validate_existing_native_runner


class ArchitectureTest(unittest.TestCase):
    def test_tp2_physical_coverage(self):
        expected={'H1':(31,1),'H2':(28,0),'H0':(27,0),'H3':(24,0)}
        for name, profile in load_profiles().items():
            pairs,idle=tp2_pairs(profile)
            self.assertEqual((len(pairs),len(idle)),expected[name])
            self.assertEqual(sorted([d for pair in pairs for d in pair]+idle),list(range(profile['dies'])))

    def test_h0_original_pair_mapping(self):
        pairs,idle=tp2_pairs(load_profiles()['H0'])
        self.assertEqual(pairs,[(2*i,2*i+1) for i in range(27)])
        self.assertEqual(idle,[])

    def test_h1_vertical_remainder_pairs_are_real_neighbors(self):
        pairs,idle=tp2_pairs(load_profiles()['H1'])
        self.assertEqual(idle,[62])
        self.assertEqual([p for p in pairs if p[1]-p[0]==7],[(6,13),(20,27),(34,41),(48,55)])

    def test_mismatch_rejected(self):
        data=json.loads((Path(__file__).parent/'profiles.json').read_text());data['candidates'][0]['dies']=54
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'profile.json';p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):load_profiles(p)

    def test_h0_memory_evidence_not_global_infeasibility(self):
        reports=h0_memory_check(load_profiles())
        self.assertEqual(reports['4B']['h0_observed_training_peak_bytes'],48588302528)
        self.assertFalse(reports['4B']['unchanged_peak_vs_capacity']['H2'])
        self.assertTrue(reports['4B']['unchanged_peak_vs_capacity']['H0'])
        self.assertEqual(contract(load_profiles()['H3'])['assumed_upper_bound_dram_path_TB_s'],2.0)

    def test_real_controller_mesh_view(self):
        for profile in load_profiles().values():
            view=mesh_view_for(profile);summary=contract(profile)
            self.assertEqual(len(view.die_ids),profile['dies'])
            self.assertEqual(len(view.link_bytes_per_ns),2*summary['mesh_undirected_edges'])
            self.assertEqual(view.coordinates[profile['dies']-1],(profile['rows']-1,profile['columns']-1))
            self.assertEqual(view.capacity_bytes[0],int(profile['dram_GB_per_die']*1e9))
            self.assertEqual(view.link_bytes_per_ns[(0,1)],profile['d2d_TB_s_per_edge_direction']*1000)

    def test_native_gate_rejects_silent_h0_fallback(self):
        profiles=load_profiles();self.assertTrue(validate_existing_native_runner(profiles['H0']))
        for name in ('H1','H2','H3'):
            with self.assertRaises(ValueError):validate_existing_native_runner(profiles[name])
        fake=dict(profiles['H0'],dram_GB_per_die=32)
        with self.assertRaises(ValueError):validate_existing_native_runner(fake)
        self.assertFalse(contract(fake)['native_final_method_supported'])

if __name__=='__main__':unittest.main()
