"""Contract-level evidence: quotas, no live R, mixed-TP coordinate coverage."""
import unittest,math
from types import SimpleNamespace
from idea1_controller.placement import largest_remainder,guillotine_layouts,assign_groups
from idea1_controller.coordinates import ParameterCoordinates
from idea1_controller.search import RiskSelector
from idea1_controller.history import GroupForecast

class PlanningContracts(unittest.TestCase):
    def test_54_die_remainder_is_never_discarded(self):
        for tp in (1,2):
            layouts=guillotine_layouts(tp)
            for count,variants in layouts.items():
                for regions in variants:
                    self.assertEqual(len(regions),count)
                    self.assertEqual(sorted(d for r in regions for d in r.dies(tp)),list(range(54)))
                    self.assertTrue(all(r.tp(tp) in (1,2,4,8) for r in regions))
    def test_192_groups_exact_quota_and_long_dispersion(self):
        groups=[str(i) for i in range(192)];forecasts={g:GroupForecast(g,530,((100,)*8,)*16,float(192-int(g)),0,1,'proxy-long' if int(g)<49 else 'proxy-short' if int(g)>=130 else 'ordinary','proxy') for g in groups}
        regions={r:list(range(3*r,3*r+3)) for r in range(9)};quotas=dict(enumerate(largest_remainder(192,[1]*27)))
        queues,longq=assign_groups(groups,forecasts,regions,quotas)
        self.assertEqual(sorted(g for q in queues.values() for g in q),sorted(groups))
        for j,q in queues.items():self.assertEqual(len(q),quotas[j])
        total=192
        for r,ws in regions.items():
            ideal=49*sum(quotas[j] for j in ws)/total
            self.assertLess(abs(longq[r]-ideal),1)
    def test_strided_qkv_ffn_tied_coordinate_restore(self):
        m=SimpleNamespace(hidden_size=16,ffn_size=32,query_width=32,kv_heads=8,head_dim=4,vocab_size=64,layers=2,qk_norm=True,qkv_bias=True,tie_word_embeddings=True)
        coords=ParameterCoordinates(m);self.assertFalse(any(c.parameter_id=='lm_head' for c in coords.components))
        for pid in sorted({c.parameter_id for c in coords.components}):
            for tp in (2,4,8):
                for old in range(2):
                    descriptors=[coords.restore_descriptor(pid,tp,old,k) for k in range(tp//2)]
                    for ci,t in enumerate(descriptors[0]['target']):
                        if t['axis'] is not None:
                            ranges=[d['source'][ci] for d in descriptors]
                            self.assertEqual(ranges[0]['start'],t['start']);self.assertEqual(ranges[-1]['end'],t['end'])
                            self.assertTrue(all(a['end']==b['start'] for a,b in zip(ranges,ranges[1:])))
                            self.assertEqual(sum(x['elements'] for x in ranges),t['elements'])
    def test_risk_gate_can_keep_legal_baseline(self):
        baseline=dict(id='base',scenario_ns=[100e9]*16,pressure_ns=110e9)
        marginal=dict(id='tiny',scenario_ns=[99e9]*16,pressure_ns=109e9)
        badtail=dict(id='badpressure',scenario_ns=[70e9]*16,pressure_ns=150e9)
        best=dict(id='good',scenario_ns=[80e9]*16,pressure_ns=105e9)
        selector=RiskSelector();winner,a=selector.select(baseline,[marginal,badtail],1)
        self.assertEqual(winner['id'],'base')
        winner,a=selector.select(baseline,[best],1);self.assertEqual(winner['id'],'good')

if __name__=='__main__':unittest.main()
