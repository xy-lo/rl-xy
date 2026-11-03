import unittest
from copy import deepcopy
from idea1_controller.workload import WorkloadSpec, Observation, DistributionSnapshot
from idea1_controller.history import digest, predict_groups
from idea1_controller.controlled_forecast import controlled_forecasts

class ControlledForecastTests(unittest.TestCase):
    def setUp(self):
        self.spec=WorkloadSpec('contract')
        self.public=[dict(group_id='known',prompt_length=12,prompt_sha256='target',library='code')]
        records=[dict(proxy_round=i,actor_update=None,samples=[
            dict(sample=i*8+j,response_length=100+i*100+j,finish_reason='stop') for j in range(8)]) for i in range(4)]
        self.source=dict(model='4B',kind='offline-independent-same-prompt-proxy',chronological=False,
            model_config_sha256='cfg',records={'known':records},scenario_seed='legacy-seed',
            task_bindings={'known':dict(prompt_sha256='target',prompt_length=12,library='code')},
            excluded_sample_ids={'known':[99]},source_sha256='source')
        self.rehash()
        rows=tuple(Observation('cal'+str(i),12,'code','s'+str(i),(10+i,)*8,(False,)*8) for i in range(4))
        self.global_snapshot=DistributionSnapshot.build(rows,self.spec,kind='offline-calibration-proxy',source_sha256='cal')
    def rehash(self):
        self.source['snapshot_id']=digest({k:v for k,v in self.source.items() if k!='snapshot_id'})
    def call(self,**kw):
        return controlled_forecasts(self.public,self.source,self.global_snapshot,self.spec,
            model='4B',config_sha256='cfg',source_file_sha256='file',**kw)
    def test_explicit_opt_in(self):
        with self.assertRaises(ValueError):self.call()
    def test_alpha_one_reproduces_old_information(self):
        old=predict_groups(self.public,self.source,proxy_adaptation=False)['known']
        snap,fs=self.call(allow_controlled_proxy=True,exact_weight=1)
        f=fs['known']
        self.assertEqual(f['response_scenarios'],old.response_scenarios)
        self.assertEqual(f['score'],old.score)
        self.assertEqual(len(f['conditional_lengths']),32)
        self.assertFalse(snap.to_dict()['chronological'])
        self.assertEqual(f['evidence_kind'],'OFFLINE_EXACT_PROXY_PLUS_GLOBAL')
    def test_runtime_preserves_raw_support_count(self):
        from types import SimpleNamespace
        from idea1_controller.native.history_help import HistoricalAdaptiveKernel
        _,fs=self.call(allow_controlled_proxy=True,exact_weight=1)
        obj=SimpleNamespace()
        HistoricalAdaptiveKernel.init_workload_help(obj,fs)
        self.assertEqual(len(obj.helper_history['known']),32)
        self.assertEqual(obj.helper_prediction_kinds['known'],'raw-offline-disjoint-same-prompt-proxy')
        fs['known']['chronological']=True
        with self.assertRaises(ValueError):HistoricalAdaptiveKernel.init_workload_help(obj,fs)
    def test_alpha_zero_global(self):
        _,fs=self.call(allow_controlled_proxy=True,exact_weight=0)
        self.assertEqual(fs['known']['evidence_kind'],'CLASS')
        self.assertFalse(fs['known']['controlled_proxy_exact'])
    def test_mixture_draws_whole_joint_vectors(self):
        _,fs=self.call(allow_controlled_proxy=True,exact_weight=.5)
        # Every vector is one exact or independent calibration K-vector.
        allowed={tuple(s['response_length'] for s in row['samples']) for row in self.source['records']['known']}
        allowed|={r.lengths for r in self.global_snapshot.records}
        self.assertTrue(set(fs['known']['response_scenarios'])<=allowed)
        self.assertEqual(fs,self.call(allow_controlled_proxy=True,exact_weight=.5)[1])
    def test_unknown_and_sparse_fallback(self):
        self.public.append(dict(self.public[0],group_id='new',prompt_sha256='unknown'))
        _,fs=self.call(allow_controlled_proxy=True)
        self.assertEqual(fs['new']['evidence_kind'],'CLASS')
        self.source['records']['known']=self.source['records']['known'][:1];self.rehash()
        self.assertEqual(self.call(allow_controlled_proxy=True)[1]['known']['evidence_kind'],'CLASS')
    def test_identity_and_current_truth_rejected(self):
        for field,value in [('prompt_sha256','wrong'),('response_length',10)]:
            before=deepcopy(self.public);self.public[0][field]=value
            with self.assertRaises(ValueError):self.call(allow_controlled_proxy=True)
            self.public=before
        self.source['model']='14B';self.rehash()
        with self.assertRaises(ValueError):self.call(allow_controlled_proxy=True)
    def test_exclusion_and_duplicate_fail_closed(self):
        for mode in ('excluded','duplicate','missing-proof'):
            before=deepcopy(self.source)
            if mode=='excluded':self.source['excluded_sample_ids']['known']=[0]
            elif mode=='duplicate':self.source['records']['known'][1]['samples'][0]['sample']=0
            else:self.source['excluded_sample_ids'].pop('known')
            self.rehash()
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.call(allow_controlled_proxy=True)
            self.source=before
    def test_invalid_weights(self):
        for w in (-1,2,float('nan')):
            with self.assertRaises(ValueError):self.call(allow_controlled_proxy=True,exact_weight=w)

if __name__=='__main__':unittest.main()
