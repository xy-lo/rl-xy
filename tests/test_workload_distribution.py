import unittest
from dataclasses import replace
from idea1_controller.workload import WorkloadSpec, Observation, DistributionSnapshot, forecasts
from idea1_controller.offline_candidates import freeze_candidates, select_precomputed

class WorkloadTests(unittest.TestCase):
    def setUp(self):
        self.spec=WorkloadSpec('4B-config-template-generation-contract', k=8)
        self.rows=tuple(Observation('cal'+str(i),10,'code','source'+str(i),(i+1,)*8,(False,)*8) for i in range(4))
        self.snapshot=DistributionSnapshot.build(self.rows,self.spec,kind='offline-calibration-proxy',source_sha256='source-digest')
        self.public=[dict(group_id='new',prompt_length=12,prompt_sha256='unseen',library='code')]
    def test_equal_conditions_have_equal_summary_not_sample_noise(self):
        p=[self.public[0],dict(self.public[0],group_id='new2',prompt_sha256='unseen2')]
        a=forecasts(p,self.snapshot,self.spec)
        for key in ('score','q25','q75','label'):
            self.assertEqual(a['new'][key],a['new2'][key])
        self.assertNotEqual(a['new']['forecast_version'],a['new2']['forecast_version'])
        self.assertNotEqual(a['new']['response_scenarios'],a['new2']['response_scenarios'])
        few=forecasts(p,self.snapshot,replace(self.spec,scenarios=3))
        self.assertEqual(a['new']['score'],few['new']['score'])
    def test_old_summary_cache_is_not_new_contract(self):
        art=freeze_candidates(self.public,self.snapshot,'catalog',dict(winner=dict(passed=True)),forecast_summary='legacy')
        _,why=select_precomputed(art,self.public,self.snapshot,'catalog',admit=lambda _:True,fallback=lambda:None)
        self.assertEqual(why,'IDENTITY_MISMATCH')
        art.pop('forecast_summary')
        from idea1_controller.workload import digest
        art['artifact_digest']=digest({k:v for k,v in art.items() if k!='artifact_digest'})
        _,why=select_precomputed(art,self.public,self.snapshot,'catalog',admit=lambda _:True,fallback=lambda:None,forecast_summary='legacy')
        self.assertEqual(why,'OFFLINE_CANDIDATE_REVALIDATED')
    def test_roundtrip_and_replay(self):
        snap=DistributionSnapshot.from_dict(self.snapshot.to_dict(),self.spec)
        a=forecasts(self.public,snap,self.spec)
        self.assertEqual(a,forecasts(self.public,snap,self.spec))
        self.assertEqual(a['new']['evidence_kind'],'CLASS')
        self.assertEqual(len(a['new']['response_scenarios']),16)
        self.assertTrue(all(len(x)==8 for x in a['new']['response_scenarios']))
    def test_global_fallback(self):
        p=[dict(self.public[0],library='unknown')]
        self.assertEqual(forecasts(p,self.snapshot,self.spec)['new']['evidence_kind'],'GLOBAL')
    def test_no_history_default_not_prediction(self):
        f=forecasts(self.public,None,self.spec)['new']
        self.assertEqual(f['evidence_kind'],'DEFAULT');self.assertEqual(f['label'],'unknown')
        self.assertEqual(f['response_scenarios'][0],(8192,)*8)
    def test_sparse_global_default(self):
        s=DistributionSnapshot.build(self.rows[:1],self.spec,kind='offline-calibration-proxy',source_sha256='source')
        self.assertEqual(forecasts(self.public,s,self.spec)['new']['evidence_kind'],'DEFAULT')
    def test_eval_overlap_rejected(self):
        with self.assertRaises(ValueError): forecasts([dict(self.public[0],prompt_sha256='cal0')],self.snapshot,self.spec)
    def test_current_response_rejected(self):
        for field in ('response_length','eos','reward','test_label'):
            with self.subTest(field=field),self.assertRaises(ValueError): forecasts([dict(self.public[0],**{field:1})],self.snapshot,self.spec)
    def test_future_and_missing_sequence(self):
        with self.assertRaises(ValueError): DistributionSnapshot.build(self.rows,self.spec,kind='committed-history',source_sha256='x')
        rows=tuple(replace(r,observed_seq=i) for i,r in enumerate(self.rows))
        s=DistributionSnapshot.build(rows,self.spec,kind='committed-history',source_sha256='x')
        with self.assertRaises(ValueError): forecasts(self.public,s,self.spec,decision_seq=3)
        self.assertIn('new',forecasts(self.public,s,self.spec,decision_seq=4))
    def test_proxy_cannot_fake_chronology_or_exact(self):
        with self.assertRaises(ValueError): DistributionSnapshot.build([replace(self.rows[0],observed_seq=0)],self.spec,kind='offline-calibration-proxy',source_sha256='x')
        with self.assertRaises(ValueError): forecasts(self.public,self.snapshot,self.spec,exact=True)
    def test_bad_vectors_and_duplicate_sources(self):
        for bad in (replace(self.rows[0],lengths=(1,)),replace(self.rows[0],lengths=(0,)*8),replace(self.rows[0],lengths=(8193,)*8),replace(self.rows[0],truncated=())):
            with self.subTest(bad=bad),self.assertRaises(ValueError): DistributionSnapshot.build([bad],self.spec,kind='offline-calibration-proxy',source_sha256='x')
        with self.assertRaises(ValueError): DistributionSnapshot.build(self.rows+self.rows,self.spec,kind='offline-calibration-proxy',source_sha256='x')
    def test_version_seed_contract(self):
        with self.assertRaises(ValueError): forecasts(self.public,self.snapshot,replace(self.spec,contract='other'))
        a=forecasts(self.public,self.snapshot,self.spec)['new']
        b=forecasts(self.public,self.snapshot,replace(self.spec,seed=163))['new']
        self.assertNotEqual(a['forecast_version'],b['forecast_version'])
        d=self.snapshot.to_dict();d['source_sha256']='changed'
        with self.assertRaises(ValueError): DistributionSnapshot.from_dict(d,self.spec)
    def test_public_limits(self):
        for p in ([dict(self.public[0],prompt_length=20000)],self.public*2,[]):
            with self.assertRaises(ValueError): forecasts(p,self.snapshot,self.spec)
    def test_offline_identity_and_current_capacity(self):
        art=freeze_candidates(self.public,self.snapshot,'catalog',dict(winner=dict(passed=True),inference={2:[2],10:[10]}))
        import json
        art=json.loads(json.dumps(art))
        fallback=lambda: {'fallback':True}
        r,why=select_precomputed(art,self.public,self.snapshot,'catalog',admit=lambda _:True,fallback=fallback)
        self.assertEqual(why,'OFFLINE_CANDIDATE_REVALIDATED')
        r,why=select_precomputed(art,self.public,self.snapshot,'different-catalog',admit=lambda _:True,fallback=fallback)
        self.assertEqual(why,'IDENTITY_MISMATCH');self.assertTrue(r['fallback'])
        r,why=select_precomputed(art,self.public,self.snapshot,'catalog',admit=lambda _:False,fallback=fallback)
        self.assertEqual(why,'CURRENT_ADMISSION_FAILED')

if __name__=='__main__': unittest.main()

class HistorySourceFallbackTests(unittest.TestCase):
    def loader(self):
        import importlib.util,sys
        from pathlib import Path
        root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts/workload_planner'))
        spec=importlib.util.spec_from_file_location('workload_distribution_cli',root/'scripts/workload_planner/run_distribution.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module.load_calibration_history
    def test_disabled_source_is_never_read_or_required(self):
        from pathlib import Path
        from unittest.mock import patch
        load=self.loader()
        with patch.object(Path,'read_bytes',side_effect=AssertionError('unused history was read')):
            self.assertEqual(load(Path('/missing/history.gz'),[],'4B','cfg',disabled=True),(None,'EXPLICIT_NO_HISTORY'))
    def test_missing_source_has_explicit_fallback_reason(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.loader()(Path(d)/'absent.gz',[],'4B','cfg'),(None,'HISTORY_FILE_MISSING'))
    def test_corrupt_and_incompatible_existing_sources_fail_closed(self):
        import tempfile,gzip,json,copy
        from pathlib import Path
        from idea1_controller.history import digest
        load=self.loader()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'history.gz'
            for data in [b'not gzip',gzip.compress(b'[]')]:
                p.write_bytes(data)
                with self.assertRaises(ValueError):load(p,[],'4B','cfg')
            good=dict(schema='idea1.independent-history/1',model='4B',model_config_sha256='cfg',records={},task_bindings={})
            good['snapshot_id']=digest(good);p.write_bytes(gzip.compress(json.dumps(good).encode()))
            self.assertEqual(load(p,[],'4B','cfg'),(good,None))
            for field,value in [('model','8B'),('model_config_sha256','other'),('schema','bad')]:
                bad=copy.deepcopy(good);bad[field]=value;bad['snapshot_id']=digest({k:v for k,v in bad.items() if k!='snapshot_id'})
                p.write_bytes(gzip.compress(json.dumps(bad).encode()))
                with self.assertRaises(ValueError):load(p,[],'4B','cfg')

class CalibrationSourceAuditTests(unittest.TestCase):
    def test_published_snapshot_rebuild_and_self_consistent_poison_rejected(self):
        import json,sys,tarfile,tempfile,copy
        from pathlib import Path
        root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
        from audit_idea1_planning import audit_calibration_source
        folder=root/'experiments/2026-11-04-idea1-workload/development'
        manifest=json.loads((folder/'bundle-manifest.json').read_text());case=next(c for c in manifest['cases'] if c['id']=='late-cap1-controlled-full-01')
        with tarfile.open(folder/case['archive'],'r:gz') as archive:
            def read(name):return json.load(archive.extractfile('planning/offline/'+name))
            raw=read('global-distribution.json');public=read('public.json');split=read('split-manifest.json')
        spec=WorkloadSpec(**split['spec']);snapshot=DistributionSnapshot.from_dict(raw,spec)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);(path/'split-manifest.json').write_text(json.dumps(split))
            audit_calibration_source(path,public,spec,'4B',raw)
            first=snapshot.records[0];lengths=list(first.lengths);lengths[0]+=1 if lengths[0]<spec.output_max else -1
            poison=DistributionSnapshot.build((replace(first,lengths=tuple(lengths)),*snapshot.records[1:]),spec,kind=snapshot.kind,source_sha256=snapshot.source_sha256).to_dict()
            # Its own digest/contract remain valid; original source reconstruction rejects it.
            DistributionSnapshot.from_dict(poison,spec)
            with self.assertRaises(AssertionError):audit_calibration_source(path,public,spec,'4B',poison)
            bad=copy.deepcopy(split);bad['calibration_prompt_hashes'].append(public[0]['prompt_sha256']);(path/'split-manifest.json').write_text(json.dumps(bad))
            with self.assertRaises(AssertionError):audit_calibration_source(path,public,spec,'4B',raw)
    def test_no_snapshot_does_not_require_history_source(self):
        import sys
        from pathlib import Path
        from unittest.mock import patch
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from audit_idea1_planning import audit_calibration_source
        with patch.object(Path,'read_bytes',side_effect=AssertionError('unused source read')):
            audit_calibration_source(Path('/missing'),[],WorkloadSpec('contract'),'4B',None)
