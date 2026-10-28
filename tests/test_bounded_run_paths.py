"""No Docker calls: only dedicated artifact ownership and mount arguments."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('bounded_run',ROOT/'scripts/bounded_run.py')
bounded=importlib.util.module_from_spec(spec);spec.loader.exec_module(bounded)


class BoundedRunPathTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name).resolve();self.repo=self.base/'rl-xy';self.repo.mkdir()
        self.store=self.base/'nvme'/'owned-artifacts';self.store.mkdir(parents=True)
        (self.repo/'runs').symlink_to(self.store,target_is_directory=True)
        self.home_runs=self.repo/'runs-home-T42';self.home_runs.mkdir()

    def test_default_resolves_existing_symlink_without_changing_it(self):
        record=self.repo/'runs'/'T42'/'new-run'
        actual,selected=bounded.resolve_artifact_paths(self.repo,None,record)
        self.assertEqual(actual,self.store);self.assertEqual(selected,self.store/'T42'/'new-run')
        self.assertTrue((self.repo/'runs').is_symlink());self.assertFalse(selected.exists())
        self.assertEqual(bounded.resolve_artifact_paths(self.repo,self.store,record),(actual,selected))

    def test_explicit_home_root_is_direct_child_and_record_can_be_nested(self):
        actual,record=bounded.resolve_artifact_paths(self.repo,self.home_runs,self.home_runs/'T42'/'W1')
        self.assertEqual(actual,self.home_runs);self.assertEqual(record,self.home_runs/'T42'/'W1')
        self.assertEqual(bounded.resolve_artifact_paths(self.repo,Path('runs-home-T42'),record),(actual,record))

    def test_root_repository_ancestor_arbitrary_and_nonexistent_roots_are_rejected(self):
        arbitrary=self.repo/'other-runs';arbitrary.mkdir()
        external=self.base/'other-user'/'runs-home-T42';external.mkdir(parents=True)
        for candidate in [Path('/'),self.repo,self.base,arbitrary,external,self.repo/'missing',self.repo/'nested'/'runs-home-T42']:
            with self.subTest(root=candidate),self.assertRaises(ValueError):
                bounded.resolve_artifact_paths(self.repo,candidate,candidate/'new')
        self.home_runs.rmdir()
        with self.assertRaisesRegex(ValueError,'existing directory'):
            bounded.resolve_artifact_paths(self.repo,self.home_runs,self.home_runs/'new')

    def test_record_equal_outside_dotdot_and_symlink_escape_are_rejected(self):
        escape=self.home_runs/'escape';escape.symlink_to(self.base,target_is_directory=True)
        for record in [self.home_runs,self.store/'W1',self.home_runs/'..'/'outside',escape/'new']:
            with self.subTest(record=record),self.assertRaisesRegex(ValueError,'strict descendant'):
                bounded.resolve_artifact_paths(self.repo,self.home_runs,record)

    def test_alternative_symlink_cannot_redirect_to_default_or_external_root(self):
        self.home_runs.rmdir();self.home_runs.symlink_to(self.store,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'real dedicated'):
            bounded.resolve_artifact_paths(self.repo,self.home_runs,self.home_runs/'W1')
        with self.assertRaisesRegex(ValueError,'real dedicated'):
            bounded.resolve_artifact_paths(self.repo,self.repo/'unused'/'..'/'runs-home-T42',self.home_runs/'W1')
        # The original default alias still works and remains unmodified.
        bounded.resolve_artifact_paths(self.repo,None,self.repo/'runs'/'new')
        self.assertEqual((self.repo/'runs').resolve(),self.store)

    def test_image_source_mounts_actual_root_and_runs_alias_without_source_bind(self):
        args=bounded.artifact_mount_args(self.repo,self.home_runs,image_source=True)
        self.assertEqual(args,['--volume',str(self.home_runs)+':'+str(self.home_runs),
            '--volume',str(self.home_runs)+':/workspace/rl-xy/runs'])
        self.assertNotIn(str(self.repo)+':/workspace/rl-xy:ro',args)
        self.assertNotIn(str(self.store),args)
        default_args=bounded.artifact_mount_args(self.repo,self.store,image_source=False)
        self.assertEqual(default_args,['--volume',str(self.repo)+':/workspace/rl-xy:ro',
            '--volume',str(self.store)+':'+str(self.store)])


if __name__=='__main__':unittest.main()
