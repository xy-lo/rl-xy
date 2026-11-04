import unittest,tempfile,json,hashlib,importlib.util,subprocess,sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('portable_mesh_builder',ROOT/'scripts/workload_planner/build_mesh_backend.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
sha=lambda value:hashlib.sha256(value).hexdigest()

class MeshBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.native=self.root/'native';self.ext=self.root/'extension';self.native.mkdir();self.ext.mkdir()
        self.binary=self.native/'build-dynarl/bin/AstraSim_Analytical_Congestion_Aware'
        self.binary.parent.mkdir(parents=True);self.binary.write_bytes(b'binary')
        (self.native/'kernel.cpp').write_bytes(b'code')
        manifest=dict(files={'kernel.cpp':dict(base_sha256=sha(b'old'),new_sha256=sha(b'code'))})
        (self.ext/'manifest.json').write_text(json.dumps(manifest))
        self.proof=dict(binary_sha256=sha(b'binary'),extension_manifest_sha256=builder.sha(self.ext/'manifest.json'),
            source_sha256={'kernel.cpp':sha(b'code')},base_commit=builder.BASE_COMMIT)
        (self.native/'rlxy-mesh-backend.json').write_text(json.dumps(self.proof))
    def test_readonly_verification(self):
        before={str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with patch.object(builder.subprocess,'check_output',return_value=builder.BASE_COMMIT+'\n'):
            self.assertEqual(builder.verify_backend(self.native,self.ext),self.proof)
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
    def test_binary_drift_rejected(self):
        self.binary.write_bytes(b'new binary')
        with self.assertRaises(ValueError):builder.verify_backend(self.native,self.ext)
    def test_source_drift_rejected(self):
        (self.native/'kernel.cpp').write_bytes(b'new source')
        with self.assertRaises(ValueError):builder.verify_backend(self.native,self.ext)
    def test_upstream_drift_rejected(self):
        with patch.object(builder.subprocess,'check_output',return_value='other'):
            with self.assertRaises(ValueError):builder.verify_backend(self.native,self.ext)
    def test_missing_verify_target_is_not_created(self):
        target=self.root/'must-stay-absent'
        result=subprocess.run([sys.executable,str(ROOT/'scripts/workload_planner/build_mesh_backend.py'),'--verify-only','--output',str(target)],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        self.assertNotEqual(result.returncode,0);self.assertFalse(target.exists())
if __name__=='__main__':unittest.main()
