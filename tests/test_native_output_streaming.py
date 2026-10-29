"""Long native diagnostic output must go to files, not memory PIPEs."""
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from rl_xy.astra_adapter.runner import run_astrasim,PINNED_ASTRA_COMMIT


class NativeOutputStreamingTests(unittest.TestCase):
    def test_large_stdout_is_streamed_and_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);binary=root/'fake-native';binary.write_text('#!/usr/bin/env python3\nimport sys,pathlib\np=next(x.split("=",1)[1] for x in sys.argv if x.startswith("--logging-folder="))\npathlib.Path(p,"log.log").write_text("native diagnostic fixture\\n")\nsys.stdout.write("x"*(2*1024*1024))\n');binary.chmod(0o755)
            for name in ['system.json','network.yml','remote.json']:(root/name).write_text('{}')
            (root/'workload.0.et').write_bytes(b'fixture')
            with patch('rl_xy.astra_adapter.runner._astra_commit',return_value=PINNED_ASTRA_COMMIT),patch('rl_xy.astra_adapter.runner.subprocess.run',wraps=subprocess.run) as run:
                r=run_astrasim(trace_prefix=root/'workload',system_config=root/'system.json',network_config=root/'network.yml',remote_memory_config=root/'remote.json',output_dir=root/'out',binary=binary)
            self.assertEqual(r.return_code,0)
            self.assertEqual(r.stdout_path.stat().st_size,2*1024*1024)
            self.assertEqual(r.stderr_path.read_text(),'')
            self.assertNotIn('capture_output',run.call_args.kwargs)
            self.assertEqual(Path(run.call_args.kwargs['stdout'].name),r.stdout_path)
            self.assertEqual(json.loads((root/'out/astra_command.json').read_text())['exit_status'],0)
