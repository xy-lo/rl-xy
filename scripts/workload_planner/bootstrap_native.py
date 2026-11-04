#!/usr/bin/env python3
"""Build an isolated Linux Idea1 backend without sudo or editing the seed."""
from pathlib import Path
import os,sys,json,time,subprocess,shutil,hashlib
h=Path.home(); w=Path(__file__).resolve().parents[2]; out=h/'outputs/idea1-final-t162'
base=out/'native'; base.mkdir(exist_ok=True); seed=h/'rl-xy/.deps/astra-sim'; native=base/'astra-sim'
status={'phase':'initializing','complete':False,'commands':[]}
state=out/'native-bootstrap-status.json'
def save():state.write_text(json.dumps(status,indent=2)+'\n')
def run(cmd,*,cwd=base,env=None,timeout=120):
 status['phase']=' '.join(map(str,cmd));save();start=time.time()
 with (out/'native-bootstrap.log').open('a') as log:
  log.write('\nCOMMAND '+repr(cmd)+'\n');log.flush()
  r=subprocess.run(list(map(str,cmd)),cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 status['commands'].append({'argv':list(map(str,cmd)),'returncode':r.returncode,'wall_s':time.time()-start});save()
 if r.returncode:raise RuntimeError('command failed; inspect native-bootstrap.log')
try:
 if not native.exists():shutil.copytree(seed,native,symlinks=True)
 expected='4ce9ecfe746a3ee5fe47559a11d91644d641fa7f'
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=native,text=True).strip()!=expected:raise ValueError('unexpected seed revision')
 run([h/'.local/bin/uv','pip','install','--python',out/'venv/bin/python3','cmake==3.31.6'],timeout=180)
 packages=base/'packages';packages.mkdir(exist_ok=True)
 run(['apt-get','download','libprotobuf-dev','libprotobuf32t64','libprotoc32t64','protobuf-compiler'],cwd=packages,timeout=180)
 prefix=base/'system-deps';prefix.mkdir(exist_ok=True)
 for deb in packages.glob('*.deb'):run(['dpkg-deb','-x',deb,prefix])
 lib=prefix/'usr/lib/x86_64-linux-gnu';env=dict(os.environ,LD_LIBRARY_PATH=str(lib)+':'+os.environ.get('LD_LIBRARY_PATH',''))
 patch=w/'experiments/2026-11-01-idea1/astra-source/astra-sequence.patch'
 run(['git','apply','--check',patch],cwd=native)
 run(['git','apply',patch],cwd=native)
 for p in (patch.parent).glob('Rlxy*.hh'):shutil.copy2(p,native/'astra-sim/workload'/p.name)
 shutil.copy2(patch.parent/'packet_fixture.cc',native/'build/astra_analytical/packet_fixture.cc')
 proto=native/'extern/graph_frontend/chakra/schema/protobuf'
 run([prefix/'usr/bin/protoc','et_def.proto','--proto_path='+str(proto),'--cpp_out='+str(proto),'--python_out='+str(proto)],env=env)
 cmake=out/'venv/bin/cmake'
 run([cmake,'-S',native/'build/astra_analytical','-B',native/'build-dynarl','-DCMAKE_BUILD_TYPE=Release','-DBUILDTARGET=congestion_aware','-DProtobuf_INCLUDE_DIR='+str(prefix/'usr/include'),'-DProtobuf_LIBRARY='+str(lib/'libprotobuf.so'),'-DProtobuf_PROTOC_EXECUTABLE='+str(prefix/'usr/bin/protoc'),'-DCMAKE_BUILD_RPATH='+str(lib)],env=env,timeout=240)
 run([cmake,'--build',native/'build-dynarl','--parallel','4'],env=env,timeout=1800)
 binary=native/'build-dynarl/bin/AstraSim_Analytical_Congestion_Aware'
 if not binary.exists():raise ValueError('build returned without expected binary')
 status.update(phase='PASS',complete=True,binary=str(binary),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),dependencies_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in packages.glob('*.deb')});save()
except Exception as e:
 status.update(phase='FAILED',error=repr(e));save();raise
