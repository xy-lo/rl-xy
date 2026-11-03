#!/usr/bin/env python3
"""Rebuild the message-boundary mesh extension in a separate remote tree."""
from pathlib import Path
import os,json,subprocess,shutil,time,hashlib
h=Path.home();w=Path(__file__).resolve().parents[2];o=h/'outputs/idea1-final-t162';base=o/'native-mesh';n=base/'astra-sim'
ext=w/'native_extensions/mesh_runtime';seed=h/'rl-xy/.deps/astra-sim';manifest=json.loads((ext/'manifest.json').read_text())
status={'complete':False,'commands':[],'extension_manifest_sha256':hashlib.sha256((ext/'manifest.json').read_bytes()).hexdigest()};state=o/'mesh-backend-build-status.json'
def save():state.write_text(json.dumps(status,indent=2))
def run(cmd,cwd=None,env=None,timeout=300):
 cmd=list(map(str,cmd));status['phase']=' '.join(cmd);save();start=time.time()
 with (o/'mesh-backend-build.log').open('a') as log:
  log.write('\nCOMMAND '+repr(cmd)+'\n');log.flush();r=subprocess.run(cmd,cwd=cwd or n,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 status['commands'].append(dict(argv=cmd,returncode=r.returncode,wall_s=time.time()-start));save()
 if r.returncode:raise RuntimeError('mesh backend command failed')
try:
 base.mkdir(exist_ok=True)
 if not n.exists():shutil.copytree(seed,n,symlinks=True)
 patch=w/'experiments/2026-11-01-idea1/astra-source/astra-sequence.patch'
 marker=n/'.rlxy-base-patch.json'
 if not marker.exists():
  clean=subprocess.run(['git','apply','--check',str(patch)],cwd=n,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
  if clean:run(['git','apply',patch])
  else:
   # A failed compile may leave a fully applied patch plus the extension.
   # Verify all other changed baseline sources against the built baseline.
   baseline=o/'native/astra-sim'
   names=subprocess.check_output(['git','diff','--name-only'],cwd=baseline,text=True).splitlines()
   for rel in names:
    if rel not in manifest['files'] and (n/rel).read_bytes()!=(baseline/rel).read_bytes():raise ValueError('non-extension baseline drift: '+rel)
  marker.write_text(json.dumps({'patch_sha256':hashlib.sha256(patch.read_bytes()).hexdigest()}))
 elif json.loads(marker.read_text())['patch_sha256']!=hashlib.sha256(patch.read_bytes()).hexdigest():raise ValueError('baseline patch changed')
 for p in patch.parent.glob('Rlxy*.hh'):shutil.copy2(p,n/'astra-sim/workload'/p.name)
 shutil.copy2(patch.parent/'packet_fixture.cc',n/'build/astra_analytical/packet_fixture.cc')
 previous=json.loads((n/'.rlxy-extension-applied.json').read_text()) if (n/'.rlxy-extension-applied.json').exists() else {}
 for rel,proof in manifest['files'].items():
  p=n/rel
  if hashlib.sha256(p.read_bytes()).hexdigest() not in (proof['base_sha256'],proof['new_sha256'],previous.get(rel)):raise ValueError('extension base differs: '+rel)
  if hashlib.sha256((ext/rel).read_bytes()).hexdigest()!=proof['new_sha256']:raise ValueError('extension bytes changed: '+rel)
  shutil.copy2(ext/rel,p)
 (n/'.rlxy-extension-applied.json').write_text(json.dumps({k:v['new_sha256'] for k,v in manifest['files'].items()},indent=2))
 prefix=o/'native/system-deps/usr';lib=prefix/'lib/x86_64-linux-gnu';env=dict(os.environ,LD_LIBRARY_PATH=str(lib))
 proto=n/'extern/graph_frontend/chakra/schema/protobuf'
 run([prefix/'bin/protoc','et_def.proto','--proto_path='+str(proto),'--cpp_out='+str(proto),'--python_out='+str(proto)],env=env)
 cmake=o/'venv/bin/cmake'
 run([cmake,'-S',n/'build/astra_analytical','-B',n/'build-dynarl','-DCMAKE_BUILD_TYPE=Release','-DBUILDTARGET=congestion_aware','-DProtobuf_INCLUDE_DIR='+str(prefix/'include'),'-DProtobuf_LIBRARY='+str(lib/'libprotobuf.so'),'-DProtobuf_PROTOC_EXECUTABLE='+str(prefix/'bin/protoc'),'-DCMAKE_BUILD_RPATH='+str(lib)],env=env)
 run([cmake,'--build',n/'build-dynarl','--parallel','4'],env=env,timeout=1800)
 binary=n/'build-dynarl/bin/AstraSim_Analytical_Congestion_Aware'
 status.update(phase='PASS',complete=True,binary=str(binary),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest());save()
except Exception as e:
 status.update(phase='FAILED',error=repr(e));save();raise
