#!/usr/bin/env python3
"""Build the pinned mesh backend in an explicit isolated directory.

--verify-only is read-only. Building never edits the seed. A complete matching
output is reused, while unrelated or drifting output trees are rejected.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,time
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
BASE_COMMIT='4ce9ecfe746a3ee5fe47559a11d91644d641fa7f'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())

def verify_backend(native,extension=ROOT/'native_extensions/mesh_runtime'):
    proof=load(native/'rlxy-mesh-backend.json');manifest=load(extension/'manifest.json')
    binary=native/'build-dynarl/bin/AstraSim_Analytical_Congestion_Aware'
    if proof['binary_sha256']!=sha(binary) or proof['extension_manifest_sha256']!=sha(extension/'manifest.json'):
        raise ValueError('binary or extension identity differs')
    expected={name:item['new_sha256'] for name,item in manifest['files'].items()}
    if proof['source_sha256']!=expected or any(sha(native/name)!=value for name,value in expected.items()):
        raise ValueError('compiled source identity differs')
    if proof['base_commit']!=BASE_COMMIT or subprocess.check_output(['git','rev-parse','HEAD'],cwd=native,text=True).strip()!=BASE_COMMIT:
        raise ValueError('unexpected upstream revision')
    return proof

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='isolated astra-sim source/build directory')
    parser.add_argument('--seed',type=Path,help='clean initialized upstream checkout at the pinned commit')
    parser.add_argument('--protobuf-prefix',type=Path,help='prefix containing include/, bin/protoc and lib/x86_64-linux-gnu/libprotobuf.so')
    parser.add_argument('--cmake',default=shutil.which('cmake') or 'cmake')
    parser.add_argument('--jobs',type=int,default=4);parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--log-root',type=Path)
    a=parser.parse_args();native=a.output.expanduser().resolve();extension=ROOT/'native_extensions/mesh_runtime'
    if a.jobs<1:parser.error('--jobs must be positive')
    if (native/'rlxy-mesh-backend.json').exists():
        proof=verify_backend(native,extension)
        print(json.dumps(dict(status='PASS_EXISTING_MESH_BACKEND',read_only=True,**proof),indent=2));return
    if a.verify_only:raise ValueError('existing compiled proof required for read-only verification')
    if not a.seed or not a.protobuf_prefix:parser.error('building requires --seed and --protobuf-prefix')
    seed=a.seed.expanduser().resolve();prefix=a.protobuf_prefix.expanduser().resolve()
    if native==seed or native in seed.parents or seed in native.parents:
        raise ValueError('seed and output must be separate non-nested trees')
    if native==ROOT or native in ROOT.parents or ROOT in native.parents:
        raise ValueError('native output must be outside the source repository')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=seed,text=True).strip()!=BASE_COMMIT:
        raise ValueError('unexpected seed revision')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=seed,text=True).strip():
        raise ValueError('seed must be clean')
    timestamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    evidence=(a.log_root or native.parent/('mesh-build-'+timestamp)).resolve();evidence.mkdir(parents=True,exist_ok=False)
    state=dict(complete=False,base_commit=BASE_COMMIT,extension_manifest_sha256=sha(extension/'manifest.json'),commands=[])
    def save(): (evidence/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    def run(cmd,timeout=300,env=None):
        cmd=list(map(str,cmd));state['phase']=cmd[0];save();start=time.monotonic()
        with (evidence/'build.log').open('a') as log:
            log.write('\nCOMMAND '+repr(cmd)+'\n');log.flush()
            result=subprocess.run(cmd,cwd=native,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        state['commands'].append(dict(argv=cmd,returncode=result.returncode,wall_s=time.monotonic()-start));save()
        if result.returncode:raise RuntimeError('build command failed; evidence '+str(evidence))
    try:
        fresh=not native.exists()
        if fresh:shutil.copytree(seed,native,symlinks=True,ignore=shutil.ignore_patterns('build-dynarl','build-dynarl-*','__pycache__','*.pyc'))
        top=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=native,text=True).strip()).resolve()
        if top!=native:raise ValueError('copied Git directory resolves outside isolated native tree')
        patch=ROOT/'experiments/2026-11-01-idea1/astra-source/astra-sequence.patch'
        marker=native/'.rlxy-base-patch.json'
        if fresh:
            run(['git','apply','--check',patch]);run(['git','apply',patch])
            for file in patch.parent.glob('Rlxy*.hh'):shutil.copy2(file,native/'astra-sim/workload'/file.name)
            shutil.copy2(patch.parent/'packet_fixture.cc',native/'build/astra_analytical/packet_fixture.cc')
            changed=[line[6:] for line in patch.read_text().splitlines() if line.startswith('+++ b/')]
            changed+=['astra-sim/workload/'+p.name for p in patch.parent.glob('Rlxy*.hh')]
            changed+=['build/astra_analytical/packet_fixture.cc']
            marker.write_text(json.dumps(dict(patch_sha256=sha(patch),files={n:sha(native/n) for n in sorted(set(changed))}),indent=2))
        baseline=load(marker)
        if baseline['patch_sha256']!=sha(patch) or 'files' not in baseline:raise ValueError('missing exact base patch certificate; use a fresh output directory')
        manifest=load(extension/'manifest.json')
        for name,value in baseline['files'].items():
            if name not in manifest['files'] and sha(native/name)!=value:raise ValueError('non-extension baseline drift '+name)
        for name,item in manifest['files'].items():
            if sha(native/name) not in (item['base_sha256'],item['new_sha256']):raise ValueError('extension base drift '+name)
            if sha(extension/name)!=item['new_sha256']:raise ValueError('extension source differs '+name)
            shutil.copy2(extension/name,native/name)
        lib=prefix/'lib/x86_64-linux-gnu'
        for file in (prefix/'bin/protoc',prefix/'include/google/protobuf/message.h',lib/'libprotobuf.so'):
            if not file.exists():raise ValueError('missing protobuf dependency '+str(file))
        env=dict(os.environ,LD_LIBRARY_PATH=str(lib)+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else ''))
        proto=native/'extern/graph_frontend/chakra/schema/protobuf'
        run([prefix/'bin/protoc','et_def.proto','--proto_path='+str(proto),'--cpp_out='+str(proto),'--python_out='+str(proto)],env=env)
        run([a.cmake,'-S',native/'build/astra_analytical','-B',native/'build-dynarl','-DCMAKE_BUILD_TYPE=Release','-DBUILDTARGET=congestion_aware',
             '-DProtobuf_INCLUDE_DIR='+str(prefix/'include'),'-DProtobuf_LIBRARY='+str(lib/'libprotobuf.so'),
             '-DProtobuf_PROTOC_EXECUTABLE='+str(prefix/'bin/protoc'),'-DCMAKE_BUILD_RPATH='+str(lib)],env=env)
        run([a.cmake,'--build',native/'build-dynarl','--parallel',str(a.jobs)],timeout=1800,env=env)
        binary=native/'build-dynarl/bin/AstraSim_Analytical_Congestion_Aware'
        proof=dict(schema='rlxy.mesh-observed/2',base_commit=BASE_COMMIT,binary_sha256=sha(binary),
            extension_manifest_sha256=sha(extension/'manifest.json'),
            source_sha256={name:item['new_sha256'] for name,item in manifest['files'].items()})
        (native/'rlxy-mesh-backend.json').write_text(json.dumps(proof,indent=2)+'\n')
        verify_backend(native,extension)
        state.update(complete=True,phase='PASS',binary_sha256=proof['binary_sha256']);save()
        print(json.dumps(dict(status='PASS_MESH_BACKEND_BUILD',evidence=str(evidence),**proof),indent=2))
    except BaseException as error:
        state.update(phase='FAILED',error=repr(error));save();raise
if __name__=='__main__':main()
