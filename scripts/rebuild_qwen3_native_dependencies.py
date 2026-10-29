from pathlib import Path
import urllib.request,tarfile,subprocess,os
root=Path.cwd();base=root/'native';prefix=base/'source-deps';deps=base/'deps'
for name,url in [('abseil-cpp-20240722.0','https://github.com/abseil/abseil-cpp/releases/download/20240722.0/abseil-cpp-20240722.0.tar.gz'),('protobuf-29.0','https://github.com/protocolbuffers/protobuf/releases/download/v29.0/protobuf-29.0.tar.gz')]:
 archive=base/(name+'.tar.gz')
 if not archive.exists():archive.write_bytes(urllib.request.urlopen(url,timeout=60).read())
 source=base/name
 if not source.exists():
  with tarfile.open(archive) as t:t.extractall(base)
 cmd=[str(deps/'bin/cmake'),'-S',str(source),'-B',str(source/'build'),'-DCMAKE_CXX_STANDARD=17','-DCMAKE_BUILD_TYPE=Release',f'-DCMAKE_INSTALL_PREFIX={prefix}',f'-DCMAKE_PREFIX_PATH={prefix}','-DCMAKE_POSITION_INDEPENDENT_CODE=ON','-DBUILD_SHARED_LIBS=ON','-DCMAKE_POLICY_VERSION_MINIMUM=3.5']
 if name.startswith('protobuf'):cmd+=['-Dprotobuf_BUILD_TESTS=OFF','-Dprotobuf_ABSL_PROVIDER=package']
 subprocess.run(cmd,check=True)
 subprocess.run([str(deps/'bin/cmake'),'--build',str(source/'build'),'--target','install','--parallel','4'],check=True)
up=base/'astra-sim';proto=up/'extern/graph_frontend/chakra/schema/protobuf'
subprocess.run([str(prefix/'bin/protoc'),'et_def.proto',f'--proto_path={proto}',f'--cpp_out={proto}',f'--python_out={proto}'],check=True)
env=dict(os.environ,PROTOBUF_FROM_SOURCE='True')
subprocess.run([str(deps/'bin/cmake'),'-S',str(up/'build/astra_analytical'),'-B',str(up/'build/astra_analytical/build'),f'-DCMAKE_PREFIX_PATH={prefix};{deps}',f'-Dprotobuf_DIR={prefix}/lib/cmake/protobuf',f'-Dabsl_DIR={prefix}/lib/cmake/absl',f'-DCMAKE_BUILD_RPATH={prefix}/lib',f'-DCMAKE_INSTALL_RPATH={prefix}/lib','-DCMAKE_POLICY_VERSION_MINIMUM=3.5'],env=env,check=True)
subprocess.run([str(deps/'bin/cmake'),'--build',str(up/'build/astra_analytical/build'),'--parallel','4'],env=env,check=True)
