#!/usr/bin/env python3
"""Actual directed-link packet checks, including unchanged default semantics."""
from pathlib import Path
import os,sys,json,subprocess,hashlib,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from idea1_controller.mesh_observation import MeshObservation
h=Path.home();o=h/'outputs/idea1-final-t162';out=o/'mesh-native-validation-01';out.mkdir(exist_ok=False)
network=out/'network.yml';network.write_text('topology: [ Ring, Ring ]\nnpus_count: [ 3, 2 ]\nbandwidth: [ 1.862645149230957, 1.862645149230957 ]\nlatency: [ 100, 100 ]\n')
old=o/'native/astra-sim/build-dynarl/RlxyPacketFixture';new=o/'native-mesh/astra-sim/build-dynarl/RlxyPacketFixture'
records=[]
def execute(label,binary,flows,policy='xy',scale=1):
 env=dict(os.environ,LD_LIBRARY_PATH=str(o/'native/system-deps/usr/lib/x86_64-linux-gnu'),RLXY_NETWORK_PACKET_BYTES='65536',RLXY_MESH_POLICY=policy,RLXY_MESH_TICKS_PER_NS=str(scale),RLXY_NETWORK_PACKET_LOG=str(out/(label+'-packets.jsonl')))
 command=[str(binary),str(network),'wafer',*flows];start=time.perf_counter()
 r=subprocess.run(command,env=env,text=True,capture_output=True,timeout=30)
 (out/(label+'.stdout')).write_text(r.stdout);(out/(label+'.stderr')).write_text(r.stderr)
 row=dict(label=label,argv=command,returncode=r.returncode,wall_s=time.perf_counter()-start,policy=policy,clock_scale=scale)
 records.append(row)
 if r.returncode:raise ValueError('native packet fixture failed: '+label)
 d=json.loads(r.stdout.strip().splitlines()[-1]);row['result']=d
 assert d['status']=='PASS'
 packets=[json.loads(s) for s in (out/(label+'-packets.jsonl')).read_text().splitlines()]
 assert len(packets)==len(flows) and all(p['bytes']==p['sent_bytes']==p['received_bytes'] and p['aggregate_callbacks']==1 for p in packets)
 assert all(x['send_callbacks']==x['recv_callbacks']==1 for x in d['flows'])
 if binary==new:
  assert d['mesh']['edges']==[], 'leaked declared bytes or actual FIFO work'
  assert d['mesh']['route_calls']==len(flows)
  for snap in d['admissions']:MeshObservation.parse(snap,1)
 return d
try:
 suites={'one':['0,5,1048576'],'shared_cut_alternatives':['0,5,1048576','1,5,1048576'],'mandatory_row':['0,2,1048576','1,2,1048576'],'disjoint':['0,1,1048576','3,4,1048576']}
 byname={}
 for name,flows in suites.items():
  a=execute(name+'-baseline',old,flows);b=execute(name+'-xy',new,flows);c=execute(name+'-adaptive',new,flows,'adaptive_xy_yx')
  assert a['flows']==b['flows'], 'default route timing differs from baseline'
  if name in ('one','mandatory_row','disjoint'):assert a['flows']==c['flows']
  if name=='shared_cut_alternatives':
   assert c['mesh']['alternate_routes']>=1
   assert max(x['end_tick'] for x in c['flows'])<max(x['end_tick'] for x in b['flows'])
  byname[name]=dict(fixed_end_tick=max(x['end_tick'] for x in b['flows']),adaptive_end_tick=max(x['end_tick'] for x in c['flows']))
 # Source-clock conversion affects observation units only, never execution ticks.
 scaled=execute('clock-scale',new,suites['one'],'xy',1000)
 unscaled=next(r['result'] for r in records if r['label']=='one-xy')
 assert scaled['flows']==unscaled['flows']
 a,b=scaled['admissions'][0],unscaled['admissions'][0]
 assert a['native_ticks_per_ns']==1000
 for ra,rb in zip(a['edges'],b['edges']):assert ra[:4]==rb[:4] and abs(ra[5]*1000-rb[5])<1e-6
 obs=MeshObservation.parse(unscaled['admissions'][0],1)
 # One full message has been declared but no first-packet event has run yet.
 assert sum(r[2] for r in obs.edges)==3*1048576 and sum(r[3] for r in obs.edges)==0
 result=dict(status='PASS_NATIVE_MESH_ROUTING',records=records,suites=byname,old_binary_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),new_binary_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),scope='native small directed-link packet tests; not full 4B or physical hardware performance')
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(status=result['status'],suites=byname),indent=2))
except Exception as e:
 (out/'failure.json').write_text(json.dumps(dict(error=repr(e),records=records),indent=2));raise
