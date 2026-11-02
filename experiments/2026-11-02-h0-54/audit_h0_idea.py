"""Audit Qwen3 54-die TP2-to-TP4 trace, lifecycle and capacity."""
import json,hashlib,sys
from pathlib import Path
from rl_xy.f2_runtime import INPUTS,make_operator
from attention_tp8_program import head_layout,kv_bytes_per_helper,attention_weight_bytes_per_helper
p=Path(sys.argv[1]);r=json.loads((p/'result.json').read_text());op=make_operator(r['model'],'wafer',cache_dir=p.parent/'audit-cache')
t=INPUTS/('Qwen3-'+r['model'])/(r['window']+'.json');rows=json.loads(t.read_text())['trajectories'];canonical=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert r['status']=='PASS_GENERATION_ONLY' and r['window']=='test1' and r['completed']==r['requests']==len(rows)==1536 and r['ranks']==54 and r['tp']==2 and r['native_exit_status']==0
assert hashlib.sha256(t.read_bytes()).hexdigest()==r['trace_sha256'] and canonical==r['canonical_requests_sha256']
for f,h in json.loads((p/'source_hashes.json').read_text()).items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
native=json.loads((p/'native/result.json').read_text());assert native['status']=='PASS' and native['native_exit_status']==0 and native['binary_sha256']==r['binary_sha256']
cap=json.loads((p/'attention-capacity.json').read_text());assert max(cap['source_bound_bytes'],cap['helper_bound_bytes'])<=cap['capacity_bytes']==64000000000
req={x['trajectory_id']:x for x in rows};migs=r['migrations'];releases=r['releases'];runs=r['helper_runs'];assert len({x['tx'] for x in migs})==len(migs)==len(releases);assert {x['tx'] for x in migs}=={x['tx'] for x in releases}
prior={}
for x in migs:
 src=x['source'];dst=x['helper'];assert 0<=src<27 and 0<=dst<27 and src!=dst and src//3==dst//3
 sr=tuple(range(2*src,2*src+2));dr=tuple(range(2*dst,2*dst+2));assert x['head_layout']==head_layout(op,sr,dr);ids=x['active_ids'];ctx=x['initial_actual_contexts'];assert len(ids)==len(ctx)==x['batch'] and len(set(ids))==len(ids) and set(ids)<=set(t['trajectory_id'] for t in rows[src::27]);assert all(i in req and c>=req[i]['prompt_length'] for i,c in zip(ids,ctx));assert x['context_sum']==sum(ctx) and x['kv_bytes_per_lane']==kv_bytes_per_helper(op,ctx) and x['weight_bytes_per_lane']==attention_weight_bytes_per_helper(op);assert x['pack_extra_read_bytes_per_lane']==x['pack_extra_write_bytes_per_lane']==x['kv_bytes_per_lane'];prior[x['tx']]=dict(zip(ids,ctx))
for seg in runs:
 ids=seg['active_ids'];ctx=seg['actual_contexts'];before=prior[seg['tx']];assert len(ids)==len(ctx)==seg['batch'] and set(ids)<=set(before) and all(before[i]==c for i,c in zip(ids,ctx));assert seg['kv_append_bytes_per_rank_per_step']==len(ids)*2*(op.model.kv_heads//4)*op.model.head_dim*op.model.actor_dtype_bytes*op.model.layers;prior[seg['tx']]={i:c+seg['repeat'] for i,c in zip(ids,ctx)}
report={'status':'PASS_LOGICAL_54_DIE_TP2_TP4','model':r['model'],'requests':len(rows),'migrations':len(migs),'segments':len(runs),'source_bound_bytes':cap['source_bound_bytes'],'helper_bound_bytes':cap['helper_bound_bytes'],'scope':'logical event/identity/capacity audit; not numerical tensor or physical calibration'}
(p/'audit-h0.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
