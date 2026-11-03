#!/usr/bin/env python3
"""Reaudit portable final TP evidence; requires no simulator or tensor runtime."""
import hashlib,json,math,tempfile
from pathlib import Path
from audit_idea1_inference import check
from check_idea1_system import extract
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'experiments/2026-11-03-idea1-elastic'
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 manifest=load(DATA/'publication-manifest.json')['files']
 actual={str(p.relative_to(DATA)) for p in DATA.rglob('*') if p.is_file() and p.name!='publication-manifest.json'}
 assert actual==set(manifest)
 for name,digest in manifest.items():assert sha(DATA/name)==digest,name
 values=load(DATA/'all-data.json');rows=values['comparison'];assert len(rows)==6
 assert {(r['model'],r['window']) for r in rows}=={(m,w) for m in ('4B','8B','14B') for w in ('test1','test2')}
 previous={(r['model'],r['window']):r for r in load(ROOT/'experiments/2026-11-03-idea1-complete/all-data.json')['comparison']}
 identities=load(DATA/'historical-identities.json');locked=load(DATA/'locked-tp-banks.json')
 banks={m:load(DATA/f'assets/tp-bank-{m}.json') for m in ('4B','8B','14B')}
 for m,bank in banks.items():
  assert not bank['provisional'] and bank['selection_window']=='tune'
  assert sha(DATA/f'assets/tp-bank-{m}.json')==locked['files'][f'tp-bank-{m}.json']
  assert bank['anchor_profile'] in {p['profile_id'] for p in bank['profiles']}
  for p in bank['profiles']:
   assert p['inference']['elastic'] and p['inference']['pending_enabled'] and p['inference']['elastic_data_wait']
   assert not p['profile_id'].endswith('-online-safe'),'v12 binding-defect candidate must not enter final bank'
 for r in values['die_snapshots']:
  assert r['total_dies']==54 and r['training_helper_subset_dies']<=r['training_dies']
  assert sum(r[k] for k in ('own_dies','generation_helper_dies','prepare_dies','training_dies','transfer_dies','prefix_dies','update_dies','idle_dies'))==54
 assert len(values['per_die'])==54*len(values['die_snapshots'])
 for r in values['candidate_search']:
  assert r['window']=='tune'
  if r['final_eligible']:
   assert r['native_exit_status']==0 and r['audit_scope']=='full safety'
 with tempfile.TemporaryDirectory() as temp:
  for r in rows:
   key=r['model']+'-'+r['window'];dst=Path(temp)/key;extract(DATA/'evidence'/(key+'.tar.gz'),dst,identities[key])
   report=check(dst/'native-run');result=load(dst/'result.json');plan=load(dst/'selected-plan.json')
   assert report['base']['samples']==1536 and report['kv_storage_hazard_checked'] and report['pending_input_fanout_checked'] and report['pending_resize_exclusion_checked']
   assert result['native_exit_status']==0 and result['total_s']==r['total_s']
   assert math.isclose(r['device_s']+r['planning_proxy_s'],r['total_s'],rel_tol=1e-12)
   assert r['fixed_TP2_system_s']==previous[r['model'],r['window']]['total_s']
   assert math.isclose(r['reduction_vs_fixed_TP2'],1-r['total_s']/r['fixed_TP2_system_s'],rel_tol=1e-10)
   assert math.isclose(r['speedup_vs_gpu'],r['gpu_s']/r['total_s'],rel_tol=1e-12)
   bank=banks[r['model']];assert plan['selected_inference_profile'] in {p['profile_id'] for p in bank['profiles']}
   assert plan['inference_selection']['reason']!='explicit controlled profile replay'
   assert plan['tp_bank_sha256']==locked['files'][f'tp-bank-{r["model"]}.json']
 print(json.dumps(dict(status='PASS_ELASTIC_SYSTEM_PUBLIC_EVIDENCE',final_runs=6,samples_per_run=1536,physical_dies=54,search_candidates=len(values['candidate_search']),die_snapshots=len(values['die_snapshots']),numeric_training_or_hardware_claim=False)))

if __name__=='__main__':main()
