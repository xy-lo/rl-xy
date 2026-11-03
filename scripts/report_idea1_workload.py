#!/usr/bin/env python3
"""Reconstruct per-die state tables and phase diagnostics from an audited run."""
import argparse,json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from idea1_controller.state_report import snapshot,default_times,CATEGORIES
from audit_idea1_inference import check
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--audit-report',type=Path)
 p.add_argument('--times',type=float,nargs='*');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 result=load(a.run/'result.json');audit=load(a.audit_report) if a.audit_report else check(a.run)
 assert audit['status']=='PASS_ACTUAL_TP_INFERENCE_NATIVE_AUDIT'
 base=audit['base'];assert base['samples']==result['requests']==result['completed']==result['trained_samples']
 assert base['total_s']==result['total_s'] and base['window']==result['window']
 rows=[json.loads(line) for line in (a.run/'occupancy.jsonl').open()]
 assert all(x['tick']<=y['tick'] for x,y in zip(rows,rows[1:]))
 executions=load(a.run/'training-batch-executions.json')
 times=default_times(base) if a.times is None else a.times
 snaps=[snapshot(rows,base,executions,t) for t in times]
 total_role_time=sum(base['role_integral_die_s'].values())
 commands=load(a.run/'command.json');native=load(a.run/'native/result.json')
 summary=dict(model=result['model'],window=result['window'],answers=result['requests'],
  total_device_s=result['total_s'],generation_end_s=base['generation_end_s'],first_training_s=base['first_training_s'],
  early_training_overlap_s=max(0,base['generation_end_s']-base['first_training_s']),
  post_generation_dynamic_tail_s=base['dynamic_end_s']-base['generation_end_s'],optimizer_suffix_s=base['native_suffix_s'],
  role_integral_die_s=base['role_integral_die_s'],post_generation_idle_die_s=base['post_generation_idle_die_s'],
  nonidle_role_occupancy=1-base['role_integral_die_s']['idle']/total_role_time,
  initial_inference_tp=result['inference_tp_by_group'],training_tp=result['training_tp_by_region'],
  native_resizes=audit['native_resizes'],training_transfers=base['native_training_transfers'],remote_training_samples=base['remote_samples'],
  regional=base['regional'],native_slowest_rank=native['slowest_rank'],native_slowest_rank_costs=native['per_rank_costs'][str(native['slowest_rank'])],
  phase_timing_scope='role occupancy and native service ledger are different views, not additive',
  scope='fixed-cost trace simulation, 54 dies; not physical compute utilization or numerical learning')
 a.output.mkdir(exist_ok=False,parents=True)
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
 (a.output/'die-snapshots.json').write_text(json.dumps(snaps,indent=2))
 paths=['result.json','command.json','occupancy.jsonl','training-batch-executions.json','source_hashes.json','native/result.json']
 (a.output/'input-hashes.json').write_text(json.dumps({name:sha(a.run/name) for name in paths},indent=2))
 lines=['# 原生事件重建的 die 状态','','模型 '+summary['model']+'，窗口 '+summary['window']+'，'+str(summary['answers'])+' 条回答。','','|时间/s|自身推理|生成helper|准备|训练|其中训练helper|传输|前缀|更新|空闲|','|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for s in snaps:
  c=s['counts'];lines.append('|%.6f|%d|%d|%d|%d|%d|%d|%d|%d|%d|'%(s['time_s'],c['own'],c['generation_helper'],c['prepare'],c['training'],s['training_helper_subset_dies'],c['transfer'],c['prefix'],c['update'],c['idle']))
 lines+=['','训练helper是训练die的子集，不能再加一次；除该子集列外，每行总计54。','',
  '这是最后一条已发生事件对应的调度角色，以物理TP2 pair为角色粒度；不是硬件计算利用率。首尾前缀与优化器更新单列，不作为空闲。']
 (a.output/'states.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(status='PASS_RECONSTRUCTED_DIE_STATES',snapshots=len(snaps),per_die_rows=54*len(snaps),total_device_s=result['total_s'])))
if __name__=='__main__':main()
