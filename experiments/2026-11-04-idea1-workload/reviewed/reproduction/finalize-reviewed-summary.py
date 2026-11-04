import argparse,csv,json,hashlib,shutil,datetime
from pathlib import Path
from collections import Counter

def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def write_csv(path,rows,fields):
 with Path(path).open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)
def shape(values):return '+'.join(f'{n}×TP{tp}' for tp,n in sorted(Counter(map(int,values)).items(),reverse=True))
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--evidence-root',type=Path,required=True);parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();p=a.evidence_root;base=p/'final-campaign-11'
 status=load(base/'status.json');assert status.get('results_ready') and not status.get('failures')
 verification=load(p/'final-bundle03-source19-check-status.json');assert verification['exit_status']==0
 comparison=load(base/'comparison.json');assert len(comparison['comparison'])==3
 assert sha(a.repo/'experiments/2026-11-03-idea1-elastic/comparison.csv')==comparison['baseline_csv_sha256']
 checks=load(p/'final-remote-checks-02-status.json');assert all(x['exit_status']==0 for x in checks)
 rows=[];snapshots=[];perdie=[];sources={}
 for row in comparison['comparison']:
  model=row['model'];folder=base/(model+'-test1');run=folder/'native-run';report=p/('final-'+model.lower()+'-state-report-03');summary=load(report/'summary.json');result=load(run/'result.json');audit=load(folder/'inference-audit.json');deployment=load(folder/'plan/deployment.json')
  assert result['requests']==result['completed']==result['trained_samples']==1536
  assert summary['total_device_s']==row['device_s']==result['total_s'] and audit['status']=='PASS_ACTUAL_TP_INFERENCE_NATIVE_AUDIT'
  assert summary['native_resizes']==len(load(run/'inference-resizes.json'))
  assert row['planning_host_s']==deployment['host_wall_s']
  events=load(run/'training-events.json');first_teacher=min(x['start_tick'] for x in events if x['phase'] in ('old_policy','reference') and x['tids'])/1e12
  executions=load(run/'training-batch-executions.json');helpers=[x for x in executions if x.get('remote',x['worker']!=x['home'])]
  points=[];helper_die_s=0
  for x in helpers:
   dies=len(x.get('physical_ranks',[])) or x.get('training_tp',2);assert dies==2
   helper_die_s+=(x['end_tick']-x['start_tick'])/1e12*dies
   points.extend([(x['start_tick'],-1,dies),(x['end_tick'],-2,-dies)])
  active=peak=0
  for _,_,delta in sorted(points):active+=delta;assert active>=0;peak=max(peak,active)
  assert active==0
  offline=load(folder/'plan/offline/result.json')
  item=dict(row,initial_inference=shape(result['inference_tp_by_group'].values()),training=shape(result['training_tp_by_region'].values()),training_dp=len(result['training_tp_by_region']),first_teacher_s=first_teacher,generation_end_s=summary['generation_end_s'],first_backward_s=summary['first_training_s'],backward_overlap_s=summary['early_training_overlap_s'],optimizer_suffix_s=summary['optimizer_suffix_s'],idle_die_s=summary['role_integral_die_s']['idle'],post_generation_idle_die_s=summary['post_generation_idle_die_s'],post_generation_dynamic_tail_s=summary['post_generation_dynamic_tail_s'],post_generation_average_idle_dies=summary['post_generation_idle_die_s']/summary['post_generation_dynamic_tail_s'],nonidle_role_occupancy=summary['nonidle_role_occupancy'],native_resizes=summary['native_resizes'],training_transfers=summary['training_transfers'],remote_training_samples=summary['remote_training_samples'],training_helper_die_s=helper_die_s,peak_training_helper_dies=peak,native_validation_host_s=load(folder/'native-status.json')['wall_s'],planning_command_host_s=load(folder/'planning-status.json')['wall_s'],search_host_s=offline['offline_search_s'],candidate_pairs=offline['counts']['candidate_pairs'],legal_pairs=offline['counts']['legal_pairs'],position_reason=deployment['position_reason'],roles_die_s=summary['role_integral_die_s'])
  rows.append(item)
  for x in load(report/'die-snapshots.json'):
   assert set(x['counts'])=={'own','generation_helper','prepare','training','transfer','prefix','update','idle'};assert x['total_dies']==54==sum(x['counts'].values())==len(x['per_die']);assert {d['die'] for d in x['per_die']}==set(range(54))
   assert x['training_helper_subset_dies']<=x['counts']['training']
   snapshots.append(dict(model=model,time_s=x['time_s'],phase=x['phase'],**x['counts'],training_helper_subset_dies=x['training_helper_subset_dies'],idle_die_ids=','.join(str(d['die']) for d in x['per_die'] if d['role']=='idle')))
   for d in x['per_die']:perdie.append(dict(model=model,time_s=x['time_s'],phase=x['phase'],**d))
  sources[model]={name:sha(folder/name) for name in ['native-run/result.json','inference-audit.json','plan/deployment.json','native-status.json','planning-status.json']}
 rows.sort(key=lambda x:int(x['model'][:-1]));out=a.output;out.mkdir(parents=True,exist_ok=False)
 payload=dict(schema='idea1.final-reviewed-summary/1',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),runtime_commit=status['source_commit'],models=rows,scope=dict(simulation=True,numerical_training=False,physical_hardware=False,wafer_dies=54,gpu_a100=48,equal_area_power_cost=False,chronological_history=False,historically_used_test1=True,heldout_native_executed=False),source_sha256=sources,baseline_sha256=comparison['baseline_csv_sha256'])
 save(out/'results.json',payload);save(out/'validation.json',dict(runtime_source=status['source_commit'],remote_checks=checks,final_portable_verification=verification,regression_accepted_and_explained=True,github_upload='deferred by user'))
 columns=['model','gpu_s','h0_s','original_generation_s','previous_innovation3_s','old_complete_system_device_s','device_s','speedup_gpu_device','reduction_old_complete_device','planning_host_s']
 write_csv(out/'comparison.csv',rows,columns)
 columns=['model','initial_inference','training','training_dp','first_teacher_s','generation_end_s','first_backward_s','backward_overlap_s','optimizer_suffix_s','idle_die_s','post_generation_idle_die_s','post_generation_dynamic_tail_s','post_generation_average_idle_dies','nonidle_role_occupancy','native_resizes','training_transfers','remote_training_samples','training_helper_die_s','peak_training_helper_dies','planning_host_s','planning_command_host_s','search_host_s','native_validation_host_s','candidate_pairs','legal_pairs','position_reason']
 write_csv(out/'system-metrics.csv',rows,columns)
 write_csv(out/'die-state-counts.csv',snapshots,['model','time_s','phase','own','generation_helper','prepare','training','training_helper_subset_dies','transfer','prefix','update','idle','idle_die_ids'])
 write_csv(out/'per-die-states.csv',perdie,['model','time_s','phase','die','row','col','worker','role','training_helper','stage'])
 (out/'die-states').mkdir()
 for model in ['4B','8B','14B']:
  report=p/('final-'+model.lower()+'-state-report-03')
  for name in ['states.md','summary.json','die-snapshots.json','input-hashes.json']:shutil.copy2(report/name,out/'die-states'/(model+'-'+name))
 (out/'analysis').mkdir()
 for name in ['mechanism-cause-attribution-01.json','mechanism-cause-attribution-01.md','result-inventory-01.json','result-inventory-01.csv']:shutil.copy2(p/'analysis'/name,out/'analysis'/name)
 lines=['# Idea1 最终统一验收与瓶颈分析','','三档均为同一冻结运行源c5f5db4、test1窗口、1536回答及一次统一更新的AstraSim固定成本时序仿真。全部原生与独立审计通过。没有数值Transformer训练或实体硬件验证；54晶圆die与48A100不是等面积、功耗或成本比较。GPU/旧系统沿用已发布冻结同窗基线，未在本次重跑GPU。','','|模型|GPU/s|H0/s|原生成机制/s|旧创新三/s|旧完整设备/s|新系统设备/s|GPU加速|较旧完整缩短|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for x in rows:lines.append('|%s|%.3f|%.3f|%.3f|%.3f|%.3f|%.3f|%.3fx|%+.2f%%|'%(x['model'],x['gpu_s'],x['h0_s'],x['original_generation_s'],x['previous_innovation3_s'],x['old_complete_system_device_s'],x['device_s'],x['speedup_gpu_device'],100*x['reduction_old_complete_device']))
 lines+=['','正的缩短值表示改善，负值表示回归。原始精度见[comparison.csv](comparison.csv)，全部指标和来源SHA见[results.json](results.json)。不能把不同信息策略、开发版本和最终版本混成一条因果消融。','','## 资源与训练接管','','|模型|初始推理组成|训练组成|训练DP|最早teacher/s|GEN结束/s|最早反向/s|反向与GEN重叠/s|']
 lines+=['|---|---|---|---:|---:|---:|---:|---:|']
 for x in rows:lines.append('|%s|%s|%s|%d|%.6f|%.6f|%.6f|%.6f|'%(x['model'],x['initial_inference'],x['training'],x['training_dp'],x['first_teacher_s'],x['generation_end_s'],x['first_backward_s'],x['backward_overlap_s']))
 lines+=['','必须区分训练准备与真正反向训练：teacher提前不等于反向提前。机制允许区域接管，但某个评估窗口不一定出现反向重叠。27个规范optimizer owners不等于训练DP数量。','','## 协作、空闲与规划开销','','|模型|推理扩组|训练捐赠批|远程样本|idle die·s|实际训练helper峰值/die|规划主机/s|搜索子步/s|候选数|原生验证主机/min|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for x in rows:lines.append('|%s|%d|%d|%d|%.3f|%d|%.6f|%.6f|%d|%.2f|'%(x['model'],x['native_resizes'],x['training_transfers'],x['remote_training_samples'],x['idle_die_s'],x['peak_training_helper_dies'],x['planning_host_s'],x['search_host_s'],x['candidate_pairs'],x['native_validation_host_s']/60))
 lines+=['','训练helper是训练占用的子集，不能重复相加；峰值由实际remote训练区间扫描得到。角色占用包含内存/通信等待，不等于FLOP利用率。各时刻角色计数见[die-state-counts.csv](die-state-counts.csv)，逐die身份见[per-die-states.csv](per-die-states.csv)，可读表见[14B状态表](die-states/14B-states.md)。每行角色总计54。','','规划为新进程内部墙钟，OS缓存未清理；搜索只是子步骤。4B/8B复用明确TUNE资源子集，不能把约1秒描述成首次全局搜索最优；14B使用完整有限模板快评。离线目录/布局库构建和native验证成本另计。设备秒与CPU实测墙钟分列，不混加成实体硬件周期。','','## 为什么部分方案回归','','机制一目标已经包含整轮，但代理没有完整重放运行时共享队列、协作资格和实际粒度。8B初验GEN56.805s、整轮108.971s；改用已验资源几何后GEN63.931s、整轮103.259s。切分与放置同时变化，不能把全部差异纯归因一个机制。','','可选B8使4B训练工作少19.678die·s，却使idle多50.887die·s；22个B8捐赠批在helper上按B4+B4执行。源die只在整批安全点发起捐赠，接收者还需完成整区恢复。更多helper或更少批数不自动缩短关键路径。该方向已保留负例并关闭最终默认；未继续增加优化轮次。','','机制二此前同信息同资源开发P0/PB有正面结果，但最终PS是否执行取决于合法cohort关系，不能把PB回退计作位置收益。全量路由开关未显著改善周期；这不证明所有拥塞已消失。详细归因与证据范围见[机制分析](analysis/mechanism-cause-attribution-01.md)。','','## 代码与证据边界','','最终三档测量源为c5f5db4。后续52ea3ea仅补强无历史文件入口、完整组prefix和独立校准来源审计；84个执行核心文件已逐SHA证明与测量源相同。原生/成本目录/binary/config/输入/命令/退出和日志在最终portable归档，开发及负例另包保存。最终代码和数据的远端Git提交与bundle校验单列。','','test1是历史使用过的窗口，同题proxy alpha1是离线额外信息，非真实actor连续更新；当前sample被排除。未启动独立heldout原生矩阵，不能据此声称跨prompt、多seed、领域漂移或学习质量泛化。没有全候选native oracle枚举，不报告全局最优或regret。GPU同helper消融、活动训练层间协作、更大训练helper TP和割面发布树未完成或未采用。','','GitHub按用户要求暂不上传；本轮交付远端代码、实际结果、回归解释和可复现证据。']
 lines+=['','## 最终数据揭示的瓶颈优先级','','|模型|GEN到首次反向/s|生成后动态阶段/s|该阶段平均空闲die|动态阶段非idle角色占比|优化器/发布后缀/ms|','|---|---:|---:|---:|---:|---:|']
 for x in rows:lines.append('|%s|%.6f|%.6f|%.3f|%.2f%%|%.3f|'%(x['model'],x['first_backward_s']-x['generation_end_s'],x['post_generation_dynamic_tail_s'],x['post_generation_average_idle_dies'],100*x['nonidle_role_occupancy'],1000*x['optimizer_suffix_s']))
 lines+=['','非idle角色占比分母为角色事件覆盖的动态区间，不含统一更新后缀；不等于全周期FLOP利用率。生成后平均空闲分别约3.50、0.70、1.27个die（分母为生成结束到动态区域计算结束，不含统一更新后缀）。因此，当前不能把几十秒的生成后阶段都称作空转：多数资源正在执行训练、准备或等待其本地内存/通信服务。继续只增加helper数量的空间有限，应优先区分活跃区域服务时间、区域释放及GEN到反向的准备依赖。角色占用不能确定算力、内存或网络中谁是硬件瓶颈。','','14B从GEN结束到最早反向仍约10.79s，明显高于4B/8B的1.18/2.01s。下一步若获授权，应先诊断teacher/完整K组优势和状态准备的关键依赖，评估可合法提前的工作；仅把更多长任务分散开，并不能消除这些依赖。目前没有隔离消融证明每项依赖各占多少秒。','','统一优化器/发布后缀仅约39.5/80.7/145.0ms，均低于整轮0.1%。即便将其完全消除，也不足以解释或解决秒级回归。这与稀疏RS组件没有端到端收益、路由开关没有完整周期收益的现有结果一致；不代表通信在所有其他工作负载上都不重要。','','4B较旧完整系统慢约0.790s；8B和14B分别快约0.381s和2.432s。不同版本同时涉及资源/放置与执行变化，目前不足以把最终差异精确拆成三个机制的独立贡献。现有数据支持条件性改进与具体实现边界，不支持所有模型必胜或已经找到native全局最优的结论。']
 (out/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
 save(out/'report-input-manifest.json',dict(comparison_sha256=sha(base/'comparison.json'),bundle_manifest_sha256=sha(p/'portable-final-bundle-03/bundle-manifest.json'),status_sha256=sha(base/'status.json'),generator_sha256=sha(__file__),source_files=sources))
 print(json.dumps(dict(status='FINAL_REVIEWED_REPORT_WRITTEN',output=str(out),models=[{'model':x['model'],'device_s':x['device_s']} for x in rows],snapshot_rows=len(snapshots),per_die_rows=len(perdie))))
if __name__=='__main__':main()
