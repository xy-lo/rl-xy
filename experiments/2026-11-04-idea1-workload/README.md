# 工作负载规划、Mesh协作与晚区域训练粒度

最终4B/8B/14B test1原生验收进行中；下表仅为完整4B TUNE开发结果。方法与复现入口见[实现合同](../../docs/IDEA1_WORKLOAD.md)。所有结果为固定成本、公开长度回放的AstraSim时序仿真，不是数值模型训练、学习质量或实体晶圆计时。

|完整4B TUNE策略|设备/s|idle die·s|训练捐赠|
|---|---:|---:|---:|
|扩展布局＋原批|74.566400|99.177207|65|
|service home（训练布局同时不同）|77.695910|140.832983|29|
|ready_lpt|75.876085|169.900037|69|
|晚区域cap2|73.526129|41.172088|76|
|晚区域cap1（选中）|73.297172|23.739210|96|

每个完整点1536回答全部生成并贡献训练；各点native和独立审计退出0。cap消融保持资源、home、推理队列、cap和历史信息一致，只改变全局rollout真实结束后启动的训练区域原子批粒度；GEN和teacher占用不变，新增训练读写收费。service与原批之间同时改变训练区域，不能作纯放置因果归因。

原生程序冻结于2ca1b01，后续证据校验/文档提交单列；运行源与后端SHA在每case归档。先用TUNE完整验收选择one_scene/score/input/late-cap1，再启动最终test1，不用最终窗口时延选策略。旧批准4B TUNE设备73.873630s作为门，选中73.297172s通过，不与GPU test1错配。

## 开发证据

[完整清单](development/bundle-manifest.json)包含5个全量TUNE点和3个64回答功能压力点；原生事件、原始日志、命令、退出、固定源代码、布局、容量、梯度一次贡献及187项成本源均可离线复查。5个TUNE点还含独立预测来源重建、评估sample排除及队列绑定。功能点不能替代全量性能结论。

- [严格匹配的cap1对照](development/late-cap1-full4b-proof-01.json)
- [严格匹配的cap2对照](development/late-cap-full4b-proof-01.json)
- [216候选的三次计时](development/planner-timing-expanded-layout-03/result.json)
- [TUNE策略选择](development/tune-policy-selection.json)
- [cap1逐die快照表](development/diagnostics/late-cap1-state-report-01/states.md)

```bash
python -m pip install -e '.[f2]'
python scripts/check_idea1_workload.py experiments/2026-11-04-idea1-workload/development
```

默认distribution为独立校准模式；本开发表使用显式controlled alpha1离线同题proxy，不是连续actor历史。test1/test2已有历史结果，不能称全新held-out。54晶圆die与48×A100并非等面积/功耗/成本，尚无GPU同helper策略消融。最终同窗表将用旧device_s，旧Mac代理规划与本次Linux主机规划分列。

## 独立分布模式的开发版本

[独立分布开发清单](distribution-development/bundle-manifest.json)另含3个完整4B TUNE点，原生及独立执行审计通过，1536回答不减。仅前两项严格配对；第三项版本不同，单列，不与前两项作纯算法归因。它们不是最终v13/cap1的无同题历史验收。

|开发版本|信息与放置|设备/s|
|---|---|---:|
|legacy点摘要P0|独立calibration，独立放置|91.188989|
|同legacy摘要PB|相同信息/资源，释放依赖放置|78.674187|
|pooled V2|未知题相同条件点分数相同，独立情景保留；后续mesh运行时|79.569265|

前两项缩短13.724%，但不能将该数字直接搬到新pooled摘要版本。该包保存原生源/事件/成本，不包含这些旧版本的完整独立预测重建；当前controlled开发包中的5点才额外包含该项重建。未知题不能提前识别真实哪条回答会长。test3/test4跨prompt的最终系统验证仍未完成。

```bash
python scripts/check_idea1_workload.py experiments/2026-11-04-idea1-workload/distribution-development
```

## 资源子集与运行时粒度的补充证据

[运行时开发清单](runtime-development/bundle-manifest.json)完整保存8个1536回答点和4个64/128回答功能点，均重新执行原生事件/预测与校准来源的离线审计。它们是已有结果的打包，不是新一轮原生实验。

|策略|4B TUNE设备/s|4B test1设备/s|
|---|---:|---:|
|原批＋晚区cap1|73.297172|69.052905|
|容量允许时拼B8|74.372965|69.630843|
|B8＋末8样本细批|75.549236|70.673202|
|同上＋ready LPT|74.095541|70.673201|

TUNE原批证据在development，test1原批证据在runtime-development；同资源/forecast执行绑定已核验。LPT改善本组TUNE但test1无可计收益。另一个test1 shortlist布局点69.195777s也保留，不以代理排名当原生最优。功能点采用明确observed-fit夹具，不能当规划性能。

[机制归因](runtime-development/analysis/mechanism-cause-attribution-01.md)区分机制一代理误差、机制二释放耦合及机制三批粒度/安全点限制。B8方向未被选为最终默认，不将更多捐赠次数或更少批数当加速。

```bash
python scripts/check_idea1_workload.py experiments/2026-11-04-idea1-workload/runtime-development
```
