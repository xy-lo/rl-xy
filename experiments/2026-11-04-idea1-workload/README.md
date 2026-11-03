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
