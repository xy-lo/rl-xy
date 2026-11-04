# Idea1 工作负载规划与 Mesh 协作：实现合同

本系统对公开长度 trace 执行完整 RL 周期的成本与事件时序仿真。它不执行数值 Transformer 训练，不提供学习质量或实体晶圆性能证明。最终实验、消融与归档入口见[实验目录](../experiments/2026-11-04-idea1-workload/README.md)。

## 分层与三个机制

|层次|职责与接口|实现位置|
|---|---|---|
|工作负载输入|不可变分布快照、规范prompt身份、完整K维向量、类别/总体/无历史回退|`src/idea1_controller/workload.py`、`controlled_forecast.py`|
|机制一：轮前资源规划|枚举可行推理TP/cap与训练区域，检查容量，离线评估并绑定部署身份|`scripts/workload_planner/run_distribution.py`、`search.py`、`release_search.py`、`materialize.py`|
|机制二：任务与物理位置|冻结区域/worker配额，按历史分配长任务、生产者与数据home，限定范围内调整完整队列位置|`region_release_placement.py`、`spatial_release.py`、`materialize_spatial.py`|
|机制三：运行时执行|未开始回答接手、合法推理TP扩组、区域接管、未开始训练批捐赠、真实Mesh链路观测与消息选路|`src/idea1_controller/native/`、`mesh_observation.py`|
|全局提交|所有样本贡献完成后按规范optimizer owner合同归约、归一化、更新一次并发布actor_v1|既有native后缀与版本/贡献审计|

统一入口`plan_system.py`依次执行离线规划、后置trace绑定与可选PS，输出`deployment.json`。部署文件表示已选择的计划，必须经native回放及独立审计后才构成实验结果。

## 信息与快照

默认`--forecast-mode distribution`只从独立calibration的完整回答向量建立总体/类别分布。新prompt只提供ID、真实输入token长度、规范文本hash与决策前可得的类别。相同条件任务不能通过随机场景的采样噪声获得不同点预测；各候选复用相同场景。

`--forecast-mode controlled-proxy --allow-controlled-proxy --proxy-weight 1`显式启用额外同题信息的离线对照。原始当前窗口sample不进入历史；规范文本hash用于排除跨集合重复。它不是连续actor训练历史，不支持跨actor更新可信度结论。test1/test2已有历史结果，不称全新held-out。

`--no-history`不读取或哈希未使用的历史文件。默认独立分布遇到缺失历史文件时明确记录`HISTORY_FILE_MISSING`并返回容量回退；空总体池同理。已有文件损坏或模型/config不兼容时明确失败。无可靠分布时选择最小可行TP的合法布局，硬上限预测不被描述为逐题长短预测。

快照与算子服务目录分别有版本。校验包括模型/生成合同、来源sample、prompt hash、目录SHA和预测摘要；committed-history接口拒绝未来observed_seq，离线proxy不伪造时间轴。独立规划审计会从原始允许的calibration样本重建分布，再重建预测和部署绑定，而非只验证快照自哈希。

## 机制一：资源与代价

物理网格为9行×6 die，共54 die；27个规范TP2控制/optimizer对不是固定推理实例数，也不等于训练DP数。当前算子目录支持TP1/2/4/8；54不能全部按TP8整除，因此布局允许混合TP。TP1使用每规范对内两个独立实例，当前没有TP1扩组协议。

默认枚举216个有限模板组合。`one_scene`快评一个场景；`full`精评16场景；`shortlist6`保留各推理族代表再精评，实际候选数可能超过6。风险目标沿用既有均值/尾部组合。有限模板、保留策略和局部位置搜索均不是全局最优证明。

推理容量包括权重、最大KV增长、常驻状态和缓冲；预测不能缩小hard预留。训练TP/DP区域同样经内存检查，运行时在实际EOS后再次验证每个训练批。全部候选非法时明确失败，不返回未验证计划。离线artifact仍需针对当前公开输入、目录与容量重新校验。

可重复的`--resource-candidate inference@fraction:training`指定显式子集，例如`box4_top@1.0:row1`；未给参数则使用完整候选族。比例只接受0.5/0.75/1.0，非法名称/空集拒绝。artifact、部署与审计绑定子集。复用TUNE已验资源几何仍会重建当前任务放置与容量，不能把单候选搜索时间称为从头发现全局最优的成本。

代价目标包括区域释放、teacher/training服务及公共前后缀，目标不是只优化GEN。但该代理没有完整重放运行时共享队列、动态接手和所有可选批细化，所以代理排名仍需native验证。服务目录拟合误差不等于完整周期的统计保证。

## 机制二：区域释放与位置

PB保持完整group、区域长任务配额和整数worker配额，联合考虑推理生产者与训练home。每个推理/训练布局对单独计算放置，缓存不能只以训练partition为键。不同DP编号不代表推理cohort可独立退出，必须同时跟踪物理占用和数据生产者依赖。

PS在固定队列内容/顺序下搜索合法位置，评估共同阻塞、helper可用性与路径竞争。当前只支持同训练区域、相同TP/cap的完整推理队列交换；跨区域cohort、PENDING或不支持的TP1场景明确保留PB。本次全量部署是否发生PS，以各自`position-proof.json`为准，不把回退算作位置收益。

此前同信息同资源4B开发P0/PB为91.188989/78.674187s；这是legacy摘要版本的对照，不可当作最终pooled版本收益。32回答合法PS功能例25.651234→25.407870s，仅证明该功能场景。

## 机制三：协作与训练接管

推理优先接手尚未开始的回答，再根据收益、权重/KV兼容、准入域、容量和真实链路状态判断是否扩组。当前是完整TP扩组并保持到EOS，不是任意活动收缩或新增FFN-only协议。推理扩组次数读取`inference-resizes.json`及audit的`native_resizes`，不能用遗留`helper_segments`代替。

区域内相关推理全部安全退出且记录齐备后，区域才执行teacher/reference、完整K8优势与反传；不必等待全片GEN。必须分别报告“训练准备提前”和“反传提前”，不能混称。某批次无反传重叠不等于接管接口未实现。

训练协作转移已经就绪、尚未开始的批。源die在step安全点发起；接收方必须完成整个训练区域和规范状态恢复，处于waiting且没有其他service/lease。活动训练批不可抢占，当前没有闲置helper在源长批中途独立拉取任务的协议。训练帮助可按域或全片范围运行，最终实验使用wafer范围并对路径争用计费。

helper以规范TP2执行可搬运程序，沿用原逐模型微批上限，必要时顺序拆成多个子批；完整payload读、网络传输、写与确认均收费。实际B8开发对照中22个捐赠批在helper上按B4+B4执行，不能假定本地主批与helper服务相同。

`--late-region-batch-cap 1`只对全局GEN已实际结束后才构建的训练区域收紧原子批。参数按TP2对缩放：TP2/4/8的上限分别1/2/4；早构建区域保持原批效率。`--training-ready-order ready_lpt`只按EOS后已知的本地服务估计调整训练任务顺序。

实验性`--adaptive-training-batch-max 8`允许本地EOS后按64GB容量拼批；`--training-tail-fine-samples 8`或16保留队列后缀细批。它们默认0，原M1 TP2训练服务目录只覆盖B4以内，故部署显式标注runtime refinement未被代理评分。完整4B对照未获得test1收益，最终选用packing0/tail0，而不是把这些试验开关默认开启。

Mesh策略读取真实FIFO和已宣布未启动bytes，对整个逻辑消息选择XY/YX，不改在途路径。13个packet检查覆盖旧XY等价、可替代路径、必经割面、bytes/ACK及clock scale。完整4B路由开关78.145321/78.145774s，无周期收益；小拓扑路径改善不能外推为模型加速，也不证明所有拥塞已消失。

## 梯度与版本合同

训练helper可在非区域根的TP2缓冲区新增梯度。生产保留27规范optimizer owners的完整RS、全局样本归一化、norm/clip、Adam、owner AG及actor_v1统一发布。训练DP数不等于归约参与者数。没有最终贡献者证书，不启用root-only稀疏图。

4B隔离后缀原始约39.2ms，稀疏根RS及重分发约39.8ms，未采用。割面发布树未启用，不能把理论字节减少算作端到端收益。审计检查全部固定样本一次贡献、训练job身份、实际TP程序、梯度和版本提交。

## 远端复现与证据检查

在新目录准备成本，14B需独立prepare。以下为复现命令模板，不表示本次又运行了新实验：

```bash
python scripts/prepare_idea1_joint.py --model 4B --output /remote/work/cost-4B
RLXY_COST_ROOT=/remote/work/cost-4B python scripts/workload_planner/plan_system.py --model 4B --window tune --mode one_scene --late-region-batch-cap 1 --output /remote/work/plan-4B
```

读取`deployment.json`中的三份输入路径后执行；同时设置当前Protobuf动态库目录：

```bash
PYTHONPATH=/remote/work/cost-4B/src python scripts/workload_planner/run_mesh_system.py --model 4B --window tune --training-scope wafer --assignment /remote/work/plan-4B/ps/selected-plan.json --resource-layout /remote/work/plan-4B/ps/selected-plan.json --inference-layout /remote/work/plan-4B/ps/inference-layout.json --workload-forecast /remote/work/plan-4B/ps/forecast.json --native-root /remote/work/native-mesh/astra-sim --fast-cost-validation /remote/work/linux-cost-validation.json --mesh-controller --mesh-routing adaptive_xy_yx --output /remote/work/run-4B
python scripts/audit_idea1_inference.py /remote/work/run-4B --output /remote/work/run-4B-audit.json
python scripts/report_idea1_workload.py /remote/work/run-4B --audit-report /remote/work/run-4B-audit.json --output /remote/work/run-4B-states
```

后端构建固定上游4ce9ecfe746a3ee5fe47559a11d91644d641fa7f与本仓库扩展，不修改原seed或系统包。Linux GCC13.3/Protobuf C++3.21.12/CMake3.31.6，新目录构建131.73s；3628个固定源字节一致，13packet检查通过。成本加速器7个大/小/ragged/FP32形状与冻结Python事件逐字段一致；这些不是硬件校准。

```bash
python scripts/workload_planner/build_mesh_backend.py --seed /remote/work/astra-seed --output /remote/work/native-mesh --protobuf-prefix /remote/work/protobuf/usr --cmake /remote/work/venv/bin/cmake
python scripts/workload_planner/build_mesh_backend.py --verify-only --output /remote/work/native-mesh
```

打包器归档原始源字节；只迁移trace和source-map的目标路径并保留原件与双SHA。检查器无需模拟器即可独立复查原生事件、187项每模型成本源、预测/校准来源、输入哈希、实际部署选项、训练贡献与更新。最终包还逐项重算全部baseline列及唯一模型行。

```bash
python scripts/check_idea1_workload.py /remote/work/portable-final-bundle --require-final
```

## 统计边界与未执行项

设备秒是仿真时间，规划/控制/构建/验证的进程墙钟独立报告。新进程计时不代表已清空OS文件缓存。旧完整系统使用device_s，不能把旧Mac规划代理与本次Linux主机耗时混加后当实体硬件周期。role occupancy包含本地内存/通信等待，不等于FLOP利用率；训练helper为训练占用子集，不能重复相加。

GPU为48 A100，晶圆54die，非等面积/功耗/成本；没有GPU同helper策略消融。一次历史窗口的条件结果不能证明跨分布/多seed/真实actor漂移或学习质量。

统一三档1536回答验收使用同一冻结运行源c5f5db4；后续无历史文件/校准审计修复只改规划与校验层，84个执行核心文件与该版本逐SHA一致。test3/test4的完整窗口各有一处canonical TUNE重复，虽已准备公开hash决定的352/632回答前缀协议及规划合同检查，按本轮收束范围未启动其原生矩阵，不作泛化结论。Oracle全候选原生枚举、活动训练层间协作通知、更大训练helper TP及发布树均未执行或实现。

三档实验已完成远端代码/数据、统一结果、分析、可复现归档及等价检查交付。用户随后授权将已验收版本上传到`https://github.com/xy-lo/rl-xy`；发布状态与CI另行核验，不重启模型实验。最终结果保留4B回归，不承诺所有模型一定超过旧完整系统。
