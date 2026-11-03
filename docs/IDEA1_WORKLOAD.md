# 工作负载分布规划与 Mesh 协作

这是已发布动态TP系统上的实现说明。三模型最终验收仍在远端进行；中间功能例不是最终性能结论。所有native结果为固定成本和公开长度轨迹驱动的AstraSim时序仿真，不是Transformer数值训练、学习质量或实体晶圆计时。

## 分层职责

|层次|入口|输入与职责|
|---|---|---|
|工作负载契约|`src/idea1_controller/workload.py`|模型/生成合同、K维样本、截断信息、分布快照；拒绝当前R/EOS及未来历史|
|受控同题实验|`controlled_forecast.py`|显式opt-in的非chronological离线代理；与独立calibration分开，alpha=1复现旧信息集，未知题回退|
|候选评估|`scripts/workload_planner/release_search.py`|有限合法TP/训练区域、当前容量、逐布局对PB放置与场景评分；缓存键包含训推双方布局|
|区域放置PB|`region_release_placement.py`|固定完整任务配额，按预测长任务配额分散、平衡队列，减少home与producer依赖|
|位置优化PS|`spatial_release.py`|固定home与训练顺序，仅置换完整队列位置；共同阻塞、helper/路径互斥，32映射/轮、最多2轮|
|运行时|`native/mesh_cohorts.py`|实际mesh队列；PENDING、合法TP扩组、就绪训练批helper净收益；过期观测退回原策略|
|物理网络|`native_extensions/mesh_runtime/`|消息级XY/YX选择，实际队列与已宣布未启动字节记账，单消息路由一致|
|统一入口|`scripts/workload_planner/plan_system.py`|离线候选→后置回放绑定→PS候选；设备执行和独立审计另行运行|

动态TP包含轮前选择初始TP和安全整组TP扩展。借到的TP保持到EOS，当前没有活动TP缩回或单独FFN租约。推理cohort与训练区域可以不同；没有合法域内借算机会时，不虚构helper收益。训练只转移尚未开始、优势已就绪的批，保留home、样本贡献一次性和全局更新分母。

PS目前支持完整推理cohort嵌套于训练区域且整队列装入cap。跨区域、TP1和未建模PENDING保留PB。两条联合动作不能复用source/helper或冲突路径。有限代理用于排名，不保证全局最优或完整原生重放等价。

## 时间与信息

- `distribution`使用独立prompt校准的总体/类别联合分布，不能在无同题信息时识别真实每题长短。
- `controlled-proxy`使用与评估sample ID分离的同prompt离线样本，不是连续actor历史。默认权重0.75；alpha=1用于与旧系统相同信息集比较。真实在线历史仍要求可核验的提交序列。
- `one_scene`快评90布局候选；`shortlist6`对少量候选追加16场景；`full`离线评估90×16。代理最优不是原生/硬件最优。
- 主机进程、搜索、容量复核分别计时，数毫秒的容量子步骤不是完整规划时间。主机墙钟与设备仿真秒单列。
- 缓存绑定精确public batch、分布版本及服务目录；新batch或漂移需重新评估/合法回退，不假设任意新题都可复用旧表。
- test1/test2有历史实验，不是全新held-out。test3/test4跨prompt全系统泛化尚未完成。

## 复现入口

先准备独立成本目录；原仓库rl_xy默认配置不代表54活动die。

```bash
python scripts/prepare_idea1_joint.py --model 4B --output /remote/work/cost-4B
RLXY_COST_ROOT=/remote/work/cost-4B python scripts/workload_planner/plan_system.py --model 4B --window tune --mode one_scene --output /remote/work/plan-4B
```

与旧同题代理对齐时，显式增加`--forecast-mode controlled-proxy --allow-controlled-proxy --proxy-weight 1`。14B须单独prepare，reference生命周期适配不同，不能复用4B成本目录。

native构建要求初始化的固定上游checkout（`4ce9ecfe746a3ee5fe47559a11d91644d641fa7f`）、CMake和Protobuf C++前缀。构建器仅改写指定的新输出，seed保持干净；帮助命令不会启动构建。

```bash
python scripts/workload_planner/build_mesh_backend.py --seed /remote/work/astra-seed --output /remote/work/native-mesh --protobuf-prefix /remote/work/protobuf/usr --cmake /remote/work/venv/bin/cmake
python scripts/workload_planner/build_mesh_backend.py --verify-only --output /remote/work/native-mesh
RLXY_COST_ROOT=/remote/work/cost-4B python scripts/workload_planner/validate_host_cost.py /remote/work/linux-cost-validation.json
```

本次依赖为Linux GCC13.3、Protobuf C++3.21.12（Ubuntu包3.21.12-8.2ubuntu0.4）、CMake3.31.6。构建器不安装系统依赖。verify-only只校验提交、源和二进制身份，不替代packet行为测试或硬件校准。

读取deployment.json的files路径，再执行：

```bash
PYTHONPATH=/remote/work/cost-4B/src python scripts/workload_planner/run_mesh_system.py --model 4B --window tune --training-scope wafer --assignment /remote/work/plan-4B/ps/selected-plan.json --resource-layout /remote/work/plan-4B/ps/selected-plan.json --inference-layout /remote/work/plan-4B/ps/inference-layout.json --workload-forecast /remote/work/plan-4B/ps/forecast.json --native-root /remote/work/native-mesh --fast-cost-validation /remote/work/linux-cost-validation.json --mesh-controller --mesh-routing adaptive_xy_yx --output /remote/work/run-4B
python scripts/audit_idea1_inference.py /remote/work/run-4B --output /remote/work/run-4B-audit.json
```

自定义Protobuf动态库前缀需将lib目录加入LD_LIBRARY_PATH。路径为需替换的远端绝对路径。列出命令不是执行成功证据；以实际退出状态、原始日志和审计验收。

## 梯度与发布合同

训练helper可能在非区域根已恢复的TP2缓冲区新增梯度。因此生产保留全27原optimizer owner的RS，不能按区域根数删除“零贡献者”。gradient_routing.py仅有显式贡献者证明时编译root-only图，否则退回原图。owner、归一化、norm/clip、Adam、owner AG及actor_v1统一发布保持一致。

独立4B后缀原始约39.2ms，root-only加重分发约39.8ms，未采用。割面发布树未启用，不能声称发布优化收益。

## 证据迁移

`scripts/package_idea1_workload.py`按显式case计划复制证据，保留改写文档原始字节和双重身份，归档源代码不改写。`scripts/check_idea1_workload.py BUNDLE --require-final`无需模拟器，重建原生事件、TP/EOS/KV、训练贡献、owner合同并校验187个成本源文件。最终表只引用同模型、同窗口已审计case；功能例、不同信息集和全量结果分开。

完整新构建已在独立远端目录执行成功（131.73秒）；与当前测量后端的3,628个固定C++/CMake源文件逐字节一致，13个packet检查通过。构建目录改变导致二进制SHA不同，两个SHA分别记录，不伪装成同一个二进制。


## 训练队列顺序与诊断

PB默认按历史预测score从长到短安排各worker训练队列，推理execution queue仍按原public顺序。home归属、长任务配额和数量保持不变；PS继续冻结PB得到的训练顺序。保留input顺序参数用于消融。固定加和服务代理对部分顺序置换不敏感，但实际微批执行与helper可用时机会改变，因此顺序需要单独的native验证。

旧输入顺序的快速4B完整TUNE原生/审计通过，75.364960s，较旧最佳73.873630s略慢。此性能门未通过，三模型尚未启动；长优先顺序的同配置对照进行中，不将预测不变当成真实等价。

逐die表可由下式重建，训练helper是训练列的子集，不重复加入总数：

```bash
python scripts/report_idea1_workload.py /remote/work/run-4B --output /remote/work/run-4B-states
```

规划证据额外由audit_idea1_planning.py重建原始16×K预测，验证评估sample排除、候选版本及native队列绑定；仅对记录的候选分数重新排名，不冒充独立重跑每个候选。
