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
- `one_scene`快评当前216布局候选；`shortlist6`对少量候选追加16场景；`full`离线评估当前216×16。代理最优不是原生/硬件最优。
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

PB默认按历史预测score从长到短安排各worker训练队列；当前推理队列同样可独立选择长优先，旧输入序保留消融。home归属、长任务配额和数量保持不变；PS继续冻结PB得到的训练顺序。保留input顺序参数用于消融。固定加和服务代理对部分顺序置换不敏感，但实际微批执行与helper可用时机会改变，因此顺序需要单独的native验证。

旧输入顺序的快速4B完整TUNE原生/审计通过，75.364960s，较旧最佳73.873630s略慢。此性能门未通过，三模型尚未启动；长优先顺序的同配置对照进行中，不将预测不变当成真实等价。

逐die表可由下式重建，训练helper是训练列的子集，不重复加入总数：

```bash
python scripts/report_idea1_workload.py /remote/work/run-4B --output /remote/work/run-4B-states
```

规划证据额外由audit_idea1_planning.py重建原始16×K预测，验证评估sample排除、候选版本及native队列绑定；仅对记录的候选分数重新排名，不冒充独立重跑每个候选。


## 条件分布点分数版本

默认pooled汇总从整个条件样本池计算score/q25/q75，相同条件的未知任务不因Monte Carlo随机种子而获得虚假的不同点分数。每题仍有独立K向量情景，用于评估真实随机波动；情景不代表提前知道该题答案。预测准入、hard capacity及真实在线历史序列门不变。

旧实验通过显式legacy汇总重放；旧未标记缓存按legacy解释，不能默认命中新pooled合同。新候选、分布绑定及native资源记录包含forecast_summary标签，独立规划审计按标签重建。受控alpha1的4B新旧数值控制字段、TP/DP、队列与cap已逐项比较一致；快照及缓存身份有意不同。当前原生TUNE训练顺序消融仍使用冻结的v5源，不能把这次API修正混作其性能收益。

legacy只是历史点分数与种子的兼容选择，不代表旧控制器。新接口可用--forecast-summary legacy明确复现旧信息处理；未使用同题历史的默认路径采用pooled。


## 候选覆盖与推理队列

通用布局增加9×3 pair网格蛇形路径的最大TP4覆盖（13×TP4＋1×TP2），以及余量行在顶部的最大TP8覆盖。每个组仍是合法相邻矩形，54die完整不重叠；不从历史native时间读取布局排名。与原五布局相比，当前八布局产生216个训推/cap候选。4B一次快速搜索1.548s、进程2.065s，重复计时另行报告；旧90候选0.69s不能冒充新规模耗时。

推理与训练队列均默认预测长任务优先，显式input顺序保留消融。两种顺序分别设定；未知题相同条件点分数相同，不制造虚假的题间长短。TP8内全数已入场时顺序影响可能很小；8B/14B cap更小时，长任务先入场可避免推迟到最后一波。因此4B的无待启动工作测试不能代替所有cap压力验证。

4B长优先训练序对照75.263858s，相对输入序75.364960s只缩短0.101102s，仍未达旧批准73.873630s，未启动最终三模型。此结果保持资源、归属、推理队列/cap及预测相同；不将微小改进夸大为解决回归。


## 可选成本感知放置

统一入口的`--home-policy service`保留原数量及长任务区域配额，以公开prompt长度、独立条件历史和已有服务目录估算各训练TP的teacher＋训练成本，再与物理producer的粗粒度释放估计相加。已经按实际TP估算的服务不能再除一次区域die数。默认`score`保持既有PB对照；该可选策略需要独立native验收，不能把代理下降当作设备收益。

存在原始同题条件K向量时，对原始独立向量成本取均值；否则以条件点score推导稳定的代表输出长度。后者不是输出均值/二阶矩的无偏估计，但相同条件、相同prompt长度不会因为不同情景随机种子获得虚假的不同placement成本。逐题独立16情景仍用于候选最终评分。生产者释放估计使用容量前沿的固定上下文吞吐近似，不声称重放真实mesh或helper。

新增策略绑定于候选artifact、放置proof和deployment元数据。当前4B单次快速搜索2.247s，选box4_top＋row2/DP18，仍处于待native验证阶段；与已有row1/DP27的性能差异不能全部归因于单个放置变化。


## 就绪训练批的运行时顺序（可选）

`--training-ready-order ready_lpt`在区域EOS数据齐备后，用已知序列形状生成的批服务估计将完整训练批从长到短排列。它不会在生成阶段读取当前R；队列仍在原生优势quorum后才发布。每批样本、计算程序、ticket身份、梯度权重和统一更新分母保持不变，teacher/advantage/gradient-zero/restore阶段不跨越。并行区域所有成员使用同一顺序及重新编号的同步门。原生独立审计检查每区域ticket序列；默认input保留旧行为。该选项需要新的全量原生验收，不能按LPT理论直接填写加速。


## 晚启动训练区域的原子任务粒度（可选）

`--late-region-batch-cap 2`仅对全局rollout已经真实完成时才构建训练程序的区域，限制每个TP2控制对形成的本地训练批样本数。早区域保持原批大小，teacher两次forward不变；原始完整K8优势已准备后才执行各训练批。目标是让晚区域更频繁到达可接手未启动批的安全点，并产生更多可供helper承接的原子任务。移到helper的批仍按现有TP2容量门执行；不新增活动批中途抢占或单独FFN租约。

默认0保持原程序。资源记录、部署元数据、微批选择和独立审计绑定参数；启用记录必须在真实generation_end之后作出，且样本/统一梯度分母守恒。此选项待原生验收，不能把更小批假定为更快：权重及梯度重复读写也需计入。
