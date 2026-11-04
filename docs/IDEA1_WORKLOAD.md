# 工作负载分布规划与 Mesh 协作

本分支在已发布动态TP系统上加入分布驱动候选规划、释放依赖感知放置、真实mesh观测和晚区域训练粒度控制。完整4B开发验收已通过，最终4B/8B/14B的test1验收进行中。所有native结果均为公开长度轨迹与固定成本驱动的AstraSim时序仿真；不是Transformer数值训练、学习质量验证或实体晶圆计时。

## 分层职责与三个机制

|层次|入口|职责|
|---|---|---|
|工作负载合同|`src/idea1_controller/workload.py`|不可变模型/生成合同、K维联合分布、版本和历史因果检查|
|受控同题信息|`controlled_forecast.py`|显式opt-in的独立同prompt回答proxy；排除当前评估sample，不冒充连续actor历史|
|机制一：候选规划|`scripts/workload_planner/release_search.py`|合法训推布局、硬容量、每布局自己的PB放置及多情景周期估计|
|机制二：区域放置PB|`region_release_placement.py`|完整任务数量配额、预测长任务区域配额、队列平衡与home/producer依赖|
|机制二：物理位置PS|`spatial_release.py`|冻结队列和home，只换物理位置；共同阻塞、helper互斥、路径竞争；每轮32候选，最多2轮|
|机制三：运行时|`native/mesh_cohorts.py`|安全PENDING接手、合法整组TP扩展、已就绪未启动训练批helper选择与真实队列观测|
|机制三：训练程序|`native/regions.py`、`native/adaptive.py`|区域接管、实际批服务程序、晚区域粒度及共享finalizer|
|物理网络|`native_extensions/mesh_runtime/`|真实FIFO及已宣布流量记账；单逻辑消息固定选XY/YX之一|
|统一入口|`scripts/workload_planner/plan_system.py`|离线规划→执行身份后置绑定→有界位置优化；输出计划不等于验收|

推理cohort与训练区域可以不同，训练区域允许异构TP。27个规范TP2控制对/优化器owner不是固定27个推理副本。动态TP包含轮前选取和安全整组扩展，借入组保留至EOS；没有活动TP缩回或独立FFN租约。训练helper接管优势已就绪、尚未启动的完整批；`--training-scope wafer`允许跨训练区域，不能描述成始终只在同DP域内协作。

## 信息边界与分布版本

默认`distribution`使用独立prompt校准的总体/类别联合分布。相同条件未知题具有相同pooled score/q25/q75，不因Monte Carlo种子获得虚假的长短身份；各题仍保留独立16×K情景。没有同题历史就不能识别真实哪题长。

`controlled-proxy`必须显式`--allow-controlled-proxy`。它使用与当前评估sample分离的同prompt原始K向量，默认权重0.75，alpha=1用于与旧同题系统对齐；不是chronological RL历史。真实在线提交API要求observed_seq严格早于decision_seq。当前R/EOS不进入轮前规划，容量始终按硬生成上限检查。

`--forecast-summary legacy`仅供重放旧点分数及种子；旧未标记artifact按legacy解释，不可命中新pooled缓存。缓存绑定public batch、模型/生成合同、快照、服务目录和训推双方布局；新batch或漂移需重新评估及容量复核。

最终同窗对比采用controlled alpha1并披露信息口径。test1/test2已有历史实验，不能称全新held-out；test3/test4跨prompt全系统泛化未完成。独立校准模式的P0/PB消融另列，不能与不同信息集结果混作单一算法收益。

## 快速候选规划

当前8种训练覆盖×9种推理模板×3种容量前沿比例，共216候选。新增蛇形最大TP4覆盖为13×TP4＋1×TP2，TP8覆盖允许残余行不同位置；每个组合法相邻，54die完整且不重叠。不读取历史native结果生成或排名模板。

|模式|216候选规划搜索中位/s|完整进程中位/s|相对完整16情景最优代理差距|
|---|---:|---:|---:|
|one_scene|1.558536|2.237521|3.5968%|
|shortlist6|2.419998|3.112540|0.8165%|
|full|12.685062|13.418036|0%（仅该代理候选集）|

这是4B三次重复的主机计时，存在并发任务影响。约9.55ms仅为已加载artifact后的摘要和容量子步骤，不是完整规划。旧90候选0.69s不能冒充当前规模时间。代理忽略部分运行时协作与精确网络排队，最终还需原生TUNE门；不保证全局最优。

推理与PB训练home队列默认按历史score长优先。未知题同分布点分数相同，不制造题间长短差异。PS冻结PB训练顺序。可选`--home-policy service`按实际训练TP估计teacher和训练服务，避免重复除以die数；该策略全量更慢，未选为最终策略。

## 4B开发验收与策略选择

下表都是完整4B TUNE、1536回答、固定成本、原生执行与独立审计PASS。后五项使用相同controlled alpha1信息，位置/训练布局变化见备注；不是GPU test1对比。

|策略|设备周期/s|idle die·s|解释|
|---|---:|---:|---|
|扩展布局＋原批|74.566400|99.177207|推理6×TP8＋TP4＋TP2，训练27×TP2|
|service home|77.695910|140.832983|训练变9×TP4＋9×TP2，不是纯单变量放置消融|
|就绪批ready_lpt|75.876085|169.900037|更多helper事件仍可能推迟关键批|
|仅晚区域cap2|73.526129|41.172088|早区域/GEN/teacher不变|
|仅晚区域cap1|73.297172|23.739210|早区域/GEN/teacher不变，最终选中|

cap1较严格匹配原批缩短1.7021%，idle减少76.0638%，训练角色服务增加6.899502 die·s，新增权重/梯度读写真实计费；432批、96捐赠、240远端样本。GEN结束51.472602s、首训练49.317864s完全相同；无推理TP扩组。角色占用不是硬件FLOP利用率，helper是训练列子集。

`final_campaign.py`先等待所有声明4B候选完整退出并审计，且最快TUNE不高于旧批准73.873630185369s，才冻结策略并启动最终三个test1。仅依据TUNE选取one_scene/score/input/late-cap1，不能用最终三档结果倒填选择。最终规划主机2.573342/2.894729/3.365557s；设备结果仍待验收。

## 晚区域训练粒度：为什么有效

长优先运行时排序会改变其他区域helper何时空闲。实际ready_lpt对照中，关键批等到72.694s才被接手，完成75.837s；仅增加捐赠次数并不降低周期。根因之一是原子大批执行期间不能发布下一批，helper无法跨越安全边界抢占。

`--late-region-batch-cap 1`仅在全局rollout已经真实完成时构建的区域训练程序中，将本地最大样本数约束为cap×(training_tp/2)。早区域保留原大批效率，teacher/reference/old-policy不变；不读取未来EOS，也不改变完整K8优势quorum。默认0保持原程序，允许1/2/4；更小批不是默认更快，额外读写必须计费并原生验收。

参数绑定资源、部署、计划ID及微批记录；独立审计检查decision_tick不早于实际GEN结束、applied/eligible一致和批大小、样本一次贡献/统一分母。helper仍执行既有TP2容量门控的可迁移程序，不是活动批中途抢占。共享finalizer覆盖base/cached/adaptive三个真正builder，排序发生在同步门编号之前，避免只修改未被调用的基类路径。

## Mesh与位置优化的实际覆盖

PS只在完整推理cohort嵌套于训练区域、整队列装入cap、可建模无PENDING时启用。跨区cohort、TP1或未建模待启动行为保留PB。最终三档计划均因crossing cohorts保留PB，不能宣称它们取得PS位置优化收益。合法32回答功能例PB25.651234→PS25.407870s（0.9487%）仅证明该场景。

消息级XY/YX策略读取真实FIFO及已宣布未启动bytes；整个逻辑消息走同一路，不改在途包路径。13个packet检查包含旧XY等价、可替代拥塞路径、必经割面、完整bytes/ACK及clock scale。可替代路径例最大tick1081545→590125，不可外推为RL周期加速；必经割面收益为0。

完整4B同配置mesh off/on为78.145321/78.145774s，未获得周期收益。64回答训练协作功能例，原批36.454723s/15捐赠/30并发helper dies；cap2为34.827264s/29捐赠/52helper dies；cap1为34.163925s/58捐赠。小例不替代完整1536回答结果。

## 梯度与发布合同

训练helper会在非区域根已恢复的TP2缓冲区新增梯度。生产保留27规范optimizer owners的完整RS、归一化、norm/clip、Adam、owner AG及actor_v1统一发布；训练区域DP数不等于optimizer owners数。没有显式最终贡献者证书就不启用root-only图。

隔离4B后缀原始约39.2ms，稀疏区域根RS加重分发约39.8ms，未采用。割面发布树未启用。当前证据支持先优化几十秒计算和原子批安全点，不能只按理论通信字节减少宣称端到端收益。

## 远端复现

在目标机器准备独立成本目录；rl_xy默认配置不代表54活动die。14B须单独prepare，reference生命周期适配不同，不能复用4B成本目录。

```bash
python scripts/prepare_idea1_joint.py --model 4B --output /remote/work/cost-4B
RLXY_COST_ROOT=/remote/work/cost-4B python scripts/workload_planner/plan_system.py --model 4B --window tune --mode one_scene --late-region-batch-cap 1 --output /remote/work/plan-4B
```

匹配同题对照时显式增加`--forecast-mode controlled-proxy --allow-controlled-proxy --proxy-weight 1`。默认命令是独立分布路径，不应预期复现受控信息结果。

固定上游seed为4ce9ecfe746a3ee5fe47559a11d91644d641fa7f，构建器只写指定新输出，seed保持干净；需要CMake和Protobuf C++前缀，不安装系统依赖。

```bash
python scripts/workload_planner/build_mesh_backend.py --seed /remote/work/astra-seed --output /remote/work/native-mesh --protobuf-prefix /remote/work/protobuf/usr --cmake /remote/work/venv/bin/cmake
python scripts/workload_planner/build_mesh_backend.py --verify-only --output /remote/work/native-mesh
RLXY_COST_ROOT=/remote/work/cost-4B python scripts/workload_planner/validate_host_cost.py /remote/work/linux-cost-validation.json
```

本次Linux GCC13.3、Protobuf C++3.21.12、CMake3.31.6。新目录重建131.73s，3628个固定C++/CMake源字节一致，13packet检查通过；构建路径不同导致二进制SHA不同，分别记录。verify-only不代替行为测试或硬件校准。

读取deployment.json路径后执行，动态库前缀需加入LD_LIBRARY_PATH：

```bash
PYTHONPATH=/remote/work/cost-4B/src python scripts/workload_planner/run_mesh_system.py --model 4B --window tune --training-scope wafer --assignment /remote/work/plan-4B/ps/selected-plan.json --resource-layout /remote/work/plan-4B/ps/selected-plan.json --inference-layout /remote/work/plan-4B/ps/inference-layout.json --workload-forecast /remote/work/plan-4B/ps/forecast.json --native-root /remote/work/native-mesh --fast-cost-validation /remote/work/linux-cost-validation.json --mesh-controller --mesh-routing adaptive_xy_yx --output /remote/work/run-4B
python scripts/audit_idea1_inference.py /remote/work/run-4B --output /remote/work/run-4B-audit.json
python scripts/report_idea1_workload.py /remote/work/run-4B --audit-report /remote/work/run-4B-audit.json --output /remote/work/run-4B-states
```

## 验证与可迁移证据

当前全回归497项总数，3skip，其余通过；docs及git diff检查退出0。这些不替代实体硬件或数值训练。独立规划审计重建原始16×K预测、sample排除、布局及队列绑定；仅按已记录候选分数重新排名，不冒充独立重跑所有评分。

`scripts/package_idea1_workload.py`按显式case计划打包，归档源字节不改写；只迁移command.trace_path和source-map目的地，保留原文及双重哈希。`scripts/check_idea1_workload.py BUNDLE --require-final`无需模拟器即可检查三档test1/1536、187项每模型成本源、预测来源、完整原生事件、TP/EOS/KV、训练贡献和统一更新。已通过功能例、完整4B以及新cap功能的归档往返和独立负例；最终三档bundle待完成。

GPU baseline为48×A100，晶圆54die，非等功耗/面积/成本；尚无GPU同helper策略消融。最终同窗表使用旧完整系统device_s，旧Mac规划代理和本次Linux主机开销分列，不把所有对GPU收益单独归因于mesh。

## 已验证资源几何的离线候选子集

统一入口和分布CLI提供可重复的`--resource-candidate inference@fraction:training`，例如`--resource-candidate box4_top@1.0:row1`。参数未给出时仍搜索216候选；给出时只在明确子集内评分、重新计算当前batch的PB/PS及hard容量，非法名字、空集或比例拒绝。artifact、部署和独立规划审计绑定此子集，不将一个候选描述成全局最优。

资源几何可从已完成TUNE原生验收的布局库中选择，任务历史和队列仍随新batch更新。本次4B子集严格来自已审计73.297172s TUNE；重建该TUNE后，全部inference/forecast及关键home/helper/cap执行绑定逐项相同。其test1冷规划0.963052s，内部单候选搜索0.028359s；离线原生验收的成本另记，不伪装成28ms首次发现最优。

初轮test1快评选13×TP4＋TP2，native/audit均PASS，但69.720098s比旧device68.262716s慢2.135%。GEN46.453401→43.843299s更快，首次训练却43.998837→45.168828s，early overlap2.454564→0。更多捐赠74→98也未减少idle。该负例说明需评估释放时间分布和完整周期，不能只优化GEN结束或helper事件数量。

test1的shortlist6和full16均选box4/row1，完整规划分别3.364356/13.690099s。布局库复用与多情景短名单的4B原生对照正在运行；此处不预填收益。当前503项测试总数，3skip，其余通过；最终三档和GitHub CI尚待完成。

## EOS后的容量感知训练拼批（开发中）

`--adaptive-training-batch-max 8`是原生训练程序构建阶段的可选上限，默认0保留原逐模型上限。只有本区域所有相关回答都已实际EOS，才可按真实长度尝试拼批；每批继续由原SystemMemory.phase检查64GB容量，装不下则缩小。teacher两次forward不变，完整K8优势和统一更新合同不变。与晚区域cap1一起使用时，晚区域仍优先使用细批，早区域可用大批减少重复状态读写。

这是执行器的局部优化，不假称已进入原M1服务目录评分。当前TP2训练目录只覆盖到B4；不为B8填造外推时间，而用同一成本源的实际原生程序验收。512项单测总数、3skip，其余通过；独立审计新增逐样本EOS时刻、配置上限、批清单和样本一次性检查。新选项的原生功能/全量性能仍待验收。

4B test1事后容量诊断（不是规划输入或性能结果）：原B4上限形成384批；尝试B8可形成177个B8和30个B4，共207批、仍1536样本，最大训练阶段峰值62.386GB，小于原64GB限制。诊断未模拟晚区域cap或实际网络，不能按批数比例直接声称周期加速。

最新功能验收：64回答双home例真实执行early B8和late B1，原生与独立审计通过；提前区域的训练program本地服务1.569020→1.504409s，但整个fixture69.629352s不变，不称整体加速。完整4B TUNE/test1正在验证。六项历史公开实验兼容检查全部退出0，检查范围见原日志，不是重新运行历史硬件实验。

最终三模型dispatcher显式携带拼批上限，并与已审计TUNE的实际resource-layout逐字段核对；可通过预先声明的resource_candidates_by_model配置使用已验证资源几何，仍重新构建当前任务的预测、放置和容量。未声明时保留完整候选搜索。用户已暂缓GitHub上传，最终交付采用远端等价检查及可复现归档，不等待认证或线上CI。

## 前段拼批、末段细批（待性能验收）

完整4B的纯B8拼批为负例：TUNE74.372965s、test169.630843s，均原生及独立审计通过，却未过既有性能门。相同test1布局下训练角色工作1046.844207→1027.166196die·s，idle90.867048→141.753694die·s；节省服务量被原子批尾部失衡抵消。GEN和teacher不变，不能把该差异归因于预测或新硬件。

`--adaptive-training-batch-max 8 --training-tail-fine-samples 8`在本区域实际EOS后，保留原队列最后8个样本为细批（每批不超过TP/2），此前样本按容量尽量拼批。所有样本顺序、teacher、完整K8优势依赖和统一更新不变；晚区域cap继续适用。参数默认0保留历史行为；此策略同样未进入原M1 B4服务目录评分，完整native验收前不称加速。独立审计从原trace及home队列重建末段成员，拒绝跨前后段混批、伪造fine标签、漏样和容量/EOS违约。

无历史入口：统一planner的`--no-history`转发已有hard-limit预测及最小可行TP回退；独立规划审计识别capacity-only-fallback-diagnostic阶段。完整4B规划/绑定合同smoke通过（无native执行或设备耗时），冷规划约1.559s，选择可行TP1；该选择只表达保守容量回退，不宣称吞吐最优。此修复不改变当前冻结v16的训练/推理运行时。

最终可迁移验收还逐一核对部署记录与实际native的packing/tail/cap/ready-order配置、明确资源候选子集及三份原始输入哈希；原生记录仅允许追加一个LF的已知序列化差异，不能用一份配置的结果搭配另一份部署参数。真实初验三档正例及18个参数/输入哈希篡改负例通过，进一步由归档往返验证。

无历史文件路径补齐：显式no-history不再读取/哈希历史文件；默认独立分布遇到缺失文件时记录HISTORY_FILE_MISSING并返回容量回退，空calibration池同理。损坏或模型/配置不匹配的现有历史明确失败。真实planner入口测试主动拒绝所有history字节读取，显式无历史和文件缺失两例均通过。此修复只在规划层，当前final11冻结运行源不修改。

为补充未知prompt验证，预先冻结两个4B独立评价batch：test3前44组/352回答、test4前79组/632回答。各窗口再下一组与TUNE存在canonical文本重复，因此前缀仅按公开hash确定；不称完整1536窗口held-out，不减少主表三档1536工作量。三种信息策略共用相同冻结候选族/执行/样本/cost：无历史最小可行TP、独立calibration分布、额外同题信息的离线controlled alpha1；最后一种不是未知prompt预测或真实时间序列。原生结果未产生前协议已固定，不按这些结果改规则，不报告未执行的oracle或全局regret。统一入口新增subset-prompts，保持完整K8组并在预测/绑定两端一致传递。

独立规划审计进一步从原始test1/test2规范prompt、TUNE/reserved排除集合及原始history样本重建calibration快照，避免只验证快照自哈希。真实初验三档均通过；已构造内容被改动且重新计算自哈希仍合法的快照，原始来源重建会拒绝，污染calibration prompt集合也会拒绝。无快照时不读取历史文件。
