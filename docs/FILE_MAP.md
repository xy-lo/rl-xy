# 公开仓库文件职责表

共187个公开文件。目录保持src/configs/docker/scripts/tests/docs分层，experiments存轻量结果与长度输入。

| 文件 | 职责 |
| --- | --- |
| `.dockerignore` | 限制 Docker build context；排除依赖缓存、历史运行结果、构建日志和验收包。 |
| `.gitignore` | 区分源码与可再生文件；排除 .deps、缓存、运行结果、验收包和构建日志。 |
| `LICENSE` | 自建工程的 MIT 许可证；上游 AstraSim/Chakra 仍遵守各自许可证。 |
| `Makefile` | Docker 脚本的快捷入口；提供 build、test、transfer、toy、two-rounds、all。 |
| `README.md` | 工程总入口、最小例子、运行命令、证据边界与文档导航。 |
| `compose.yaml` | 可选 Compose 构建/交互声明；正式结果使用 docker.sh 保留镜像身份。 |
| `configs/fixture_transfer/astra_network.yml` | 两 rank FullyConnected；精确表达放大 1000 倍后的带宽与链路时延。 |
| `configs/fixture_transfer/astra_remote_memory.json` | 固定 NO_MEMORY_EXPANSION；禁止隐式借用远程内存。 |
| `configs/fixture_transfer/astra_system.json` | FIFO/native ring、trace 开关、零额外 endpoint/reduction delay 和后端设置。 |
| `configs/fixture_transfer/case.json` | 该案例的工作负载、硬件、策略与合成成本合同；CLI 的主输入。 |
| `configs/gpu/a100_48_optimistic.json` | 刻意偏向GPU的GEMM TensorCore/零launch敏感性，不是物理FP32吞吐。 |
| `configs/gpu/a100_48_reference.json` | 48A100-SXM80GB真实名义规格、声明输入精度模型及未校准launch/延迟/共享带宽假设。 |
| `configs/gpu/wafer_typed_rate_sensitivity.json` | 晶圆FP32/vector降到64的未知率敏感性，不替换正式H0来源值。 |
| `configs/hardware/fixture_4die.json` | 独立4die/4核native回归来源，不是正式H0。 |
| `configs/hardware/xy_h0_case3_performance.json` | 54die Case3来源合同；Fig3几何仅模板参考，不缩计算die或宣称制造布局验证。 |
| `configs/hardware/xy_h0_wsc_case3.json` | 用户H0原表的独立来源合同：54die、FP16 1020FLOP/ns及原始几何；未知值不补写。 |
| `configs/hardware/xy_h1h2h3_54.json` | 正式H1–H3候选参数、单位、来源、空值与设计/实测状态；不是可执行硬件精度声明。 |
| `configs/qwen2_group_fixture/astra_network.yml` | TP2教学FullyConnected带宽8B/ns和时延1model ns，不是正式晶圆mesh。 |
| `configs/qwen2_group_fixture/astra_remote_memory.json` | 无外部内存扩展，所有容量由原生资源合同检查。 |
| `configs/qwen2_group_fixture/astra_system.json` | FIFO、TP ring、roofline关闭的原生系统合同。 |
| `configs/qwen2_group_fixture/case.json` | 小GQA/绑权/RoPE、两group四异长trajectory手工fixture及明确教学硬件。 |
| `configs/qwen2_updates_fixture/astra_network.yml` | 教学两节点FullyConnected，8B/ns及1model ns。 |
| `configs/qwen2_updates_fixture/astra_remote_memory.json` | 无外部内存扩展；容量由资源hook拒绝超额。 |
| `configs/qwen2_updates_fixture/astra_system.json` | FIFO与TP ring原生执行合同，关闭roofline避免重复计时。 |
| `configs/qwen2_updates_fixture/case.json` | 2epoch、minibatch3的四trajectory教学fixture；TP2/单逻辑核，不是H0。 |
| `configs/toy_grpo_2die/astra_network.yml` | 两 rank FullyConnected；精确表达放大 1000 倍后的带宽与链路时延。 |
| `configs/toy_grpo_2die/astra_remote_memory.json` | 固定 NO_MEMORY_EXPANSION；禁止隐式借用远程内存。 |
| `configs/toy_grpo_2die/astra_system.json` | FIFO/native ring、trace 开关、零额外 endpoint/reduction delay 和后端设置。 |
| `configs/toy_grpo_2die/case.json` | 该案例的工作负载、硬件、策略与合成成本合同；CLI 的主输入。 |
| `configs/wafer4_fixture/astra_network.yml` | 历史TP2模板元数据；run-wafer从显式scenario生成真实mesh配置。 |
| `configs/wafer4_fixture/astra_remote_memory.json` | 无外部扩展内存，实际容量仍由native资源hook校验。 |
| `configs/wafer4_fixture/astra_system.json` | 基础FIFO/TP ring模板，运行时生成两维物理metadata及明确通信组。 |
| `configs/wafer4_fixture/case.json` | 明确合成完整group工作负载，原始TP2逻辑模板，hierarchical入口展开DP。 |
| `configs/wafer4_fixture/wafer.json` | mesh/DP、来源H路径及每项未校准执行假设，绝不混成论文实测参数。 |
| `configs/wafer54_fixture/astra_network.yml` | 历史TP2模板元数据；run-wafer从显式scenario生成真实mesh配置。 |
| `configs/wafer54_fixture/astra_remote_memory.json` | 无外部扩展内存，实际容量仍由native资源hook校验。 |
| `configs/wafer54_fixture/astra_system.json` | 基础FIFO/TP ring模板，运行时生成两维物理metadata及明确通信组。 |
| `configs/wafer54_fixture/case.json` | 明确合成完整group工作负载，原始TP2逻辑模板，hierarchical入口展开DP。 |
| `configs/wafer54_fixture/wafer.json` | mesh/DP、来源H路径及每项未校准执行假设，绝不混成论文实测参数。 |
| `docker/Dockerfile` | 基于指定本地镜像构建 Abseil、Protobuf、固定 AstraSim 和隔离 Python 环境。 |
| `docker/Dockerfile.execution` | 复用0.3原生后端，打包0.4 Python执行层，不安装GPU训练依赖。 |
| `docker/Dockerfile.gpu` | 在0.5上编译独立GPU网络扩展，保存patch/header与二进制身份。 |
| `docker/Dockerfile.inputs` | 复用已验证0.2原生镜像，打包0.3 Python输入接口，无新GPU依赖。 |
| `docker/Dockerfile.mapped` | 在固定0.1镜像上接入资源、CPU统计修正和mesh扩展，编译两个analytical后端并冻结patch hash。 |
| `docker/Dockerfile.schedule` | 复用0.4原生Astra镜像、打包0.4.1 Python更新层；不使用GPU。 |
| `docker/Dockerfile.wafer` | 固定0.4.1基础上编译逐核/压缩资源hook，记录新增header hash并打包0.5 Python。 |
| `docker/astrasim/RlxyGpu.hh` | GPU端点、每节点switch、共享uplink及spine，实际native Link竞争。 |
| `docker/astrasim/RlxyMesh.hh` | 新增无wrap二维mesh、XY路由；使用原生Device/Link/Chunk排队并输出实际路径。 |
| `docker/astrasim/RlxyResources.hh` | 实际Astra时钟的ALLOC/READY/FREE、容量、KV预留、SRAM和排他local-memory准入；错误强制非零退出。 |
| `docker/astrasim/gpu.patch` | 固定aware入口增加GPU topology选择，不改默认mesh分支。 |
| `docker/astrasim/mesh.patch` | 在congestion-aware frontend显式选择mesh2d override；无override保留上游行为。 |
| `docker/astrasim/statistics.patch` | 允许CPU/控制节点参与回放；以GPU/COMM区间并集计算overlap，避免CPU空隙误计或无符号下溢。 |
| `docker/astrasim/workload.patch` | 向上游Workload的can_issue、issue、callback插入资源hook，不另建事件时钟。 |
| `docs/09_H0_SOURCE_AND_GEOMETRY.md` | H0逐项单位/来源、FP16口径、48/54几何冲突和缺参说明。 |
| `docs/10_OPTIMIZER_SCHEDULE.md` | 多epoch/minibatch组件职责、冻结behavior、梯度清零/归一化/发布和执行限制。 |
| `docs/ARCHITECTURE.md` | W/H/S 分层、模块边界、rank-local DAG、时间编码与当前扩展边界。 |
| `docs/EVIDENCE_BOUNDARIES.md` | 区分合成回放、解析假设、硬件校准和真实训练；规定指标可支持的结论。 |
| `docs/FILE_MAP.md` | 公开仓库全部文件职责与分层目录。 |
| `docs/README.md` | 当前组件/历史说明与实验结果的文档导航。 |
| `docs/REPRODUCIBILITY.md` | 环境、执行顺序、结果包、hash 复查和失败后的复现规则。 |
| `docs/REPRODUCING.md` | 完整镜像依赖链、公开64轨迹复现及证据可用范围。 |
| `docs/components/astra_adapter.md` | Chakra 映射、官方 CLI、单位、日志契约及 SEND/RECV 口径。 |
| `docs/components/config.md` | 配置组件的输入、输出、字段约束和失败条件。 |
| `docs/components/data_sources.md` | 旧数据和公开数据各自可提供的证据、许可与不可替代项。 |
| `docs/components/docker.md` | 指定基础镜像、独立 venv、项目持久容器、临时代理和正式运行说明。 |
| `docs/components/gpu_simulation.md` | 论文方法、48卡近预算、GPU独立成本/网络、精度与硬件不确定性及四组比较边界。 |
| `docs/components/ledger.md` | storage 和 KV 容量账本的字段、公式、释放规则及覆盖范围。 |
| `docs/components/mapped_costs.md` | 映射/流量公式、计时合同、analytic/lookup来源与DNN边界。 |
| `docs/components/migration_inputs.md` | 输入接入模块职责、命令、clip审计、执行边界与下一步。 |
| `docs/components/network.md` | mesh适配、原生排队、四rank反事实、YAML override和拓扑适用边界。 |
| `docs/components/qwen_execution.md` | Qwen2/GQA执行职责、数据流、验收方式、固定来源和不覆盖的能力。 |
| `docs/components/reporting.md` | run bundle 每类产物、指标定义、来源和哈希/状态约束。 |
| `docs/components/tensor_resources.md` | 张量类别、保守pin与真实流量的区别、准入时序、资源合同及未建模buffer。 |
| `docs/components/validation.md` | 独立验收项、fail-closed 行为、手算链路和负例要求。 |
| `docs/components/wafer_execution.md` | 54die/多核/DP语义、几何和假设、GPU隔离/复现、对比口径、资源压缩及未实现边界。 |
| `docs/components/workload.md` | 工作流组件的阶段、操作、采样/TP/损失/反向与版本语义。 |
| `docs/完整流程预览.html` | 本文的轻量预览：运行结果、12 阶段流程、阶段时间与文件清单。 |
| `docs/工程文件与完整流程.md` | 本文：全部自建文件索引、完整案例、实际阶段时间、复现入口及结果解释。 |
| `docs/资源版完整流程预览.html` | v0.2自包含轻量预览，含实际指标和逐阶段数据。 |
| `docs/资源版工程说明与完整流程.md` | v0.2自动生成的验收、指标、阶段时间、模块及全部文件索引；不替换历史报告。 |
| `examples/toy_grpo_2die/README.md` | 教学例子的模型、两条轨迹、参数/KV/版本公式和逐步使用说明。 |
| `experiments/2026-10-27-v0.6/acceptance_summary.json` | 原20命令、退出状态、日志hash和native检查数量。 |
| `experiments/2026-10-27-v0.6/comparisons.json` | 原始四组机器可读比较，完整预算与假设。 |
| `experiments/2026-10-27-v0.6/core_runtime_hashes.json` | 45个原始运行核心源码/C++文件SHA256。 |
| `experiments/2026-10-27-v0.6/input_hashes.json` | 公开输入精确SHA256。 |
| `experiments/2026-10-27-v0.6/inputs/astra_network.yml` | 原实验逐字节复制的结构/长度/来源输入。 |
| `experiments/2026-10-27-v0.6/inputs/astra_remote_memory.json` | 原实验逐字节复制的结构/长度/来源输入。 |
| `experiments/2026-10-27-v0.6/inputs/astra_system.json` | 原实验逐字节复制的结构/长度/来源输入。 |
| `experiments/2026-10-27-v0.6/inputs/case.json` | 原实验逐字节复制的结构/长度/来源输入。 |
| `experiments/2026-10-27-v0.6/inputs/compilation.json` | 原实验逐字节复制的结构/长度/来源输入。 |
| `experiments/2026-10-27-v0.6/inputs/gpu_provenance.json` | 原实验逐字节复制的结构/长度/来源输入。 |
| `experiments/2026-10-27-v0.6/inputs/model_config.json` | 原实验逐字节复制的结构/长度/来源输入。 |
| `experiments/2026-10-27-v0.6/local_verification_summary.json` | 已执行的本地原始事件/资源独立复核结果。 |
| `experiments/2026-10-27-v0.6/metrics_summary.json` | 六份运行选定指标及原始metrics/manifest SHA。 |
| `experiments/2026-10-27-v0.6/model_bottlenecks.json` | 模型服务与实际Astra事件的分解数据。 |
| `experiments/2026-10-27-v0.6/provenance.json` | 历史源身份/归档hash与公开整理差异。 |
| `experiments/2026-10-27-v0.6/实验结果.md` | 本版完整实验设置、预算、四组结果、验证及局限。 |
| `experiments/2026-10-27-v0.6/模型内部瓶颈分解.md` | 模型服务分量对照，非硬件计数器。 |
| `pyproject.toml` | Python 包名/版本/依赖、src 布局及 rlxy CLI 注册。 |
| `scripts/acceptance.py` | 依次执行容器测试、链路、单轮、两轮、重复运行和完整性门禁，保存命令日志。 |
| `scripts/build_images.sh` | 无sudo按0.1到0.6顺序构建全部本地镜像。 |
| `scripts/check_public_results.py` | 核对公开摘要、历史核心代码及输入hash，不冒充原始日志重放。 |
| `scripts/check_result.py` | 复查成功状态、非空检查集、固定上游版本、Docker 证据和全部文件 SHA-256。 |
| `scripts/compare_gpu_wafer.py` | 严格匹配模型/trace/group/长度/算法更新后输出GPU观测与场景预测；失配拒绝比值。 |
| `scripts/compare_simulated_platforms.py` | 两边都仿真；严格全step/输入/算法/近预算合同，输出所有资源差异，拒绝用实测秒数预测GPU。 |
| `scripts/docker.sh` | 无sudo Docker构建/受控容器复用，精确UID/项目/镜像检查；可挂载已核对链接的自有外部证据目录。 |
| `scripts/execution_acceptance.py` | 0.4 Docker编排：多组/两轮/重复/带宽/实际模型/旧长度短探针和原生回归。 |
| `scripts/gpu_capture_wrapper.py` | 只在私有容器适配既有collector，另存未舍入阶段时间和完整解析配置，不激活主机flag。 |
| `scripts/gpu_network_fixtures.py` | 原生验证节点共享上行竞争、独立双向路径和同节点绕过spine，保持完整统计门禁。 |
| `scripts/gpu_sim_acceptance.py` | 先GPU4/旧mesh/native门禁，再独立完整GPU/偏向GPU/晶圆低rate运行与四组比较，不预设赢家。 |
| `scripts/gpu_smoke.py` | 专用GPU容器中受限veRL真训练/长度采集；模型/数据/包/源码hash及失败日志。 |
| `scripts/input_acceptance.py` | 宿主编排0.3 Docker验收、预期拒绝、旧数据复用及原生流程回归。 |
| `scripts/integration_fixtures.py` | Docker内真实执行资源竞争/READY门控/七类C++负例及4rank共享/独立mesh路由反事实。 |
| `scripts/make_execution_case.py` | 实际HF形状+指定完整长度group编译；记录来源、未知数值和显式反事实response cap。 |
| `scripts/prepare_matched_case.py` | 真实GPU成功step的全部group导入，保持每条原长度，无cap改动及GPU计时注入。 |
| `scripts/private_sram_fixtures.py` | 原生同核/异核争用、单核溢出/错误总和/核数负例与raw/compact时钟等价。 |
| `scripts/remote_session.py` | 既有转发端口发现、主机key核验、保活复用和不确定写状态保护；不保存密码。 |
| `scripts/reproduce_v06.sh` | 不申请GPU，从公开完整长度case生成新control与四组对照。 |
| `scripts/resource_acceptance.py` | 宿主只编排Docker命令；单轮/两轮/repeat、带宽/SRAM敏感性、OOM、native fixture、完整性与版本冻结。 |
| `scripts/reuse_legacy.py` | Docker内核对旧日志/CSV/summary/state，生成3种子长度负载和来源hash。 |
| `scripts/schedule_acceptance.py` | Docker门禁：87测试、H0预期拒绝、五原生bundle、重复/两轮/旧结果及mesh资源回归。 |
| `scripts/wafer_acceptance.py` | Docker单测、逐核/旧native、4/54die、原toy、64条真实GPU全step与对比/完整性门禁。 |
| `scripts/write_execution_report.py` | 从最终PASS和匹配源码生成本报告及全文件说明。 |
| `scripts/write_gpu_sim_report.py` | 仅从绑定实际PASS生成结果、所有文件说明及HTML预览。 |
| `scripts/write_input_report.py` | 读取本轮PASS证据，生成报告/轻量预览及完整文件说明；拒绝源码hash失配。 |
| `scripts/write_resource_report.py` | 从资源版PASS证据生成本文件与HTML预览；逐文件说明缺项即拒绝。 |
| `scripts/write_schedule_report.py` | 从匹配源码hash的PASS生成工程文档、全文件清单及轻量HTML。 |
| `scripts/write_wafer_report.py` | 读取实际匹配源码的PASS与比较报告，生成全部文件说明和轻量HTML。 |
| `scripts/write_walkthrough.py` | 读取 PASS 验收包生成本文及 HTML 预览；逐文件说明缺项时立即失败。 |
| `src/rl_xy/__init__.py` | Python 包入口与包版本。 |
| `src/rl_xy/astra_adapter/__init__.py` | 公开配置核验、trace 生成、进程执行、日志解析的边界接口。 |
| `src/rl_xy/astra_adapter/config.py` | 交叉核验 system/network/remote-memory 与 case 的 rank、时间及带宽单位。 |
| `src/rl_xy/astra_adapter/generator.py` | 生成每 rank 的官方 Chakra ET；验证节点、tag、参与者、本地依赖和 collective。 |
| `src/rl_xy/astra_adapter/parser.py` | 配对 issue/callback，核验节点类型、所有 rank 完成与统计，不接受残缺结果。 |
| `src/rl_xy/astra_adapter/runner.py` | 核验官方 commit 和可执行文件；无 shell 调用 AstraSim，保存命令与原始日志。 |
| `src/rl_xy/cli.py` | argparse 单入口；run 执行 toy，fixture-transfer 执行链路测试。 |
| `src/rl_xy/communication_lowering.py` | 跨die通信展开端点读/原生网络/端点写，并补反向残差相加；重建rank-local依赖。 |
| `src/rl_xy/config.py` | 严格读取 case JSON；拒绝未知字段、类型漂移及不支持的并行/学习合同。 |
| `src/rl_xy/cost_model.py` | 解析inclusive-local成本和带来源/OOD拒绝的精确profile lookup接口；未训练DNN。 |
| `src/rl_xy/costs.py` | 声明的 FLOP/vector 合成成本；保留局部 DMA 公式；无硬件实测校准。 |
| `src/rl_xy/execution_contract.py` | 多prompt/group语义、独立组内demean和compact QKV宽度；未知reward不伪造。 |
| `src/rl_xy/execution_validation.py` | 通用阶段/版本/训练/形状及实际Astra事件检查，独立于旧A/B曲线。 |
| `src/rl_xy/fixture.py` | 独立的 21 ns 数据移动链、out-of-place SRAM 分配与 63 B 容量负例。 |
| `src/rl_xy/gpu_model.py` | 独立GPU SM/CTA分区/FLOP/私有shared memory/共享HBM成本，有理峰值分摊；GPU与晶圆预算审计。 |
| `src/rl_xy/hardware_profile.py` | H合同、几何/聚合检查、缺失参数与后端能力门禁。 |
| `src/rl_xy/ledger.py` | 参数/优化器/梯度持久 storage 与 rollout KV 的身份、生命周期和准入容量账。 |
| `src/rl_xy/legacy_workload.py` | 旧原始日志/Hydra导入；统计约束合成；clip参考口径显式选择。 |
| `src/rl_xy/mapped_costs.py` | 在操作图不变的条件下应用typed GEMM和显式向量/状态RMW成本，不生成时间戳。 |
| `src/rl_xy/mapping.py` | 单逻辑核OS合法tile枚举与精确ragged/batch流量；不将非法tile静默塞入SRAM。 |
| `src/rl_xy/migration.py` | W/H/模型输入联合预检和状态容量必要条件，不产生晶圆时间。 |
| `src/rl_xy/model.py` | MHA/GQA、QKV bias与绑权后的精确unique/local参数、持久状态和compact KV。 |
| `src/rl_xy/model_import.py` | 实际Qwen2配置/参数检查和受限算子回放能力；正式H门禁仍保持。 |
| `src/rl_xy/multicore.py` | N/M/batch输出分区、ragged输出/FLOP守恒、私有SRAM和共享DRAM/模块瓶颈，BF16/FP32独立rate。 |
| `src/rl_xy/operator_ir.py` | 有类型的张量shape/dtype/layout、Matmul方向与KernelSpec及互斥计时合同。 |
| `src/rl_xy/reporting.py` | 写 CSV/JSON/JSONL、round.phase 时间、指标、SUMMARY 与 hash manifest。 |
| `src/rl_xy/resource_encoding.py` | 严格rank顺序的读集合delta和内容寻址workspace，无损还原与篡改拒绝。 |
| `src/rl_xy/resource_validation.py` | 独立重放C++资源日志并与实际Astra ticks逐节点交叉核验，生成主流程内存峰值。 |
| `src/rl_xy/run.py` | 串接配置复制、trace、AstraSim 执行、解析、账本检查及成功/失败归档。 |
| `src/rl_xy/tensor_plan.py` | 完整声明对象、保守tape/反向partial保留、KV预留兑现和每节点可执行资源合同。 |
| `src/rl_xy/training_schedule.py` | 确定顺序minibatch/epoch图扩展、每更新命名/版本/分母、rank本地依赖重建。 |
| `src/rl_xy/training_validation.py` | 独立重建更新计划并复用逐更新完整训练检查；拒绝版本/清零/优势/分母篡改。 |
| `src/rl_xy/types.py` | 类型合同与TokenExtent长度对象；长度不假装token值。 |
| `src/rl_xy/validation.py` | 从独立清单核对阶段、反向、清零、Adam 读写、版本、KV、通信与真实事件。 |
| `src/rl_xy/verl_trace.py` | veRL L1长度/分组/版本/阶段记录的严格读取与未知数值标记。 |
| `src/rl_xy/wafer_graph.py` | 完整group轮转分DP、全局token分母、真实TP/DP通信组与梯度SUM、round fence及唯一资源映射。 |
| `src/rl_xy/wafer_run.py` | 来源绑定的显式H场景、rank分文件、原生mesh与独立完整事件/内存/梯度检查、证据归档。 |
| `src/rl_xy/workflow.py` | 兼容旧toy；Qwen2 compact QKV/bias/RoPE/GQA、独立group和全局单次更新事件。 |
| `tests/test_astra_config.py` | 带宽缩放/两套配置正例，带宽漂移和 bool 冒充整数负例。 |
| `tests/test_check_result.py` | 完整 bundle 校验正例，以及篡改 metrics 后拒绝的负例。 |
| `tests/test_comparison.py` | 纯合成单测验证真实比较的门禁：cap、模型/日志hash、更新次数失配不能输出ratio。 |
| `tests/test_config_model.py` | 参数/KV 手算、未知字段、证据等级、策略字段和 rounds 覆盖测试。 |
| `tests/test_fixture.py` | T01 结构/容量手算与 63 B out-of-place 不可行负例。 |
| `tests/test_generator_contract.py` | P2P 参与者错误、tag 越界、不支持 collective 的 fail-closed 测试。 |
| `tests/test_gpu_model.py` | GPU CTA ragged/FLOP/私有容量/聚合峰值/带宽/精度敏感性与近预算差异测试。 |
| `tests/test_ledger.py` | 长期状态、KV reservation/materialization、NIC 计划容量及两轮无泄漏检查。 |
| `tests/test_legacy_workload.py` | 日志/配置一致性、合成矩约束、确定性及batch-max截断口径反例。 |
| `tests/test_mapping_cost_model.py` | OS手算、ragged独立循环、SRAM合法性/自动重映射、带宽/dtype/profile失配等测试。 |
| `tests/test_migration_inputs.py` | H几何/单位/来源/null、trace语义、真实模型参数检查及负例。 |
| `tests/test_parser.py` | 事件配对、缺 callback、乱序日志、rank-local 节点和提前 RECV 等待测试。 |
| `tests/test_qwen_execution.py` | Qwen参数/KV金标准、多组/未知reward/绑权/生命周期、RoPE/GQA梯度恒等式和合同变异负例。 |
| `tests/test_reporting.py` | 两轮 phase 不混桶，以及空参数表仍有固定 CSV schema。 |
| `tests/test_simulated_comparison.py` | 纯合成比较门禁：可翻转赢家、禁止实测替代、不同capture和预算不足拒绝比值。 |
| `tests/test_tensor_resources.py` | tape/梯度shape、延迟READY、容量/生命周期/预留失败、只读Reference及两轮无增长。 |
| `tests/test_training_schedule.py` | H0来源/几何/单位负例；尾批、拆组优势、两轮版本、梯度存储和多种篡改负例。 |
| `tests/test_validation.py` | 完整证据正例；删依赖、漏反向/Adam 状态、容量越界/漏事件等负例。 |
| `tests/test_wafer_graph.py` | 私有映射/守恒、共享带宽、DP不均匀长度分母/组、几何scope、无损压缩与拒绝负例。 |
| `tests/test_workflow.py` | 本地依赖链、完整前后向、embedding TP 求和、梯度清零及 v2 发布。 |
