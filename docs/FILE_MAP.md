# 文件索引

当前职责以本表为准。主链是`run-aligned`；旧入口和报告生成器保留兼容，不能把旧版本结果改名当前基线。新旧运行身份与公开证据范围见[架构](ARCHITECTURE.md)、[结果](RESULTS.md)。

按实际公开文件生成，一路径一条，无重复追加版本表。原始运行日志、缓存、依赖、凭据及归档大包不属于源码索引。

## 根目录与检查流程

| 文件 | 职责 |
|---|---|
| `.dockerignore` | 限制 Docker build context；排除依赖缓存、历史运行结果、构建日志和验收包。 |
| `.github/workflows/ci.yml` | 固定官方Actions提交的Python回归、公开证据和文档检查；不冒称native或硬件校准。 |
| `.gitignore` | 区分源码与可再生文件；排除 .deps、缓存、运行结果、验收包和构建日志。 |
| `LICENSE` | 自建工程的 MIT 许可证；上游 AstraSim/Chakra 仍遵守各自许可证。 |
| `Makefile` | 当前0.9.5镜像与受限Docker测试、公开检查和历史小例快捷入口。 |
| `README.md` | 当前状态、3B结果与5份核心文档入口。 |
| `compose.yaml` | 可选 Compose 构建/交互声明；正式结果使用 docker.sh 保留镜像身份。 |
| `pyproject.toml` | Python 包名/版本/依赖、src 布局及 rlxy CLI 注册。 |

## 实现

| 文件 | 职责 |
|---|---|
| `src/rl_xy/__init__.py` | Python 包入口与包版本。 |
| `src/rl_xy/aligned_graph.py` | 固定48活动rank同步阶段、DO状态/更新、训推权重转换与精确组件模板缓存。 |
| `src/rl_xy/aligned_run.py` | 原生宏事件/全局资源/阶段和版本验收、独立逻辑证明与本轮GEMM域。 |
| `src/rl_xy/astra_adapter/__init__.py` | 公开配置核验、trace 生成、进程执行、日志解析的边界接口。 |
| `src/rl_xy/astra_adapter/config.py` | 交叉核验 system/network/remote-memory 与 case 的 rank、时间及带宽单位。 |
| `src/rl_xy/astra_adapter/generator.py` | 生成每 rank 的官方 Chakra ET；验证节点、tag、参与者、本地依赖和 collective。 |
| `src/rl_xy/astra_adapter/parser.py` | 配对 issue/callback，核验节点类型、所有 rank 完成与统计，不接受残缺结果。 |
| `src/rl_xy/astra_adapter/runner.py` | 核验官方 commit 和可执行文件；无 shell 调用 AstraSim，保存命令与原始日志。 |
| `src/rl_xy/baseline_contract.py` | 区分官方batch/切分合法性、IR能力和严格迁移身份，不把候选合法当作最优或已实现。 |
| `src/rl_xy/batching.py` | 同一步独立生成请求线性GEMM打包、ragged注意力隔离、FLOP/样本守恒。 |
| `src/rl_xy/capacity_mapping.py` | 按真实容量与声明成本搜索合法多核输出网格和OS分块，不使用最终胜负。 |
| `src/rl_xy/cli.py` | argparse 单入口；run 执行 toy，fixture-transfer 执行链路测试。 |
| `src/rl_xy/colocation.py` | 固定veRL同步协议、连续TP训推切换、全局屏障、权重/样本重分片和驻留账本。 |
| `src/rl_xy/colocation_validation.py` | 从native实际事件校验全局阶段屏障、版本与完整资源生命周期。 |
| `src/rl_xy/communication_lowering.py` | 跨die通信展开端点读/原生网络/端点写，并补反向残差相加；重建rank-local依赖。 |
| `src/rl_xy/component_proof_cache.py` | 同一immutable串行模板首次完整replay，复用时逐宏时钟检查与结束签名复核。 |
| `src/rl_xy/config.py` | 严格读取 case JSON；拒绝未知字段、类型漂移及不支持的并行/学习合同。 |
| `src/rl_xy/cost_model.py` | 解析inclusive-local成本和带来源/OOD拒绝的精确profile lookup接口；未训练DNN。 |
| `src/rl_xy/costs.py` | 声明的 FLOP/vector 合成成本；保留局部 DMA 公式；无硬件实测校准。 |
| `src/rl_xy/distributed_optimizer.py` | MCore参数padding、DP owned intervals、RS/AG、唯一norm和状态字节代数。 |
| `src/rl_xy/evidence_io.py` | 浅层只读dataclass字段导出、三层外壳流式/叶值C编码，保持canonical JSON与gzip表格字节/SHA；显式格式索引及Linux阶段超时，不删原始事件或放宽容量。 |
| `src/rl_xy/execution_contract.py` | 多prompt/group语义、独立组内demean和compact QKV宽度；未知reward不伪造。 |
| `src/rl_xy/execution_validation.py` | 通用阶段/版本/训练/形状及实际Astra事件检查，独立于旧A/B曲线。 |
| `src/rl_xy/fixture.py` | 独立的 21 ns 数据移动链、out-of-place SRAM 分配与 63 B 容量负例。 |
| `src/rl_xy/gemm_census.py` | 运行时与CLI共用的唯一GEMM签名域/次数统计库，不依赖scripts命名空间。 |
| `src/rl_xy/gpu_model.py` | 独立GPU SM/CTA分区/FLOP/私有shared memory/共享HBM成本，有理峰值分摊；GPU与晶圆预算审计。 |
| `src/rl_xy/grpo_math.py` | 独立小型GRPO数学oracle：sample-std、PPO双clip、k3 KL、mask与DP归约；不执行模型训练。 |
| `src/rl_xy/hardware_profile.py` | H合同、几何/聚合检查、缺失参数与后端能力门禁。 |
| `src/rl_xy/inference_lifetimes.py` | 推理层级活跃区间，释放无backward中间量并保持输出到导出边界。 |
| `src/rl_xy/ledger.py` | 参数/优化器/梯度持久 storage 与 rollout KV 的身份、生命周期和准入容量账。 |
| `src/rl_xy/legacy_workload.py` | 旧原始日志/Hydra导入；统计约束合成；clip参考口径显式选择。 |
| `src/rl_xy/local_chain_contraction.py` | 本地串行链整数时长精确收缩；网络保留、逻辑ledger独立重放。 |
| `src/rl_xy/mapped_costs.py` | 在操作图不变的条件下应用typed GEMM和显式向量/状态RMW成本，不生成时间戳。 |
| `src/rl_xy/mapping.py` | 单逻辑核OS合法tile枚举与精确ragged/batch流量；不将非法tile静默塞入SRAM。 |
| `src/rl_xy/memoized_costs.py` | 完整shape/hardware键的只读成本与元数据复用，不改变数值。 |
| `src/rl_xy/migration.py` | W/H/模型输入联合预检和状态容量必要条件，不产生晶圆时间。 |
| `src/rl_xy/model.py` | MHA/GQA、QKV bias与绑权后的精确unique/local参数、持久状态和compact KV。 |
| `src/rl_xy/model_import.py` | 实际Qwen2配置/参数检查和受限算子回放能力；正式H门禁仍保持。 |
| `src/rl_xy/multicore.py` | N/M/batch输出分区、ragged输出/FLOP守恒、私有SRAM和共享DRAM/模块瓶颈，BF16/FP32独立rate。 |
| `src/rl_xy/operator_ir.py` | 有类型的张量shape/dtype/layout、Matmul方向与KernelSpec及互斥计时合同。 |
| `src/rl_xy/reporting.py` | 写 CSV/JSON/JSONL、round.phase 时间、指标、SUMMARY 与 hash manifest。 |
| `src/rl_xy/resource_encoding.py` | 严格rank顺序的读集合delta、完整workspace值和可选read-ID符号表；无损还原原storage ID，拒绝重复表、非法索引和delta链。 |
| `src/rl_xy/resource_validation.py` | 逐原始C++行核对Astra时钟、快照、READY、容量和峰值；完整整数向量+cores+当前私有容量缓存仅复用派生值，串行零态省逐核加减，重叠/浮点保留原算术；每node仍核workspace总和。 |
| `src/rl_xy/run.py` | 串接配置复制、trace、AstraSim 执行、解析、账本检查及成功/失败归档。 |
| `src/rl_xy/tensor_plan.py` | 完整声明对象、保守tape/反向partial保留、KV预留兑现和每节点可执行资源合同。 |
| `src/rl_xy/training_batching.py` | 真实聚合训练/logprob微批，保留causal attention、有效token分母、梯度累积与对象生命周期。 |
| `src/rl_xy/training_schedule.py` | 确定顺序minibatch/epoch图扩展、每更新命名/版本/分母、rank本地依赖重建。 |
| `src/rl_xy/training_validation.py` | 独立重建更新计划并复用逐更新完整训练检查；拒绝版本/清零/优势/分母篡改。 |
| `src/rl_xy/types.py` | 类型合同与TokenExtent长度对象；长度不假装token值。 |
| `src/rl_xy/uniform_generation.py` | 等长独立请求紧凑合批，分离请求轴KV并重新估计批量成本。 |
| `src/rl_xy/validation.py` | 从独立清单核对阶段、反向、清零、Adam 读写、版本、KV、通信与真实事件。 |
| `src/rl_xy/verl_trace.py` | veRL L1长度/分组/版本/阶段记录的严格读取与未知数值标记。 |
| `src/rl_xy/wafer_graph.py` | 完整group轮转分DP、全局token分母、真实TP/DP通信组与梯度SUM、round fence及唯一资源映射。 |
| `src/rl_xy/wafer_run.py` | 来源绑定的显式H场景、rank分文件、原生mesh与独立完整事件/内存/梯度检查、证据归档。 |
| `src/rl_xy/workflow.py` | 兼容旧toy；Qwen2 compact QKV/bias/RoPE/GQA、独立group和全局单次更新事件。 |

## 配置

| 文件 | 职责 |
|---|---|
| `configs/aligned_3b/W0-tp1-mb2/astra_system.json` | W0-tp1-mb2：原生collective后端配置源；aligned runner记录实际覆盖后的副本。 |
| `configs/aligned_3b/W0-tp1-mb2/case.json` | W0-tp1-mb2：固定模型与完整轨迹长度；不含实际采样token值或reward。 |
| `configs/aligned_3b/W0-tp1-mb2/model_config.json` | W0-tp1-mb2：固定Qwen官方结构源，模型参数可追溯；不是权重。 |
| `configs/aligned_3b/W0-tp1-mb2/policy.json` | W0-tp1-mb2：完整算法/TP/DP/microbatch/DO/SP/overlap参数，双平台必须同SHA与fingerprint。 |
| `configs/aligned_3b/W0-tp1-mb2/workload_provenance.json` | W0-tp1-mb2：固定长度合成控制的来源、B/N/P/R及无学习质量声明。 |
| `configs/aligned_3b/W0-tp2-mb2/astra_system.json` | W0-tp2-mb2：原生collective后端配置源；aligned runner记录实际覆盖后的副本。 |
| `configs/aligned_3b/W0-tp2-mb2/case.json` | W0-tp2-mb2：固定模型与完整轨迹长度；不含实际采样token值或reward。 |
| `configs/aligned_3b/W0-tp2-mb2/model_config.json` | W0-tp2-mb2：固定Qwen官方结构源，模型参数可追溯；不是权重。 |
| `configs/aligned_3b/W0-tp2-mb2/policy.json` | W0-tp2-mb2：完整算法/TP/DP/microbatch/DO/SP/overlap参数，双平台必须同SHA与fingerprint。 |
| `configs/aligned_3b/W0-tp2-mb2/workload_provenance.json` | W0-tp2-mb2：固定长度合成控制的来源、B/N/P/R及无学习质量声明。 |
| `configs/aligned_3b/W1-tp2-mb4/astra_system.json` | W1-tp2-mb4：原生collective后端配置源；aligned runner记录实际覆盖后的副本。 |
| `configs/aligned_3b/W1-tp2-mb4/case.json` | W1-tp2-mb4：固定模型与完整轨迹长度；不含实际采样token值或reward。 |
| `configs/aligned_3b/W1-tp2-mb4/model_config.json` | W1-tp2-mb4：固定Qwen官方结构源，模型参数可追溯；不是权重。 |
| `configs/aligned_3b/W1-tp2-mb4/policy.json` | W1-tp2-mb4：完整算法/TP/DP/microbatch/DO/SP/overlap参数，双平台必须同SHA与fingerprint。 |
| `configs/aligned_3b/W1-tp2-mb4/workload_provenance.json` | W1-tp2-mb4：固定长度合成控制的来源、B/N/P/R及无学习质量声明。 |
| `configs/aligned_3b/W1-tp2-mb8/astra_system.json` | W1-tp2-mb8：原生collective后端配置源；aligned runner记录实际覆盖后的副本。 |
| `configs/aligned_3b/W1-tp2-mb8/case.json` | W1-tp2-mb8：固定模型与完整轨迹长度；不含实际采样token值或reward。 |
| `configs/aligned_3b/W1-tp2-mb8/model_config.json` | W1-tp2-mb8：固定Qwen官方结构源，模型参数可追溯；不是权重。 |
| `configs/aligned_3b/W1-tp2-mb8/policy.json` | W1-tp2-mb8：完整算法/TP/DP/microbatch/DO/SP/overlap参数，双平台必须同SHA与fingerprint。 |
| `configs/aligned_3b/W1-tp2-mb8/workload_provenance.json` | W1-tp2-mb8：固定长度合成控制的来源、B/N/P/R及无学习质量声明。 |
| `configs/aligned_fixture/astra_network.yml` | 历史/小例网络输入；aligned runner实际网络另存inputs。 |
| `configs/aligned_fixture/astra_remote_memory.json` | 禁用外部内存扩展的小例配置。 |
| `configs/aligned_fixture/astra_system.json` | Astra基础系统配置；runner保存实际覆盖值。 |
| `configs/aligned_fixture/case.json` | 小模型完整流程，不能替代3B性能负载。 |
| `configs/aligned_fixture/gpu.json` | 四GPU小例成本/网络模型，不代表48卡主场景。 |
| `configs/aligned_fixture/policy.json` | 四rank、四trajectory的算法/并行控制；只做小例证明。 |
| `configs/aligned_fixture/wafer.json` | 小例2x2 mesh与显式假设。 |
| `configs/colocation/sources.json` | 上游veRL固定commit和官方模型配置revision/URL/SHA。 |
| `configs/colocation/tp2_to_tp1.json` | 0.5B/3B训练TP2到生成TP1的固定协议与假设。 |
| `configs/colocation/tp4_to_tp2.json` | 7B训练TP4到生成TP2的固定协议与假设。 |
| `configs/fixture_transfer/astra_network.yml` | 两 rank FullyConnected；精确表达放大 1000 倍后的带宽与链路时延。 |
| `configs/fixture_transfer/astra_remote_memory.json` | 固定 NO_MEMORY_EXPANSION；禁止隐式借用远程内存。 |
| `configs/fixture_transfer/astra_system.json` | FIFO/native ring、trace 开关、零额外 endpoint/reduction delay 和后端设置。 |
| `configs/fixture_transfer/case.json` | 该案例的工作负载、硬件、策略与合成成本合同；CLI 的主输入。 |
| `configs/gpu/a100_48_W0_hot.json` | 精确热GEMM加解析非GEMM的条件性GPU成本场景；缺签名拒绝。 |
| `configs/gpu/a100_48_W1_mb4_hot.json` | W1 TP2/mb4条件性GPU成本输入：精确hot-cuBLAS GEMM标签与解析非GEMM/网络；缺GEMM签名拒绝，不代表默认或最优veRL部署。 |
| `configs/gpu/a100_48_optimistic.json` | 刻意偏向GPU的GEMM TensorCore/零launch敏感性，不是物理FP32吞吐。 |
| `configs/gpu/a100_48_reference.json` | 48A100-SXM80GB真实名义规格、声明输入精度模型及未校准launch/延迟/共享带宽假设。 |
| `configs/gpu/wafer_typed_rate_sensitivity.json` | 晶圆FP32/vector降到64的未知率敏感性，不替换正式H0来源值。 |
| `configs/hardware/fixture_4die.json` | 独立4die/4核native回归来源，不是正式H0。 |
| `configs/hardware/xy_h0_case3_performance.json` | 54die Case3来源合同；Fig3几何仅模板参考，不缩计算die或宣称制造布局验证。 |
| `configs/hardware/xy_h0_wsc_case3.json` | 用户H0原表的独立来源合同：54die、FP16 1020FLOP/ns及原始几何；未知值不补写。 |
| `configs/hardware/xy_h1h2h3_54.json` | 正式H1–H3候选参数、单位、来源、空值与设计/实测状态；不是可执行硬件精度声明。 |
| `configs/models/qwen25_3b.json` | 官方Qwen2.5-3B结构原样配置，SHA与来源独立记录。 |
| `configs/models/qwen25_7b.json` | 官方Qwen2.5-7B结构原样配置，SHA与来源独立记录。 |
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
| `configs/verl_sync_cases/qwen05b/astra_network.yml` | 案例默认网络模板，运行时按平台显式生成native拓扑。 |
| `configs/verl_sync_cases/qwen05b/astra_remote_memory.json` | 关闭隐式远端内存扩展。 |
| `configs/verl_sync_cases/qwen05b/astra_system.json` | 原生通信和trace系统输入。 |
| `configs/verl_sync_cases/qwen05b/case.json` | 模型结构/逐轨迹长度/同步更新输入；真实和合成性质见相邻来源。 |
| `configs/verl_sync_cases/qwen05b/model_config.json` | 所选模型官方结构配置。 |
| `configs/verl_sync_cases/qwen05b/workload_provenance.json` | 长度来源、结构反事实和非数值训练边界。 |
| `configs/verl_sync_cases/qwen3b/astra_network.yml` | 案例默认网络模板，运行时按平台显式生成native拓扑。 |
| `configs/verl_sync_cases/qwen3b/astra_remote_memory.json` | 关闭隐式远端内存扩展。 |
| `configs/verl_sync_cases/qwen3b/astra_system.json` | 原生通信和trace系统输入。 |
| `configs/verl_sync_cases/qwen3b/case.json` | 模型结构/逐轨迹长度/同步更新输入；真实和合成性质见相邻来源。 |
| `configs/verl_sync_cases/qwen3b/model_config.json` | 所选模型官方结构配置。 |
| `configs/verl_sync_cases/qwen3b/workload_provenance.json` | 长度来源、结构反事实和非数值训练边界。 |
| `configs/verl_sync_cases/qwen7b/astra_network.yml` | 案例默认网络模板，运行时按平台显式生成native拓扑。 |
| `configs/verl_sync_cases/qwen7b/astra_remote_memory.json` | 关闭隐式远端内存扩展。 |
| `configs/verl_sync_cases/qwen7b/astra_system.json` | 原生通信和trace系统输入。 |
| `configs/verl_sync_cases/qwen7b/case.json` | 模型结构/逐轨迹长度/同步更新输入；真实和合成性质见相邻来源。 |
| `configs/verl_sync_cases/qwen7b/model_config.json` | 所选模型官方结构配置。 |
| `configs/verl_sync_cases/qwen7b/workload_provenance.json` | 长度来源、结构反事实和非数值训练边界。 |
| `configs/verl_sync_standard/qwen05b/astra_network.yml` | 案例默认网络模板，运行时按平台显式生成native拓扑。 |
| `configs/verl_sync_standard/qwen05b/astra_remote_memory.json` | 关闭隐式远端内存扩展。 |
| `configs/verl_sync_standard/qwen05b/astra_system.json` | 原生通信和trace系统输入。 |
| `configs/verl_sync_standard/qwen05b/case.json` | 模型结构/逐轨迹长度/同步更新输入；真实和合成性质见相邻来源。 |
| `configs/verl_sync_standard/qwen05b/model_config.json` | 所选模型官方结构配置。 |
| `configs/verl_sync_standard/qwen05b/workload_provenance.json` | 长度来源、结构反事实和非数值训练边界。 |
| `configs/verl_sync_standard/qwen3b/astra_network.yml` | 案例默认网络模板，运行时按平台显式生成native拓扑。 |
| `configs/verl_sync_standard/qwen3b/astra_remote_memory.json` | 关闭隐式远端内存扩展。 |
| `configs/verl_sync_standard/qwen3b/astra_system.json` | 原生通信和trace系统输入。 |
| `configs/verl_sync_standard/qwen3b/case.json` | 模型结构/逐轨迹长度/同步更新输入；真实和合成性质见相邻来源。 |
| `configs/verl_sync_standard/qwen3b/model_config.json` | 所选模型官方结构配置。 |
| `configs/verl_sync_standard/qwen3b/workload_provenance.json` | 长度来源、结构反事实和非数值训练边界。 |
| `configs/verl_sync_standard/qwen7b/astra_network.yml` | 案例默认网络模板，运行时按平台显式生成native拓扑。 |
| `configs/verl_sync_standard/qwen7b/astra_remote_memory.json` | 关闭隐式远端内存扩展。 |
| `configs/verl_sync_standard/qwen7b/astra_system.json` | 原生通信和trace系统输入。 |
| `configs/verl_sync_standard/qwen7b/case.json` | 模型结构/逐轨迹长度/同步更新输入；真实和合成性质见相邻来源。 |
| `configs/verl_sync_standard/qwen7b/model_config.json` | 所选模型官方结构配置。 |
| `configs/verl_sync_standard/qwen7b/workload_provenance.json` | 长度来源、结构反事实和非数值训练边界。 |
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

## 运行、分析与兼容工具

| 文件 | 职责 |
|---|---|
| `scripts/acceptance.py` | 依次执行容器测试、链路、单轮、两轮、重复运行和完整性门禁，保存命令日志。 |
| `scripts/baseline_preflight.py` | CPU生成3B候选策略卡与容量/对齐缺口，不选赢家、不启动完整仿真。 |
| `scripts/benchmark_gemm.py` | 受限单GPU cuBLAS热GEMM微基准、数值fixture与重复性；不代表全框架校准。 |
| `scripts/benchmark_gemm_capped.py` | 原cuBLAS脚本分配器fraction上限与完整来源记录。 |
| `scripts/bounded_run.py` | 专用无GPU/无网络Docker执行器，限制memory/swap/CPU/pids/timeout并保存OOM和退出证据。 |
| `scripts/bridge_pruning_acceptance.py` | 双平台完整native事件和资源峰A/B证明。 |
| `scripts/build_images.sh` | 按固定依赖链构建原生镜像，最终protocol标签0.9.5。 |
| `scripts/check_aligned_result.py` | 新gzip/index格式的独立完整性、时钟与指标验收。 |
| `scripts/check_docs.py` | 校验6份当前文档、2份冻结历史MD、相对链接、Bash语法和文件索引覆盖。 |
| `scripts/check_public_results.py` | 校验公开轻量证据的SHA/原manifest/MVP绑定与新旧汇总算术；不重放未分发日志。 |
| `scripts/check_result.py` | 复查成功状态、非空检查集、固定上游版本、Docker 证据和全部文件 SHA-256。 |
| `scripts/collect_aligned_shapes.py` | 逐调用/逐rank加权GEMM签名域，不用占位census时长发布性能。 |
| `scripts/collect_gemm_shapes.py` | 枚举精确BF16 GEMM签名，采集标签前固定问题域。 |
| `scripts/colocation_acceptance.py` | 唯一ID下原生小例/多轮/TP4/idle与三模型双平台运行，保存退出与日志SHA。 |
| `scripts/compare_aligned_results.py` | 以两份manifest绑定的独立PASS为前提，比较相同W/策略/算子/通信，失败时禁止发布比值。 |
| `scripts/compare_colocated_platforms.py` | 同模型/负载/协议/源码门禁与两侧时长、近预算和局限对照。 |
| `scripts/compare_gpu_wafer.py` | 严格匹配模型/trace/group/长度/算法更新后输出GPU观测与场景预测；失配拒绝比值。 |
| `scripts/compare_native_replay.py` | 冻结后新增stdlib只读分析：流式核对固定工作负载的rank-local原生issue/callback和资源原字节；记录原失败wrapper，不能把native等价改成整流程PASS，不发布跨平台比值。 |
| `scripts/compare_simulated_platforms.py` | 两边都仿真；严格全step/输入/算法/近预算合同，输出所有资源差异，拒绝用实测秒数预测GPU。 |
| `scripts/distributed_optimizer_fixtures.py` | 原生DP2/4、大payload与RS/AG网络/隐式mem口径独立fixture。 |
| `scripts/docker.sh` | 无sudo Docker构建/受控容器复用，精确UID/项目/镜像检查；可挂载已核对链接的自有外部证据目录。 |
| `scripts/execution_acceptance.py` | 0.4 Docker编排：多组/两轮/重复/带宽/实际模型/旧长度短探针和原生回归。 |
| `scripts/gpu_capture_wrapper.py` | 只在私有容器适配既有collector，另存未舍入阶段时间和完整解析配置，不激活主机flag。 |
| `scripts/gpu_network_fixtures.py` | 原生验证节点共享上行竞争、独立双向路径和同节点绕过spine，保持完整统计门禁。 |
| `scripts/gpu_sim_acceptance.py` | 先GPU4/旧mesh/native门禁，再独立完整GPU/偏向GPU/晶圆低rate运行与四组比较，不预设赢家。 |
| `scripts/gpu_smoke.py` | 专用GPU容器中受限veRL真训练/长度采集；模型/数据/包/源码hash及失败日志。 |
| `scripts/input_acceptance.py` | 宿主编排0.3 Docker验收、预期拒绝、旧数据复用及原生流程回归。 |
| `scripts/integration_fixtures.py` | Docker内真实执行资源竞争/READY门控/七类C++负例及4rank共享/独立mesh路由反事实。 |
| `scripts/make_execution_case.py` | 实际HF形状+指定完整长度group编译；记录来源、未知数值和显式反事实response cap。 |
| `scripts/prepare_aligned_inputs.py` | W0/W1/W2固定形状配置，明确未知token/reward与legacy schema override。 |
| `scripts/prepare_colocation_cases.py` | 从官方模型结构和显式长度生成三模型控制，记录真实/合成来源。 |
| `scripts/prepare_matched_case.py` | 真实GPU成功step的全部group导入，保持每条原长度，无cap改动及GPU计时注入。 |
| `scripts/private_sram_fixtures.py` | 原生同核/异核争用、单核溢出/错误总和/核数负例与raw/compact时钟等价。 |
| `scripts/probe_aligned.py` | 完整模型预算探针，输出RAM、建图墙钟、容量和节点数。 |
| `scripts/probe_training_capacity.py` | 一训练microbatch精确DRAM需求与全DO常驻状态下界，不伪装完整流程fit/性能。 |
| `scripts/read_id_encoding_acceptance.py` | tiny GPU/wafer 的字符串/eager 与interned/buffered native A/B；核对全部操作/事件、展开rank合同、资源日志原字节和峰值；不是3B或硬件性能结果。 |
| `scripts/remote_session.py` | 既有转发端口发现、主机key核验、保活复用和不确定写状态保护；不保存密码。 |
| `scripts/reproduce_v06.sh` | 不申请GPU，从公开完整长度case生成新control与四组对照。 |
| `scripts/resource_acceptance.py` | 宿主只编排Docker命令；单轮/两轮/repeat、带宽/SRAM敏感性、OOM、native fixture、完整性与版本冻结。 |
| `scripts/reuse_legacy.py` | Docker内核对旧日志/CSV/summary/state，生成3种子长度负载和来源hash。 |
| `scripts/schedule_acceptance.py` | Docker门禁：87测试、H0预期拒绝、五原生bundle、重复/两轮/旧结果及mesh资源回归。 |
| `scripts/validate_gemm_cache_regimes.py` | 有GPU资源上限的小选择hot/eviction/sequence缓存regime实测工具。 |
| `scripts/validate_gpu_costs.py` | CPU逐签名对比解析GPU服务与独立热GEMM标签，报告覆盖、失配与cache边界。 |
| `scripts/wafer_acceptance.py` | Docker单测、逐核/旧native、4/54die、原toy、64条真实GPU全step与对比/完整性门禁。 |
| `scripts/write_colocation_report.py` | 从PASS验收生成0.7轻量实验摘要、JSON及HTML，明确未校准边界。 |
| `scripts/write_execution_report.py` | 历史版本验收报告兼容生成器；只用于对应原版本，不维护当前核心文档。 |
| `scripts/write_gpu_sim_report.py` | 历史版本验收报告兼容生成器；只用于对应原版本，不维护当前核心文档。 |
| `scripts/write_input_report.py` | 历史版本验收报告兼容生成器；只用于对应原版本，不维护当前核心文档。 |
| `scripts/write_resource_report.py` | 历史版本验收报告兼容生成器；只用于对应原版本，不维护当前核心文档。 |
| `scripts/write_schedule_report.py` | 历史版本验收报告兼容生成器；只用于对应原版本，不维护当前核心文档。 |
| `scripts/write_wafer_report.py` | 历史版本验收报告兼容生成器；只用于对应原版本，不维护当前核心文档。 |
| `scripts/write_walkthrough.py` | 历史版本验收报告兼容生成器；只用于对应原版本，不维护当前核心文档。 |

## 测试

| 文件 | 职责 |
|---|---|
| `tests/fixtures/read_id_header_test.cpp` | 独立C++ header harness：恢复read-ID并拒绝bool/float/越界/重复符号；须显式提供预编译harness，测试不自行联网构建。 |
| `tests/test_aligned_graph.py` | 活动rank/DO/版本/真实microbatch与桥接读权限等价/变异检查。 |
| `tests/test_aligned_result_check.py` | 新格式checksum/事件/原生退出的独立正负例。 |
| `tests/test_aligned_run.py` | 新runner时钟、发布、CLI和组件映射负例。 |
| `tests/test_aligned_shapes.py` | 实际GEMM调用加权、完整签名域、编译占位成本禁作性能。 |
| `tests/test_astra_config.py` | 带宽缩放/两套配置正例，带宽漂移和 bool 冒充整数负例。 |
| `tests/test_baseline_contract.py` | prompt/trajectory单位、整除、软件fingerprint、DO未实现、成本覆盖与诚实门禁负例。 |
| `tests/test_bounded_run_paths.py` | 专属artifact-root、record包含关系、symlink/路径逃逸和immutable-source挂载门禁。 |
| `tests/test_check_result.py` | 完整 bundle 校验正例，以及篡改 metrics 后拒绝的负例。 |
| `tests/test_colocation.py` | 全局依赖、多轮无泄漏、连续TP4、打包生成隔离与合法映射检查。 |
| `tests/test_compare_aligned_results.py` | 配对输入/来源/通信/成本域或验收绑定失配的拒绝检查。 |
| `tests/test_compare_native_replay.py` | 后加分析的日志轮转、完整覆盖、时钟/资源/输入失配拒绝与失败wrapper保留检查；不改变冻结runtime测试身份。 |
| `tests/test_comparison.py` | 纯合成单测验证真实比较的门禁：cap、模型/日志hash、更新次数失配不能输出ratio。 |
| `tests/test_component_proof_cache.py` | 模板不可变及全部实际宏时钟、coverage/deps失败拒绝。 |
| `tests/test_config_model.py` | 参数/KV 手算、未知字段、证据等级、策略字段和 rounds 覆盖测试。 |
| `tests/test_distributed_optimizer.py` | 分片/填充/范数owner/全局缩放与RS/AG字节代数。 |
| `tests/test_evidence_io.py` | 流式gzip/minified语义与原表字节一致、原子失败、阶段超时恢复。 |
| `tests/test_fixture.py` | T01 结构/容量手算与 63 B out-of-place 不可行负例。 |
| `tests/test_gemm_cache_regimes.py` | 缓存probe选择、GPU resident内存预算和重复协议门禁。 |
| `tests/test_generator_contract.py` | P2P 参与者错误、tag 越界、不支持 collective 的 fail-closed 测试。 |
| `tests/test_gpu_model.py` | GPU CTA ragged/FLOP/私有容量/聚合峰值/带宽/精度敏感性与近预算差异测试。 |
| `tests/test_grpo_math.py` | 手算组标准差、双clip/KL、mask和DP均值不变性，不验证LLM学习效果。 |
| `tests/test_inference_lifetimes.py` | 早/晚推理读权限与生存期等价，拒绝训练tape。 |
| `tests/test_ledger.py` | 长期状态、KV reservation/materialization、NIC 计划容量及两轮无泄漏检查。 |
| `tests/test_legacy_workload.py` | 日志/配置一致性、合成矩约束、确定性及batch-max截断口径反例。 |
| `tests/test_local_chain_contraction.py` | 严格串行收缩/展开、资源ledger及native A/B可选检查。 |
| `tests/test_mapping_cost_model.py` | OS手算、ragged独立循环、SRAM合法性/自动重映射、带宽/dtype/profile失配等测试。 |
| `tests/test_memoized_costs.py` | 完整key、成本完全一致和只读对象复用。 |
| `tests/test_migration_inputs.py` | H几何/单位/来源/null、trace语义、真实模型参数检查及负例。 |
| `tests/test_parser.py` | 事件配对、缺 callback、乱序日志、rank-local 节点和提前 RECV 等待测试。 |
| `tests/test_qwen_execution.py` | Qwen参数/KV金标准、多组/未知reward/绑权/生命周期、RoPE/GQA梯度恒等式和合同变异负例。 |
| `tests/test_reporting.py` | 两轮 phase 不混桶，以及空参数表仍有固定 CSV schema。 |
| `tests/test_resource_read_symbols.py` | Python read-ID往返、旧格式兼容、delta/索引负例；可选C++ harness交叉核验。 |
| `tests/test_resource_replay_cache.py` | 嵌入冻结a8旧实现，逐checks/全部MemorySnapshot/peak及数值类型对照；覆盖重复serial、overlap、非法事件/向量/容量、source mutation与cache eviction。 |
| `tests/test_runtime_efficiency.py` | 增量账本、编码往返、完整资源回放及坏输入拒绝的等价检查。 |
| `tests/test_simulated_comparison.py` | 纯合成比较门禁：可翻转赢家、禁止实测替代、不同capture和预算不足拒绝比值。 |
| `tests/test_tensor_resources.py` | tape/梯度shape、延迟READY、容量/生命周期/预留失败、只读Reference及两轮无增长。 |
| `tests/test_training_batching.py` | 聚合形状/分母/隔离注意力、梯度与生命周期及未支持配置拒绝回归。 |
| `tests/test_training_capacity_probe.py` | 容量测量ceiling与真实目标分离、完整batch/globalT保持及峰值比较。 |
| `tests/test_training_schedule.py` | H0来源/几何/单位负例；尾批、拆组优势、两轮版本、梯度存储和多种篡改负例。 |
| `tests/test_uniform_generation.py` | 独立请求批轴、字节/FLOP/host输入与旧模板控制。 |
| `tests/test_validation.py` | 完整证据正例；删依赖、漏反向/Adam 状态、容量越界/漏事件等负例。 |
| `tests/test_wafer_graph.py` | 私有映射/守恒、共享带宽、DP不均匀长度分母/组、几何scope、无损压缩与拒绝负例。 |
| `tests/test_workflow.py` | 本地依赖链、完整前后向、embedding TP 求和、梯度清零及 v2 发布。 |

## Docker与原生扩展

| 文件 | 职责 |
|---|---|
| `docker/Dockerfile` | 基于指定本地镜像构建 Abseil、Protobuf、固定 AstraSim 和隔离 Python 环境。 |
| `docker/Dockerfile.alignment` | 保留0.7原生后端，添加0.8对齐门禁与资源安全工程层。 |
| `docker/Dockerfile.colocation` | 基于0.6固定原生后端添加0.7Python协议层，不修改native网络实现。 |
| `docker/Dockerfile.execution` | 复用0.3原生后端，打包0.4 Python执行层，不安装GPU训练依赖。 |
| `docker/Dockerfile.gpu` | 在0.5上编译独立GPU网络扩展，保存patch/header与二进制身份。 |
| `docker/Dockerfile.inputs` | 复用已验证0.2原生镜像，打包0.3 Python输入接口，无新GPU依赖。 |
| `docker/Dockerfile.mapped` | 在固定0.1镜像上接入资源、CPU统计修正和mesh扩展，编译两个analytical后端并冻结patch hash。 |
| `docker/Dockerfile.protocol` | 0.8后端上重编read-ID资源hook并复制PP1/DO协议实现，标签0.9.5。 |
| `docker/Dockerfile.schedule` | 复用0.4原生Astra镜像、打包0.4.1 Python更新层；不使用GPU。 |
| `docker/Dockerfile.wafer` | 固定0.4.1基础上编译逐核/压缩资源hook，记录新增header hash并打包0.5 Python。 |
| `docker/astrasim/RlxyGpu.hh` | GPU端点、每节点switch、共享uplink及spine，实际native Link竞争。 |
| `docker/astrasim/RlxyMesh.hh` | 新增无wrap二维mesh、XY路由；使用原生Device/Link/Chunk排队并输出实际路径。 |
| `docker/astrasim/RlxyResources.hh` | 实际Astra时钟的ALLOC/READY/FREE、容量、KV预留、SRAM和排他local-memory准入；严格解码read-ID符号表与unsigned索引；64KiB日志缓冲只改变主机I/O，正常结束日志逐字节对照，错误仍非零退出。 |
| `docker/astrasim/gpu.patch` | 固定aware入口增加GPU topology选择，不改默认mesh分支。 |
| `docker/astrasim/mesh.patch` | 在congestion-aware frontend显式选择mesh2d override；无override保留上游行为。 |
| `docker/astrasim/statistics.patch` | 允许CPU/控制节点参与回放；以GPU/COMM区间并集计算overlap，避免CPU空隙误计或无符号下溢。 |
| `docker/astrasim/workload.patch` | 向上游Workload的can_issue、issue、callback插入资源hook，不另建事件时钟。 |

## 校准标签

| 文件 | 职责 |
|---|---|
| `calibration/20261028/W0-a100-hot.json` | 285个W0精确签名的自采hot-cuBLAS标签；不覆盖融合/全步/跨域。 |
| `calibration/20261028/W1-mb4-a100-hot.json` | W1原生形状域291个canonical NN热GEMM标签及来源/测量；不覆盖transpose/stride/fusion/FlashAttention、非GEMM、网络或整步校准。 |

## 文档

| 文件 | 职责 |
|---|---|
| `docs/ARCHITECTURE.md` | 分层模块、执行/资源/通信合同、H0来源和证据边界。 |
| `docs/EXPERIMENT_DESIGN.md` | 模型/算法/数据/切分/硬件/统计/预算参数及逐项原因、原论文和源码依据。 |
| `docs/FILE_MAP.md` | 当前公开工程逐文件职责，单一清单，无旧版追加重复表。 |
| `docs/REPRODUCING.md` | Docker依赖、快速检查、小例/完整W1受限命令、原包与公开摘要的检查区别。 |
| `docs/RESULTS.md` | 已验收3B时间、阶段、完整证据身份、失败与论文使用范围。 |

## 公开实验与冻结历史

| 文件 | 职责 |
|---|---|
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
| `experiments/2026-10-27-v0.6/实验结果.md` | 冻结0.6历史实验文档，字节不变，按原commit上下文读取，不与当前结果混用。 |
| `experiments/2026-10-27-v0.6/模型内部瓶颈分解.md` | 冻结0.6历史实验文档，字节不变，按原commit上下文读取，不与当前结果混用。 |
| `experiments/2026-10-28-v0.9.5/dataset_audit.json` | 固定GSM8K记录、分词/拆分审计回执；没有生成回答或质量结果。 |
| `experiments/2026-10-28-v0.9.5/design_contract.json` | 数值不变的便携设计合同；导航字段变更显式记录，初始状态不是最新运行状态。 |
| `experiments/2026-10-28-v0.9.5/design_sources.json` | 公开引用和本地材料SHA索引；不分发论文PDF或本地原文。 |
| `experiments/2026-10-28-v0.9.5/execution_history.json` | 紧凑派生执行记录，保留42位置/29执行与失败状态；原完整inventory离线保存。 |
| `experiments/2026-10-28-v0.9.5/gemm_error_audit.json` | 目标热标签与旧解析kernel服务的阶段/频次误差；不是整步误差。 |
| `experiments/2026-10-28-v0.9.5/gemm_measurement_verification.json` | 291个新W1 canonical NN热标签的采集/数值fixture/预算回执；状态按采集时点保留。 |
| `experiments/2026-10-28-v0.9.5/gpu/actual_gemm_census.json` | 原提交图精确逻辑GEMM域、调用次数和FLOPs。 |
| `experiments/2026-10-28-v0.9.5/gpu/global_stage_times.json` | 原全局阶段时钟包络，含等待，不能机械相加。 |
| `experiments/2026-10-28-v0.9.5/gpu/manifest.json` | 原全量run manifest；其artifact全集并非全部随Git分发。 |
| `experiments/2026-10-28-v0.9.5/gpu/metrics.json` | 原样指标文件，与原manifest SHA绑定，不回填当前代码身份。 |
| `experiments/2026-10-28-v0.9.5/gpu/run.json` | 原wrapper命令、预算、真实退出/OOM与墙钟。 |
| `experiments/2026-10-28-v0.9.5/gpu/verification.json` | 原单包独立MVP回执，绑定其manifest；不是在公开子集上重新运行MVP。 |
| `experiments/2026-10-28-v0.9.5/gpu/wallclock_timings.json` | 原仿真程序构图/编码/native/后验子计时，与模型时间分开。 |
| `experiments/2026-10-28-v0.9.5/inputs/case.json` | 原样固定W1输入；与两平台原manifest的输入SHA绑定。 |
| `experiments/2026-10-28-v0.9.5/inputs/model_config.json` | 原样固定W1输入；与两平台原manifest的输入SHA绑定。 |
| `experiments/2026-10-28-v0.9.5/inputs/policy.json` | 原样固定W1输入；与两平台原manifest的输入SHA绑定。 |
| `experiments/2026-10-28-v0.9.5/inputs/workload_provenance.json` | 原样固定W1输入；与两平台原manifest的输入SHA绑定。 |
| `experiments/2026-10-28-v0.9.5/native_equivalence.json` | 实际a8→a9全rank原生资源/事件时钟/ET字节等价回执。 |
| `experiments/2026-10-28-v0.9.5/native_equivalence_run.json` | 等价分析实际资源上限、退出与墙钟。 |
| `experiments/2026-10-28-v0.9.5/pair_analysis.json` | 实际配对分析墙钟、RSS、工具SHA与退出回执。 |
| `experiments/2026-10-28-v0.9.5/pair_verification.json` | 实际严格配对结果、原manifest/MVP/source/image绑定与idle特例证明。 |
| `experiments/2026-10-28-v0.9.5/publication_manifest.json` | 当前实际分发的轻量文件精确inventory、SHA/字节及不含大raw日志的范围。 |
| `experiments/2026-10-28-v0.9.5/results.json` | 从已放行配对生成的参数、时间、比值与阶段事实。 |
| `experiments/2026-10-28-v0.9.5/source_freeze.json` | 原0.9.5 a9的325文件冻结清单、测试和镜像身份，不是当前Git源树。 |
| `experiments/2026-10-28-v0.9.5/wafer/actual_gemm_census.json` | 原提交图精确逻辑GEMM域、调用次数和FLOPs。 |
| `experiments/2026-10-28-v0.9.5/wafer/global_stage_times.json` | 原全局阶段时钟包络，含等待，不能机械相加。 |
| `experiments/2026-10-28-v0.9.5/wafer/manifest.json` | 原全量run manifest；其artifact全集并非全部随Git分发。 |
| `experiments/2026-10-28-v0.9.5/wafer/metrics.json` | 原样指标文件，与原manifest SHA绑定，不回填当前代码身份。 |
| `experiments/2026-10-28-v0.9.5/wafer/run.json` | 原wrapper命令、预算、真实退出/OOM与墙钟。 |
| `experiments/2026-10-28-v0.9.5/wafer/verification.json` | 原单包独立MVP回执，绑定其manifest；不是在公开子集上重新运行MVP。 |
| `experiments/2026-10-28-v0.9.5/wafer/wallclock_timings.json` | 原仿真程序构图/编码/native/后验子计时，与模型时间分开。 |
