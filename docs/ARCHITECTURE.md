# 架构与端到端数据流

0.2扩展：在原操作DAG与Chakra之间插入typed mapping/cost、通信端点lowering和tensor/resource合同；Workload hook在原生issue/callback时执行真实准入，Python独立重放其日志。详见 [mapped_costs](components/mapped_costs.md)、[tensor_resources](components/tensor_resources.md)、[network](components/network.md)。下文保留0.1接口背景；旧ledger只跟踪persistent/KV，不替代新的全声明对象计划。

## 1. 目标与原则

这个 MVP 的目标不是堆叠功能，而是建立一条可审计的最短闭环：同一份配置经过严格解析后，生成工作负载操作、资源生命周期和每 rank Chakra trace；AstraSim 执行 trace；解析器把原始输出还原为事件和指标；验证器再用相互独立的合同检查判断这次运行能否被接受。

核心原则如下：

1. **W/H/S 分离**：工作负载、硬件假设和映射/调度策略各有独立配置语义，避免把算法逻辑写死在硬件模型中。
2. **AstraSim 是时序真源**：规划器提供工作量与依赖，不能以 Python 估算完成时间冒充 AstraSim 输出。
3. **对象先于内存数字**：当前为参数、梯度、优化器状态和 rollout KV 记录身份、版本、位置及生命周期；内存指标只覆盖这些已建模对象，不把未实现的激活 tape 或 SRAM 工作区当成零占用。
4. **失败关闭**：二进制退出异常、rank 不完整、event 缺失、容量越界、版本错误或 hash 不一致均使 run 失败。
5. **证据不越级**：synthetic fixture 只证明框架闭环，不自动升级为硬件校准或真实训练证据。

## 2. 分层

### 2.1 合同层

`config.py` 读取 `case.json`，要求字段集合完全匹配 schema，并完成跨字段约束。例如，当前 MVP 要求一 die 对应一个 TP rank、`PP=1`、`DP=1`、两个 trajectory、每个训练 microbatch 一条 trajectory，以及严格同步调度。

输出是冻结 dataclass，而不是在后续阶段继续修改的自由字典。解析后的配置另存为 `resolved_config.json`，作为本次运行的实际输入。

### 2.2 模型与成本层

`model.py` 枚举 toy Transformer 参数，明确 unique element、每 die 本地 element、复制参数和 TP 分片布局；同时给出 actor、reference、master、gradient 与 Adam 状态的字节账。

`costs.py` 把 FLOP、vector op 和局部 DMA 字节映射为合成时长。局部 DMA 公式目前用于独立 transfer fixture，主 toy 不展开逐 tile 的本地搬运。这里的速率来自 fixture 配置，只用于构造可追踪工作量，不是测得的芯片性能。

### 2.3 工作流层

`workflow.py` 把一轮同步 GRPO 展开为细粒度 `Operation`：

- `compute`：含持续时间、FLOP、vector op、tensor bytes 与语义元数据；
- `collective`：含 collective 类型与 payload；
- `p2p`：含源/目的 rank 与 payload。

每个 operation 有稳定的 `node_id`、`op_id`、phase、`participant_ranks` 和 `dependencies_by_rank`。首个调度策略在**每个参与 rank 内**把节点连接为确定性的本地严格依赖链；collective 与 p2p 通信节点负责 rank 间显式同步。未参与某个操作的 rank 不产生节点，也不承担虚构时长。数据结构保留 rank-local 依赖边，后续可以换成更宽松的 DAG 调度器，而无需改 W/H 合同或结果格式。

### 2.4 对象与资源层

账本为持久状态和 rollout KV allocation increment 记录：存储身份、逻辑 tensor 身份、dtype、字节数、版本、位置、创建/就绪/最后消费/释放时点、是否持久以及所有 rank。快照核对已跟踪的 DRAM、KV reservation 与计划 p2p NIC 容量。主 toy 未展开所有中间张量、激活 tape、collective 临时缓冲、SRAM tile 工作区或资源锁；报告中的 DRAM 峰值是已跟踪小计，不是全轮或全设备峰值。独立 transfer fixture 单独验证 SRAM 生命周期与容量负例。

这个层回答“什么数据在何时存在于哪里”，不负责决定 AstraSim 网络完成时间。

### 2.5 AstraSim 适配层

适配器把统一 operation 序列降为两个 rank 的 Chakra ET：

- compute 映射为计算节点；
- collective 映射为 AstraSim 支持的 collective；
- p2p 在源 rank 生成 send、目的 rank 生成 receive，并共享匹配 tag；
- 每个 trace 只包含该 rank 实际参与的节点，`ctrl_deps` 来自该 rank 的本地前驱；host/reward/advantage 等 rank0 synthetic replay 不在 rank1 镜像；
- collective 覆盖两个 TP rank，p2p 覆盖其源/目的 rank，二者都是本地依赖链之间的显式同步点。

适配器随后使用固定的 system、network、remote-memory 配置调用 official AstraSim analytical binary，并保存命令、stdout、stderr 和退出状态。解析器只消费本次进程真实产生的输出；没有静默 fallback。当前 binary 是 `congestion_unaware`：它不会模拟多流共享链路时的争用或排队，通信结果只适用于无拥塞 analytical 假设。

### 2.6 验证与报告层

验证器从四类彼此可交叉核对的证据做门禁：

- 配置与公式推导的不变量；
- planner 产生的 operation 和对象生命周期；
- AstraSim issue/callback、rank completion 与统计输出；
- 报告文件本身的完整性和 hash。

reporter 在验证结束后生成不可混淆的 run bundle。`summary.json` 是索引，不是原始证据的替代品。

## 3. 2-die 同步 GRPO 时序

```text
actor v0
  │
  ├─ 输入 prompt 到 rank0，再复制到 rank1
  ├─ trajectory A/B rollout：prefill + decode + TP collective + token p2p
  ├─ response 回 host，规则 reward
  ├─ reference 只读 teacher-forcing log-prob
  ├─ actor v0 只读 old-policy log-prob
  ├─ 组内 advantage：A=+0.5，B=-0.5
  ├─ 清零本轮梯度缓冲；train_A forward/loss/loss_backward/model_backward，梯度累积
  ├─ train_B forward/loss/backward，梯度累积
  ├─ 全局梯度范数、clip
  ├─ shape-only AdamW：actor v0 → v1
  ├─ 全分片更新 barrier
  └─ 发布 v1，释放本轮临时对象；reference 仍为 v0
```

由于 `DP=1`，不会存在数据并行梯度 all-reduce。TP collectives、分布式 log-softmax reductions、全局 norm 标量 reduction 和 publish barrier 必须按其各自语义区分，不能都笼统记为“同步通信”。

publish barrier 后的 `publish_actor_version` 和 cleanup 的 `release_round_temporaries` 各自带 `control_marker_ns=1`。这两个 1 ns 是为了让控制事件在 trace 中可见而显式加入的 synthetic control cost，不是测量出的版本切换或内存释放硬件时延。

## 4. 时间编码

Chakra compute duration 的字段粒度是微秒，而 toy fixture 的合同以整数纳秒表达。为保留小整数例子的可手算性，配置固定：

```text
astra_time_scale = 1000
```

编码规则是：

- 合同中的 `1 ns` 以 Chakra `1 µs` 写入，即 AstraSim 时间轴放大 1000 倍；
- D2D 网络按相同时间倍率缩放。analytical backend 会把配置中的 GiB/s 换算为 B/ns，因此合同的 `8 B/ns` 写成 `0.007450580596923828`，换算后正好是 `0.008 B/ns`；transfer fixture 的本地 DMA 显式降低为 COMP，不由 Astra 的 `local-mem-bw` 重复计费；主 toy 尚未逐 tile 展开本地 DMA；
- 链路时延乘以 1000，例如合同的 `1 ns` 写成 `1000 ns`；
- 解析后的 AstraSim tick/cycle 再除以 1000，恢复 fixture 的合同 ns。

这是一种确定性整数编码技巧，不是单位校准，也不允许据此推断真实硬件频率。

AstraSim stdout 中名为 `GPU time` 的原始计数在报告中统一改称 **compute-lane occupancy**。它包含 trace 中建模为 COMP 的 synthetic host/local-control replay，因此既不是实体 GPU 利用率，也不是 accelerator kernel profiler 时间。类似地，`COMM_RECV` 从 issue 到 callback 可能含接收方提前 posted 后等待发送方的时间；T01 transfer fixture 的链路服务时延必须用 rank0 `COMM_SEND` 的 issue→callback 计算，不能用 rank1 `COMM_RECV` 的完整驻留时间。

## 5. 关键接口

| 边界 | 输入 | 输出 | 主要不变量 |
|---|---|---|---|
| 配置 → 规划 | `CaseConfig` | `Operation[]` | 字段完整、跨字段合法、phase 齐全 |
| 配置 → 模型 | model/strategy | `ParameterRow[]` | unique/local/replicated 一致 |
| 规划 → 账本 | operations + case | object/snapshot/transfer | 生命周期闭合、容量不越界 |
| 规划 → adapter | operations + Astra config | 每 rank `.et` | ID、依赖、collective、p2p 匹配 |
| AstraSim → parser | stdout/stderr/trace log | events + rank statistics | 每个参与 rank/node 的 issue/callback 成对 |
| 全部 → validation | config/ops/ledger/events | checks | 任一必需检查失败则 run 失败 |
| 全部 → reporting | 证据记录 | run bundle | 命令、退出码、hash、原始日志齐全 |

## 6. 明确不做的事

- 不模拟真实神经网络数值、采样随机性或学习收敛。
- 不声称 analytical backend 等价于具体晶圆产品。
- 不把 rank0 的 synthetic host/control replay 当成实体 accelerator kernel，也不在未参与 rank 上镜像这些节点。
- 不模拟链路争用；`congestion_unaware` 结果不能用于拥塞敏感结论。
- 不在第一版引入流水并行、数据并行、MoE、异步 rollout、KV 共享或抢占。
- 不从参考仓库继承路径、容器、结果文件或凭据。

## 7. 扩展方向

后续扩展应保持现有合同与证据格式：先增加新的策略实现和相应验证，再引入 PP/DP、多 prompt、动态 batching、KV 共享、异步 actor 或真实硬件校准。任何 calibrated 配置都需要单独的数据来源、拟合过程、误差和适用范围，不能直接修改 synthetic fixture 的标签。

