# 对象与资源账本组件

## 职责

账本组件把“模型和流程需要多少资源”转成可审计的对象记录与容量快照。它明确区分物理占用、有效数据和已预留但尚未物化的未来增长空间，避免把 reservation 与 materialized bytes 重复计费。

实现入口：`src/rl_xy/ledger.py`；记录类型：`ObjectRecord`、`MemorySnapshot`。

## 输入

- `CaseConfig`：模型形状、dtype bytes、die/DRAM/SRAM/NIC 容量、KV admission、round 和 trajectory；
- 可选 `Operation[]`：用于把创建、就绪、最后消费者和释放时点绑定到稳定 `op_id`；
- `model.py` 的参数枚举与 `kv_bytes_per_processed_token_per_die` 公式。

## 输出

`LedgerBundle` 包含：

- `objects`：每个 persistent state 和实际 KV allocation increment 的对象行；
- `memory_snapshots`：已建模 storage 的物理占用快照，不代表全设备实际峰值；
- `kv_snapshots`：显式标为 logical view 的 KV 曲线，不构成第二份分配；
- `kv_bytes_written_per_die`：按 round/die 的 KV 累计写入量；
- `all_snapshots`：用于统一导出的物理快照加逻辑视图。

对象记录字段包括 storage/tensor identity、kind、location、bytes、dtype、version、created/ready/last-consumer/released、persistent 和 owner rank。

## 持久状态

每个 die、每个参数分片分别记录：

- actor BF16；
- reference BF16；
- FP32 master weight；
- FP32 Adam `m`；
- FP32 Adam `v`；
- FP32 gradient buffer。

复制参数按 replicated 规则完整计入每个 die；分片参数按 TP local elements 计入。reference 版本在本例中保持初始版本且只读；actor/master/optimizer 的版本推进由 workflow/validation 交叉核对。持久对象跨轮存在，不因单轮 cleanup 被释放。

## KV 生命周期

当前实现只接受冻结的 A/B toy case。最后生成的 token 不再反馈给模型，因此每轮实际处理位置为：

- A：prompt 2 个位置，加两次 decode 回灌；
- B：prompt 2 个位置，加一次 decode 回灌。

每个 die 每处理一个 token 的 KV 字节由层数、本地 KV heads、head dim 和 actor dtype 推导。`reserve_public_max_length` 在 admission 时保留未来增长预算；KV 逐步物化时，相同字节从 `unmaterialized` 移到 `allocated`，不会同时占两份。

对当前配置，以下数值是**公式推导的账本不变量**，不是 AstraSim 或硬件实测：

- 每 token、每 die KV：32 B；
- 总 reservation：256 B/die；
- materialized KV 曲线：0 → 128 → 192 → 96 → 128 → 0 B/die；
- 每轮累计 KV 写入：224 B/die。

持久状态与 KV 的和构成当前已跟踪的 DRAM 小计曲线；训练激活 tape、中间张量和 collective 缓冲未纳入，不能把小计当作整轮完整峰值。`*.dram.kv_view` 仅为解释视图，不能再次相加。

## SRAM 与 NIC

主 toy case 的 NIC 快照按计划 p2p packet 统计占用峰值，不包含 AstraSim 原生 collective 的内部缓冲。SRAM 只有资源存在占位行，reporter 不把它当作有效的零峰值，指标明确为未建模。网络时序由 AstraSim 建模，主 toy 不展开本地 DMA。可手算 transfer fixture 则显式记录 out-of-place compute 的 SRAM 输入/输出同时驻留，并用 63 B 容量负例验证 64 B 峰值必须被拒绝。

## 不做什么

- 不替代 AstraSim 的网络/执行时序；
- 不以进程 RSS 代表模拟 DRAM/SRAM；
- 不把逻辑 KV 视图和物理占用相加；
- 不隐式纳入 Python/C++ runtime、allocator metadata 或未建模 tensor；
- 不自动泛化到任意 trajectory 组合、KV 共享或并发调度；
- 不把账本峰值表述为实体硬件测量。

## 失败条件

- trajectory ID/长度不符合冻结的 A=3、B=2 toy 合同；
- 推导对象大小为非法值；
- `allocated_bytes`、`valid_bytes` 或 reservation 关系不成立；
- 任一物理位置占用超过独立 capacity；
- 临时对象在 cleanup 后仍存活，或多轮累计产生泄漏；
- reference 被写、版本不连续或对象生命周期引用不存在的 operation。

前两类由 ledger 构建时拒绝；其余由 validation 以结构化 check 复核。

