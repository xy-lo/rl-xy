# AstraSim 适配器

## 职责

适配器是规范 `Operation` 与 official AstraSim 之间的唯一边界。它负责生成每 rank Chakra ET、校验并调用固定版本 analytical binary、保存原始进程证据，再严格解析 rank completion、统计和节点 issue/callback。

实现目录：`src/rl_xy/astra_adapter/`。

## 输入

- 已验证且按依赖顺序排列的 `Operation[]`；
- rank 数和 `astra_time_scale=1000`；
- trace prefix；
- `astra_system.json`、`astra_network.yml`、`astra_remote_memory.json`；
- `ASTRA_SIM_ROOT` 和 `ASTRA_SIM_BIN` 指向的容器内固定版本；
- 输出目录与超时。

## Trace 生成

每个 rank 生成 `<prefix>.<rank>.et`，metadata version 为 `0.0.4`。映射规则：

- compute → `COMP_NODE`，`duration_micros=duration_ns`，并记录 `num_ops` 和 `tensor_size`；
- collective → `COMM_COLL_NODE`，映射 `ALL_REDUCE`、`ALL_GATHER`、`REDUCE_SCATTER`、`ALL_TO_ALL`；
- p2p → 源 rank 的 `COMM_SEND_NODE` 和目的 rank 的 `COMM_RECV_NODE`，tag 使用稳定 `node_id`；
- `ctrl_deps` 来自该 rank 的 `dependencies_by_rank`；每个 rank 只写入自己位于 `participant_ranks` 中的 operation。

当前 topology 严格为两 rank。每个参与 rank 维持自己的本地严格依赖链；collective 显式覆盖两个 TP rank，p2p 显式覆盖源/目的 rank 并完成两条本地链的同步。适配器不会为未参与的 rank 插入改变时间语义的占位节点。

## 时间缩放

Chakra compute duration 使用微秒字段，而 fixture 合同需要整数纳秒。适配器固定放大 1000 倍：合同 `N ns` 写为 `N µs`；D2D network 带宽按同一倍率换算、链路时延乘以 1000；parser 再将 AstraSim tick 除以 1000 得到合同 ns。本地 DMA 已作为显式 COMP fixture 计时，避免由 `local-mem-bw` 重复收费。

例如目标 `8 B/ns` 在 1000 倍时间轴上需要 `0.008 B/ns`；考虑 analytical backend 的 GiB/s→B/ns 换算，YAML 写为 `0.007450580596923828`。`1 ns` 链路时延写为 `1000 ns`。这是合成 fixture 的整数编码，不是硬件频率或性能校准。

## 运行输出

runner 使用参数数组直接启动二进制，不经过 shell。四个核心参数为：

```text
--workload-configuration=<trace-prefix>
--system-configuration=<system.json>
--network-configuration=<network.yml>
--remote-memory-configuration=<remote-memory.json>
```

同时设置独立 logging folder。runner 返回并保留：实际 argv、UTC 起止时间、退出码、stdout、stderr、debug log、二进制路径及 SHA-256、AstraSim commit。

parser 输出：

- 每个 operation 在其**参与 rank**上的 issue/callback tick、duration 及缩放后的 ns；
- 每 rank completion cycles、exposed communication、Wall cycles、compute-lane occupancy 和 communication-node occupancy。后两项分别来自 AstraSim 原始标签 `GPU time` 与 `Comm time`。

原始 `GPU time` 只表示 AstraSim compute lane 被 COMP 节点占用的计数；本项目统一称为 **compute-lane occupancy**，绝不解释为实体 GPU 利用率。当前 trace 还把 synthetic host/local-control replay 编码为 COMP，因此该值也不是纯模型 kernel 时间。

对于 p2p，接收方可以先 issue `COMM_RECV` 等待尚未发出的消息，所以 RECV issue→callback 可能包含 posted wait。T01 transfer fixture 的链路服务时延按 rank0 `COMM_SEND` issue→callback 取值；rank1 RECV 的更长驻留时间只可解释为 communication-node occupancy，不能当作链路本身的服务时延。

当前运行的是 `AstraSim_Analytical_Congestion_Unaware`。它建模给定带宽、时延与通信操作，但不模拟共享链路上的多流争用、拥塞排队或拥塞控制；任何通信时间和扩展性结论都必须标注这一限制。

## 不做什么

- 不使用参考仓库的旧二进制、旧 CLI 或离线拟合表；
- 不在 AstraSim 失败时返回 Python 估时结果；
- 不吞掉 stderr、error/critical log、缺 rank 或缺事件；
- 不下载“最近版本”覆盖固定 commit；
- 不支持当前合同之外的 BARRIER 节点或多 rank p2p 占位语义；
- 不把 parser 计算的 ns 宣称为实体硬件测量。
- 不把 `GPU time` 政名义误写成硬件 GPU 利用率，也不从 congestion-unaware 结果推断链路争用行为。

## 失败条件

Trace 生成阶段会拒绝：空 workload、重复/非正 node ID、未来依赖、无效 duration/payload、越界 p2p、未知 collective、非 `1000` 时间比例、缺少官方 Chakra schema 或空 trace。

运行阶段会拒绝：二进制不可执行、配置/trace 缺失、实际 git commit 与固定值不符、进程启动失败/超时、非零退出或没有 debug log。

解析阶段会拒绝：stderr 非空、error/critical 记录、completion/stat 重复或缺失、rank 集合不完整、任一参与 rank/node 的 issue/callback 缺失/重复/多出、节点 ID/名称/类型不一致、callback 早于 issue、completion cycles 与 Wall/final callback 不一致。

