# 验证组件

## 职责

验证组件决定一次 run 是否可接受。它不只检查“进程退出 0”，而是将配置推导、不依赖 Astra 的 planner/ledger 事实，与 AstraSim 实际事件和统计做交叉验证，并为每项检查输出 expected、actual、evidence 和 scope。

正式结果采用失败关闭策略：任一必需 check 失败，run 状态必须为 `FAIL`；缺失证据本身也是失败。

## 输入

- `CaseConfig` 或 transfer fixture config；
- 参数枚举和公式推导 totals；
- `Operation[]`；
- `LedgerBundle` 的对象和快照；
- AstraSim `AstraEvent[]` 与每 rank summary；
- runner provenance：命令、退出码、二进制 hash、AstraSim commit；
- 对应 evidence kind。

## 输出

`CheckResult[]`，每一项至少包含：

- `check_id`：稳定检查名；
- `passed`：布尔状态；
- `expected`：合同期望；
- `actual`：实际观察；
- `evidence`：可定位的文件、事件或推导说明；
- `scope`：`synthetic_fixture`、负向 fixture 或其他明确范围。

reporter 将其写入 `checks.json`，并据此产生 summary 状态。

## 主例子检查域

至少覆盖：

1. **配置与参数**：unique/local parameter totals、dtype byte accounting 和 TP/replicated 规则；
2. **阶段覆盖**：input 到 cleanup 的每个必需 phase 存在且顺序合法；
3. **轨迹语义**：prompt/response、reward、advantage 和全局 response-token loss 分母一致；
4. **训练语义**：A/B 两个 microbatch 均有 forward/loss/backward，gradient buffer 使用 read-modify-write；
5. **并行语义**：TP collective 和 p2p 端点合法；`DP=1` 时没有 DP gradient all-reduce；
6. **版本语义**：rollout/old-policy 使用行为版本，optimizer 只更新 actor，barrier 后发布 `v+1`，reference 不写；
7. **对象生命周期**：临时 KV 在 rollout 后释放，多轮不泄漏，持久对象保留；
8. **容量**：每个 DRAM/SRAM/NIC 位置独立检查，不以总容量互相借用；
9. **Astra 事件**：每个 operation 只在其参与 rank 上有唯一 issue/callback，未参与 rank 不应出现该节点；完成时间与各 rank final callback 一致；
10. **provenance**：固定 AstraSim commit、成功退出码、原始日志、argv 和 hash 齐全。

## Transfer fixture 检查域

transfer fixture 通过一条可手算串行链验证时间编码与资源语义：

- completion 的配置期望为 21 ns；
- 32 B p2p 在 8 B/ns、1 ns latency 下的配置期望为 5 ns；该值取 rank0 `COMM_SEND` issue→callback；
- out-of-place compute 的 SRAM 期望峰值为 64 B；
- D2D payload 期望为 32 B；
- 六次显式 DRAM 端点移动的期望 traffic 为 192 B；
- 将 SRAM 改为 63 B 时，构建必须因输入/输出不能同时驻留而失败。

这些首先是 fixture 的期望值；只有 AstraSim 实际事件与检查相符后，才可报告该 run 已验证它们。

T01 中 rank1 可以在 rank0 完成前序 DMA 时提前 posted `COMM_RECV`，所以 RECV issue→callback 会把等待 SEND 的时间算进去。验证器不得用这个 RECV 驻留时间检查 5 ns 链路服务时延。主例子的 publish 与 cleanup 节点还应分别核对配置中的 `control_marker_ns=1`；它们是 synthetic control cost，而不是硬件测量。

## 不做什么

- 不用一个总 `PASS` 掩盖单项失败；
- 不把单元测试或守恒关系当作硬件校准；
- 不用 operation 的计划 duration 填补缺失 Astra event；
- 不把 Astra 原始 `GPU time` 当硬件 GPU 利用率，统一按 compute-lane occupancy 解释；
- 不用 congestion-unaware backend 验证链路争用或拥塞效应；
- 不验证真实梯度数值、AdamW 数值或模型学习质量；
- 不自动修改配置让失败用例通过；
- 不把 warning/缺失日志降级为成功。

## 失败条件

包括但不限于：expected 与 actual 不一致；phase/对象/版本缺失；容量越界；生命周期不闭合；存在被禁止的 DP 通信；reference 被写；Astra 进程或 parser 失败；参与 rank/node event 覆盖不全或未参与 rank 多出事件；completion 与 Wall/final callback 不一致；provenance 不完整。

验证异常也必须被视为失败，不能生成 `PASS` summary。

