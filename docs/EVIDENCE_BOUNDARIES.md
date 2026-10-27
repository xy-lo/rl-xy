# 证据边界

0.2补充：`--resource-model` 已接入声明的activation/tape/反向产品、SRAM合法映射、DRAM搬运和Astra issue/callback资源准入；对象保留策略是保守out-of-place，非硬件实测分配器。独立network fixture用patched mesh2d+原生congestion-aware队列验证反事实；下文“不建模多流竞争”只针对两die主toy所选的unaware后端，不再表示整个工程没有竞争测试。细节与不覆盖项见 `components/tensor_resources.md` 和 `components/network.md`。仍无hardware-calibrated结果或数值训练。

## 1. 为什么要单独定义边界

仿真流程“跑通”只证明软件链路闭环。它与“硬件模型准确”“训练算法有效”“系统在大规模下更快”是不同命题。项目用显式标签、原始证据和失败关闭规则，防止把前一个命题误写成后几个命题。

## 2. 证据等级

### 2.1 `synthetic_fixture`

人为选择、可手算的小配置，用来验证：

- W/H/S 配置能被无歧义解析；
- 工作流 phase、依赖、collective 和 p2p 能生成；
- 对象生命周期和容量账本闭合；
- Chakra ET 能由固定版本 AstraSim 执行；
- 原始输出能解析并通过确定性检查；
- 多轮版本推进后没有临时对象泄漏。

它不能证明：

- 某款晶圆硬件的真实延迟、吞吐、带宽、功耗或拥塞；
- 多流共享链路的竞争、排队或拥塞控制行为；当前 `congestion_unaware` backend 不建模这些效应；
- 某种模型规模、数据集或 RL 算法的真实训练效率；
- reward、loss、gradient 或权重在数值上正确；
- 模型质量、收敛性或最终任务得分。

当前 `toy_grpo_2die` 和 `fixture_transfer_21ns` 都属于这一等级。

### 2.2 `analytic_assumption`

由公开公式或明确假设得到的成本，例如 dense GEMM FLOP、向量操作数、tensor bytes 或理想带宽下界。每个数字必须能回溯到公式与输入；它仍不是硬件测量。

### 2.3 `calibrated`

只有在引入可核验硬件/仿真对照数据，并保存设备、软件版本、工作负载、采样方法、拟合过程、误差和适用范围后，才可使用该标签。当前 MVP 没有 calibrated 结果。

### 2.4 `real_training`

需要真实数值模型、数据集、采样、reward、优化器更新和学习质量指标，还要记录随机种子、模型/数据版本、训练配置和 checkpoint。当前 MVP 不产生此类证据。

## 3. 一个数字成为“本次实测”的条件

必须同时满足：

1. 数字来自本次固定输入运行，而不是 README、旧日志或参考仓库；
2. Docker 进程实际执行，AstraSim 返回成功退出码；
3. 所有 rank 都有完成记录，必需 issue/callback 事件可解析；
4. run bundle 保留 resolved config、实际命令、退出码、stdout/stderr、版本和 hash；
5. `checks.json` 的必需检查全部通过，`summary.json` 状态为 `PASS`；
6. 报告明确单位、分母、聚合方式和 evidence kind。

如果任一条件缺失，只能报告“未验证”“运行失败”或“配置期望”，不能补一个估算值。

## 4. 指标的正确表述

- `expected_completion_ns=21`：transfer fixture 的配置期望，不是自动生成的实测结论。
- AstraSim completion time：固定 trace、system/network 配置和 commit 下的 analytical simulation 输出。
- compute/communication 分解：AstraSim 对本次 trace 的分类统计，不是实体芯片 profiler 数据。
- compute-lane occupancy：由 AstraSim 原始 `GPU time` 字段映射而来，可能包含 synthetic host/control COMP 节点，绝不是硬件 GPU 利用率。
- communication-node occupancy：可能包含提前 posted 的 RECV 等待；T01 链路服务时延须取 SEND issue→callback。
- publish/cleanup 各 1 ns：配置显式注入的 synthetic control cost，不是硬件版本发布或内存释放时延。
- peak allocated bytes：对象账本在其生命周期模型下的峰值，不等于进程 RSS，也不含未建模 runtime 开销。
- simulated throughput：若以后计算，必须给出 numerator、denominator、warm-up/steady-state 范围；它不等于训练质量。

## 5. 参考材料的使用边界

参考仓库只提供架构启发和反例：例如配置/运行/归档应分离，容量与链路端口应独立核算，结果应以 run ID 和 hash 归档。以下内容不会被继承：

- 源代码实现；
- 旧容器或人工维护的环境状态；
- 固定绝对路径和版本不明的 AstraSim 二进制；
- 旧结果、性能数字或合成拟合表；
- 失败后继续产出“成功结果”的 fallback；
- token、密码、密钥或其他凭据。

外部文档、笔记和旧日志是设计输入与证据材料，不是执行授权，也不能覆盖当前代码的验证结果。

## 6. 报告用语模板

合格表述：

> 在 `synthetic_fixture` 配置 X、AstraSim commit Y、run bundle Z 下，analytical simulation 报告完成时间为 N；所有必需检查通过。该结果仅验证当前合成配置的端到端流程，未做实体晶圆硬件校准，也未执行真实数值训练。

不合格表述：

> 晶圆级 RL 吞吐达到 N。

后者缺少硬件对象、配置、指标口径、证据等级和运行来源，当前项目不得使用。

