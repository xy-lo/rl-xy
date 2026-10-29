# Qwen3 GPU—晶圆迁移 baseline 实验设计

> 本页记录历史实验设计。当前 f2 成本、固定策略与用户接受的单窗基座见 [f2 接手说明](F2_HANDOFF.md)。

本页记录已执行的三档设置与当前证据边界。数字见[结果](RESULTS.md)，命令见[复现](REPRODUCING.md)。此前3B/7B/14B/32B规划属于0.9.5历史；本次明确采用Qwen3-4B/8B/14B。

## 1. 对照对象与研究问题

对照 GPU 调优所得系统策略与同策略晶圆迁移。两边固定模型、逐轨迹P/R、同步GRPO阶段、48活动设备、PP1、SP/设备间overlap关闭、层级activation checkpoint与distributed optimizer。晶圆底层采用正常的容量感知core/tile后端，系统层仅作必要容量适配。硬件迁移收益与同晶圆专用方法收益分别评价；后者本次尚未实现。

原因：保持工作量和协议公平，避免将硬件差异归为算法收益；直接迁移可以更快或更慢，不以预期方向作为验收。

## 2. 模型与公开长度输入

| 模型 | 参数量 | 层数 | hidden / Q宽 | query / KV heads |
|---|---:|---:|---:|---:|
| Qwen3-4B | 4,022,468,096 | 36 | 2560 / 4096 | 32 / 8 |
| Qwen3-8B | 8,190,735,360 | 36 | 4096 / 4096 | 32 / 8 |
| Qwen3-14B | 14,768,307,200 | 40 | 5120 / 5120 | 40 / 8 |

模型固定revision/config SHA见 [inputs/manifest.json](../experiments/2026-10-29-qwen3/inputs/manifest.json)。公开回答长度来自 [samuki-hf/thinking-rollouts](https://huggingface.co/datasets/samuki-hf/thinking-rollouts)，revision `a726a001d0145eddb2d5a2f5ede60591950b141f` 的DS-1000 thinking、T0.6三档分区；题目来源固定 [DS-1000 revision](https://github.com/xlang-ai/DS-1000/tree/b39aab71da6d23ef8d3cac59a7c5f834516ab334)。这是生成基准的长度回放，不是在线RL训练日志。Prompt长度按固定官方chat template重建；没有重新生成回答。

固定种子20261028，按题目library分层选择1个调优＋4个测试窗口，每窗口192题×8回答；五窗口共960题互斥，40题保留。模型/平台共用题目和sample ID。回答长度完整保留，最大8192，存在明显长度长尾；长度尾部不能直接等同硬件执行尾部。

原因：同源逐轨迹输入提供可追溯工作量，保留变长性；测试时间不用于选择切分或调整硬件。

## 3. 同步执行与状态移动

顺序为weight_sync→generation→samples→old_policy→reference→advantage→training→optimizer→publish。FIFO有界并发在单副本内请求完成后补位；结束请求释放KV，所有副本完成后进入后续阶段。GEMM线性层打包有效token，attention保留各请求独立因果长度；训练包含forward、层重算与backward。权重训推布局不同时，按既定完整张量导出/重分片路径计通信和本地读写。

容量统一预留C×(最大P2869+最大R8192)的KV，并检查训练/推理状态、激活和临时buffer；这是一套保守准入规则，降档不等于实际轨迹必然OOM。公开数据同时保留实际运行准入记录。

原因：RL阶段和模型状态移动属于研究范围，不能只把一次forward当作整步；checkpoint节省空间时相应重算也要付费。

## 4. GPU与晶圆建模

GPU为48×A100-SXM80GB（6×8），每卡108 SM、164KiB共享存储/SM、2.039TB/s HBM和312TFLOPS稠密BF16名义峰值。**本次Qwen3走纯解析成本，没有加载旧3B实测GEMM标签。** 矩阵形状经SM/CTA映射估算本地服务，网络由原生ASTRA执行。

晶圆为6×9逻辑mesh、48活动die；每die256 core、64GB DRAM、2TB/s DRAM带宽。每core1.25MB SRAM，合计每die320MB私有存储，不是统一池。容量来源为WSC-LLM §5.1.1；BF16速率、SRAM带宽、向量率、D2D方向与延迟等假设分别在配置中标记。粒度到core分区和私有tile，不逐指令/逐路由器周期仿真。

原因：保持算子、容量、带宽与网络的分层精度以研究资源组织；优先校准主导成本，不用微架构复杂度替代准确性。GPU片上复用/流水与晶圆参数假设仍阻止硬件主结论放行。

## 5. 选参、适配、指标与结论范围

| 模型 | GPU训/推TP；训练mb；logprob mb；并发 | 晶圆适配 |
|---|---|---|
| 4B | 4/4；16；16；64 | 训练mb降至8 |
| 8B | 2/4；4；4；64 | 训练mb降至2，并发降至32 |
| 14B | 8/4；8；64；64 | logprob mb降至32 |

这些是历史调优所得策略，不称全局最优。声明网格先解析筛选，再对每模型最多10个详细候选做原生核查，最终四窗口按冻结GPU策略及晶圆最小容量适配配对。后续审计发现C128模型内反例，说明原搜索边界仍不充分；原数值保持历史身份，不能只替换GPU列。

主指标为设备侧RL单步时间及有效Σ(P+R)/Σ时间吞吐；另给阶段分解、等待和内存。WSC-LLM使用请求E2E与请求/s，WATOS使用F+B iteration和FLOPs/s；评价维度相近，分母与统计对象不同。四窗口波动不作为模型误差或显著性证明。

原因：时间覆盖整个协议，有效token吞吐不奖励额外重算。工程一致性、硬件精度、学习质量分开验收；当前仅前者及记录复算已完成。
