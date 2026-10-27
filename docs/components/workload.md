# 工作流与操作规划组件

## 职责

工作流组件把已验证的 W/H/S 合同降为确定性 `Operation` 序列。每个操作都有稳定 ID、所属 round/phase、类型、前驱、工作量/字节数和语义 metadata，使同一份计划能同时驱动 Chakra trace、资源账本、验证器和报告器。

实现入口：`src/rl_xy/workflow.py`；成本公式：`src/rl_xy/costs.py`；参数枚举：`src/rl_xy/model.py`。

## 输入

- `CaseConfig`；
- toy Transformer 形状与 TP 分片规则；
- prompt、两条 response、reward 与版本；
- synthetic FLOP/vector/DMA 速率；
- `rounds`，单轮默认 1，多轮验证使用 2。

## 输出

按 `node_id` 排序的 `Operation[]`。操作类型包括：

- `compute`：`duration_ns`、`flops`、`vector_ops`、`tensor_bytes`；
- `collective`：`collective` 和 `payload_bytes`；
- `p2p`：`src_rank`、`dst_rank`、`payload_bytes`。

每个 operation 还带 `op_id`、phase、round、`participant_ranks`、`dependencies_by_rank` 和 metadata。第一版在每个参与 rank 内让节点依赖该 rank 的前一个节点，形成可重复的本地严格依赖链；collective 与 p2p 作为 rank 间显式同步点。未参与该 operation 的 rank 不生成节点，而不是复制一个占位事件。

## 阶段与功能

必须覆盖以下 phase：

1. `input`：host 输入 prompt，rank0 向 rank1 复制；
2. `rollout`：A/B prefill、逐 token decode、TP reductions、单 owner sampling 的 logits/token p2p；
3. `reward`：trajectory 回 host 和规则 reward；
4. `reference`：只读 reference teacher-forcing log-prob；
5. `old_policy`：只读 actor 行为版本 log-prob；
6. `advantage`：组内去均值并复制结果；
7. `train_A`、`train_B`：各自 forward、loss、显式 backward 和梯度 read-modify-write 累积；
8. `clip`：本地平方和、全局 norm 标量 reduction、裁剪；
9. `optimizer`：shape/cost-only AdamW，版本从 `v` 到 `v+1`；
10. `publish`：全分片 barrier 后原子发布新 actor 版本；
11. `cleanup`：释放本轮临时对象。

Transformer 前向显式展开 embedding、RMSNorm、QKV、attention score/softmax/value、O projection、FFN、LM head 和分布式 log-softmax；反向显式标注 dinput/dweight、保存张量依赖和梯度累积。

词表并行 embedding 之后必须有 hidden 部分和 AllReduce。每轮 train_A 开始前显式清零 FP32 梯度缓冲；train_B 只累积、不再清零。loss 后另有损失到本地 logits 梯度的操作，再进入模型反向。

## 关键语义

- A/B reward 为 `1/0` 时，demean-only advantage 为 `+0.5/-0.5`；
- loss 的分母是两条 response 的全部 token 数，不是 microbatch 局部分母；
- `DP=1`，因此不存在 DP gradient all-reduce；
- reference 在优化器阶段不可写；
- host link、reward 和 advantage 等独占动作只在 rank0 以 synthetic host replay 编码，metadata 保留 `logical_executor` 与 `rank0_synthetic_host_replay`；rank1 不镜像这些 duration；
- 两条 rollout 可同时持有 KV reservation/已物化 KV，但共享同一 TP=2 执行组，当前 rank-local 严格链按固定 trajectory/stage 顺序执行计算；`rollout_max_concurrency=2` 不是 kernel 并行度；
- publish barrier 使用 AstraSim 支持的小 payload `ALL_REDUCE` 表达，不宣称它是模型梯度通信。
- publish barrier 之后的版本发布和 cleanup 各使用 `control_marker_ns=1`。这 1 ns 是显式 synthetic control cost，只用于保留可观察控制节点，不代表真实发布或释放时延。

## 不做什么

- 不进行浮点张量计算、随机采样或数值 AdamW；
- 不输出 reward model 或策略模型 checkpoint；
- 不根据 AstraSim 结果动态改写计划；
- 不模拟 PP/DP、异步 rollout、重叠调度、MoE 或抢占；
- 不把 FLOP 公式当作硬件校准。

`costs.py` 保留了局部 DMA 公式，供后续显式 DMA operation 使用；当前完整 toy trace 没有调用该函数，`dram/module/sram bytes_per_ns` 不影响 toy Astra 时间。T01 的局部搬运由 fixture 自己的显式 COMP duration 表达。任何报告都不能把这些尚未接入时序的 H 字段说成已由 AstraSim 校准。

## 失败条件

- compute duration 非正；
- collective/p2p payload 非正；
- collective 类型不在当前 AstraSim 映射集合；
- p2p 端点相同或越界；
- 必需 phase 缺失、ID 重复，或某参与 rank 的依赖不存在/指向未来；
- 版本、reference 只读、loss 分母或梯度累积合同在验证阶段不成立。

任何失败都应阻止 trace 被认定为可运行工作负载。

