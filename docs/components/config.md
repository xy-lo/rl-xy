# 配置组件

## 职责

配置组件是 W/H/S 合同的唯一入口。它把 JSON 解析为冻结的 dataclass，拒绝未知字段、缺失字段、歧义值和当前 MVP 不支持的组合，并把本次实际使用的配置保存为 resolved 版本。

实现入口：`src/rl_xy/config.py`；Astra 文件合同：`src/rl_xy/astra_adapter/config.py`；共享类型：`src/rl_xy/types.py`。

## 输入

主例子输入为 `configs/toy_grpo_2die/case.json`，包含五部分：

- 根信息：`schema_version`、`case_id`、`evidence_kind`、`rounds`；
- `model`：层数、hidden/head/FFN/vocab 形状及各状态 dtype bytes；
- `workload`：round/group、prompt、两条 trajectory、reward、advantage/loss/optimizer 规则；
- `hardware`：die/core/DRAM/SRAM/NIC 容量、计算/移动速率、D2D/host 参数和 `astra_time_scale`；
- `strategy`：TP/PP/DP、microbatch、rollout concurrency、布局兼容性、schedule 和 KV admission；
- `costs`：reward/advantage、toy kernel 的操作成本常数，以及 `control_marker_ns=1`。后者只为 publish/cleanup 提供显式 synthetic control cost，不代表硬件控制时延。

CLI 可以只覆盖 `rounds`；覆盖后的值必须写进 `resolved_config.json`，不能让原文件和实际运行参数分叉而无记录。

system/network/remote-memory 三份 AstraSim 文件在运行前还要做第二层交叉校验：字段和值必须严格匹配冻结合同（布尔值不能冒充整数），`npus_count=2`，原始时钟中的 bandwidth/latency 必须与 case 的 `8 B/model-ns`、`1 model-ns` 和 `astra_time_scale=1000` 精确对应。解析结果写入 `astra_config_contract.json`。

## 输出

- `CaseConfig`：后续 planner、model、ledger、adapter 和 validation 共用的只读配置；
- `resolved`：可序列化的最终配置快照；
- `source_path`：原配置绝对路径，用于 provenance 和 hash。

## 主要约束

- schema 当前只接受 `1.0`；
- 当前未校准实现只接受 `evidence_kind=synthetic_fixture`，不能由输入自报为 calibrated；
- 所有正整数硬件/模型/成本字段必须大于零；
- `hidden_size = query_heads × head_dim`；
- query heads、KV heads、hidden、FFN、vocab 必须能被 TP 整除；
- 一 die 对应一个 TP rank；
- 当前固定 `PP=1, DP=1`；
- rollout 同时准入/保留两条活跃轨迹，`rollout_max_concurrency=2`；它表示 KV admission/residency 上限，不表示同一组 TP rank 同时执行两条 compute；
- 固定 `share_prompt_kv=false` 和 actor 训推布局兼容；
- 恰好两条、ID 唯一的 trajectory，response 不超过公开最大长度；
- 每个训练 microbatch 只含一条 trajectory；
- 固定 `demean_only`、`global_response_token_mean`、`adamw_shape_only`；
- 固定 `strict_synchronous` 和 `reserve_public_max_length`。
- 一个 D2D packet 必须分别放入每 die 的 TX/RX 容量。

## 不做什么

- 不自动补默认字段或忽略拼写错误；
- 不猜 dtype、单位、并行度或网络拓扑；
- 不把 environment variable 的临时值悄悄写进模型合同；
- 不证明给定硬件数字来自实体设备；本版本强制标为 synthetic fixture；
- 不负责运行 AstraSim 或聚合结果。

## 失败条件

出现下列任一情况立即抛出配置错误：JSON 根不是对象、字段集合不精确、token 越界、数值类型/范围不合法、trajectory 重名、response 超长、模型形状不整除、TP 与 die 数不匹配，或请求当前不支持的 PP/DP/调度/优化器语义。

配置失败时不得生成 partial run 或回退到宽松解析。

