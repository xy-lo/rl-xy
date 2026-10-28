# 架构与执行合同

当前主入口是 `rl_xy.cli run-aligned`。历史 `run`、`run-wafer`、`run-colocated` 保留用于兼容；它们的旧协议和性能数字不能替代本版控制。逐文件职责见 [FILE_MAP](FILE_MAP.md)，命令见 [REPRODUCING](REPRODUCING.md)。

## 分层与模块

| 层 | 主要文件 | 输入、功能与输出 |
|---|---|---|
| 输入/能力合同 | `config.py`、`baseline_contract.py`、`model.py`、`hardware.py` | 校验模型、完整轨迹、batch/并行组、H0 来源及实现能力；配置合法不等于已经实现或成本可信 |
| 同步协议 | `colocation.py`、`colocation_validation.py` | 全局阶段边界、训练/生成 TP 布局、权重重分片、行为/Actor/Reference 版本 |
| 真实微批/DO | `training_batching.py`、`distributed_optimizer.py`、`aligned_graph.py` | 聚合真实形状，保留轨迹/causal attention/有效token分母，RS→owned Adam→AG 与状态生命周期 |
| 推理生命周期 | `uniform_generation.py`、`inference_lifetimes.py` | 固定长度生成、KV 预留/增长/回收、最后 hidden row 的 sampling LM head；不复制实际 server router |
| 算子与硬件成本 | `operator_ir.py`、`mapped_costs.py`、`gpu_model.py`、`mapping.py`、`multicore.py` | typed GEMM/向量/读写；GPU SM/CTA/标签或 wafer tile/核私有 SRAM、共享 DRAM 服务 |
| 成本域与缓存 | `gemm_census.py`、`memoized_costs.py` | 按实际调用次数列精确逻辑签名，查表缺域拒绝，完整 shape/precision/cache 口径不得以统一倍数替代 |
| 张量/资源 | `tensor_plan.py`、`resource_encoding.py`、`resource_validation.py` | alloc/read/READY/free、持久状态、workspace、私有 SRAM、DRAM/NIC、完整生命周期和峰值 |
| 原生适配 | `communication_lowering.py`、`astra_adapter/`、`docker/astrasim/` | 每 rank Chakra ET、通信端点与资源 hook；原生网络排队和 issue/callback 时钟 |
| 等价执行优化 | `local_chain_contraction.py`、`component_proof_cache.py`、`evidence_io.py` | 严格串行本地链精确收缩、不可变组件证明复用、流式证据；不删通信或资源验证 |
| 运行/证据 | `aligned_run.py`、`reporting.py`；`scripts/bounded_run.py` | 预算、退出、镜像/源/输入身份、原始日志、SHA、指标与失败保留 |
| 独立分析 | `scripts/check_aligned_result.py`、`compare_aligned_results.py`、`compare_native_replay.py` | 绑定实际 manifest 与工具 SHA；单包、同策略、原生等价分别检查，不提升硬件精度 |

## 一次同步更新

1. 持久 Actor/Reference/DO 状态准入；按所选 naive 路径导出推理权重并计重分片通信和本地读写。
2. 经过阶段屏障，执行 prefill 与逐 token decode；Transformer/KV 处理全 P，普通 sampling 只对每请求最后 hidden row 做 LM head。
3. 搬运样本、回收生成状态、切回训练；执行 reward、old-policy logprob、冻结 Reference、GRPO group 优势阶段。
4. 执行真实训练 microbatch 的 forward/backward 与梯度累积；teacher 使用完整 S=P+R−1，不把采样剪枝套到训练。
5. DP 梯度 ReduceScatter、唯一参数所有权的 norm/clip、owned FP32 Adam 更新、BF16 参数 AllGather。
6. Actor 版本发布和清理。当前一次更新 Actor0→1，Reference0→0；下一轮不能提前读取未发布版本。

W1 训练 DP 连续分配完整 group；训练 TP 块内生成使用 `local[j::micro]` 静态交错，回收后按源 owner 显式搬运。每生成 replica 收到16个group各4条，不能误写成8个完整group。两侧 mapping/dispatch 相同，但这不是 veRL 实际 worker/server 路由、continuous batching 或 cache 实现。算法 oracle 及合成输入不等于执行数值 LLM 训练。

## 内存、通信与资源准入

对象可先分配、后 READY；读必须满足生产依赖，free 必须晚于最后消费。KV、保守激活 tape、反向 partial、FP32 主梯度/master/moments、通信 staging 和外层样本 carrier 均有生命周期。DO 只分片归属状态，不把所有内存免费除 DP。arena 是原逻辑对象/预留的保守准入峰值，不是新算法节约；它不是生产 veRL 的显存严格下界。

GROUP AllGather 的 `comm_size` 是每 rank 输入 shard，ReduceScatter 是每 rank 完整输入；输出空间、端点读写、ring SUM 和 staging 分别保留。Astra 内部高 `local-mem-bw` 仅禁用重复 memory proxy，真实本地成本由显式服务给出，不是芯片带宽。操作级 payload 之和不等于 ring/链路 wire traffic。

`RlxyResources.hh` 在原生 issue/callback 执行准入/释放并输出实际快照。统一合同和逐 rank 合同必须能完整解码、语义和 SHA 对齐。read-ID 字典仅压缩重复 storage 字符串，非法类型/索引被拒绝；完整向量值缓存仍验证每行快照和私有容量；64 KiB 日志缓冲只改变宿主 I/O，kill 后的部分日志仍是失败。

严格本地链只合并没有中间竞争边界的串行 compute，保留整数服务总和；通信、全局边界和 cleanup 不免费消失。原始逻辑对象在实际 native macro 时钟上回放。宏内推导事件不能冒称每个算子都有独立 native callback。组件实例证明和语义签名去重分开，不能以字典去重省掉 invocation 覆盖。

## 硬件与单位

| 参数 | GPU 模型 | H0 晶圆模型 |
|---|---|---|
| 分配/活动 | 48/48，6节点×8 A100-SXM80GB | 54/48，6×9逻辑mesh，256核/die |
| 名义 FP16/BF16 算力 | 312TFLOPS/卡；14.976PFLOPS合计 | FP16 1.02TFLOPS/核；active12.53376PF，allocated14.10048PF；BF16同率是假设 |
| 主存 | 80GB/卡，2.039TB/s | 64GB/die，2TB/s |
| 片上 | 164KiB/SM参考预算，不是实测cuBLAS occupancy | 1.25MB/core私有SRAM；容量不当带宽 |
| 网络主场景 | NVLink300GB/s/方向；节点uplink400GB/s/方向共享 | D2D2TB/s/edge/方向的执行口径假设 |

H0 来源保留于 `configs/hardware/`；用户原表/论文值、单位解释、执行假设分别记录。WSC-LLM Fig.3 模板几何与 Case3 54die 资源不能直接拼成已验证封装；不缩小计算die掩盖布局问题。NoC/SRAM带宽、FP32/向量率、D2D方向/延迟、NIC/host等未定项不能冒充论文实测。细粒度 NoC/DMA、有限 NIC 队列与物理布局未完整校准。主参数和引用见[实验设计](EXPERIMENT_DESIGN.md)。

模型成本用 ns，当前 native tick 每1000tick对应1model ns；日志和 manifest 保存缩放。以真实完成 callback 还原时钟，程序墙钟另记。阶段表是全 rank 最早 issue 到最晚 callback 的 envelope；barrier 的全步包络不是屏障服务时间，阶段跨度不能直接相加。

## 证据能支持什么

每个 run 保存实际 argv、模型/数据/算法/硬件/策略、源树与镜像/二进制、输入/产物 SHA、退出/OOM、预算、原始事件/资源与检查。失败保留；超时或超容量不填性能数值，不从多次尝试拼出伪单次成功。

单测/数学 oracle/守恒/原生小例证明工程一致性；热GEMM标签只覆盖指定调用协议，不证明真实layout/fusion/网络/host或整步校准；仿真吞吐不证明学习质量。当前 `hardware_calibrated=false`、`full_verl_runtime_reproduced=false`、`paper_performance_ready=false`。G7受限3B控制已通过，G8强GPU/四模型/代表性负载仍未放行，详见[结果](RESULTS.md)。
