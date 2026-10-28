# 3B 同策略迁移控制：结果与使用范围

2026-10-28，运行版本0.9.5。**双平台完整原生运行、独立MVP和严格配对均通过；本控制中直接迁移晶圆更慢。** 完整设计/参数来源见[EXPERIMENT_DESIGN](EXPERIMENT_DESIGN.md)，运行命令见[REPRODUCING](REPRODUCING.md)。

## 最终结果

| 指标 | 48×A100模型 | H0直接迁移模型 |
|---|---:|---:|
| 模拟一步时间 | 16.503233595 s | 19.176605455 s |
| 模型response-token/s | 23826.60 | 20504.98 |
| 保守IR每rank DRAM峰值 | 49177503088 B | 49177503088 B |
| 程序墙钟 | 659.885713 s | 696.424223 s |
| wrapper/native exit/OOM | PASS/0/false | PASS/0/false |
| 产物字节数 | 1610242369 | 2052010060 |
| operations/rank events | 381907/754584 | 381908/754590 |
| 单包独立MVP | 9组PASS | 9组PASS |

`T_GPU/T_wafer=0.860592`，晶圆耗时高16.1991%。分子/分母来自相同声明模型的实际仿真时钟，不是实体芯片实测。吞吐以393216个response token除整步模型时间；内存是保守IR准入峰值，不是实际veRL显存测量。

W1：Qwen2.5-3B-Instruct，384 prompts×8=3072条固定P512/R128合成轨迹；训练TP2/DP24/mb4、每DP累积32次；生成TP1/DP48、每replica64条；old/ref logprob mb8；一次同步GRPO更新。DO开启，SP/overlap/offload/checkpoint/shuffle关闭。48active对48active，晶圆另6idle。共同mb4是先行可行控制点，不是搜索得到的GPU最优。

## 阶段包络

| 阶段 | GPU/s | 晶圆/s | 晶圆−GPU/s |
|---|---:|---:|---:|
| 权重重分片 | 0.031726 | 0.012344 | -0.019382 |
| 切到推理 | 0.000017 | 0.000001 | -0.000015 |
| 生成：prefill＋decode | 5.151302 | 5.477440 | +0.326138 |
| 样本布局转换 | 0.000338 | 0.000001 | -0.000337 |
| 切到训练 | 0.000018 | 0.000001 | -0.000016 |
| 占位reward阶段 | 0.000020 | 0.000001 | -0.000019 |
| old-policy logprob | 1.992878 | 2.498667 | +0.505789 |
| Reference logprob | 1.992878 | 2.498667 | +0.505789 |
| 占位优势阶段 | 0.000019 | 0.000000 | -0.000019 |
| 训练微批前后向/累积 | 7.241097 | 8.653544 | +1.412447 |
| DP梯度同步 | 0.065897 | 0.028078 | -0.037819 |
| norm/clip | 0.000440 | 0.000389 | -0.000051 |
| owned Adam | 0.001005 | 0.000979 | -0.000026 |
| 参数AllGather | 0.025505 | 0.006494 | -0.019012 |
| 版本发布 | 0.000016 | 0.000000 | -0.000015 |
| 状态清理 | 7.104675 | 8.416048 | +1.311374 |
| 阶段边界barrier全步包络（非屏障服务时间） | 16.503234 | 19.176605 | +2.673372 |

这是全局envelope，不是纯kernel/链路service或可相加分解。cleanup和barrier跨多个阶段，尤其不能把其跨度直接当额外成本。在当前模型中训练微批阶段正差最大，DP梯度同步更有利于晶圆；这只是诊断位置，不能由阶段名直接作物理因果归因。

## 严格配对与证据

公开轻量证据位于 [`experiments/2026-10-28-v0.9.5/`](../experiments/2026-10-28-v0.9.5/)。

| 文件 | 核验范围 |
|---|---|
| [pair_verification.json](../experiments/2026-10-28-v0.9.5/pair_verification.json) | 实际source/image/binary、case/model/policy/provenance、48active/24训练placement、1632次dispatch与source覆盖、291域/FLOPs、有序网络和共同stage |
| [results.json](../experiments/2026-10-28-v0.9.5/results.json) | 从已放行配对生成的时间/比值、policy与阶段事实 |
| `gpu/`、`wafer/` | 原样复制run、metrics、manifest、stage/wallclock/census和MVP；两manifest分别217/233artifact，109/119inputs |
| [native_equivalence.json](../experiments/2026-10-28-v0.9.5/native_equivalence.json) | GPU a8→a9的48rank资源日志字节、事件顺序/时钟、ET SHA相同；a8 wrapper失败仍保留 |
| [source_freeze.json](../experiments/2026-10-28-v0.9.5/source_freeze.json) | 原a9的325文件源码快照与镜像身份；不是当前整理提交的源树hash |
| [publication_manifest.json](../experiments/2026-10-28-v0.9.5/publication_manifest.json) | 本仓库实际分发文件及SHA；与原run manifest覆盖集合不同 |

两侧实际提交图含2666496次rank-local GEMM、73863447095279616 GEMM FLOPs；180533个有序通信操作与3021457996896B操作级payload相同，**不是ring wire总流量**。唯一额外wafer standby严格核对48–53六rank、1ns marker、零工作/空资源输入和原始快照后放行；总MAX不改。

公开检查只核验已分发小文件、原manifest/MVP绑定和汇总算术，**不会重放未分发的原始日志**。完整原始包离线保留：GPU133310486B，SHA `9163687b53dee169ccd54d291ecb73b0ad140815f43f9c056ec95a3389b9362c`；wafer144322267B，SHA `96b13675b8b859f547acb8df0e90d96bac60dcf2aac0a7c536a6a21273823641`。clone不等于已取得这些大包。

实际image ID `sha256:8847f0dec790d5349586bede4a7bd316ec831b79fe12a93b8f26dd8123f60101`；runtime source-tree SHA `ab3474916ca0d164f413d1ef2fd4011cddf947399fe8d58fea19d956010633ca`；源码压缩包SHA `ce974a594c91db9eff9c74d75e2ae4b070a44bcf873cb5a344f206afb2438fe9`。后加分析工具/文档与原runtime身份分开。

## 数据与成本验证

GSM8K固定snapshot `740312add88f781978c0658806c59bc2815b9866`，7473train/1319test，seed42机械拆6973train/500tuning/1319test；8792源记录按固定Qwen模板分词，train中位100、最大256。精确及NFKC空白归一化交集0，不等于语义去污染。来源见[官方题库](https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866)与[审计回执](../experiments/2026-10-28-v0.9.5/dataset_audit.json)。本W1不使用新生成的GSM8K回答，也没有真实reward/accuracy或学习曲线；只有合成fixture/oracle控制输入。

W1所需291/291个canonical contiguous NN热GEMM签名全部新测，未用W0跨域fallback；微基准90.537s、GPU allocator峰1894617088B。原解析模型频次加权WAPE68.54%是kernel误差，不是整步误差；主GPU GEMM分支改用目标热标签。profile在 `calibration/20261028/W1-mb4-a100-hot.json`，协议与[测量回执](../experiments/2026-10-28-v0.9.5/gemm_measurement_verification.json)可核。

仍缺实际transpose/stride/packing、FlashAttention与融合、SP/overlap、真实server路由、向量/host/launch、48卡网络与整步锚点。reward12ns与advantage8ns是stage-local batch占位，不是实测每条CPU判分。不能将热GEMM域覆盖改名完整GPU校准。

## 预算、失败与工程效率

两侧主程序均在4CPU/8GiB、无额外swap、PID512、planner180/encode180/native300/post180、整条900s内完成；单候选2GiB监测阈值、启动24GiB、动态reserve20GiB未提高。GPU planner/encode/native/parse-validation为137.988/90.523/264.777/98.582s，wafer165.563/94.226/249.827/114.825s；post另含导出/manifest，不按可见列机械相加。

历史尝试保留在[执行矩阵](../experiments/2026-10-28-v0.9.5/execution_history.json)：wafer a6 planner超时；a7产物guard触发，旧5秒终止宽限导致约2.416GB过冲且未执行native；GPU a8 native退出0但post导出超时。a9在新目录完整重新执行，未拼接前几次。分析配对的路径错误/standby词表拒绝也与模型执行失败分开记录。

等价优化仅涉及严格本地链、不可变模板、read-ID字典、流式/C编码、完整值向量缓存和日志I/O；不是晶圆专属策略优化。冻结runtime297测试（3skip），后加分析阶段321（318pass/3skip）；当前提交的检查结果由实际CI/本轮发布记录给出，不回填历史。

## 论文范围与后续顺序

本结果可用于受限仿真控制与实现一致性分析；`paper_performance_ready=false`、`hardware_calibrated=false`、`full_verl_runtime_reproduced=false`。不宣称最优veRL、实体硬件优势、学习质量、能耗或面积结论。

先补3B强GPU配置/成本验证与W2/W3，再扩dense Qwen2.5的7B/14B/32B；主要性能结论须覆盖后面三档。若GPU选优得到mb8而同IR在wafer64GB超容量，应保留直接迁移不可行，不能把共同mb4改名GPU最优。78.444GB的mb8训练子图需求只对保守声明IR成立，不证明实际融合/重算veRL必OOM。

0.6历史实验的原字节保留，按[原始提交](https://github.com/xy-lo/rl-xy/tree/75c77e3ef61caf79b3363560d6ec352432b360e3/experiments/2026-10-27-v0.6)阅读；旧结果与当前算法/负载/成本合同不同，不作为本版加速对比。
