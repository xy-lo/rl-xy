# 架构与执行合同

当前Qwen3入口是 `qwen3_baselines.py` 和 `replay_qwen3_policies.py`。原 `rl_xy.cli run-aligned` 等0.9.5入口保留兼容；两条路径的输入、成本域和运行身份分别解释。

## 分层职责

| 层 | 模块 | 合同 |
|---|---|---|
| 输入与模型 | `model_import.py`、`model.py`、`config.py` | Qwen3参数精确枚举、4B独立Q投影宽度、Q/K RMSNorm；固定逐轨迹P/R和分组 |
| 本地成本 | `qwen3_cost.py`、`gpu_model.py`、`capacity_mapping.py` | 线性/attention/向量操作；GPU SM/CTA与晶圆core/tile容量映射 |
| RL阶段 | `qwen3_experiment.py` | 训推权重转换、样本搬运、old/ref logprob、训练重算、DO和版本发布 |
| 生成调度 | `qwen3_rollout.py` | 副本内FIFO补位、KV增长/释放；逻辑token轮次的精确事件压缩 |
| 原生重复程序 | `qwen3_sequence.py`、`RlxySequence.hh` | 压缩宿主表示，仍执行每个原生子步/collective，不以压缩省略设备工作 |
| 原生网络 | `qwen3_native.py`、`RlxyMesh.hh`、`RlxyGpu.hh` | rank依赖、die间/节点间网络及issue/callback完成时间 |
| 身份与路径 | `qwen3_source.py`、`qwen3_paths.py` | 内容树与二进制SHA；显式输入/输出/后端路径，缓存不写入冻结输入 |
| 公开校验 | `check_qwen3_results.py` | 校验已发布输入、记录账本与统计；不冒充原日志重放或硬件校准 |

## 粒度与存储

系统级追踪同步阶段和TP/DP组；晶圆逻辑网格为54die、活动计算rank0–47。die内部把矩阵分给多个core，按最慢核工作和共享DRAM服务估算；每core的tile单独受1,250,000B SRAM限制。256core合计320MB，但单核不能把它当作池化空间。每die外部DRAM为64GB，状态与KV等另做准入。

GPU实际使用108SM、164KiB共享存储/SM和合法CTA候选。本次Qwen3的profile没有`gemm_profile`；旧3B热GEMM查表路径仍存在，不能混称本轮实测成本。两边本地计算+访存目前相加，GPU跨CTA的L2复用没有建模，仍需修订/校准。片内NoC、SRAM bank和PE阵列没有逐周期模拟。

## RL单步与通信

权重准备→生成→样本转移→old/ref→优势→forward/重算/backward→DO归约/Adam/参数发布。训练/生成TP不同时，历史baseline通过完整参数导出与目标分片装载处理；端点读写与通信计费。生成FIFO只在副本内补位，未引入晶圆专用跨副本任务重分配。

Network模型使用固定ASTRA后端和透明扩展；不同逻辑组的通信仍在同一原生网络资源上执行。compact程序与expanded小例按各rank时钟检查等价。表示压缩、宿主缓存、流式日志省宿主开销，不免费减少模拟计算或通信。

## 时间与容量边界

Qwen3各大阶段之间有全局屏障，每阶段取并行依赖/网络执行后的完成时间，因此单步可求阶段之和。阶段内不能把48个rank时间直接相加。0.9.5旧aligned报告的stage envelope可能重叠，不能把旧表任意求和；该口径不自动继承到新路径。

时间为模拟设备ns；native内部采用固定tick缩放，程序墙钟独立记录。容量检查包括Actor/Reference/DO持久状态、导出buffer、KV、激活和前向临时buffer。GPU80GB、晶圆64GB均按声明生命周期上界；模型通过准入不代表实体框架显存已经验证。

## 当前限制与证据身份

源树、策略、输入和二进制共同绑定每份结果，失败尝试保留。新发布包装器与历史运行是不同身份；冻结结果继续绑定历史树d7530f67…。目前没有真实48GPU或晶圆整步校准、学习质量、能耗/面积或专用方法收益结论。后续改进不得通过改变参数迎合预期快慢方向。
