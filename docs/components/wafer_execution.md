# DP×TP2晶圆执行、真实veRL采集与GPU对照

本组件是基于官方Astra事件时钟的工作负载迁移，不是实体晶圆的数值runtime。
核心代码独立实现；训练侧复用用户只读veRL环境和既有长度collector，私有容器flag不影响主机其它任务。

## H、几何与场景

用户选择继续54die/6×9、256核/die、1GHz、1.02TFLOPS TensorFP16/core、1.25MB SRAM/core、64GB DRAM/die、DRAM/D2D2TB/s。
`xy_h0_case3_performance.json` hardware/3把Fig.3的外部模块与48模块派生阵列尺寸标为模板参考；
不缩小计算die、不把215除以行列数编成物理pitch，不声称制造布局通过。
旧hardware/2原表组合审计及冻结结果不改写；最新用户意图取代曾提出但未实施的48切换。

`wafer.json`把每个未知执行取值独立写为test_assumption：BF16/FP32 GEMM、向量吞吐、SRAM/NoC路径、D2D时延/口径、NIC、host。
BF161020和FP32512FLOP/ns、vector256op/ns等只是本工程场景值，不是论文或硬件实测。
D2D2TB/s暂按每边每方向的显式假设进入工程场景，不能忽略与总接口预算的潜在差别；正式结论需要其它口径/值的敏感性及可信来源。
NoC当前是模块共享带宽上界，不模拟逐包、VC、bank或布线时延；不把SRAM汇总容量当共享池。

## 数据并行语义

Qwen2.5-0.5B为14Q heads/2KV heads，当前均匀TP不能直接设成54。因此采用DP27×TP2，物理rank线性相邻成TP对。
每个完整prompt group按输入顺序轮转分配给DP副本，每副本至少一个group；不复制成27份独立假训练。
每轮先完整采集/Ref/old-policy/组内优势；每副本一条trajectory/microbatch，局部梯度按**全step有效response token总数**归一化。
再按同TP位置组成两个DP组执行真实native ALL_REDUCE SUM，不再除以DP数；避免异长/不均匀副本的局部均值再等权平均错误。
随后各TP副本norm/clip→Adam形状RMW→TP发布。每轮增加显式全rank fan-in/fan-out fence，保证所有副本发布后再进入下一轮。

当前分布式基线支持整个采集批次一次更新/epoch及多个epoch；`optimizer_minibatch_trajectories!=0`在hierarchical入口拒绝。
0.4.1直接TP2入口的多minibatch能力仍保留。通用DP多minibatch空副本/尾批同步、PP、动态区域/异步版本、packing、checkpointing、MoE等尚未覆盖。
不把这个受限同步基线叫做veRL所有功能已迁移。

## 多核与内存

`multicore.py`按N→M→batch确定矩形输出分区，K不拆；输出不重叠，FLOP/输出守恒；小输出允许核心空闲。
每个分区独立选择合法OS tile与私有SRAM工作集；各核算力并行，但共享DRAM和模块路径服务不能乘以256。
Matmul组串行，计算及本地服务串行，不隐含DMA重叠；BF16与FP32输入采用分别声明的GEMM rate。
原生C++检查每核workspace与并发占用，记录每核资源；总量够而单核超限仍拒绝。
同核并发请求串行、不同核可同时准入的独立native fixture不依赖local_memory排他锁。

每个native rank仅读取自己的资源文件。严格rank顺序下读集合以base/add/remove无损编码，私有workspace数组按SHA256内容寻址去重。
原始完整合同保留在Python计划中独立回放；compact→expand必须逐项相同，raw/encoded真实Astra ticks必须相同。
节点成本按内容hash去重写入cost_profiles.json；operations引用profile，不删除shape或因果检查。
资源日志采用等价的稀疏/稠密每核计数；所有JSON/原始日志/配置/hash随证据保存。
显式跟踪完整transport缓冲和padding，Astra未暴露的内部碎片/额外缓冲仍不声称覆盖。

## 原生通信组与时钟

Chakra节点`pg_name`绑定输入groups.json中的精确成员；未知/不匹配组拒绝。
TP对与DP同shard组分别使用上游原生group ring；6×9物理mesh用已有透明XY扩展、无wrap，Link/Chunk原生队列负责争用。
小collective按group FP32元素和上游total_nodes最小传输约束显式padding；logical payload和transport payload分开。
DP网络与本地reduce/endpoint服务分别计时，采用保守串行，不把collective内部算术默认为免费。
模型ns→Chakra us/raw tick scale仍为1000，所有issue/callback按严格整数tick核对，不由Python另造网络时钟。

## GPU采集与比较

`gpu_smoke.py`实际运行一张空闲A100上的veRL单step；32prompt×2回答、response cap4、microbatch1、GRPO demean、token-mean、epoch1、无activation checkpoint。
这是明确设置的短跑，不是把原训练负载事后cap4。collector同时保存实际逐条prompt/response长度、group/behavior、未四舍五入的timing_raw和解析配置。
模型权重/config、数据、训练源码、collector、Python头文件及包版本入报告。运行日志保留失败，不因rc0缺训练记录而判成功。
GPU首step有JIT和框架冷启动、4token回答奖励可能全0；不代表学习质量，也不能作为正式稳态基线。

复现GPU容器需：只暴露已获授权的空闲GPU；rl-xy读写挂载、RLInfra只读挂载；Python3.12头文件与平台pyconfig头来自服务器可核验路径；
为无passwd UID设置LOGNAME/USER及独立缓存路径。只在专用容器内建立/root/RLInfra/scripts、/RLInfra别名及私有collector/flag，退出删除flag。
不改变HOME、不激活主机全局flag，不用sudo，不删除/停止其它用户进程。
GPU4被其它任务占用时只停止本任务容器并改用空闲5；不把他人的显存占用误诊为芯片或训练代码结果。

`prepare_matched_case.py`导入全部实录group，拒绝模型hash失配；compile保留原ID映射和所有原长度，不读取GPU秒数作为chip cost。
`compare_gpu_wafer.py`核对模型/trace hash、每trajectory长度/group、完整step、无cap改动、归一化/epoch/更新/padding/checkpoint设置和PASS证据。
匹配通过后也只输出**单步GPU观测 vs 未校准晶圆场景预测**，不称实测晶圆speedup；GPU1卡与54die资源预算也并不相等。
host/reward成本仍是场景假设，DNN/可信profile与硬件校准仍未完成。正式论文主表需要多步稳态、多种子、资源口径和独立校准。

## 命令

```bash
python3 -m rl_xy.cli run-wafer --case configs/wafer54_fixture/case.json --scenario configs/wafer54_fixture/wafer.json --out runs/wafer/<唯一ID>
python3 scripts/private_sram_fixtures.py --out runs/wafer/<唯一fixtureID>
python3 scripts/prepare_matched_case.py --gpu runs/gpu_matched/<GPU_ID> --template configs/wafer54_fixture --model-config <原模型config> --out runs/wafer/<CASE_ID>
python3 -m rl_xy.cli run-wafer --case runs/wafer/<CASE_ID>/case.json --scenario configs/wafer54_fixture/wafer.json --out runs/wafer/<RUN_ID>
python3 scripts/compare_gpu_wafer.py --gpu runs/gpu_matched/<GPU_ID> --wafer runs/wafer/<RUN_ID> --out runs/comparison/<唯一ID>
```

上述原生仿真命令在wafer Docker镜像内运行，镜像ID和源码树hash由manifest固定；GPU训练在专用GPU容器运行，不能互换环境身份。

## 大型证据的存储

完整24层/64trajectory的逐对象资源合同会产生GB级证据，不能用系统盘剩余空间估计fixture规模。
本轮首次完整回放因系统盘不足实际失败（Errno28），报告和日志保留，不能计入通过。
源码仍在家目录；运行证据可以放在当前UID拥有的独立NVMe目录，并将项目`runs`链接到该目录。
迁移前后逐文件比较，不合并覆盖既有目录，不删除其它用户数据；`RLXY_RUNS_ROOT`显式指定真实runs根。
`docker.sh`检查所有者、项目链接和精确读写bind，将外部路径以相同绝对路径挂入新容器，避免host/container的Path.resolve身份分歧。
设置该变量后不能误复用缺少此挂载的旧容器；验收报告同时记录实际证据根目录。
完整模型的debug日志也会超过原生默认10份轮转保留上限。wafer镜像将10MiB分段的保留数明确扩为1000，并把Logging.cc哈希与二进制哈希一起记录。
这是日志完整性扩展，不改变仿真时钟；解析器仍要求全部分段连续、每个issue/callback恰好一次，不能放宽覆盖条件来接受被轮转丢失的日志。
首次完整原生回放虽完成事件检查，交付入口漏写summary.json、父验收又遭SSH stdout断连，因此该轮总验收未通过。
后续修正为先写统一机器摘要再封存manifest；4/54die fixture立即检查全bundle完整性后才允许昂贵大模型回放。
进度输出断连只关闭控制台输出，不影响已经持久化的命令退出码/日志hash和后续门禁；旧失败报告及旧源码身份保留，不能给旧bundle补文件后追认。
