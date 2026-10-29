# 独立接手：RL × 晶圆级实验与 DynaRL

本文面向**无法访问原研究者本地工作区**的智能体或研究者。所需代码、公开数据和本文均从 GitHub 获取；文中的仓库路径都相对于 `rl-xy/`。不要假设能读取其他电脑上的 `outputs/`、`scratch/`、Zotero、远程 GPU 或历史聊天。

## 1. 项目目标和接下来要做什么

研究问题：将大模型 RL post-training 的计算、通信与资源调度迁移到晶圆级架构时，哪些地方形成瓶颈，晶圆感知的资源组织和调度能否改善效率？研究对象是 **Qwen3-4B、8B、14B** 的 RL 计算与通信时间，而不是训练出模型权重或比较学习效果。

用户已经接受目前的 **f2 条件架构仿真模型和分时共置结果**作为下一阶段的共同基座。下一任务的优先目标是实现并评估 **GPU 上的 DynaRL 动态调度对照**，随后设计和验证晶圆级优化。最终目标有四个对象：

| 对象 | 当前状态 | 作用 |
|---|---|---|
| GPU 分时共置 baseline | 三个模型 test1 已完成 | 静态系统对照 |
| GPU DynaRL 动态调度 | **尚未实现** | GPU 上的动态调度对照 |
| 晶圆分时共置迁移 baseline | 三个模型 test1 已完成 | 将 GPU 方案在晶圆上按容量适配 |
| 晶圆优化方案 | **尚未设计和实现** | 在同一晶圆模型上验证方法收益 |

不要预设晶圆迁移必须慢，也不要调整配置以强行让优化方案获胜。GPU 内的方法比较应共用 GPU 硬件与计时规则；晶圆内的方法比较应共用晶圆硬件、算存和网络模型。跨硬件比较要声明两者不是等面积或等功耗配额。f2 公共计时器是四组实验的共同基础，**不能把旧计时模型换成 f2 的差额当成优化方案收益**。

## 2. 获取仓库和理解已有结果

```bash
git clone https://github.com/xy-lo/rl-xy.git
cd rl-xy
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[f2]'
python -m unittest discover -s tests -v
python scripts/check_docs.py
```

f2 代码基座来自提交 `f3f5b2a9142daae63f3884a9d2d99754625f6b59`；本文件可能在其后续文档提交中更新。执行前记录当前 `git rev-parse HEAD`。该版本干净检出上的本地结果为 **355 项测试、352 通过、3 跳过**，GitHub 的 [Checks 记录](https://github.com/xy-lo/rl-xy/actions/runs/40865569338)也已通过。Python 测试和下表结果**不需要 48 张真实 GPU**。

同一模型的 GPU 与晶圆使用完全相同的 `test1` 轨迹。每模型是 **192 prompts × 8 responses = 1536 条轨迹**；三档模型各自的输出长度不同，不能把跨模型秒数变化完全归因于参数量。

| Qwen3 | GPU 单步 / s | 晶圆单步 / s | GPU 有效 tokens/s | 晶圆有效 tokens/s |
|---|---:|---:|---:|---:|
| 4B | 132.965929 | 159.060157 | 36794.659 | 30758.400 |
| 8B | 178.511874 | 208.237818 | 28248.793 | 24216.279 |
| 14B | 260.754746 | 266.746950 | 16565.670 | 16193.538 |

**时间口径**：当前静态仿真将 weight sync、generation、samples、old policy、reference、advantage、training、optimizer、publish 九个同步阶段的模拟设备时间求和。**吞吐口径**：`Σ(每条轨迹 prompt tokens + response tokens) / 模拟 step 秒数`。不包含离线搜索、建图、编译、仿真器主机运行时间；不是在线请求 TPS，也不是实际训练质量。

结果和配置入口：

- `experiments/2026-10-29-qwen3-f2/results/baseline-f2-test1.csv`：上表完整精度及 token 数；其中 `gpu_raw_result` / `wafer_raw_result` 是**历史归档相对路径，不是仓库中的可访问文件**。
- `experiments/2026-10-29-qwen3-f2/results/baseline-f2-test1-phases.csv`：各模型、两平台九阶段及关键路径账本。
- `experiments/2026-10-29-qwen3-f2/policies/`：各平台固定策略和容量迁移规则。
- `experiments/2026-10-29-qwen3/inputs/Qwen3-{4B,8B,14B}/`：已发布的 `config.json`、`tune.json`、`test1.json` … `test4.json`。
- `experiments/2026-10-29-qwen3-f2/assets-manifest.json`：发布资产和上述输入的哈希，含原始实验身份信息。

输出长度有长尾并在 8192 token 处截断：test1 的响应 p50，4B/8B/14B 分别为 **2070/2325.5/1957**，p95 分别为 **8192/7948.5/6623.25**；`finish_reason=length` 分别是 **86/71/28 条**。这些是截断后观测值，不能推断截断前到底多长。

## 3. 基座建模与对照约束

| 项目 | 已声明的模型 |
|---|---|
| GPU | 48 × A100-SXM4-80GB 模型，6 节点、每节点 8 卡；单卡 HBM 约 2.039 TB/s、BF16 稠密峰值 312 TFLOPS；节点上行 400 GB/s 是模型假设。 |
| 晶圆 | 6×9 逻辑 mesh，其中 48 个 die 活动；每 die 256 core、64 GB DRAM；每 core **私有 1.25 MB SRAM**，每 die 合计 320 MB，不是共享 SRAM 池。 |
| 晶圆本地成本 | core 私有 SRAM、tile 和共享 die 带宽层次；GEMM 使用单输入 buffer、允许跨核自然算存重叠；SRAM 读/写按 400/270 GB/s/core 代理，D2D 时延 100 ns、每方向带宽按模型假设。 |
| 网络 | GPU 和晶圆都经 f2 AstraSim packet-streaming 后端；64 KiB 是数值分组，不是硬件 flit。 |
| GPU 校准 | A100 实测 GEMM/Attention 成本及 2 卡部分 collective 的协议校准；TP4/8、其他 rank、跨节点和 GATHER/BCAST 存在外推或假设。 |

晶圆的硬件是由论文与代理参数组成的**假想架构模型**，不是实体 Dojo 或其他某个 SKU 的完整复刻，也没有目标晶圆的端到端时延实测。GPU 实测组件误差不能当成 48 卡完整 RL step 的误差。用户接受这套模型作为研究基座；因此继续使用时要准确声明假设，不把它说成真实硬件端到端测量。

固定策略格式为 `trainTP / rolloutTP / train microbatch / logprob microbatch / rollout 并发`：

| 模型 | GPU | 晶圆迁移 |
|---|---|---|
| 4B | 4 / 4 / 16 / 16 / 128 | 4 / 4 / 8 / 16 / 128 |
| 8B | 4 / 4 / 8 / 16 / 128 | 4 / 4 / 4 / 16 / 64 |
| 14B | 8 / 8 / 8 / 16 / 128 | 相同 |

GPU 方案由 `tune` 上的有限网格解析 Screen 选出，是 **best-found 的固定方案**，没有证明原生全网格或真实硬件全局最优。晶圆只按预先冻结的五窗口容量规则调整，不按晶圆 test 耗时挑选。14B 的声明模型容量准入通过，峰值约 60.035 GB，低于晶圆每 die 64 GB 的限额。4B/8B 的调整不是免费优化，比较中必须保留。

三档晶圆 generation 分别比 GPU 慢约 **28.84/36.84/25.32 秒**；训练和 old/reference 阶段节省部分时间，所以总差较小。14B 总差仅 **5.99 秒、2.30%**，没有系统级误差界时不能据此认定实体硬件显著胜负。训练阶段的晶圆总时长较短也不等于纯 GEMM 更快：账本的“本地服务”还包含访存、通信端点和 GPU 协议服务。

## 4. 代码入口、原生后端和复现边界

- `src/rl_xy/qwen3_experiment.py`：当前静态 GRPO 阶段与 `run_candidate`；**各阶段分别运行并求和**。
- `src/rl_xy/qwen3_rollout.py`、`qwen3_sequence.py`、`qwen3_native.py`：FIFO 长度回放、紧凑序列及原生事件执行。
- `src/rl_xy/f2_runtime.py`、`gpu_calibration.py`、`communication_service.py`：发布版 f2 成本入口及 GPU 成本/协议；晶圆 provider 在 `experiments/2026-10-29-qwen3-f2/wafer/`。
- `scripts/search_f2.py`：只用 `tune` 做新的有限解析筛选，支持并发 128。它不是 DynaRL，也不提供物理最优证明；本次发布未完整重做历史搜索。
- `scripts/replay_f2.py`：在正确构建的 f2 原生后端上回放固定策略；`scripts/check_f2_compaction.py`：两平台小型原生集成核查。

正式回放需要 AstraSim 上游固定 revision `4ce9ecfe746a3ee5fe47559a11d91644d641fa7f`。以下命令在仓库根目录准备源码与补丁：

```bash
mkdir -p native
git clone https://github.com/astra-sim/astra-sim.git native/astra-sim
git -C native/astra-sim checkout 4ce9ecfe746a3ee5fe47559a11d91644d641fa7f
git -C native/astra-sim submodule update --init --recursive
git -C native/astra-sim apply "$PWD/experiments/2026-10-29-qwen3-f2/native/native-combined.patch"
cp experiments/2026-10-29-qwen3-f2/native/headers/*.hh native/astra-sim/astra-sim/workload/
cp experiments/2026-10-29-qwen3-f2/native/packet_fixture.cc native/astra-sim/build/astra_analytical/
```

该补丁已包含历史 Qwen3 变更和 f2 packet 改动，**不要再叠加旧 `qwen3-native.patch`**。原生构建的 Abseil、Protobuf、CMake、Boost 等要求见 [构建说明](REPRODUCING.md)。其中 `scripts/rebuild_qwen3_native_dependencies.py` 假设已有 `native/deps/bin/cmake` 等依赖，它不是一键安装所有系统依赖的脚本；按接手环境准备后再构建。若构建失败，记录具体依赖问题，不冒充原生验证通过。

构建成功后，将环境变量指向**实际构建出的 f2 congestion-aware 二进制**：

```bash
export ASTRA_SIM_ROOT="$PWD/native/astra-sim"
export ASTRA_SIM_BIN="/path/to/built/f2/AstraSim_Analytical_Congestion_Aware"
python scripts/check_f2_compaction.py --output runs/f2-smoke
python scripts/replay_f2.py --model 4B --platform gpu --window test1 --output runs/f2-4b-gpu
python scripts/replay_f2.py --model 4B --platform wafer --window test1 --output runs/f2-4b-wafer
```

这些是**新运行**，应记录新源码、数据、配置、资产、二进制与命令身份。输出目录必须全新。正式回放可能耗时较长；重试未完成阶段时保留失败日志，不把部分 network segment 当作已验证检查点。发布时用既有 f2 原生二进制验证过小例，并确认补丁应用到固定上游后与已用原生源码一致；**尚未在干净系统从零编译原生依赖，也未用发布后的新源码完整重跑六点**。

历史运行的 source tree SHA 为 `798fea6b769764bc028cd520f40053d6b95d965679f308751a5957c675f82315`，原 bundle SHA 为 `2f52a646a0ac71b6e071f8eb4622e3e67142922534fe47588f8694d6a2093df4`，原 native binary SHA 为 `7f13bf22147f779bb6fd3fe69e5cf5a7b22e305ae1a7420a8b061e95cb2816c0`。发布版修订了路径和测试，且清理了 profile 内本机路径，属于**新身份**；不能把新的 Git commit 或资产哈希回填成历史运行身份。GitHub 只分发结果摘要、成本资产、配置与补丁，**不含六个案例的完整原生事件日志、原始本机 bundle 或原生二进制**。如果需要逐事件独立审查，先确认能否获取原始归档；不能根据缺失文件声称已完成该审查。

## 5. DynaRL 下一任务：论文对齐、实现和验收

准确论文是 **DynaRL: Flexible and Dynamic Scheduling of Large-Scale Reinforcement Learning Training**，OSDI 2026：[USENIX 页面](https://www.usenix.org/conference/osdi26/presentation/wang-yuanqing)、[公开 PDF](https://www.usenix.org/system/files/osdi26-wang-yuanqing.pdf)。可同时查看 [RLinf 开源仓库](https://github.com/RLinf/RLinf)和其[动态调度指南](https://rlinf.readthedocs.io/en/latest/rst_source/guides/dynamic_scheduling.html)，使用时记录具体版本/commit，核对它与论文是否一致。论文使用真实 H100 集群及其工作负载；它的加速比不能直接填到当前 48 A100 / Qwen3 / 公开长度回放结果中。

论文的核心并不是简单改变一个固定 TP：动态超图记录组件和运行状态；调度器依据持续的资源过量/瓶颈信号，在 WorkerGroup、Worker、请求三个层级决定资源和数据分配；迁移需要安全中断点、状态转移及数据重分配。论文描述弱状态 rollout 的 workload 迁移、训练器的参数与优化器状态 P2P 重分片等不同机制。根据任务范围实现可追溯的仿真抽象；未实现的论文机制要标明，不能称为完整复现。

推荐按以下顺序推进：

1. **先读论文并检查公开实现**，形成论文机制到代码/仿真模型的对应表：调度触发、候选资源、迁移安全点、状态/通信成本、数据路由及不支持项。固定参考版本和建模范围，不直接复用论文的加速比。
2. **建立可重叠的 GPU 事件时间线**：rollout、推理和训练按数据依赖逐步就绪；同时约束实际分配的 GPU 数、显存、通信链路与共享资源。记录参数/优化器/KV、工作负载迁移和重新分片等需要计时的工作。不能把原静态九阶段时间简单求和后贴上 DynaRL 标签，也不能让不同阶段免费占用同一资源。
3. **先做退化与守恒验证**：关闭动态迁移/重叠时，与对应同步 baseline 的计算、通信、工作量、阶段依赖和时间一致；每个请求被处理且不重复，GRPO 组在需要时包含完整 8 条响应，状态版本、容量及资源占用一致。再用小型可手算案例验证动态迁移、并发、网络竞争及非零开销。
4. **运行三档 GPU DynaRL 对照**：沿用本仓库的 Qwen3、公开 `test1`、48 卡模型、精度和 f2 组件成本；对照原 GPU 分时共置。参数选取只用 `tune`，不能依据 test 结果迎合胜负。分别报告端到端/稳态指标口径、每阶段耗时、迁移次数与成本、资源占用和有效 tokens/s；若动态方法跨迭代流水，明确测量窗口与工作量分子，避免把同步单步和跨迭代吞吐误当成同一指标。
5. **再开展晶圆优化**：在同一晶圆硬件和底层模型上，与晶圆迁移 baseline 配对；用可解释的瓶颈及消融说明收益来源。保留 GPU DynaRL、GPU 静态、晶圆静态、晶圆优化四个对象，展示结果而非预定结论。若要报告四窗口均值，需要额外运行 `test2`–`test4`，不能用已有单窗冒充。

每组结果应留下：commit、源码/成本/后端身份、输入 SHA、模型与策略、硬件参数、命令、退出状态、阶段和汇总、运行日志及已知假设。小例通过不等于全量结果或真实硬件误差已验证。研究者真实可用 GPU 通常 **1–3 张 A100、最多 8 张**；48 卡是目标仿真配置，不能把 1–8 卡组件测量说成 48 卡实测。优先复用已有成本，不为细节无止境扩展仿真；需要真实硬件的补充验证应与正式仿真结论明确区分。

晶圆研究背景可参考 [WSC-LLM](https://doi.org/10.1145/3695053.3731101)、[WATOS](https://arxiv.org/abs/2512.12279)、[WaferLLM](https://www.usenix.org/conference/osdi25/presentation/he) 和 [TEMP](https://arxiv.org/abs/2512.14256)。这些论文的硬件、负载和验证方法不同，不能直接替换本项目假设或把它们的结果数值移作本项目实验结果。
