# Qwen3 f2 分时共置基座：结果、复现与后续

本仓库保留 f2 的 GPU 组件成本 profile、晶圆单缓冲成本后端、网络补丁、固定策略与三个模型 test1 结果；这是条件架构仿真数据，不是实体 48 GPU 或目标晶圆的端到端计时。用户接受此模型作为后续工作的当前基座；相关假设及未测范围依然必须随结果披露。历史 0.10.0 四窗口使用旧成本模型，不能与 f2 拼表。

| Qwen3 | GPU step/s | 晶圆 step/s | GPU 策略 trainTP/rollTP/MB/logMB/cap | 晶圆策略 |
|---|---:|---:|---|---|
| 4B | 132.965929 | 159.060157 | 4/4/16/16/128 | 4/4/8/16/128 |
| 8B | 178.511874 | 208.237818 | 4/4/8/16/128 | 4/4/4/16/64 |
| 14B | 260.754746 | 266.746950 | 8/8/8/16/128 | 不变 |

每档都是相同 test1 的192 prompts × 8 responses。GPU选策为 tune 上有限网格快速模型的首名，不称原生或硬件全局最优；晶圆只按冻结容量规则调整，未按晶圆时间选择策略。有效吞吐为Σ(P+R)/step；stage、精确结果与容量策略在 `experiments/2026-10-29-qwen3-f2/`。test2–4 未在 f2 上执行，14B 的2.30%差距不能作为显著物理胜负。

本仓库可用 `python3 scripts/search_f2.py --model 4B --output runs/f2-new/4B-tune` 进行新的有限网格解析筛选，包含并发128；它不证明原生或物理最优。本次发布未完整重做三档筛选，历史选策依据仍是原始实验记录。

## 用当前仓库运行

```bash
python3 -m pip install -e '.[f2]'
python3 -m unittest discover -s tests -v
```

固定 AstraSim 上游 revision 为 `4ce9ecfe746a3ee5fe47559a11d91644d641fa7f`；在干净 checkout 上应用 `experiments/2026-10-29-qwen3-f2/native/native-combined.patch`，复制 `native/headers/*.hh` 到 `astra-sim/workload/`，并复制 `native/packet_fixture.cc` 到 `build/astra_analytical/`。该补丁包含原 Qwen3 改动和 f2 分包改动，**不要额外叠加旧 Qwen3 patch**。所需上游依赖和历史子模块见 [原生构建说明](REPRODUCING.md)。建成 congestion-aware 后端后显式指定：

```bash
export ASTRA_SIM_ROOT=/path/to/f2-patched-astra-sim
export ASTRA_SIM_BIN=/path/to/f2-patched-astra-sim/build/bin/AstraSim_Analytical_Congestion_Aware
python3 scripts/replay_f2.py --model 4B --platform gpu --window test1 --output runs/f2-new/4B-gpu
python3 scripts/replay_f2.py --model 4B --platform wafer --window test1 --output runs/f2-new/4B-wafer
```

`ASTRA_SIM_BIN` 必须指向实际构建出来的 f2 二进制。runner 显式设置数值 packet 64 KiB、主机超时7200秒，真实目标工作量耗时较长；输出是新源码/资产/二进制身份，不能冒充原历史运行身份。原生补丁在固定上游的适用性、本轮使用既有 f2 后端的最小验证以及完整 350+ 单测结果见发布记录；干净跨平台完整重建与六点重放不在本次声明范围。

## 身份与模型边界

历史运行源树内容 SHA `798fea6b769764bc028cd520f40053d6b95d965679f308751a5957c675f82315`；原本机 bundle SHA `2f52a646a0ac71b6e071f8eb4622e3e67142922534fe47588f8694d6a2093df4`；原 native binary SHA `7f13bf22147f779bb6fd3fe69e5cf5a7b22e305ae1a7420a8b061e95cb2816c0`。本仓库更新了源码、测试、相对路径并清理 profile 的本机路径字段，形成新身份；原历史 hash 仅用于溯源，不会声称本仓库的字节等于原 bundle。`assets-manifest.json` 绑定仓库内可复算资产和输入哈希。

GPU 的本地 GEMM/FA 使用 A100 组件测量，协议只在 TP2 的部分 collective 上测量；其他 rank、跨节点和 GATHER/BCAST 存在外推/假设。晶圆成本模型为256核/die，每核1.25MB私有 SRAM，方向 SRAM 400/270 GB/s/core、D2D 100ns 与其余带宽/算力为已声明的代理或假设；无实体 WSC 延时校准。64 KiB 是模拟数值分组，不是硬件 flit。三档差异来自整个模型和容量适配，不强加预期胜负。

## 下一步 DynaRL 与晶圆优化

从 DynaRL 论文复现动态资源划分、阶段重叠、依赖与竞争；当前 `run_candidate` 对九个同步阶段分别模拟后求和，不能直接当成 DynaRL 的重叠时间线。先让新调度器的无重叠退化例对齐同步基线，再分别比较 GPU 动态调度、晶圆迁移基线和晶圆优化。固定相同请求、精度、计时器和资源模型；不要把 f1→f2 成本模型修订算成优化方法收益。完整原始日志、报告层失败与修复及历史 source 快照保留在独立实验工作区，不在本仓库冒称已全部分发。
