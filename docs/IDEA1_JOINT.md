# Idea1：生成长尾协作与提前训练

Idea1 将 KV 局部性感知的生成长尾协作与空闲资源提前准备/训练统一调度，优化完整 RL step。主实验为 H0 54 die、九个六 die 域、每域三个 TP2；192 prompt × 8 回答的完整 test1。GPU 为既有 48 A100 条件模型。

| 模型 | GPU 分时共置 / s | H0 分时共置 / s | 仅生成协作 / s | 完整 Idea1 / s |
|---|---:|---:|---:|---:|
| 4B | 132.966 | 162.372 | 107.623 | 108.836 |
| 8B | 178.512 | 245.501 | 187.004 | 186.396 |
| 14B | 260.755 | 348.756 | 285.635 | 276.267 |

较仅生成协作，4B 慢 1.13%、8B 快 0.33%、14B 快 3.28%。这是单窗条件仿真；不证明实体晶圆性能或学习质量，也不预设完整方案一定优于消融。

## 资源与执行语义

- **长尾协作**：完成自身任务的 TP2 可作为同域 helper，与源 TP2 组成 TP4 attention；FFN 留在源 TP2。显式搬运 attention 权重和 KV，保留租约并记录增量 KV。局部性估计使用当前上下文、完成历史和物理距离，不读取未完成请求真实剩余长度。
- **提前训练**：六 die 域的生成/helper 租约全部结束、训练组完整 GRPO 数据 EOS 且实际传到、容量通过后，该组进入 old policy、reference、advantage 和训练。默认零固定预留。域级准入是保守实现，不等于每个刚空闲的 die 都能立即训练。
- **单次同步更新**：actor/reference 本轮版本固定，全部样本处理一次，按全局 1,536 样本口径累加/归一化，最后 DP 规约、优化器和版本发布一次。单轮内部重叠不等于跨轮异步 RL。成本回放不执行数值梯度。

4B/8B 沿用冻结基线的完整 actor/reference/梯度驻留，参数分别为训练微批 4/2、策略微批 8/1、生成并发 32/16。14B 使用训练/策略微批 1、并发 16，reference 分片在生成前通过两次 27-rank ALL_GATHER 重建并完整收费；本组 reference 计算后释放临时副本，再分配梯度。容量上界为 60.795/55.844/62.325 GB/die，均低于 64 GB。

三模型保留原 grouped-core GQA 复用假设，包含重排、并行度和 SRAM 约束；该假设未经过实体晶圆校准。helper 租约期间暂不服务跨组样本传输，训练组间也不迁移剩余微批。这两点仍会形成下游等待和训练尾部空闲，见[瓶颈分析与下一步](IDEA1_INSIGHTS.md)。

## 发布内容与历史身份

[实验目录](../experiments/2026-11-02-idea1-joint/README.md)包含全部汇总表、Excel、三模型事件、原生日志/计时账本和逐批算子程序的 gzip 副本。4B/8B 来源于 T125，14B 来源于 T118；本次发布不冒充重新运行三档完整实验。原始文件 SHA 与发布文件 SHA 分开保存；含本机路径的字段替换为 `<historical-workspace>`，数值、输入和模型条件不变。

已编译的逐批程序随仓库分发，因此无需原工作区的 baseline 日志才能运行完整联合控制器。复用的是算子描述，不是阶段耗时；所有计算、样本传输与集合通信在新的 ASTRA 共享时钟中重新执行。控制器和成本工作区分别冻结，发布路径适配不会回填历史源码身份。

## 准备与复跑

先安装本仓库 `.[f2]`，按[既有原生后端说明](IDEA1.md#准备原生后端)准备包含 sequence/RPC/P2P 扩展的 ASTRA 二进制。完整固定上游与补丁保留在仓库内；执行时显式提供二进制，不回退到其他后端。

```bash
python -m pip install -e '.[f2]'
python scripts/prepare_idea1_joint.py --model 4B --output /tmp/idea1-small
python scripts/prepare_idea1_joint.py --model 14B --output /tmp/idea1-14b
export ASTRA_SIM_ROOT=/path/to/astra-sim
export ASTRA_SIM_BIN="$ASTRA_SIM_ROOT/build-dynarl/bin/AstraSim_Analytical_Congestion_Aware"
PYTHONPATH=/tmp/idea1-small/src python scripts/idea1_joint/run_joint.py --model 4B --native-root "$ASTRA_SIM_ROOT" --output /tmp/idea1-runs/4B
PYTHONPATH=/tmp/idea1-small/src python scripts/idea1_joint/run_joint.py --model 8B --native-root "$ASTRA_SIM_ROOT" --output /tmp/idea1-runs/8B
PYTHONPATH=/tmp/idea1-14b/src python scripts/idea1_joint/run_joint.py --model 14B --native-root "$ASTRA_SIM_ROOT" --output /tmp/idea1-runs/14B
python scripts/audit_idea1_joint.py /tmp/idea1-runs/4B
python scripts/audit_idea1_joint.py /tmp/idea1-runs/8B
python scripts/audit_idea1_joint.py /tmp/idea1-runs/14B
```

输出目录必须不存在；程序拒绝覆盖。`--subset-prompts 1` 可先跑原 trace 的小例；`--no-early` 是关闭提前准入的检查开关。正式复现必须不带子集参数，保留完整 test1。原始正式运行的二进制 SHA 为 `3e21f5ee4374aee825e6987d43216ed507b62747c5268852de3315167b8152fe`；用新编译二进制得到的时间必须记录新身份。

```bash
python scripts/check_idea1_joint.py
```

该命令对公开压缩证据检查原生 issue/callback 覆盖、样本因果、逐批程序一致性、容量和状态积分；不重新仿真、不放行物理精度。发布适配另通过 4B/14B 原始小例的真实原生执行和独立审计，两种成本工作区分别核对 187 个文件；不把小例当正式性能结果。

## 约 30B 扩展核查

当前公开长度来源为 [samuki-hf/thinking-rollouts](https://huggingface.co/datasets/samuki-hf/thinking-rollouts)，revision `a726a001d0145eddb2d5a2f5ede60591950b141f`。本轮逐项核对 370 个文件与 832 个递归目录项，模型标签只有 Qwen3 1.7B、4B、8B、14B 的 think/nothink；未发现 27–34B 分区。因此没有约 30B 同源实验，不用 14B 长度冒充 32B，不把 30B-A3B MoE 当稠密 30B。

[目录核查数据](../experiments/2026-11-02-idea1-joint/trace-scale-audit.json)保留 revision、所有模型标签和分区路径。结论只针对当前来源，不代表其他公开数据不存在。未来若获得匹配模型的真实长度/终止原因，先确认同一 DS-1000 prompt/sample 子集、模型配置、thinking/温度和容量可行，再扩展 GPU/H0/消融/完整 Idea1 四组实验。
