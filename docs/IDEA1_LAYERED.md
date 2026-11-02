# Idea1 分层控制器与训练长尾接管

当前实现将既有生成协作、提前训练和训练微批接管接入分层控制器。4B、8B、14B完整test1与旧完整机制精确一致；固定输入、放置、任务分配、成本和准入条件，不把分层迁移当作加速来源。

| 模型 | GPU分时/s | H0晶圆分时/s | 原生成协作/s | 提前训练固定分配/s | 旧完整机制/s | 新分层控制器/s | 新旧差/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4B | 132.965929 | 162.371769 | 107.622598 | 108.835799 | 80.824606 | 80.824606 | 0 |
| 8B | 178.511874 | 245.501036 | 187.004415 | 186.396312 | 149.550215 | 149.550215 | 0 |
| 14B | 260.754746 | 348.755848 | 285.635449 | 276.267177 | 226.353700 | 226.353700 | 0 |

三档各192 prompt×8回答、1536样本。H0为54die、每die64GB/256core、DRAM与D2D 2TB/s，27个推理TP2和27个训练TP2副本；九个六die域是准入域，不是九个DP副本。GPU为既有48 A100条件模型。本表是单窗口条件仿真，不是实体硬件或学习质量测量。

## 分层与执行

- `HierarchicalController`统一编排计划复核、STAGING/激活、事件去重与版本检查、候选、仲裁、命令提交及一次更新/发布屏障。
- `NativeContext`管理原生事实；角色与helper/idle活动、执行阶段分别记录。`FixedPlanner`只导入原分组/FIFO与完整GRPO归属，不实现资源搜索或历史预测。
- `RoleScheduler`、`RegionController`、`RolloutCooperator`、`TrainingCooperator`分别处理角色、域/数据准入、attention帮助和未开始训练微批接管。
- `Eligibility`、`ResourceManager`、`LegacyOrderArbiter`保持原安全边界、容量和分支优先级；`NativeWorker`只执行批准动作。
- `NativeKernel.step()`禁用旧整体调度，调用会报错；复用旧attention局部内核、逐批算子和原生通信，不更改数学/容量/网络模型。
- 前置reference重建和最后梯度规约/优化器/发布留在同一原生DAG。框架确认前置就绪、全部任务及原生覆盖后发布，不额外执行或省略一次更新。

旧策略兼容模式保留：六die域全部释放才进入准备；训练微批接管等全局EOS且接收组完成自身队列；同域生成协作和原距离/收益规则。PENDING回答接管、资源规划、历史配额、异构TP和放宽旧准入规则都未启用。后续策略必须单独消融，不能用本表声称这些能力已完成。

4B/8B训练接管分别98/155微批，14B为260微批；生成后平均空闲4.789/2.391/1.217die。容量上界60.803732/55.851907/62.332500GB/die。角色占用包括访存、通信与等待，不是算术利用率。

## 复跑

先按[原生后端说明](IDEA1.md#准备原生后端)准备ASTRA。已冻结的算子程序仍来自[联合证据](../experiments/2026-11-02-idea1-joint/README.md)。

```bash
python -m pip install -e '.[f2]'
python scripts/prepare_idea1_joint.py --model 4B --output /tmp/layered-cost-small
python scripts/prepare_idea1_joint.py --model 14B --output /tmp/layered-cost-14b
export ASTRA_SIM_ROOT=/path/to/astra-sim
export ASTRA_SIM_BIN="$ASTRA_SIM_ROOT/build-dynarl/bin/AstraSim_Analytical_Congestion_Aware"
PYTHONPATH=/tmp/layered-cost-small/src python scripts/idea1_layered/run_layered.py --model 4B --training-scope wafer --native-root "$ASTRA_SIM_ROOT" --output /tmp/layered-runs/4B
PYTHONPATH=/tmp/layered-cost-small/src python scripts/idea1_layered/run_layered.py --model 8B --training-scope wafer --native-root "$ASTRA_SIM_ROOT" --output /tmp/layered-runs/8B
PYTHONPATH=/tmp/layered-cost-14b/src python scripts/idea1_layered/run_layered.py --model 14B --training-scope wafer --native-root "$ASTRA_SIM_ROOT" --output /tmp/layered-runs/14B
python scripts/audit_idea1_layered.py /tmp/layered-runs/14B
python scripts/check_idea1_layered.py
```

输出目录必须不存在。先用`--subset-prompts 1`检查原始8样本；完整实验不带此参数。执行时保留prepared成本目录在PYTHONPATH，控制器包与成本源分离；prepared工具逐项核对原187成本/配置文件，不把新控制器混入数学基座。

## 证据与范围

[发布证据](../experiments/2026-11-02-idea1-layered/README.md)包含Excel/CSV、三档结果、控制与样本/微批事件、native issue/callback日志和54rank统计的压缩副本，并含旧对照证据。独立检查验证1536样本、冻结程序、容量、原生覆盖及精确对照；基础框架16项测试在常规测试集中执行。

原始运行身份以历史源SHA为准，发布路径适配与脱敏身份分开保存。本次发布不冒充重新执行历史完整实验。未完成实体晶圆、Transformer数值梯度、GPU同机制与多窗验证，既有物理/论文性能边界不因发布解除。
