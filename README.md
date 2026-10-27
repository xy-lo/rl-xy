# rl-xy

基于固定版本AstraSim的晶圆级RL post-training性能仿真框架。当前版本0.6.0，支持同步GRPO算子/张量图、54die多核晶圆、独立48-A100模型及可追溯的近预算比较。

**当前实验四组均GPU更快。** 本项目提供未校准的性能预测和实现合同验证，不是实体晶圆数值训练后端，也不证明学习质量、功耗或面积优势。

| 入口 | 内容 |
| --- | --- |
| [本版实验结果](experiments/2026-10-27-v0.6/实验结果.md) | 模型、负载、硬件、四组结果、来源、验收与局限 |
| [文档导航](docs/README.md) | 当前组件与历史文档的阅读顺序 |
| [文件职责表](docs/FILE_MAP.md) | 全部公开源码、配置、脚本与实验摘要文件 |
| [构建与复现](docs/REPRODUCING.md) | 完整Docker构建链、小例和64轨迹复现 |

## 当前结果

Qwen2.5-0.5B结构，32个完整group×2回答、64条真实长度轨迹，prompt为70–127 token、回答实际均4 token；一次同步更新。晶圆DP27×TP2，GPU DP24×TP2，两侧保留相同全step有效token分母与更新目标。

| 场景 | 晶圆预测ms | GPU预测ms | 当前模型结果 |
| --- | ---: | ---: | --- |
| 主对照 | 195.475331 | 129.083868 | GPU更快 |
| GPU计算采用更乐观设置 | 195.475331 | 75.417170 | GPU更快 |
| 晶圆FP32/向量率敏感性 | 221.357898 | 129.083868 | GPU更快 |
| 联合敏感性 | 221.357898 | 75.417170 | GPU更快 |

GPU名义密集FP16峰值高6.209%、显存容量高11.111%，其它资源差异单列；没有把旧单卡42.55秒冷首步按48卡缩放。当前wafer仍采用固定分区与OS32分块，GPU在4种合法CTA间选择；双方都没有完整kernel/硬件校准，这不是架构最优性能排名。

## 执行链路

```text
模型结构 + 完整轨迹长度 + 算法合同 + 硬件/策略
  → 同步GRPO操作DAG与张量生命周期
  → wafer私有核SRAM/共享DRAM成本 或 GPU SM/CTA/HBM成本
  → 每rank Chakra execution trace与资源合同
  → 原生AstraSim：wafer mesh 或 GPU node-switch/shared-uplink/spine
  → 原始issue/callback日志、资源准入与事件验证
  → 来源/负载/预算门禁 → 四组条件性比较
```

计算与本地存储成本来自声明公式/参数，通信时序和排队由原生后端执行。未知reward/logprob不伪造；数值训练只曾用于独立历史长度采集，仿真不产生可评价学习质量的权重。

## 快速开始

Linux上已有Docker访问权限；不使用宿主sudo，不需要GPU。

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
bash scripts/build_images.sh
bash scripts/docker.sh test rlxy-astrasim:0.6.0
bash scripts/reproduce_v06.sh my-unique-run
```

`build_images.sh`按依赖顺序构建0.1→0.6；这些是本地镜像tag。完整实验生成的大日志放在被Git忽略的`runs/`和`acceptance/`，运行需要充足磁盘与内存。小例、外置runs挂载及错误恢复见[复现说明](docs/REPRODUCING.md)。

## 目录

| 目录 | 职责 |
| --- | --- |
| `src/rl_xy/` | 严格配置、模型/算子/张量/资源合同、映射、Astra适配、验证与报告 |
| `configs/` | 小例、H0来源、54die执行假设及GPU参考/敏感性配置 |
| `docker/` | 固定上游构建、C++资源hook、mesh和GPU网络扩展 |
| `scripts/` | Docker运行、采集导入、分层验收、复现与比较工具 |
| `tests/` | 合同、守恒、负例和比较门禁回归 |
| `docs/` | 当前组件、设计边界、历史演进与文件索引 |
| `experiments/2026-10-27-v0.6/` | 本版结果MD、公开长度输入、来源hash和轻量结果JSON |

## 验证与剩余工作

历史完整0.6验收：112项测试、20条命令、21项native断言、5个新bundle通过。归档下载后从原始日志独立重解析1,645,238条事件，时钟与全部资源峰值一致；四组比较与四份模型分解独立重算一致。公开仓库只带轻量摘要与原始归档hash，不包含全部历史原始日志；验证边界见实验报告。

待完成：容量感知wafer分区/tile/复用、双方并行策略搜索、更长负载、GPU融合/缓存与可信成本校验、NIC有限队列/细NoC/DMA、通用多更新场景，以及非零学习质量/稳态和实体设备runtime。不能将本例闭环标记为全部veRL实体迁移完成。

代码独立实现，MIT许可。上游AstraSim及其依赖按各自许可获取，未将其完整源码或论文PDF复制入本仓库。
