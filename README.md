# rl-xy

基于 AstraSim 的 RL post-training 性能仿真框架。当前版本 **0.10.0**，新增 Qwen3-4B、8B、14B 公开长度回放，以及 48×A100 模型与 48 活动 die 晶圆模型的同步 GRPO 配对。

**当前结果尚未完成硬件计时校准，论文性能主结论未放行。** 下表是冻结成本模型的可复算输出。GPU 使用解析 SM/CTA 模型；晶圆使用 core 私有 SRAM 分块和共享带宽模型。成功执行、容量检查与求和一致性不等于已证明真实硬件精度或 GPU 全空间最优。

| 模型 | 平台 | 设备单步时间：均值 ± SD（s） | 有效吞吐（token/s） |
| --- | --- | --- | --- |
| Qwen3-4B | GPU | 234.05 ± 10.41 | 22,084 |
| Qwen3-4B | 晶圆迁移 | 202.11 ± 10.49 | 25,574 |
| Qwen3-8B | GPU | 328.04 ± 17.00 | 16,091 |
| Qwen3-8B | 晶圆迁移 | 291.58 ± 16.51 | 18,103 |
| Qwen3-14B | GPU | 446.03 ± 14.65 | 9,994 |
| Qwen3-14B | 晶圆迁移 | 359.11 ± 11.74 | 12,413 |

每窗口 192 prompt×8 回答，四个正式测试窗口。时间为设备侧完整 RL 单步时间；SD 是窗口负载波动。有效吞吐为四窗口 Σ(P+R)/Σ设备时间，单位 token/s；不同于在线 serving 的请求 TPS。三档每条轨迹均在 GPU—晶圆之间严格配对。

![当前模型的时间与吞吐](experiments/2026-10-29-qwen3/figures/baseline_normalized.png)

## 从这里开始

| 文档 | 内容 |
|---|---|
| [结果与可靠性](docs/RESULTS.md) | 绝对数值、图、阶段分解、已知问题及历史数据身份 |
| [实验设计](docs/EXPERIMENT_DESIGN.md) | 模型、公开 trace、窗口、策略、硬件与指标的五点设计 |
| [构建与复现](docs/REPRODUCING.md) | 纯 Python 校验、原生后端、单候选与冻结策略配对回放 |
| [架构](docs/ARCHITECTURE.md) | RL 阶段、die/core/tile 粒度、容量、通信和计时口径 |
| [文件索引](docs/FILE_MAP.md) | 实现、脚本、配置与兼容入口职责 |

## 快速核对

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
python3 scripts/check_public_results.py
python3 scripts/check_qwen3_results.py
python3 scripts/check_docs.py
```

这些命令不需要 GPU。纯 Python 校验检查公开输入、历史结果身份、阶段账本与统计公式；原生小例另见复现文档。完整历史原始日志、编译依赖、模型权重和本地研究工作区不进入本仓库。

```text
公开长度 + 官方模型配置 + 同步协议
  → 分阶段算子/矩阵形状与状态移动
  → GPU SM/CTA 或晶圆 core/tile/容量/带宽成本
  → 原生 AstraSim 网络和依赖执行
  → 阶段完成时间、账本与配对验收
```

自建实现为 MIT。AstraSim、Chakra 及数据/模型元数据使用各自许可；来源和固定 revision 见实验设计与发布数据中的 provenance。0.6 与 0.9.5 的历史小型控制及其原始身份仍保留，不能与当前三档数字混用。
