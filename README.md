# rl-xy

基于 AstraSim 的晶圆级 RL post-training **性能仿真与执行合同验证框架**。当前版本 0.9.5，主路径是同步 GRPO：同一软件策略分别映射到 48 张 A100 模型与 54-die 晶圆 H0 模型，计算、状态移动、通信和内存生命周期显式计费。

当前完成首组 **Qwen2.5-3B** 严格控制：384×8＝3072 条固定 P512/R128 合成轨迹，训练 TP2/DP24/mb4 → 生成 TP1/DP48，48 active；DO 开启，SP/overlap 关闭。

| 指标 | 48×A100 模型 | 晶圆直接迁移模型 |
|---|---:|---:|
| 模拟一步时间 | 16.503234 s | 19.176605 s |
| 仿真程序墙钟 | 659.886 s | 696.424 s |
| 每 rank 的声明 DRAM 峰值 | 49.1775 GB | 49.1775 GB |
| 完整运行／独立 MVP／严格配对 | PASS | PASS |

本控制中晶圆耗时高 **16.20%**。这是受限显式 IR 和声明成本下的结果；没有数值训练或学习质量结果，不称最优/default veRL、实体晶圆部署或已校准硬件加速。两边相同的软件策略不自动代表最强 GPU 基线。

## 从这里开始

| 文档 | 解决的问题 |
|---|---|
| [构建与复现](docs/REPRODUCING.md) | Docker、测试、小例、完整 3B 命令、预算与结果检查 |
| [架构](docs/ARCHITECTURE.md) | 分层职责、执行链、资源合同、硬件与时间口径 |
| [实验设计](docs/EXPERIMENT_DESIGN.md) | 模型/算法/数据/切分/硬件/统计参数及逐项理由、来源 |
| [实验结果](docs/RESULTS.md) | 最新 3B 阶段结果、证据、失败和论文使用范围 |
| [文件索引](docs/FILE_MAP.md) | 每个源码、配置、脚本、测试的职责 |

只保留上述当前文档入口。两份 0.6 历史实验 MD 保留原字节，按原提交理解；历史数字不与本版混用。

## 最小检查

Linux、Docker，宿主不需要 sudo 或 GPU；镜像构建需要网络。正式性能运行使用受限 runner。

```bash
bash scripts/build_images.sh
make test
python3 scripts/check_public_results.py
python3 scripts/check_docs.py
```

完整构建链最后生成 `rlxy-astrasim:0.9.5`。本仓库的文档/分析整理提交与历史 a9 镜像是不同身份；重建必须保存新 image/source SHA，不能声称取得原 image ID。构建环境与现成镜像复现见复现文档。

```text
case/model/policy/hardware → typed算子与张量图
  → GPU热GEMM查表/解析模型 或 wafer核/存储映射
  → Chakra traces + 资源合同 → 原生AstraSim调度/网络
  → issue/callback与资源日志 → 独立验收 → 有范围限制的结果
```

`src/` 是实现，`configs/` 是输入，`docker/` 是固定后端与透明扩展，`scripts/` 是运行/校验/兼容工具，`tests/` 是回归，`calibration/` 是微基准标签，`experiments/` 只保存轻量可审核证据。大日志、依赖缓存、模型权重和凭据不入 Git。

下一阶段先补齐 3B 实际 layout/fusion/FlashAttention、SP/overlap、GPU 选优与 host/网络/整步验证及代表性长响应，再扩展 7B/14B/32B 和晶圆优化。自建代码 MIT；AstraSim/Chakra 按各自许可获取。
