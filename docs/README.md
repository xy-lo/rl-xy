# 文档导航

当前公开版为0.6.0。仓库首页与本页描述当前能力；带旧版本标题或旧验收ID的文档只说明对应阶段。

| 阅读目的 | 入口 |
| --- | --- |
| 本版实验设置、四组结果、证据与局限 | [0.6实验结果](../experiments/2026-10-27-v0.6/实验结果.md) |
| 当前文件职责索引 | [FILE_MAP.md](FILE_MAP.md) |
| 从干净环境构建与复现 | [REPRODUCING.md](REPRODUCING.md) |
| 分层架构 | [ARCHITECTURE.md](ARCHITECTURE.md) |
| 54die、多核、DP×TP与张量生命周期 | [wafer_execution.md](components/wafer_execution.md) |
| 独立GPU模型与节点共享上行 | [gpu_simulation.md](components/gpu_simulation.md) |
| 算子映射、工作集及本地成本 | [mapped_costs.md](components/mapped_costs.md) |
| 资源合同与实际准入 | [tensor_resources.md](components/tensor_resources.md) |
| 原生链路竞争 | [network.md](components/network.md) |
| H0来源及几何解释的演进 | [09_H0_SOURCE_AND_GEOMETRY.md](09_H0_SOURCE_AND_GEOMETRY.md)；当前Case3 scope见硬件配置 |
| 证据等级与禁止的外推 | [EVIDENCE_BOUNDARIES.md](EVIDENCE_BOUNDARIES.md) |

0.1/0.2生成的`工程文件与完整流程.md`、`资源版工程说明与完整流程.md`及对应HTML保留作历史说明，不能替代0.6结果。架构文档中关于两die、congestion-unaware的旧限制须结合当前wafer/GPU组件阅读；当前主流程已使用扩展的原生mesh/shared-uplink排队。

公开整理只调整入口、构建/复现脚本及个人环境参数，不重新定义已归档的实验身份。原始大日志、模型权重、数值训练环境和认证信息不进入Git；轻量摘要的可验证范围见实验报告。
