# H0：用户原表、论文来源与几何边界

本轮唯一新H0输入为 `configs/hardware/xy_h0_wsc_case3.json`。历史
`xy_h1h2h3_54.json` 保留，不再当作本轮H0。schema hardware/2在来源字段外
显式保存整体布局、模块阵列、计算die的几何值，不静默改尺寸或拓扑。

| 项目 | H0保留值 | 转换/边界 |
| --- | --- | --- |
| 核/工艺/频率 | Dojo-style、7nm、1GHz | 不是实测芯片身份 |
| 总体/阵列 | 215×215、199.92×199.50mm | 分别保留，不互相替代 |
| 模块/计算die | 24.99×33.25、21.92×22.81mm | 计算die面积为相乘派生值 |
| 拓扑 | 6×9=54die | x=6、y=9只定义逻辑mesh坐标 |
| 每模块核心 | 16×16=256 | WSC-LLM §5.1.1；沿用户参考WSC设置 |
| FP16算力 | 1.02TFLOPS/core | 1020FLOP/ns，不是旧候选1024 |
| SRAM容量 | 1.25MB/core | 十进制1250000B，非带宽、非共享池 |
| DRAM | 64GB/die、2TB/s | 64000000000B、2000B/ns |
| D2D | 2TB/s，论文表中口径 | 数值已知；每边/每方向口径仍未确认 |

来源：用户2026-10-26给定H0表；WSC-LLM原论文
https://doi.org/10.1145/3695053.3731101 ，PDF第9页§5.1.1和第10页Table1 Case3。
论文的每核Tensor FP16、SRAM容量、54die、DRAM/D2D数值可核对；论文推理工作负载使用FP16，
不能据此推出BF16、FP32反向/优化器向量吞吐。§5.1.1的四方向合计6TB/s互联描述，
不直接当作SRAM带宽、模块NoC带宽或Table1每条D2D边的有效吞吐。

## 几何不一致的验收

原表模块固定尺寸下，199.92=8×24.99、199.50=6×33.25，对应48模块。
54模块按6×9得到149.94×299.25mm，旋转每个模块得到199.50×224.91mm。
两种都不是原阵列占地，也都放不进215×215mm边界；模块面积总和也超过声明阵列面积。
报告应拒绝“几何已可实现”，但不把逻辑拓扑擅自减为48。这里没有加入原表未给定的
300mm晶圆或26×33mm曝光场约束；不声称几何布局是用户最终封装实现。

## 明确保留未知项

SRAM/NoC带宽、BF16/向量吞吐、D2D时延/packet、NIC缓冲、host链路及晶圆直径均保持null。
外部历史配置中的10TB/s/die SRAM是另一56die设计的C级假设，不是本H0来源。
可以另外定义显式场景假设做敏感性，但不能回填成本表为“论文值”。
当前完整执行后端仍TP2、单逻辑核、主路径FullyConnected；H0检查不会绕过能力门禁。

```bash
python3 -m rl_xy.cli inspect-hardware --profile configs/hardware/xy_h0_wsc_case3.json
# 下条应rc2：来源完整性检查不等于当前后端能执行H0。
python3 -m rl_xy.cli inspect-hardware --profile configs/hardware/xy_h0_wsc_case3.json --require-executable
```
