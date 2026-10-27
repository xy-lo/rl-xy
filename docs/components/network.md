# 网络适配与竞争fixture

主toy是固定两die TP2，不宣称已经扩展为任意规模晶圆。其native collective/P2P仍使用analytical congestion-unaware后端，因而主toy指标不支持拥塞性能结论。`communication_lowering.py` 显式展开源DRAM读→native网络→目标DRAM写；native网络只负责传输，局部服务不重复计入网络时长。

## 二维mesh扩展

固定上游commit的congestion-aware helper只支持一维拓扑。`docker/astrasim/RlxyMesh.hh` 新增矩形mesh与dimension-order XY路由，`mesh.patch` 只在明确设置 `RLXY_NETWORK_TOPOLOGY=mesh2d` 时选择该拓扑。每个相邻节点间各有两个有向Link，不跨边界wrap-around；编号 `id=x+y*width`，路由先改变x再改变y。

传输、序列化、Link排队和callback继续使用上游原生Device/Link/Chunk/EventQueue，未另造Python网络时钟。runner仅允许patched congestion-aware binary接收 `network_model='mesh2d'`；默认清除override，避免环境污染。扩展来源与patch/header hash由镜像marker保存。

输入复用NetworkParser的两维数组读取尺寸、带宽与时延。由于原parser无Mesh枚举，YAML使用 `Ring` 作为维度字段载体，**显式mesh2d override才定义真实邻接**，stdout和fixture execution.json均记录override，不能仅凭YAML标签解释实际网络。原生unaware二维Ring仅在本测试的2×2情况下邻接与mesh相同；不能推广到3×3等无wrap mesh的无拥塞对照。

## 反事实测试

`scripts/integration_fixtures.py` 使用4rank、2×2邻接，8192B probe 0→3（XY路径0→1→3）。比较solo、独立路径2→0和共享末跳1→3，在aware/unaware两个二进制各执行一次。检查每个rank实际完成、每个节点issue/callback覆盖和依赖，再比较同一probe从源issue到目标callback的时差，不用两条不同长度flow的总wall time冒充拥塞差。

第二条flow源端固定延迟512 model ns，使其1024ns序列化区间与probe第二跳重叠。该backend为store-and-forward；同时启动反而可能在第二跳错开，不能把“路径相交”自动当成时间上争用。门禁要求：aware共享链路probe变慢，独立链路probe不变；unaware三者相同。原始ET、network/system、argv、exit、binary/source/image hash和日志全部保留，mesh stdout另输出实际路由路径。没有Comm节点的rank不要求不存在的Comm统计，但仍要求所有计划节点和rank完成；主流程parser不因此放松。

## 限制

这是原生链路排队模型上的小拓扑验证，不是已校准的wafer NoC、虚通道、router arbitration、backpressure或电路级模型；不证明大规模网络或任意TP/PP/DP工作流正确。扩展到更多die时仍需扩workload/config/mapping合同、collective路由和held-out拓扑测试。
