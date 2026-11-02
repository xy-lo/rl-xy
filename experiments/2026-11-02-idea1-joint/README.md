# 完整 Idea1 联合实验与可复核证据

方法与命令见[完整 Idea1](../../docs/IDEA1_JOINT.md)，分析见[关键路径与改进](../../docs/IDEA1_INSIGHTS.md)。每模型完整 test1、1,536 条轨迹、54 活动 die，原生退出 0。三模型完整 step 为 108.835799/186.396312/276.267177 秒，仅为单窗条件成本仿真。

## 数据与身份

- `comparison.csv`：GPU/H0/仅生成协作/完整 Idea1 四方案。
- `joint-metrics.csv`：阶段时间、批量、容量、空闲、传输及有效 P+R 吞吐。
- `die-snapshots.csv`：76 个时刻六类角色，时间以生成开始为零点；每行合计 54。
- `stage-events-summary.csv`、`worker-timings.csv`：15 个阶段汇总和 81 个固定训练组的时序。
- `all-data.json`、`critical-waits.json`：统一数据、原工作区相对来源与关键样本等待证据。`provenance.source` 保留历史相对引用；发布证据位于当前目录各模型子目录。
- `Idea1三模型实验汇总.xlsx`：时间与参数、重叠与资源、die 状态三张表。
- `trace-scale-audit.json`：公开源当前目录核查，未发现约 30B 模型。
- `bottleneck-analysis.json`、`bottleneck-bounds.csv`：固定生成/服务时间的回溯分析，不是新调度器仿真。

4B/8B 是 T125 完整原生运行；14B 是 T118 冻结运行。三个模型子目录保留结果、审计、容量、命令与逐批摘要，并以 `*-program.json.gz` 分发完整冻结策略/训练算子程序。`baseline-metadata.json` 记录原输入与成本源码身份，供准备工作区与复跑时核验。

每个 `raw/` 中保留压缩的控制器/占用轨迹、逐批训练事件、样本传输、helper/KV 事件、native operations、原生完整 debug 日志和 rank 结束账本。`historical-identities.json` 同时记录原始 SHA 与发布解压后的 SHA；本机路径替换为 `<historical-workspace>`，替换不改变数值。`publication-manifest.json` 绑定当前发布字节，不伪称与未脱敏文件相同。

逐 rank protobuf 工作负载、成本映射缓存、编译目录和原生二进制不入仓库；控制器、冻结逐批程序、固定输入、H0 源码补丁和原生构建补丁可生成新的运行。原始正式二进制身份在各模型结果中记录。物理晶圆时延、数值梯度和学习质量不因事件检查通过而获证。
