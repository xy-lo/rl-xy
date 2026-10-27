# 结果归档组件

## 职责

报告组件把一次运行的输入、计划、原始仿真证据、资源账本、验证结果和汇总指标写入唯一 run 目录。其目标是让第三方不依赖终端滚屏或人工记忆即可回答：运行了什么、用的哪个版本、命令是什么、是否成功、数字如何得到、原始证据在哪里。

## 输入

- run ID、case ID、UTC 起止时间；
- source config 与 resolved config；
- 项目版本/代码状态、Docker image 信息、AstraSim commit 和 binary hash；
- 参数、对象、operation、event、memory、transfer 记录；
- runner 的 argv、exit code、stdout/stderr/debug log；
- validation checks 和汇总所需的明确定义指标。

## 输出

写入 `runs/<case>/<run_id>/`：

- `manifest.json`：provenance、hash 和文件索引；
- `astra_config_contract.json`：Astra 配置与 case 的单位/字段交叉校验结果；
- `resolved_config.json`：实际生效配置；
- `parameters.csv`：参数拆分；
- `objects.csv`：对象身份与生命周期；
- `operations.jsonl`：规范计划；
- `events.jsonl`：Astra issue/callback；
- `astra_rank_summary.csv`：每 rank completion 与原始统计计数；
- `memory.csv`：物理快照与明确标记的逻辑视图；
- `transfers.csv`：collective/p2p 通信记录；本地 DMA 仍在 operations 中，不伪装成已测 transfer；
- `checks.csv`、`checks.json`：逐项验证的表格/结构化版本；
- `metrics.json`：带单位和口径的机器可读指标；
- `summary.json`、`SUMMARY.md`：状态与人机可读索引；
- `status.json`：running/success/failed 终态与时间；
- `astra_stdout.log`、`astra_stderr.log`：原始进程流；
- `astra_command.json`：实际 argv 和退出码；Astra commit/binary hash 在 manifest。

`inputs/` 保存 case、system/network/remote-memory 副本与每 rank Chakra ET；`astra_logs/log.log` 保存 debug 事件。全部文件都由 manifest 记录 hash。manifest 同时记录 Docker image tag 与不可变 image ID；正式运行缺少这两项会失败关闭。

## 写入规则

1. run 目录必须是新目录；禁止静默覆盖同名证据。
2. 原始日志按字节保留；解析后的 JSON/CSV 不能替代它们。
3. JSON 使用稳定键/编码，JSONL 一行一个稳定 ID 记录，CSV 首行固定 schema。
4. 输入、trace、配置、二进制和全部 bundle 产物使用 SHA-256；manifest 记录生成时所见值。
5. `summary.json` 只引用已经通过定义的指标，并包含单位、evidence kind 和检查状态。
6. `PASS` 只能由全部必需 checks 通过且 Astra 返回 0 推导；reporter 不接受调用者手工指定成功。

## 指标口径

允许汇总的指标包括固定配置下的 completion、每 rank wall、compute-lane occupancy、communication-node occupancy、exposed communication、operation/参与-rank event 数、按 `round.phase` 的时间/字节、物理峰值内存、KV reservation/materialization、通信 payload 和版本推进。AstraSim stdout 的原始 `GPU time` 只能映射为 `astra_compute_lane_busy_ns/ratio`，不得命名为硬件 GPU 利用率。

每个值必须说明是 AstraSim 输出、parser 差值、ledger 推导还是配置期望。配置期望不能混入“实测结果”字段。communication-node occupancy 可能包含 posted RECV wait；T01 的链路服务时延单独由 SEND issue→callback 导出。所有网络指标还必须注明 backend 为 congestion-unaware、未模拟链路争用。publish/cleanup 的 1 ns 必须标记为 synthetic control cost，不能并入硬件控制开销结论。

当前 toy 的 DRAM 峰值仅覆盖常驻状态与 rollout KV，不包括 activation tape/所有中间工作区；metrics 中用 memory_accounting_scope 明确记录。SRAM 初始零占用占位行不参与峰值汇总；尚未实现的主 toy 工作区峰值为 null。

## 不做什么

- 不自动提交 Git、推送远程或上传外部平台；
- 不写入密码、token、私钥、Docker socket 信息或完整环境变量；
- 不复制参考仓库旧结果；
- 不覆盖、清理或合并已有 run；
- 不从日志中挑选部分成功 rank 生成 `PASS`；
- 不以 summary 取代原始命令、退出码和日志。

## 失败条件

- run ID 冲突或目标目录已存在；
- 必需输入/日志缺失；
- 序列化失败或记录 ID 不唯一；
- hash 无法计算或写后复核不一致；
- Astra 非零退出或 checks 不全/失败却请求 `PASS`；
- 输出中检测到不应归档的凭据材料。

失败时应尽可能保留已有诊断证据，并明确标记失败；不得把不完整目录当作正式结果引用。

