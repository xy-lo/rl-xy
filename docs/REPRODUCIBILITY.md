# 复现与审计

0.2资源版使用 `docker/Dockerfile.mapped` 在固定0.1镜像上编译两个后端及透明扩展。执行 `bash scripts/docker.sh build-mapped rlxy-astrasim:0.2.0`，然后使用独立容器 `rl-xy-mapped`。正式门禁为 `python3 scripts/resource_acceptance.py --container rl-xy-mapped --run-id <ID>`，包括实际硬件参数敏感性、容量失败、原生资源与mesh竞争。报告由 `scripts/write_resource_report.py --acceptance acceptance/<ID>/report.json` 生成。旧 `acceptance.py` 专用于0.1兼容路径；两个版本结果不能混在同一性能对比中。

## 1. 唯一正式环境

正式结果必须由项目Dockerfile构建的镜像产生（基础版 `docker/Dockerfile`，资源版另叠加 `docker/Dockerfile.mapped`）。宿主机 Python 或手工安装的 AstraSim 可用于只读排查，但其结果不能写成正式 run。

镜像的上游版本合同和本次实际环境：

- Dockerfile 默认基础镜像为 `ubuntu:22.04`（本次未验收此默认构建）；本服务器实际采用 `pjlab/deeplearning:v2.2.3-cuda12.8.0-torch2.7.0`，实际系统 Ubuntu 24.04；
- official AstraSim commit：`4ce9ecfe746a3ee5fe47559a11d91644d641fa7f`；
- analytical backend：`congestion_unaware`，只用于无拥塞 analytical 假设，不模拟共享链路的多流争用与排队；
- Protobuf C++ toolchain：`29.0`；Python runtime：`5.29.0`；
- 项目包版本：见 `pyproject.toml`。

Docker 脚本不包含 `sudo`。运行账户必须已经能访问 Docker daemon；项目不会代替管理员修改用户组、socket 权限或 daemon 配置。

同一不可变 image ID 与源码/配置 hash 是本次复现基准。基础镜像 tag、apt 和 pip 工具的获取不是位级锁定构建，不能承诺未来从网络重建得到相同 image ID；本次成功的实际 ID 保留在 manifest 中。

## 2. 从干净目录运行

在项目根目录执行：

```bash
RLXY_BASE_IMAGE=pjlab/deeplearning:v2.2.3-cuda12.8.0-torch2.7.0 ./scripts/docker.sh build rlxy-astrasim:0.1.0
./scripts/docker.sh test rlxy-astrasim:0.1.0
./scripts/docker.sh transfer rlxy-astrasim:0.1.0 repro-001
./scripts/docker.sh toy rlxy-astrasim:0.1.0 repro-001
./scripts/docker.sh two-rounds rlxy-astrasim:0.1.0 repro-001
```

也可用 Makefile：

```bash
make docker-build
make test
make transfer RUN_ID=repro-001
make toy RUN_ID=repro-001
make two-rounds RUN_ID=repro-001
```

`run_id` 应唯一且能关联实验记录。不要复用已有目录，也不要删除旧 run 后制造同名结果。

## 3. 门禁顺序

1. `test`：纯逻辑和负向测试；
2. `transfer`：可手算数据移动 fixture，检查 AstraSim 时间缩放、数据路径和容量失败分支；
3. `toy`：完整单轮同步 GRPO 流程；
4. `two-rounds`：版本连续性和轮间资源释放。

只有前一层通过后才应解释后一层指标。单元测试通过不能替代 AstraSim 运行；transfer fixture 通过也不能替代完整 toy case。

## 4. run bundle 最低内容

每个 run 目录至少包含：

| 文件 | 用途 |
|---|---|
| `manifest.json` | run ID、时间、代码/依赖版本、输入和产物 hash、退出状态索引 |
| `astra_config_contract.json` | system/network/remote-memory 与 case 单位合同的解析结果 |
| `resolved_config.json` | 本次真正生效的完整配置 |
| `parameters.csv` | 参数分片和复制规则 |
| `objects.csv` | 对象身份、版本、位置与生命周期 |
| `operations.jsonl` | planner 生成的规范 operation 序列 |
| `events.jsonl` | 从 AstraSim trace log 解析的各参与 rank/node issue/callback 事件 |
| `astra_rank_summary.csv` | 每 rank completion/Wall/compute-lane/communication 原始计数 |
| `memory.csv` | 各位置内存快照与容量 |
| `transfers.csv` | collective/p2p 的结构化记录；本地 DMA 仍是 operation |
| `checks.csv` / `checks.json` | 每项 expected、actual、evidence、scope 与 pass/fail |
| `metrics.json` | 机器可读指标与明确口径 |
| `summary.json` / `SUMMARY.md` | 状态和汇总指标；仅作索引 |
| `status.json` | 本次 run 的 running/success/failed 终态 |
| `astra_stdout.log` | 未改写的标准输出 |
| `astra_stderr.log` | 未改写的标准错误 |
| `astra_command.json` | 实际 argv、环境入口和进程退出码 |
| `astra_logs/log.log` | AstraSim DEBUG issue/callback 原始日志 |
| `inputs/` | 输入配置副本与 `<prefix>.<rank>.et` trace |

Chakra ET、AstraSim debug log 或其他中间证据若由本次运行生成，也应保留在 run 目录的明确子目录中并纳入 manifest hash。

## 5. 完整性检查

运行后执行：

```bash
python3 scripts/check_result.py runs/<case>/<run_id>
```

这个脚本复核：必需文件存在、manifest/summary/status 三种状态一致、check 非空且全通过、counter 一致、输入/metrics/全部产物的 SHA-256 与完整文件集合一致；成功时打印 `manifest.json` 的 SHA-256。它不是第二个仿真器，也不替代 `checks.json` 中的语义验证。

人工复核至少包括：

```bash
sed -n '1,200p' runs/<case>/<run_id>/astra_command.json
sed -n '1,240p' runs/<case>/<run_id>/summary.json
sed -n '1,260p' runs/<case>/<run_id>/checks.json
tail -n 80 runs/<case>/<run_id>/astra_stdout.log
tail -n 80 runs/<case>/<run_id>/astra_stderr.log
```

复核 timing 时，Astra stdout 的 `GPU time` 必须记作 compute-lane occupancy，而不是硬件 GPU 利用率。T01 的 RECV 事件可能从提前 posted 开始计时；链路服务时延应检查 SEND issue→callback。publish/cleanup 的 1 ns 来自 `control_marker_ns`，属于 synthetic control cost。

## 6. 重复运行比较

比较两个 run 时，先比较 resolved config、项目版本、AstraSim commit、Docker image tag 与 image ID、命令。只有这些相同时，才可把 completion、事件计数、通信字节或峰值内存当成同配置重复性检查。

若结果不同，按以下顺序定位：

1. manifest 中的输入 hash；
2. operation/trace hash；
3. 实际 AstraSim argv 与配置副本；
4. 每 rank completion 与 issue/callback 事件；
5. parser/validation 版本；
6. reporter 的聚合口径。

不要直接比较两个 `summary.json` 的少数字段后宣称回归。

## 7. 失败处理

- Docker daemon 不可访问：停止，不切换到宿主机“正式运行”，也不在脚本中加入 sudo。
- 镜像构建失败：保留 build log；不使用来源不明的已有容器替代。
- AstraSim 非零退出：归档 stdout/stderr/argv/退出码，run 标记失败。
- rank 或 event 缺失：run 标记失败，不用 planner 估时补齐。
- 容量、生命周期或版本检查失败：保留证据，修复实现后创建新 run ID。
- 文档中的“实测结果”只能引用通过全部门禁的 run bundle。

