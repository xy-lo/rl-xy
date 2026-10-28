# 构建、复现与检查

当前命令针对0.9.5同步控制。所有完整仿真使用CPU/Docker，不需要GPU；GPU微基准是独立可选测量。设计见[EXPERIMENT_DESIGN](EXPERIMENT_DESIGN.md)，结果与原包可用范围见[RESULTS](RESULTS.md)。

## 构建与快速检查

Linux上已有Docker访问权限，宿主Python≥3.10。公开构建入口默认基础镜像为`ubuntu:22.04`；可用`RLXY_BASE_IMAGE`指定本地镜像（服务器历史基础镜像为`pjlab/deeplearning:v2.2.3-cuda12.8.0-torch2.7.0`）。AstraSim固定commit在Dockerfile中。网络代理通过本地环境/构建参数注入，不把账号、密码或token写入仓库。

```bash
bash scripts/build_images.sh
make test
python3 scripts/check_public_results.py
python3 scripts/check_docs.py
```

构建链按0.1→mapped→inputs→execution→schedule→wafer→gpu→colocation→alignment→protocol执行，最终tag0.9.5。构建脚本保留兼容链，当前整理没有在新机器重复整条编译；服务器已有原0.9.5镜像，原生控制的实际验收见结果页。

宿主做纯Python检查可以先`python3 -m pip install -e .`，再`PYTHONPATH=src python3 -m unittest discover -s tests`。宿主无native/可选依赖的skip不算native验收。`make test`使用指定Docker镜像，镜像identity与输出必须记录。

## 小型原生例子

同样使用受限runner，先创建专属输出根。下面选择小型aligned fixture，不缩减或替代正式3B W1。

```bash
stamp=$(date -u +%Y%m%dT%H%M%SZ)
output_root="$PWD/runs-home-T42"
mkdir -p "$output_root"
python3 scripts/bounded_run.py \
  --name "rl-xy-T42-smoke-$stamp" --image rlxy-astrasim:0.9.5 --image-source \
  --runs-root "$output_root" --record "$output_root/smoke-$stamp" \
  --memory-gib 8 --cpus 4 --pids-limit 512 --timeout-s 180 \
  --max-output-mib 128 --minimum-free-gib 24 --disk-reserve-gib 20 \
  python3 -m rl_xy.cli run-aligned \
  --case configs/aligned_fixture/case.json --scenario configs/aligned_fixture/wafer.json \
  --policy configs/aligned_fixture/policy.json --active 4 --out "runs/smoke-$stamp/result"
```

`--image-source`执行镜像内源码；修改宿主代码后想验收该代码必须先构建新镜像或明确采用runner默认只读host-bind模式并记录新源hash，不能混淆两种身份。唯一目录/容器名防覆盖；不会删除旧容器或旧证据。

## 运行身份与分析身份

0.9.5-a9 源码归档包含 325 个文件，SHA-256 为 `ce974a594c91db9eff9c74d75e2ae4b070a44bcf873cb5a344f206afb2438fe9`；冻结测试为 297 项、3 项跳过、整体 PASS。实际 `rlxy-astrasim:0.9.5` 的 image ID 为 `sha256:8847f0dec790d5349586bede4a7bd316ec831b79fe12a93b8f26dd8123f60101`。它运行镜像内源码，仅挂载专属输出目录；不能用当前 mirror 的文档或分析文件替换这一历史身份。

固定 stdlib MVP checker 的实际副本为 analysis-tools-a3，source SHA 为 `f8921b2d72d64de7bd24f8d9f83ebaec357a0e0fe657b0de7cb402d280f855db`。它在原包外写 `verification.json`，绑定运行 manifest SHA 与 checker source SHA。后加的 `compare_native_replay.py`、更新后的 `compare_aligned_results.py` 及其测试属于分析层；分析版本变化不需要重建仿真镜像，也不回填原包。两侧独立 PASS 只满足单包前提；配对必须另核同一工作负载、策略、成本域、有效通信、idle例外与两份验收绑定。最终配对及有范围限制的比值见结果页。

Linux、Docker、Python≥3.10 和固定 AstraSim commit `4ce9ecfe746a3ee5fe47559a11d91644d641fa7f` 是运行依赖。`Dockerfile.protocol` 从 0.8.0 编译资源 hook 并复制协议源码；完整旧依赖链仍由 `build_images.sh` 保留，该脚本已将最终 protocol tag 统一为0.9.5。若要从核对后的冻结源码另行构建，应给新镜像独立标签并保存新 image ID；不能把一次重建自动视为 a9 原镜像。

```bash
# 在已核对的源码目录中；此命令会建立新的镜像身份。
docker build --network host --file docker/Dockerfile.protocol --tag rlxy-astrasim:0.9.5-rebuild .
```

本轮没有在另一台干净主机重新完成依赖链构建。现成镜像的身份可只读检查：

```bash
docker image inspect rlxy-astrasim:0.9.5 --format '{{.Id}}'
```

## 已完成的 W1 a9 与原始证据

两侧原输入 `case.json`、`model_config.json`、`policy.json` 逐字节相同：3B、384 prompts×8 responses、P512/R128固定长度合成控制、训练TP2/DP24、microbatch4、rolloutTP1。SP/overlap/offload/checkpoint关闭，保留完整3072 trajectories和393216 response tokens；不截短长度来换取成功。

| 记录 | GPU `W1-gpu-hot-mb4-a9` | wafer `W1-wafer-mb4-a9` |
|---|---:|---:|
| wrapper / native exit / OOM | PASS / 0 / false | PASS / 0 / false |
| wrapper实际墙钟 | 659.885713 s | 696.424223 s |
| 原指标模型时间 | 16.503233595006 s | 19.176605455346 s |
| allocated / active | 48 / 48 | 54 / 48 |
| operations / rank events | 381907 / 754584 | 381908 / 754590 |
| 原wrapper artifact bytes | 1610242369 | 2052010060 |
| 固定独立MVP | 9组PASS | 9组PASS |
| MVP实际墙钟 / 最大RSS | 111.459757 s / 929349632 B | 110.397441 s / 932823040 B |

wafer多出的一个`inactive_rank_marker`节点精确覆盖48–53六个idle rank；不得静默删去未知阶段或改变总时钟口径。两边native DRAM峰值均为49,177,503,088 B。上表分别记录事实，没有计算比值，不能替代严格配对放行。

原运行在 `/home/xuyin.p/rl-xy/runs-home-T42/T42/`。完整原包没有删除或修改；包外专属export的完整tar.gz、全部raw文件、双端SHA/inventory和独立验收保存在工作区：

- `outputs/rl-xy-paper-experiment-20261028/evidence/W1-gpu-a9-independent/`：220源文件、217 artifacts、109 inputs；完整包133310486 B，SHA `9163687b53dee169ccd54d291ecb73b0ad140815f43f9c056ec95a3389b9362c`。
- `outputs/rl-xy-paper-experiment-20261028/evidence/W1-wafer-a9-independent/`：236源文件、233 artifacts、119 inputs；完整包144322267 B，SHA `96b13675b8b859f547acb8df0e90d96bac60dcf2aac0a7c536a6a21273823641`。

这些完整大包不等同于Git仓库内容，仅clone工程不能宣称已经取得或重放了原始a9。安全解压拒绝absolute、`..`、symlink、hardlink、special成员；双端分别核对manifest全量artifact/input SHA及精确inventory。原包、压缩包和包外分析分开保留。

## 新运行的完整护栏与命令

`runs-home-T42` 必须是工程根下已有的真实专属目录，不能是symlink或任意路径。`bounded_run.py`还允许原`runs`解析后的既有目录；本轮不改旧NVMe `runs` symlink。record必须严格包含于选定artifact root，root不能是工程根、`/`或祖先。`--image-source`只挂载输出，保留镜像内冻结源码。

T42实际限制为 **4 CPU、8 GiB、memory-swap同为8 GiB（无额外swap）、PID512、无网络/无GPU设备、整条900 s、每candidate 2048 MiB（2147483648 B）**；启动前可用至少24 GiB、运行时持续reserve20 GiB。启动阈值是显式T42参数，runner默认30 GiB没有调低。采样护栏触发时立即kill这个自有容器，保留失败原因和部分包；不把OOM、timeout或artifact超限填成性能结果。

Linux runner内部 **planner180 s / encode180 s / native300 s / post180 s**。planner含唯一模板签名准备；post含解析、验证和后续证据写出，`parse_and_validation_seconds`只计其中一个子区间，不能把四个可见计时列简单相加当wrapper总墙钟。a9原记录如下：

| 原始计时键 | GPU秒 | wafer秒 |
|---|---:|---:|
| planner_including_unique_signatures_seconds | 137.987749 | 165.563319 |
| serialization_and_trace_seconds | 90.522526 | 94.225518 |
| native_seconds | 264.777234 | 249.827152 |
| parse_and_validation_seconds | 98.581891 | 114.824836 |

下列命令会产生新证据，必须在确认成本/空间后手动执行；本次文档任务未执行。输出ID使用时间戳，不能重用a9目录。`engineering_regression`保留其工程口径；`formal_baseline`另需同时通过runtime/cost gate，不能只凭配置合法改标签。

```bash
RLXY_REPRO_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RLXY_REPRO_RUNS="$(pwd)/runs-home-T42"
mkdir -p "$RLXY_REPRO_RUNS"
python3 scripts/bounded_run.py \
  --name "rl-xy-T42-W1-gpu-repro-${RLXY_REPRO_STAMP}" \
  --image rlxy-astrasim:0.9.5 \
  --image-source \
  --runs-root "$RLXY_REPRO_RUNS" \
  --record "$RLXY_REPRO_RUNS/T42/W1-gpu-repro-${RLXY_REPRO_STAMP}" \
  --memory-gib 8 --cpus 4 --pids-limit 512 --timeout-s 900 \
  --max-output-mib 2048 --minimum-free-gib 24 --disk-reserve-gib 20 \
  --purpose engineering_regression \
  python3 -m rl_xy.cli run-aligned \
  --case configs/aligned_3b/W1-tp2-mb4/case.json \
  --scenario configs/wafer54_fixture/wafer.json \
  --policy configs/aligned_3b/W1-tp2-mb4/policy.json \
  --gpu-profile configs/gpu/a100_48_W1_mb4_hot.json --active 48 \
  --out "runs/T42/W1-gpu-repro-${RLXY_REPRO_STAMP}/result"

python3 scripts/bounded_run.py \
  --name "rl-xy-T42-W1-wafer-repro-${RLXY_REPRO_STAMP}" \
  --image rlxy-astrasim:0.9.5 \
  --image-source \
  --runs-root "$RLXY_REPRO_RUNS" \
  --record "$RLXY_REPRO_RUNS/T42/W1-wafer-repro-${RLXY_REPRO_STAMP}" \
  --memory-gib 8 --cpus 4 --pids-limit 512 --timeout-s 900 \
  --max-output-mib 2048 --minimum-free-gib 24 --disk-reserve-gib 20 \
  --purpose engineering_regression \
  python3 -m rl_xy.cli run-aligned \
  --case configs/aligned_3b/W1-tp2-mb4/case.json \
  --scenario configs/wafer54_fixture/wafer.json \
  --policy configs/aligned_3b/W1-tp2-mb4/policy.json --active 48 \
  --out "runs/T42/W1-wafer-repro-${RLXY_REPRO_STAMP}/result"
```

新包的独立验收输出必须放在bundle外；默认单JSON value/native JSON cap为512 MiB，不因整份unified大于512 MiB而提高该cap。实际checker用流式全顶层语义重编码，只保留当前rank对象对及原MVP索引；它不保证所有数据规模RSS恒定。固定a3在两份完整a9实际测得峰值均小于1 GiB，300秒外层timeout未触发。

```bash
sha256sum scripts/check_aligned_result.py
python3 scripts/check_aligned_result.py \
  "$RLXY_REPRO_RUNS/T42/W1-gpu-repro-${RLXY_REPRO_STAMP}/result" \
  --out \
  "$RLXY_REPRO_RUNS/T42/analysis-gpu-${RLXY_REPRO_STAMP}/verification.json"
python3 scripts/check_aligned_result.py \
  "$RLXY_REPRO_RUNS/T42/W1-wafer-repro-${RLXY_REPRO_STAMP}/result" \
  --out \
  "$RLXY_REPRO_RUNS/T42/analysis-wafer-${RLXY_REPRO_STAMP}/verification.json"
```

checker自身命令没有300秒timeout选项，a9本地通过外层进程组runner实际限制300秒，并用`/usr/bin/time -l`记录RSS；其脚本与完整stderr保留在独立验收目录。不要声称上面裸checker命令自身含有这一外层超时。

## 无损优化与尚未放行的证据

read-ID符号表只压缩重复storage字符串，旧格式仍能展开；C++严格拒绝bool/float/越界索引。资源validator用完整向量值、cores、当前私有capacity复用dense/sparse/range/peak派生工作；每node仍核workspace总和，每原始行仍核真实C++ snapshot，合法重叠和浮点算术沿用旧路径。证据导出用浅层只读dataclass字段和三层外壳流式/叶值C编码，canonical字节/SHA保持一致。资源日志64KiB缓冲只影响主机I/O，正常结束的tiny A/B日志逐字节一致；异常kill留下的部分包仍按失败处理。

固定MVP包括完整artifact/input SHA、表格存储/解压SHA/bytes/rows、统一/每rank资源合同语义、事件矩阵/compute ticks/deps、模板/invocation证明及结束绑定；不包括raw debug对导出事件全重解析、完整对象/私有SRAM ledger独立重放、原逻辑IR/native macro源重建、成本校准或学习质量。三者不能互相替代。

W1只有291个canonical NN hot-cuBLAS GEMM标签；transpose/stride/fusion/FlashAttention、非GEMM、网络及整步held-out校准仍缺。热标签加解析成本不是完整GPU测量，更不是最优/default veRL配置。`hardware_calibrated`、`numerical_training`、`paper_performance_ready`在a9均为false；强GPU论文主表继续受独立门禁约束。


## 配对、公开摘要与原始日志

```bash
python3 scripts/compare_aligned_results.py \
  --gpu "$RLXY_REPRO_RUNS/T42/W1-gpu-repro-${RLXY_REPRO_STAMP}/result" \
  --wafer "$RLXY_REPRO_RUNS/T42/W1-wafer-repro-${RLXY_REPRO_STAMP}/result" \
  --gpu-verification "$RLXY_REPRO_RUNS/T42/analysis-gpu-${RLXY_REPRO_STAMP}/verification.json" \
  --wafer-verification "$RLXY_REPRO_RUNS/T42/analysis-wafer-${RLXY_REPRO_STAMP}/verification.json" \
  --out "$RLXY_REPRO_RUNS/T42/pair-${RLXY_REPRO_STAMP}.json"
```

`check_public_results.py`只检查仓库轻量证据、输入/原manifest/MVP绑定及汇总算术，不会因缺少大原包而假装重放成功。完整原包存在时再运行单包MVP与配对。默认512MiB是单JSON value上限，不按统一大文件总大小放宽。原始包安全解压须拒绝absolute、`..`、symlink/hardlink/special成员，并按原manifest核全量SHA。

历史0.6兼容入口是`bash scripts/reproduce_v06.sh unique-run-id`；其协议、资源和旧结果单独解释，不用于当前严格3B主表。历史报告生成器只为原版本兼容，不是维护当前6份文档的入口。
