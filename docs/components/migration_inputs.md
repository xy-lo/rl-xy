# veRL 输入、正式 H 与旧数据复用（0.3）

本组件将已有 veRL 负载证据、真实模型结构与用户 H 设计放入明确的数据接口。它输出可检查的输入、容量必要条件和当前后端缺口；当前原生完整执行仍是两 die/单逻辑核 toy。接口接通不等于 GQA、多核和正式晶圆执行已经完成。

## 模块与文件

| 文件 | 内容与功能 |
| --- | --- |
| `src/rl_xy/hardware_profile.py` | 严格读取带值、单位、来源、状态的 H；几何检查、聚合及执行能力检查；null 不转成零 |
| `configs/hardware/xy_h1h2h3_54.json` | 用户 H1–H3 候选：9×6模块、256核/模块、64GB DRAM/模块、1.25MB SRAM/核等；保留未定义参数 |
| `src/rl_xy/verl_trace.py` | 读取训练侧既有 run_meta/traj/stage_times JSONL；校验完整组、步覆盖、版本、长度和来源 |
| `src/rl_xy/model_import.py` | 读取真实 Qwen2配置，计入GQA、QKV bias和绑权参数；输出参数/KV规模与缺少的执行模板 |
| `src/rl_xy/legacy_workload.py` | 从原始日志提取白名单标量；核对Hydra参数；按明确参考口径生成统计约束长度 |
| `src/rl_xy/migration.py` | 连接 W、H、模型来源，输出模型状态容量必要条件和均匀head分片TP候选；不输出芯片时间 |
| `scripts/reuse_legacy.py` | 只读旧训练目录，核对raw log/steps.csv/summary/state，生成固定种子的合成长度及来源清单 |
| `scripts/input_acceptance.py` | 宿主编排Docker构建、单测、H负例、旧数据复用与原生资源流程回归；不启动GPU训练 |
| `scripts/remote_session.py` | 保活复用、从本机既有转发发现新端口、核验已知主机公钥；不保存密码，不盲目重试未知状态的写命令 |
| `docker/Dockerfile.inputs` | 继承已验证0.2原生镜像，加入0.3 Python输入接口；无需重编译Astra |
| `tests/test_migration_inputs.py` | 硬件来源/单位/几何/空值，trace分组/步/版本/缺失值，真实Qwen参数规模等检查 |
| `tests/test_legacy_workload.py` | 原始日志解析、统计可行性、确定性、精确整数矩、clip口径反例 |

## 可以不新跑 GPU 吗

可以。性能模拟需要工作量、模型结构、学习/更新规则、硬件和执行成本；每次运行不必真实计算所有张量。现阶段优先复用已有GPU日志，缺失逐轨迹记录时使用明确声明的统计约束合成数据。真实数值训练主要用于采集、语义对照与校准，不是离散事件模拟的固定前置步骤。

当前旧日志只有逐步长度min/mean/max、旧clip_ratio和阶段汇总等。它们无法恢复真实group内相关性、token IDs、逐轨迹reward/logprob、KV动态行为或完整学习过程。生成器只保留已知边际约束，随机种子和未知信息写入 `synthesis.json`。不将GPU阶段秒数写成晶圆算子成本。

## 旧 clip_ratio 的特殊检查

原始E1第71步配置上限512，观测最大503，仍记录clip_ratio=1/512；E1第112步最大498也有同样情况。原始E3也有多个最大长度低于2048而该比率非零的步骤。因此不能一律把旧字段解释为“达到配置上限/被截断的比例”。

本轮读取的 veRL `metric_utils.py` 将该指标参照值取为 `batch.batch["responses"].shape[-1]`，与动态padded宽度有关。当前源码不能独立证明历史版本完全一致，因此生成器要求显式选择 `--clip-reference public_cap` 或 `observed_max`。已有两组数据复用选择 `observed_max`，记录这是建模选择，并另外报告 `public_cap_hit_ratio_generated`。真实EOS/截断原因仍未知；不修改旧CSV或旧图。

## 命令

```bash
# 已存在的测试H执行仍保留。
RLXY_CONTAINER=rl-xy-inputs bash scripts/docker.sh toy-resource rlxy-astrasim:0.3.0 <新run-id>

# inspect成功表示合同可读取；require-executable会因明确缺口退出2。
python3 -m rl_xy.cli inspect-hardware --profile configs/hardware/xy_h1h2h3_54.json
python3 -m rl_xy.cli inspect-hardware --profile configs/hardware/xy_h1h2h3_54.json --require-executable

python3 -m rl_xy.cli import-verl-trace --trace <捕获JSONL> --origin framework_capture --out <新文件.json>
python3 -m rl_xy.cli import-legacy-run --log <旧原始日志> --overrides <归档overrides.yaml> --run-id <原实验名> --out <统计.json>
python3 -m rl_xy.cli synthesize-workload --statistics <统计.json> --seed 0 --first-step 1 --last-step 116 --clip-reference observed_max --out <新目录>
python3 -m rl_xy.cli preflight-migration --trace <生成lengths.jsonl> --origin statistics_constrained_synthetic --hardware configs/hardware/xy_h1h2h3_54.json --model-config <真实config.json> --out <新文件.json>

# 宿主执行，内部所有验收命令进入Docker；旧训练目录只读挂载。
python3 scripts/input_acceptance.py --run-id <新run-id>
```

CLI输出目录/文件拒绝覆盖已存在产物。输入解析不执行日志、脚本、网页中的内容。

## 正式 H 与模型剩余边界

原来源的1.024 TFLOPS/核为FP16口径，不能直接给BF16模型。SRAM有效带宽、模块NoC、D2D方向口径/时延、端点缓冲与主机接口仍有未定义项。当前完整GRPO后端也尚无54模块/256核映射。

实际Qwen2.5-0.5B配置为24层、896 hidden、14 query heads、2 KV heads、绑词嵌入与输出头；计QKV biases后unique参数494032768。现有toy缺GQA反向、RoPE、QKV bias以及绑权存储/梯度模板。`inspect-hf-model`会列出这些缺口，不能把参数检查输出解释为真实模型已执行。

源路径、来源SHA与运行命令在验收包保留；旧训练的精确软件commit、历史权重身份和完整生效算法配置仍需继续溯源。
