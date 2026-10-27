#!/usr/bin/env python3
"""Generate the Chinese file guide and workflow report from accepted run bundles."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


FILE_INFO = {
    "docs/README.md": "当前组件/历史说明与实验结果的文档导航。",
    "docs/FILE_MAP.md": "公开仓库全部文件职责与分层目录。",
    "docs/REPRODUCING.md": "完整镜像依赖链、公开64轨迹复现及证据可用范围。",
    "scripts/build_images.sh": "无sudo按0.1到0.6顺序构建全部本地镜像。",
    "scripts/reproduce_v06.sh": "不申请GPU，从公开完整长度case生成新control与四组对照。",
    "scripts/check_public_results.py": "核对公开摘要、历史核心代码及输入hash，不冒充原始日志重放。",
    ".dockerignore": "限制 Docker build context；排除依赖缓存、历史运行结果、构建日志和验收包。",
    ".gitignore": "区分源码与可再生文件；排除 .deps、缓存、运行结果、验收包和构建日志。",
    "LICENSE": "自建工程的 MIT 许可证；上游 AstraSim/Chakra 仍遵守各自许可证。",
    "Makefile": "Docker 脚本的快捷入口；提供 build、test、transfer、toy、two-rounds、all。",
    "README.md": "工程总入口、最小例子、运行命令、证据边界与文档导航。",
    "compose.yaml": "可选 Compose 构建/交互声明；正式结果使用 docker.sh 保留镜像身份。",
    "pyproject.toml": "Python 包名/版本/依赖、src 布局及 rlxy CLI 注册。",
    "docker/Dockerfile": "基于指定本地镜像构建 Abseil、Protobuf、固定 AstraSim 和隔离 Python 环境。",
    "scripts/docker.sh": "无 sudo 的 Docker build/create/exec/run 统一入口；核验挂载、UID/GID、镜像 ID。",
    "scripts/check_result.py": "复查成功状态、非空检查集、固定上游版本、Docker 证据和全部文件 SHA-256。",
    "scripts/acceptance.py": "依次执行容器测试、链路、单轮、两轮、重复运行和完整性门禁，保存命令日志。",
    "scripts/write_walkthrough.py": "读取 PASS 验收包生成本文及 HTML 预览；逐文件说明缺项时立即失败。",
    "src/rl_xy/__init__.py": "Python 包入口与包版本。",
    "src/rl_xy/types.py": "配置、Operation、ObjectRecord、MemorySnapshot、AstraEvent、CheckResult 等共享类型。",
    "src/rl_xy/config.py": "严格读取 case JSON；拒绝未知字段、类型漂移及不支持的并行/学习合同。",
    "src/rl_xy/model.py": "枚举 Transformer 参数分片/复制、各状态字节及每 token KV 字节。",
    "src/rl_xy/costs.py": "声明的 FLOP/vector 合成成本；保留局部 DMA 公式；无硬件实测校准。",
    "src/rl_xy/workflow.py": "将同步 GRPO 展开成 rank-local DAG，显式生成 TP、采样、损失反向和版本节点。",
    "src/rl_xy/fixture.py": "独立的 21 ns 数据移动链、out-of-place SRAM 分配与 63 B 容量负例。",
    "src/rl_xy/ledger.py": "参数/优化器/梯度持久 storage 与 rollout KV 的身份、生命周期和准入容量账。",
    "src/rl_xy/validation.py": "从独立清单核对阶段、反向、清零、Adam 读写、版本、KV、通信与真实事件。",
    "src/rl_xy/reporting.py": "写 CSV/JSON/JSONL、round.phase 时间、指标、SUMMARY 与 hash manifest。",
    "src/rl_xy/run.py": "串接配置复制、trace、AstraSim 执行、解析、账本检查及成功/失败归档。",
    "src/rl_xy/cli.py": "argparse 单入口；run 执行 toy，fixture-transfer 执行链路测试。",
    "src/rl_xy/astra_adapter/__init__.py": "公开配置核验、trace 生成、进程执行、日志解析的边界接口。",
    "src/rl_xy/astra_adapter/config.py": "交叉核验 system/network/remote-memory 与 case 的 rank、时间及带宽单位。",
    "src/rl_xy/astra_adapter/generator.py": "生成每 rank 的官方 Chakra ET；验证节点、tag、参与者、本地依赖和 collective。",
    "src/rl_xy/astra_adapter/runner.py": "核验官方 commit 和可执行文件；无 shell 调用 AstraSim，保存命令与原始日志。",
    "src/rl_xy/astra_adapter/parser.py": "配对 issue/callback，核验节点类型、所有 rank 完成与统计，不接受残缺结果。",
    "tests/test_astra_config.py": "带宽缩放/两套配置正例，带宽漂移和 bool 冒充整数负例。",
    "tests/test_check_result.py": "完整 bundle 校验正例，以及篡改 metrics 后拒绝的负例。",
    "tests/test_config_model.py": "参数/KV 手算、未知字段、证据等级、策略字段和 rounds 覆盖测试。",
    "tests/test_fixture.py": "T01 结构/容量手算与 63 B out-of-place 不可行负例。",
    "tests/test_generator_contract.py": "P2P 参与者错误、tag 越界、不支持 collective 的 fail-closed 测试。",
    "tests/test_ledger.py": "长期状态、KV reservation/materialization、NIC 计划容量及两轮无泄漏检查。",
    "tests/test_parser.py": "事件配对、缺 callback、乱序日志、rank-local 节点和提前 RECV 等待测试。",
    "tests/test_reporting.py": "两轮 phase 不混桶，以及空参数表仍有固定 CSV schema。",
    "tests/test_validation.py": "完整证据正例；删依赖、漏反向/Adam 状态、容量越界/漏事件等负例。",
    "tests/test_workflow.py": "本地依赖链、完整前后向、embedding TP 求和、梯度清零及 v2 发布。",
    "docs/ARCHITECTURE.md": "W/H/S 分层、模块边界、rank-local DAG、时间编码与当前扩展边界。",
    "docs/EVIDENCE_BOUNDARIES.md": "区分合成回放、解析假设、硬件校准和真实训练；规定指标可支持的结论。",
    "docs/REPRODUCIBILITY.md": "环境、执行顺序、结果包、hash 复查和失败后的复现规则。",
    "docs/components/config.md": "配置组件的输入、输出、字段约束和失败条件。",
    "docs/components/workload.md": "工作流组件的阶段、操作、采样/TP/损失/反向与版本语义。",
    "docs/components/ledger.md": "storage 和 KV 容量账本的字段、公式、释放规则及覆盖范围。",
    "docs/components/astra_adapter.md": "Chakra 映射、官方 CLI、单位、日志契约及 SEND/RECV 口径。",
    "docs/components/validation.md": "独立验收项、fail-closed 行为、手算链路和负例要求。",
    "docs/components/reporting.md": "run bundle 每类产物、指标定义、来源和哈希/状态约束。",
    "docs/components/docker.md": "指定基础镜像、独立 venv、项目持久容器、临时代理和正式运行说明。",
    "docs/工程文件与完整流程.md": "本文：全部自建文件索引、完整案例、实际阶段时间、复现入口及结果解释。",
    "docs/完整流程预览.html": "本文的轻量预览：运行结果、12 阶段流程、阶段时间与文件清单。",
    "examples/toy_grpo_2die/README.md": "教学例子的模型、两条轨迹、参数/KV/版本公式和逐步使用说明。",
}
for _case in ("fixture_transfer", "toy_grpo_2die"):
    FILE_INFO[f"configs/{_case}/case.json"] = "该案例的工作负载、硬件、策略与合成成本合同；CLI 的主输入。"
    FILE_INFO[f"configs/{_case}/astra_system.json"] = "FIFO/native ring、trace 开关、零额外 endpoint/reduction delay 和后端设置。"
    FILE_INFO[f"configs/{_case}/astra_network.yml"] = "两 rank FullyConnected；精确表达放大 1000 倍后的带宽与链路时延。"
    FILE_INFO[f"configs/{_case}/astra_remote_memory.json"] = "固定 NO_MEMORY_EXPANSION；禁止隐式借用远程内存。"

PHASES = [
    ("input", "输入", "host 将 prompt [1,2] 送入 D0，再通过 P2P 复制到 D1。"),
    ("rollout", "生成", "A/B 各 prefill 两 token，随后 A1/B1 decode、释放 B KV，再 decode A2、释放 A KV。D1 logits 发给 D0，由 D0 采样并广播 token。"),
    ("reward", "奖励", "已完成回答送回 host；固定规则奖励 A=1、B=0，并向 D1 发送奖励结果和阶段围栏。"),
    ("reference", "Reference 概率", "冻结 Reference 对 A/B 的 teacher-forcing 输入执行前向，取 3/2 个回答位置的 log-prob。"),
    ("old_policy", "旧策略概率", "更新前 actor v0 对同一 A/B 输入执行无梯度前向，输出 old log-prob。"),
    ("advantage", "组内优势", "组均值为0.5，A/B 优势为+0.5/-0.5；D0 将结果复制给 D1。"),
    ("train_A", "训练 A", "先清零本轮 FP32 梯度缓冲；执行 A 的前向、全局分母5的损失、损失到 logits 的反向、模型反向及梯度累积。"),
    ("train_B", "训练 B", "执行 B 的同类前后向并累积到同一梯度缓冲；B 不会再次清零。"),
    ("clip", "梯度裁剪", "分片贡献局部平方和；复制 norm 参数由 owner 贡献；标量 AllReduce 后应用裁剪。"),
    ("optimizer", "优化器", "shape-only AdamW 读取 gradient/master/m/v，写 master/m/v/actor；Reference 不写。"),
    ("publish", "发布", "全分片小 payload AllReduce 围栏后发布 actor v1；publish marker 显式为1 ns。"),
    ("cleanup", "回收", "记录本轮临时 KV 已释放；cleanup marker 显式为1 ns；下一轮从新 actor 版本继续。"),
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


for _name, _description in {
    "实验结果.md": "本版完整实验设置、预算、四组结果、验证及局限。",
    "comparisons.json": "原始四组机器可读比较，完整预算与假设。",
    "metrics_summary.json": "六份运行选定指标及原始metrics/manifest SHA。",
    "acceptance_summary.json": "原20命令、退出状态、日志hash和native检查数量。",
    "local_verification_summary.json": "已执行的本地原始事件/资源独立复核结果。",
    "provenance.json": "历史源身份/归档hash与公开整理差异。",
    "input_hashes.json": "公开输入精确SHA256。",
    "core_runtime_hashes.json": "45个原始运行核心源码/C++文件SHA256。",
    "model_bottlenecks.json": "模型服务与实际Astra事件的分解数据。",
    "模型内部瓶颈分解.md": "模型服务分量对照，非硬件计数器。",
}.items():
    FILE_INFO[f"experiments/2026-10-27-v0.6/{_name}"] = _description
for _name in ("case.json", "compilation.json", "gpu_provenance.json", "model_config.json",
              "astra_system.json", "astra_network.yml", "astra_remote_memory.json"):
    FILE_INFO[f"experiments/2026-10-27-v0.6/inputs/{_name}"] = "原实验逐字节复制的结构/长度/来源输入。"


def project_files(root: Path) -> list[str]:
    excluded = {".git", ".deps", ".venv", "__pycache__", "runs", "acceptance", "build_logs", "build", "dist"}
    paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not any(part in excluded for part in path.relative_to(root).parts)
        and not path.name.endswith((".pyc", ".pyo"))
    }
    paths.update({"docs/工程文件与完整流程.md", "docs/完整流程预览.html"})
    missing = sorted(paths - set(FILE_INFO))
    if missing:
        raise ValueError(f"missing per-file documentation: {missing}")
    return sorted(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    acceptance_path = args.acceptance.resolve()
    accepted = load_json(acceptance_path)
    if accepted.get("status") != "PASS":
        raise ValueError("walkthrough requires a PASS acceptance report")
    bundles = {name: root / value for name, value in accepted["bundles"].items()}
    metrics = {name: load_json(path / "metrics.json") for name, path in bundles.items()}
    manifests = {name: load_json(path / "manifest.json") for name, path in bundles.items()}
    for name, manifest in manifests.items():
        if manifest["status"] != "success":
            raise ValueError(f"run bundle {name} is not successful")
    toy = metrics["toy"]
    software = manifests["toy"]["software"]
    paths = project_files(root)
    unit_log = root / next(row["log"] for row in accepted["commands"] if row["label"] == "unit_tests")
    unit_summary = next(line.strip() for line in unit_log.read_text(encoding="utf-8").splitlines() if line.startswith("Ran "))
    bt = chr(96)
    fence = bt * 3
    md = [
        "# rl-xy：仿真器搭建、逐文件说明与完整流程",
        "",
        "交付日期：2026-10-26（Asia/Shanghai）。本报告由实际通过的 Docker/AstraSim 验收包生成。",
        "",
        "## 1. 本次搭建结果",
        "",
        f"服务器工程根目录：{bt}/path/to/research/rl-xy{bt}；容器：{bt}{accepted['container']}{bt}；容器内目录：{bt}/workspace/rl-xy{bt}。",
        "",
        "已经跑通配置→细粒度操作图→每 rank Chakra ET→官方 AstraSim→日志配对→独立检查→带哈希结果包的完整链路。",
        "当前交付定位是两 die TP=2 的同步 GRPO 操作级合成回放基础版，包含单轮与两轮；大模型前后向、采样和 AdamW 不做数值张量计算。",
        "",
        "| 验收场景 | AstraSim 总时间 / model ns | 逻辑操作 | 参与 rank 节点 | 检查 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("transfer", "toy", "two_rounds", "toy_repeat"):
        value = metrics[name]
        checks = load_json(bundles[name] / "checks.json")["checks"]
        md.append(f"| {name} | {value['total_simulated_time_ns']:.3f} | {value['operation_count']} | {value['operation_rank_event_count']} | {sum(row['passed'] for row in checks)}/{len(checks)} |")
    md += [
        "",
        "全部 CLI/完整性检查退出码均为0。重复单轮的 events.jsonl 逐字节一致；两轮 actor v0→v1→v2，Reference v0 保持冻结。",
        f"单元测试实际日志：{bt}{unit_summary}{bt}，最终 OK；总验收断言 {len(accepted['checks'])} 项全部通过。",
        f"总验收报告：{bt}{acceptance_path.relative_to(root)}{bt}，保留每条命令、退出码、日志路径和日志 SHA-256。",
        "",
        "## 2. 环境和版本",
        "",
        "| 项目 | 本次值 |",
        "|---|---|",
        f"| 项目镜像引用 | {bt}{software['container_image_tag']}{bt} |",
        f"| 不可变镜像 ID | {bt}{software['container_image_id']}{bt} |",
        f"| 官方 AstraSim commit | {bt}{software['astrasim_commit']}{bt} |",
        f"| AstraSim binary SHA-256 | {bt}{software['astrasim_binary_sha256']}{bt} |",
        f"| 本次源码树 SHA-256 | {bt}{software['rl_xy_source_tree_sha256']}{bt} |",
        f"| Python | {software['python']} |",
        "| 基础镜像 | pjlab/deeplearning:v2.2.3-cuda12.8.0-torch2.7.0；宿主已有镜像 |",
        "| 工具链 | Abseil 20240722.0、Protobuf C++29.0 / Python5.29.0 |",
        "| 时序后端 | AstraSim_Analytical_Congestion_Unaware |",
        "| 模拟设备 | 2 die × 1 logical core；TP2/PP1/DP1 |",
        "| 初始化边界 | 参数/优化器/梯度缓冲在 t=0 已初始化；测量稳态轮次 |",
        "",
        "未自动提交或推送 Git；git revision 为 null 时，以本次 manifest 的源码树 SHA-256 定位运行内容。基础镜像不等于模拟硬件；仿真没有占用 GPU。",
        "源码树 hash 覆盖源码、配置、测试、脚本和手写文档；排除本页及 HTML 预览这两个派生报告，避免报告引用自身 hash 的循环定义，也排除缓存及运行产物。验收另行检查四次运行的 hash 均与当前源码一致。",
        "",
        "## 3. 架构与各层职责",
        "",
        fence + "mermaid",
        "flowchart TD",
        '  A["case.json：W/H/S"] --> B["严格配置 + 参数/字节计数"]',
        '  B --> C["workflow：rank-local 操作 DAG"]',
        '  C --> D["ledger：持久状态 + KV 准入/生命周期"]',
        '  C --> E["generator：rank0/rank1 Chakra ET"]',
        '  E --> F["官方 AstraSim analytical"]',
        '  F --> G["原始 stdout/stderr/debug"]',
        '  G --> H["parser：issue/callback + rank 完成"]',
        '  H --> I["独立 validation"]',
        "  D --> I",
        '  I --> J["CSV/JSON/JSONL + manifest + 完整性检查"]',
        fence,
        "",
        "W 定义模型、token、group、reward、损失分母和版本；H 定义容量与合成速率；S 定义 TP 布局、微批和严格同步顺序。AstraSim 负责实际时间推进，外层不补造缺失时序。",
        "rank0 独占的输入、采样、reward 和 advantage 不在 rank1 镜像。每 rank 只依赖自己前一个本地节点；P2P/collective 将本地链同步。",
        "",
        "## 4. 所有自建文件逐一说明",
        "",
        f"本节逐项覆盖 {len(paths)} 个自建文件。第三方 .deps/.git、缓存和重复 run 文件不作为自建源码；其作用分别见第9节和结果目录清单。",
        "",
        "| 文件（相对于工程根） | 职责、输入/输出及使用时机 |",
        "|---|---|",
    ]
    for path in paths:
        md.append(f"| {bt}{path}{bt} | {FILE_INFO[path]} |")
    md += [
        "",
        "## 5. 完整例子的 W/H/S 合同",
        "",
        "模型为2层微型 Transformer：hidden8、query/KV heads2/2、head_dim4、FFN16、vocab16、RMSNorm/SwiGLU、无 bias、词嵌入和输出头不绑权。Actor/Reference/KV 为BF16；master/gradient/Adam m/v为FP32。",
        "输入 prompt=[1,2]；A response=[3,4,5]、reward=1；B response=[6,7]、reward=0。最后生成的 token 不再送回前向，因此 rollout/Reference/old/训练前向各处理7个输入位置；回答概率和损失覆盖5个位置。",
        "训练微批为单条轨迹；A、B 都使用同一个全局损失分母5，累积后只做一次裁剪/更新。max_response=3 是公开准入上界；固定 A/B/stage 顺序不按未来真实长度优先排序。",
        "",
        "| 参数/状态 | 每 die 数量或字节 |",
        "|---|---:|",
        "| 唯一逻辑参数 | 全模型1576个 |",
        "| 每 die 本地参数 | 808个（含复制 norm 参数） |",
        "| Actor BF16 / Reference BF16 | 1616 / 1616 B |",
        "| master / Adam m / Adam v | 各3232 B |",
        "| 长期状态 | 12928 B |",
        "| FP32 梯度累积缓冲 | 3232 B |",
        "| 常驻状态与梯度合计 | 16160 B |",
        "| 每个已处理 token 的本地 KV | 32 B |",
        "| 两轨迹 KV 准入预留 | 256 B |",
        "| rollout KV 实际峰值 / 每轮累计写入 | 192 / 224 B |",
        "",
        "两个回答可同时保留 KV，计算共享同一 TP2 组按固定顺序运行。KV 曲线为0→128→192→96→128→0；B 结束时，B 已物化存储和未兑现预留一起释放。",
        "",
        "## 6. 一轮实际发生的12个阶段",
        "",
        fence + "mermaid",
        "flowchart LR",
        '  A["输入"] --> B["Rollout A/B"] --> C["Reward"] --> D["Reference"] --> E["Old policy"] --> F["Advantage"]',
        '  F --> G["清零 + Train A"] --> H["Train B"] --> I["Clip"] --> J["AdamW"] --> K["发布 v1"] --> L["Cleanup"]',
        fence,
        "",
        "| 阶段 | 完整数据/控制流 |",
        "|---|---|",
    ]
    for key, title, description in PHASES:
        md.append(f"| {title} / {bt}{key}{bt} | {description} |")
    md += [
        "",
        "每次前向都展开 embedding→embedding TP 求和→每层 norm/QKV/attention score/softmax/AV/O projection/TP 求和/residual/norm/gate-up/SwiGLU/down/TP 求和/residual→final norm→LM head。",
        "概率阶段保持词表分片，对目标位置执行 max、sum-exp 和 target-logit 三次 AllReduce。反向展开 loss→logits gradient→LM head dinput/dweight→TP input-gradient 求和→final norm backward→逆层序 FFN/attention/QKV/norm backward→embedding backward→gradient RMW。",
        "DP=1，因此没有 DP 梯度 AllReduce。发布围栏使用4 B的 native AllReduce；publish/cleanup 各1 ns为显式控制 fixture。",
        "",
        "### 本次单轮阶段时间",
        "",
        "下面由真实 issue/callback 取各阶段最早 issue 与最晚 callback，再除以1000。单位 model ns；通信驻留可能含 posted RECV 等待。",
        "",
        "| 阶段 | 起点 / ns | 终点 / ns | 跨度 / ns | 节点数（含参与rank） |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, title, _ in PHASES:
        value = toy["phase_timing"]["r00." + key]
        md.append(f"| {title} | {value['start_ns']:.3f} | {value['end_ns']:.3f} | {value['span_ns']:.3f} | {value['operation_rank_events']} |")
    md += [
        "",
        "### 时间单位怎样恢复",
        "",
        "Chakra 的 duration_micros 为整数微秒。外层用1 model ns→1 Chakra us，将时间放大1000倍；网络 latency=1000 raw ns，bandwidth=0.007450580596923828 GiB/s，在后端换算为0.008 B/raw ns，恢复模型的8 B/ns。所有 metrics ns 都经过除1000；原始 tick 保留在 events.jsonl。",
        "",
        "## 7. 可手算的21 ns链路验证",
        "",
        "32 B输入，out-of-place input/output 同在 SRAM 时占64 B；本地DMA16 B/ns，每步2 ns；计算2 ns；D2D8 B/ns、启动1 ns，因此 SEND 服务为5 ns。",
        "",
        fence + "text",
        "D0: DRAM→SRAM 0–2 → compute 2–4 → SRAM→DRAM 4–6 → DRAM→TX 6–8 → SEND 8–13",
        "D1: RECV posted 0–13 → RX→DRAM 13–15 → DRAM→SRAM 15–17 → compute 17–19 → SRAM→DRAM 19–21",
        fence,
        "",
        f"本次实际：completion={metrics['transfer']['total_simulated_time_ns']} ns；SEND={metrics['transfer']['p2p_send_service_ns']} ns；RECV驻留={metrics['transfer']['p2p_receive_occupancy_ns']} ns；SRAM峰值64 B；D2D逻辑payload32 B；DRAM端点流量192 B。63 B SRAM负例在发起AstraSim前被拒绝。",
        "RECV的13 ns包含前8 ns等待发送方，不能把它当作链路服务时间。rank0只生成n1–n5，rank1只生成n5–n9，没有镜像计算制造的假延迟。",
        "",
        "## 8. 运行、验收和接手命令",
        "",
        fence + "bash",
        'ssh "$RLXY_SSH_HOST"',
        "cd /path/to/research/rl-xy",
        "",
        "# 镜像已构建；需要从源码重新构建时，临时代理由当前环境提供",
        "RLXY_BASE_IMAGE=pjlab/deeplearning:v2.2.3-cuda12.8.0-torch2.7.0 ./scripts/docker.sh build",
        "./scripts/docker.sh create-container",
        "",
        "# 每次使用新的 run-id；下面一次执行全部验收",
        "python3 scripts/acceptance.py --container rl-xy --run-id my-new-run",
        "",
        "# 或分别执行",
        "RLXY_CONTAINER=rl-xy ./scripts/docker.sh transfer rlxy-astrasim:0.1.0 my-transfer",
        "RLXY_CONTAINER=rl-xy ./scripts/docker.sh toy rlxy-astrasim:0.1.0 my-toy",
        "RLXY_CONTAINER=rl-xy ./scripts/docker.sh two-rounds rlxy-astrasim:0.1.0 my-two-rounds",
        "python3 scripts/check_result.py runs/toy_grpo_2die/my-toy",
        "",
        "# 交互查看容器；执行命令不会占用GPU",
        "docker exec -it rl-xy /bin/bash",
        fence,
        "",
        "验收脚本先运行单元测试，再执行 transfer、toy、two-rounds、toy-repeat；接着核验所有bundle、21/5/13 ns不变量、版本链、Reference冻结、重复事件一致及两轮时间。失败立即停止并保留诊断；不会覆盖同名run。",
        "",
        "## 9. 每类运行文件与第三方依赖",
        "",
        "| 本次场景 | 相对于工程根的结果目录 |",
        "|---|---|",
    ]
    for name, path in accepted["bundles"].items():
        md.append(f"| {name} | {bt}{path}{bt} |")
    md += [
        "",
        "每个结果目录都有以下文件；manifest 中逐一列出全部产物哈希。inputs 内保存原配置和两个二进制 ET。",
        "",
        "| 产物 | 如何阅读 |",
        "|---|---|",
        "| manifest.json | 成功状态、命令、退出码、Docker/image/上游/binary/source/input/产物身份。 |",
        "| resolved_config.json / astra_config_contract.json | 最终 W/H/S 与网络单位交叉验证结果。 |",
        "| operations.jsonl | 计划的操作、工作量、payload、参与rank及rank-local前驱。 |",
        "| events.jsonl | 本次真实 rank/node 的 issue/callback、原始tick和恢复后的ns。 |",
        "| parameters.csv / objects.csv | 参数拆分及已跟踪的持久/KV storage 身份和生命周期。 |",
        "| memory.csv / transfers.csv | 容量快照与逻辑通信记录；KV view不要重复计入DRAM。 |",
        "| checks.json / checks.csv | 每项 expected/actual/pass/evidence/scope。 |",
        "| astra_rank_summary.csv | 后端原始计数；GPU字段解读为compute lane。 |",
        "| metrics.json / summary.json / SUMMARY.md | 指标、PASS索引及单次运行概述。 |",
        "| status.json / astra_command.json | 进程/归档状态、实际argv与退出码。 |",
        "| astra_stdout.log / astra_stderr.log / astra_logs/log.log | 原始完成记录和节点DEBUG事件。 |",
        "| inputs/case.json、astra_system.json、astra_network.yml、astra_remote_memory.json | 本次输入副本，避免原配置后改导致结果不可解释。 |",
        "| inputs/workload.0.et、inputs/workload.1.et | 官方Chakra protobuf编码；按rank分别供AstraSim打开。 |",
        "",
        "宿主 .deps/astra-sim 是下载的官方源码参考缓存，正式运行使用镜像 /opt/astra-sim。其 Chakra、fmt、spdlog、analytical network、analytical memory 及递归子模块锁定在父仓库记录的gitlink版本。不会把上游所有源文件复制成自建模块。",
        "build_logs/保存构建尝试；acceptance/<run-id>/保存总验收日志；runs/保存每次输入/输出。它们与src/configs/docs分开。",
        "",
        "## 10. 指标可参考的范围与原方案对应",
        "",
        f"本次全流程单轮耗时为 {toy['total_simulated_time_ns']:.3f} model ns，属于声明的合成成本和无拥塞网络假设下的AstraSim结果。",
        "当前基础版已实现原方案的W/H/S分离、两die完整阶段/反向/更新/版本、持久状态和KV账、21ns链路、容量负例、两轮及证据归档。原方案的细粒度tile/SRAM/所有中间激活与资源锁是后续精细模型，不能由本次PASS推断已经完成。",
        "",
        "| 口径 | 当前覆盖和解释 |",
        "|---|---|",
        "| DRAM已跟踪峰值 | 16352 B/die =16160常驻+192KV；不含训练activation tape、所有临时张量、allocator元数据及碎片。 |",
        "| SRAM | transfer fixture有显式64 B out-of-place容量检查；主toy未建立tile工作区峰值，报告为null，不写成0 B。 |",
        "| 本地DRAM/SRAM带宽 | transfer fixture显式计时；主toy compute由FLOP/vector成本回放，未加入逐tile本地DMA。 |",
        "| 通信字节 | collective/p2p字段是父操作逻辑payload；不是实际网络wire字节、协议开销或byte-hop总量。 |",
        "| 并发/拥塞 | 每rank严格链，无跨算子重叠优化；congestion_unaware不建模多流链路争用。 |",
        "| GPU time | compute-lane occupancy，含synthetic host/control replay；不是GPU硬件利用率。 |",
        "| 算法数值 | reward/advantage合同可算；前后向、采样和AdamW为形状/成本回放，无模型checkpoint或学习质量。 |",
        "",
        "因此可以把本项目作为可复现的AstraSim RL流程骨架和回归基线；扩大模型/硬件或用于论文性能结论之前，要补足局部数据移动/激活/tile资源模型、真实workload和硬件校准。",
        "",
    ]
    document = root / "docs" / "工程文件与完整流程.md"
    document.write_text("\n".join(md), encoding="utf-8")
    phase_rows = "".join(
        f"<tr><td>{html.escape(title)}</td><td>{toy['phase_timing']['r00.'+key]['span_ns']:.3f}</td><td>{html.escape(description)}</td></tr>"
        for key, title, description in PHASES
    )
    file_rows = "".join(
        f"<tr><td><code>{html.escape(path)}</code></td><td>{html.escape(FILE_INFO[path])}</td></tr>"
        for path in paths
    )
    cards = "".join(
        f"<div class='card'><h3>{html.escape(name)}</h3><strong>{metrics[name]['total_simulated_time_ns']:,.3f} ns</strong><p>{metrics[name]['operation_count']} operations · {metrics[name]['operation_rank_event_count']} rank nodes</p></div>"
        for name in ("transfer", "toy", "two_rounds")
    )
    preview = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rl-xy 完整流程与文件说明</title>
<style>body{font:16px/1.75 system-ui,sans-serif;margin:36px auto;max-width:1180px;padding:0 24px;color:#172b4d;background:#f8fafc}h1,h2{line-height:1.3}h2{margin-top:42px}.cards{display:flex;gap:16px;flex-wrap:wrap}.card{background:white;border:1px solid #dbe3ed;border-radius:12px;padding:20px;min-width:260px;flex:1}.card strong{font-size:28px;color:#075b9b}.flow{padding:22px;background:#e8f2fb;border-radius:10px}table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{padding:10px 12px;border:1px solid #dbe3ed;text-align:left;vertical-align:top}code{font-size:13px;overflow-wrap:anywhere}.note{padding:18px;background:#fff6dd;border-left:4px solid #d09a24}a{color:#075b9b}</style>
<h1>rl-xy · AstraSim RL 基础仿真器</h1>
<p>2026-10-26 · Docker 实际验收 · 2 die / TP2 / 1 prompt / 2 responses</p>
""" + f"""<div class="cards">{cards}</div>
<h2>从配置到结果</h2><p class="flow">W/H/S 配置 → 参数/字节计数 → rank-local DAG → Chakra ET → 官方 AstraSim → 原始日志 → 独立检查 → manifest 结果包</p>
<h2>完整同步 GRPO</h2><p class="flow">输入 → Rollout A/B → Reward → Reference → Old policy → Advantage → 梯度清零+Train A → Train B → Clip → AdamW → 发布 v1 → Cleanup</p>
<table><tr><th>阶段</th><th>实际跨度 / ns</th><th>数据与控制流</th></tr>{phase_rows}</table>
<h2>结果口径</h2><p class="note">这是操作级 synthetic fixture，未做硬件校准或数值训练。16352 B/die 是常驻状态+KV 的已跟踪小计；主toy尚未展开activation tape、tile/SRAM与本地DMA，不能当作整轮真实内存峰值。详细边界及可复现命令见 Markdown 主文档。</p>
<h2>全部 {len(paths)} 个自建文件</h2><table><tr><th>文件</th><th>说明</th></tr>{file_rows}</table>
<p>主文档：<a href="工程文件与完整流程.md">工程文件与完整流程.md</a></p></html>"""
    (root / "docs" / "完整流程预览.html").write_text(preview, encoding="utf-8")
    print(json.dumps({"document": str(document), "project_files_documented": len(paths)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
