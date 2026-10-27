#!/usr/bin/env python3
"""Generate an evidence-backed Chinese input/reuse report and file inventory."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from write_resource_report import FILE_INFO, project_files
from rl_xy.reporting import deterministic_tree_hash


FILE_INFO.update({
    "configs/hardware/xy_h1h2h3_54.json": "正式H1–H3候选参数、单位、来源、空值与设计/实测状态；不是可执行硬件精度声明。",
    "src/rl_xy/hardware_profile.py": "H合同、几何/聚合检查、缺失参数与后端能力门禁。",
    "src/rl_xy/verl_trace.py": "veRL L1长度/分组/版本/阶段记录的严格读取与未知数值标记。",
    "src/rl_xy/model_import.py": "Qwen2实际结构、GQA/绑权/QKV bias参数量与执行模板缺口。",
    "src/rl_xy/legacy_workload.py": "旧原始日志/Hydra导入；统计约束合成；clip参考口径显式选择。",
    "src/rl_xy/migration.py": "W/H/模型输入联合预检和状态容量必要条件，不产生晶圆时间。",
    "scripts/remote_session.py": "既有转发端口发现、主机key核验、保活复用和不确定写状态保护；不保存密码。",
    "scripts/reuse_legacy.py": "Docker内核对旧日志/CSV/summary/state，生成3种子长度负载和来源hash。",
    "scripts/input_acceptance.py": "宿主编排0.3 Docker验收、预期拒绝、旧数据复用及原生流程回归。",
    "scripts/write_input_report.py": "读取本轮PASS证据，生成报告/轻量预览及完整文件说明；拒绝源码hash失配。",
    "docker/Dockerfile.inputs": "复用已验证0.2原生镜像，打包0.3 Python输入接口，无新GPU依赖。",
    "tests/test_migration_inputs.py": "H几何/单位/来源/null、trace语义、真实模型参数检查及负例。",
    "tests/test_legacy_workload.py": "日志/配置一致性、合成矩约束、确定性及batch-max截断口径反例。",
    "docs/components/migration_inputs.md": "输入接入模块职责、命令、clip审计、执行边界与下一步。",
    "docs/components/data_sources.md": "旧数据和公开数据各自可提供的证据、许可与不可替代项。",
})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acceptance", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    accepted = json.loads(a.acceptance.read_text())
    if accepted["status"] != "PASS" or accepted["source_tree_sha256"] != deterministic_tree_hash(root):
        raise ValueError("requires PASS evidence with matching source tree")
    reuse = root / accepted["legacy_reuse"]
    rows = json.loads((reuse / "report.json").read_text())
    paths = project_files(root)
    unit = root / next(x["log"] for x in accepted["commands"] if x["label"] == "unit_tests")
    unit_summary = next(x for x in unit.read_text().splitlines() if x.startswith("Ran "))
    md = ["# rl-xy 0.3：正式H合同、旧veRL数据复用与输入接入", "",
          "日期：2026-10-26（Asia/Shanghai）。本报告从本轮Docker验收结果生成。", "",
          "本轮完成输入与证据接口；完整原生执行仍为两die/单逻辑核toy。真实Qwen2/GQA/绑权执行和54模块多核运行尚未完成。", "",
          f"Docker镜像：`{accepted['image']}`；ID：`{accepted['image_id']}`。", "",
          f"源码树：`{accepted['source_tree_sha256']}`；单测：`{unit_summary}`；验收断言：{len(accepted['checks'])}项。", "",
          "## 1. 本轮已完成与尚未完成", "",
          "| 工程项 | 当前结果 | 边界 |", "| --- | --- | --- |",
          "| 自动恢复连接 | 已验证本机VS Code重认证、重建转发并发现新端口；脚本支持key核验和保活 | 不能保证VPN/跳板永不中断；不保存密码 |",
          "| 正式H合同 | 54模块、256核/模块等按来源录入；10个未定义数值和D2D口径显式列出 | 正式H未执行，不混入测试H时间 |",
          "| 旧veRL数据复用 | 原始日志、CSV、summary、退出state逐项核对 | 历史软件commit/权重revision尚未补齐 |",
          "| 长度trace | 两组负载×3种子；保留逐步边际约束 | group内相关性、EOS原因、token/reward/logprob未恢复 |",
          "| 模型接入 | 真实Qwen2参数/形状导入，unique参数494032768 | GQA/RoPE/QKV bias/绑权执行模板仍需实现 |",
          "| Astra回归 | 原两die资源流程实际重新执行，时间108832.270 model ns | 这是旧toy回归，不是正式H的新性能 |", "",
          "## 2. 旧数据复用结果", "",
          "| 旧实验 | 原始步数 | GPU逐步时间均值/s | 每种子生成轨迹数 | 来源 |", "| --- | ---: | ---: | ---: | --- |"]
    for row in rows["runs"]:
        md.append(f"| {row['run_id']} | {row['observed_steps']} | {row['gpu_step_mean_s']:.9f} | {row['synthetic_workloads'][0]['trajectories']} | 旧GPU实测统计→合成长表 |")
    md += ["", "上述GPU秒数仅为旧观测。本轮没有新GPU训练，没有从GPU秒数推导晶圆算子时间。E3对应dapo8k子集，不是全量DAPO-Math-17K验收。", "",
           "clip_ratio审计：E1第71步最大503<配置上限512，但旧比率仍为1/512；第112步最大498也如此。生成器要求显式参考口径，本轮选择observed_max；另列真正达到公开上限的生成比例。该选择与当前读取的veRL指标源码相符，但历史commit尚未固定。", "",
           "## 3. 完整本轮流程", "",
           "1. 只读发现旧日志和归档Hydra覆盖项，绑定source SHA。",
           "2. 比对raw log/steps.csv/summary/state，检查步数与成功退出。",
           "3. 读取实际模型配置，导入用户正式H，明确保留unknown/null。",
           "4. 逐步按prompt/response min/mean/max和指定clip口径生成整数长度；种子0/1/2。",
           "5. 转为标准veRL长度接口，核对完整组、token总量、同步版本；缺少的张量值不伪造。",
           "6. 生成W/H/模型联合预检：容量必要条件、TP整除候选与未完成执行能力；不输出正式H时间。",
           "7. 在新Docker镜像中重新执行现有Astra资源流程，独立核查事件与结果包。", "",
           "## 4. 下一步", "",
           "| 次序 | 工作 | 完成依据 |", "| --- | --- | --- |",
           "| 1 | 实现真实Qwen2的GQA/RoPE/bias/绑权与通用microbatch更新合同 | 参数、张量与通信逐项覆盖；小模型独立数值对照 |",
           "| 2 | 实现多核映射及模块NoC/DRAM共享，接通mesh全流程 | 单模块多核→2×2→正式规模的资源/路由验收 |",
           "| 3 | 正式H缺参采用明确设计场景或可信来源补齐，接入成本模型 | 参数来源、精度口径和假设可追溯；无隐含默认值 |",
           "| 4 | 旧数据多个种子的敏感性与公开输入扩展 | 分布、相关性、单位与源模型差异明确 |",
           "| 5 | 独立校准与统一baseline | 算子/存储/网络/整步未见样本误差；再讨论性能收益 |", "",
           "## 5. 全工程文件说明", "", f"共{len(paths)}个源码、配置、测试与说明文件；runs/acceptance属于运行产物，单独见验收清单。", "",
           "| 文件 | 包含内容与功能 |", "| --- | --- |"]
    md += [f"| `{path}` | {FILE_INFO[path]} |" for path in paths]
    md += ["", "## 6. 复现与结果入口", "", "```bash",
           "cd /path/to/research/rl-xy", "python3 scripts/input_acceptance.py --run-id <新的唯一ID>",
           "PYTHONPATH=src python3 scripts/write_input_report.py --acceptance acceptance/<ID>/report.json --out runs/input_delivery/<ID>",
           "```", "", f"验收报告：`{a.acceptance}`；旧数据复用：`{accepted['legacy_reuse']}`；原生回归：`{accepted['regression_bundle']}`。",
           "", "公开数据来源与适用范围见docs/components/data_sources.md；本轮未将公开预览混入用户旧GPU观测。", ""]
    a.out.mkdir(parents=True, exist_ok=False)
    text = "\n".join(md)
    (a.out / "工程说明与完整流程.md").write_text(text)
    (a.out / "流程预览.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><title>rl-xy 0.3输入接入</title><style>body{max-width:1160px;margin:32px auto;padding:0 24px;background:#fafafa;color:#20242b}pre{white-space:pre-wrap;font:15px/1.8 system-ui,sans-serif;overflow-wrap:anywhere}</style><pre>' + html.escape(text) + '</pre></html>')
    print(f"generated {len(paths)} file descriptions at {a.out}")


if __name__ == "__main__":
    main()
