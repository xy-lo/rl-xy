#!/usr/bin/env python3
"""Generate full source inventory and flow from actual matching Docker PASS."""
import argparse
import html
import json
from pathlib import Path
from write_execution_report import FILE_INFO, project_files
from rl_xy.reporting import deterministic_tree_hash

FILE_INFO.update({
    'configs/hardware/xy_h0_wsc_case3.json':'用户H0原表的独立来源合同：54die、FP16 1020FLOP/ns及原始几何；未知值不补写。',
    'src/rl_xy/training_schedule.py':'确定顺序minibatch/epoch图扩展、每更新命名/版本/分母、rank本地依赖重建。',
    'src/rl_xy/training_validation.py':'独立重建更新计划并复用逐更新完整训练检查；拒绝版本/清零/优势/分母篡改。',
    'tests/test_training_schedule.py':'H0来源/几何/单位负例；尾批、拆组优势、两轮版本、梯度存储和多种篡改负例。',
    'docs/09_H0_SOURCE_AND_GEOMETRY.md':'H0逐项单位/来源、FP16口径、48/54几何冲突和缺参说明。',
    'docs/10_OPTIMIZER_SCHEDULE.md':'多epoch/minibatch组件职责、冻结behavior、梯度清零/归一化/发布和执行限制。',
    'docker/Dockerfile.schedule':'复用0.4原生Astra镜像、打包0.4.1 Python更新层；不使用GPU。',
    'scripts/schedule_acceptance.py':'Docker门禁：87测试、H0预期拒绝、五原生bundle、重复/两轮/旧结果及mesh资源回归。',
    'scripts/write_schedule_report.py':'从匹配源码hash的PASS生成工程文档、全文件清单及轻量HTML。',
})
for name in ('case.json','astra_system.json','astra_network.yml','astra_remote_memory.json'):
    FILE_INFO['configs/qwen2_updates_fixture/'+name] = {
        'case.json':'2epoch、minibatch3的四trajectory教学fixture；TP2/单逻辑核，不是H0。',
        'astra_system.json':'FIFO与TP ring原生执行合同，关闭roofline避免重复计时。',
        'astra_network.yml':'教学两节点FullyConnected，8B/ns及1model ns。',
        'astra_remote_memory.json':'无外部内存扩展；容量由资源hook拒绝超额。'}[name]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--acceptance',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    r=json.loads(a.acceptance.read_text())
    if r['status']!='PASS' or r['source_tree_sha256']!=deterministic_tree_hash(root):
        raise ValueError('actual PASS with matching source required')
    paths=project_files(root)
    unit=(root/next(c['log'] for c in r['commands'] if c['label']=='unit_tests')).read_text()
    unit_summary=next(s for s in unit.splitlines() if s.startswith('Ran '))
    lines=['# rl-xy 0.4.1：H0来源纠正与多更新执行','',
        '实际Docker门禁通过后生成；活动源仅服务器 `/path/to/research/rl-xy`。不是完成全54模块迁移的声明。','',
        f"镜像 `{r['image']}`；ID `{r['image_id']}`。",'',f"源码树 `{r['source_tree_sha256']}`。",'',
        f"{unit_summary}；{len(r['commands'])}条命令符合预期；{len(r['checks'])}项总断言；五bundle完整性通过。",'',
        '## 本轮工程状态','',
        '| 模块 | 已完成 | 未完成/边界 |','| --- | --- | --- |',
        '| H0 | 用户表全部保留；54die/256核、1020FP16、SRAM/DRAM/D2D；独立几何拒绝 | 几何冲突、未知带宽/时延/精度口径未擅自补齐 |',
        '| 更新 | 尾批不丢弃、多epoch、每批独立token分母、zero/clip/Adam/barrier/publish | 仍单trajectory microbatch，无打包、shuffle、数值训练 |',
        '| 策略版本 | 一轮采集behavior冻结，Actor每次更新+1，Reference固定 | 无异步/陈旧权重策略 |',
        '| 执行 | 原生Astra与资源时钟、TP2教学H、旧A/B及GQA回归 | 主RL未切换54die mesh/256核；非H0性能 |',
        '| 全迁移 | 本轮独立子阶段完成 | 多核/NoC、54节点完整流程、未截短真实形状、校准仍未完成 |','',
        '## 本次实际仿真结果','',
        '| 运行 | model ns | 操作数 | 优化器更新 | Actor末版本 | DRAM峰值B |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for label,m in r['metrics'].items():
        peaks={k:v for k,v in m['memory_peak_bytes_by_location'].items() if k.endswith('.dram')}
        lines.append(f"| {label} | {m['total_simulated_time_ns']:.3f} | {m['operation_count']} | {m['optimizer_updates']} | {m['actor_version_end']} | {peaks} |")
    lines+=['','时间来自原生Astra事件；输入为小模型合成fixture，本地成本为未校准解析假设，不能解读成正式H0芯片耗时或学习质量。','',
        '## 完整流程与复现','',
        '1. 独立读H0原表并转单位，报告48/54几何不一致及未知项；require-executable应rc2。',
        '2. 读schema1.1实际形状/完整group，按原有流程rollout→reward→reference→old-policy→完整组advantage。',
        '3. 原顺序分批；每epoch/每minibatch：zero→单trajectory forward/loss/backward累加→norm/clip→Adam RMW→barrier→publish。',
        '4. 模板展开后才建立唯一张量存储和资源合同；behavior/旧logprob全轮保留，tape逐trajectory释放，round末统一cleanup。',
        '5. Chakra trace进入Docker中的固定commit Astra；独立核对issue/callback、梯度形状和容量/READY/生命周期。',
        '6. 两轮8次更新、拆组优势保持、分母6/3、重复事件逐字节相同、旧toy/GQA不变、独立native mesh和资源负例回归。','',
        '```bash','cd /path/to/research/rl-xy',
        'python3 scripts/schedule_acceptance.py --run-id <唯一ID> --container <新容器名>',
        'PYTHONPATH=src python3 scripts/write_schedule_report.py --acceptance acceptance/<ID>/report.json --out runs/schedule_delivery/<ID>',
        '```','',
        '细节：`docs/09_H0_SOURCE_AND_GEOMETRY.md` 和 `docs/10_OPTIMIZER_SCHEDULE.md`。','',
        '## 全部工程文件说明','',f'共{len(paths)}文件；运行产物通过acceptance/report与各bundle manifest索引，不混入源码。','',
        '| 文件 | 内容与功能 |','| --- | --- |']
    lines += [f'| `{name}` | {FILE_INFO[name]} |' for name in paths]
    a.out.mkdir(parents=True,exist_ok=False)
    text='\n'.join(lines)+'\n'
    (a.out/'工程说明与完整流程.md').write_text(text)
    (a.out/'流程预览.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>H0与多更新执行</title><style>body{max-width:1180px;margin:32px auto;padding:0 24px;background:#fafafa;color:#20242b}pre{white-space:pre-wrap;font:15px/1.8 system-ui;overflow-wrap:anywhere}</style><pre>'+html.escape(text)+'</pre></html>')
    print(f'{len(paths)} described files; {a.out}')


if __name__=='__main__':main()
