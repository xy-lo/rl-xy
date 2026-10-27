#!/usr/bin/env python3
"""Generate evidence-led near-budget comparison and complete source inventory."""
import argparse
import html
import json
from pathlib import Path
from write_wafer_report import FILE_INFO,project_files
from rl_xy.reporting import deterministic_tree_hash

FILE_INFO.update({
    'src/rl_xy/gpu_model.py':'独立GPU SM/CTA分区/FLOP/私有shared memory/共享HBM成本，有理峰值分摊；GPU与晶圆预算审计。',
    'docker/astrasim/RlxyGpu.hh':'GPU端点、每节点switch、共享uplink及spine，实际native Link竞争。',
    'docker/astrasim/gpu.patch':'固定aware入口增加GPU topology选择，不改默认mesh分支。',
    'docker/Dockerfile.gpu':'在0.5上编译独立GPU网络扩展，保存patch/header与二进制身份。',
    'configs/gpu/a100_48_reference.json':'48A100-SXM80GB真实名义规格、声明输入精度模型及未校准launch/延迟/共享带宽假设。',
    'configs/gpu/a100_48_optimistic.json':'刻意偏向GPU的GEMM TensorCore/零launch敏感性，不是物理FP32吞吐。',
    'configs/gpu/wafer_typed_rate_sensitivity.json':'晶圆FP32/vector降到64的未知率敏感性，不替换正式H0来源值。',
    'scripts/gpu_network_fixtures.py':'原生验证节点共享上行竞争、独立双向路径和同节点绕过spine，保持完整统计门禁。',
    'scripts/compare_simulated_platforms.py':'两边都仿真；严格全step/输入/算法/近预算合同，输出所有资源差异，拒绝用实测秒数预测GPU。',
    'scripts/gpu_sim_acceptance.py':'先GPU4/旧mesh/native门禁，再独立完整GPU/偏向GPU/晶圆低rate运行与四组比较，不预设赢家。',
    'scripts/write_gpu_sim_report.py':'仅从绑定实际PASS生成结果、所有文件说明及HTML预览。',
    'tests/test_gpu_model.py':'GPU CTA ragged/FLOP/私有容量/聚合峰值/带宽/精度敏感性与近预算差异测试。',
    'tests/test_simulated_comparison.py':'纯合成比较门禁：可翻转赢家、禁止实测替代、不同capture和预算不足拒绝比值。',
    'docs/components/gpu_simulation.md':'论文方法、48卡近预算、GPU独立成本/网络、精度与硬件不确定性及四组比较边界。',
})


def main():
    p=argparse.ArgumentParser();p.add_argument('--acceptance',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1];r=json.loads(a.acceptance.read_text())
    if r['status']!='PASS' or deterministic_tree_hash(root)!=r['source_tree_sha256']:raise ValueError('actual matching PASS required')
    paths=project_files(root);w=json.loads((root/r['frozen_wafer_control']/'metrics.json').read_text())
    unit=(root/next(c['log'] for c in r['commands'] if c['label']=='unit_tests')).read_text()
    lines=['# rl-xy0.6：48GPU仿真与54die晶圆近预算对照','',
        '**不是单卡实测缩放，不是芯片实测，不是校准精度/学习质量证明。原始0.5证据保持冻结。**','',
        f"源码树`{r['source_tree_sha256']}`，镜像`{r['image_id']}`。",'',
        next(x for x in unit.splitlines() if x.startswith('Ran '))+'；'+str(len(r['commands']))+'条命令符合预期。','',
        '## 运行结果','', '| 运行 | model ns | 设备/核或SM | 操作/rank事件 |','| --- | ---: | --- | --- |',
        f"| frozen wafer0.5 | {w['total_simulated_time_ns']} | 54/256 | {w['operation_count']}/{w['operation_rank_event_count']} |"]
    for label,m in r['metrics'].items():
        lines.append(f"| {label} | {m['total_simulated_time_ns']} | {m['dies']}/{m['cores_per_die']} | {m['operation_count']}/{m['operation_rank_event_count']} |")
    lines+=['','## 四组条件性比较','',
        '| 组 | 晶圆预测秒 | GPU预测秒 | GPU/晶圆时间比 | 本场景晶圆更快 |',
        '| --- | ---: | ---: | ---: | --- |']
    for label,c in r['outcomes'].items():
        lines.append(f"| {label} | {c['wafer_predicted_s']} | {c['gpu_predicted_s']} | {c['gpu_over_wafer_time_ratio_SCENARIO_ONLY']} | {c['wafer_faster_in_this_scenario']} |")
    b=r['outcomes']['primary']['budget_audit']
    lines+=['','## 预算合同','', '| 总资源 | 晶圆 | GPU | GPU相对差异% |','| --- | ---: | ---: | ---: |']
    lines += [f'| {k} | {v} | {b["gpu"][k]} | {b["gpu_minus_wafer_percent"][k]:.6f} |' for k,v in b['wafer'].items()]
    lines+=['','## 如何解释','',
        '采用WSC-LLM第11页的6×8A100近预算思路；WATOS第11页也有显式调整GPU容量的比较，但本轮不造虚拟GPU以强求所有资源相等。',
        'GPU的FP16/显存更高，不代表所有FP32/向量/SRAM/带宽都更高。所有主/敏感性结果原样报告，不因GPU胜出而调参到晶圆胜出。',
        '若晶圆在四组中均较快，只能增强当前建模假设下可行性的证据；若发生翻转，说明优势依赖假设，必须保留翻转。',
        'GPU是声明精度、未融合CTA参考，not a calibrated PyTorch/vLLM model；偏向GPU组用TensorCore峰值处理FP32/混合GEMM和零launch，非真实硬件峰值，也非整个runtime严格下界。',
        '晶圆低rate组将FP32/vector变64，只是敏感性，未改写原H0。GPU/晶圆主流程TP2固定，不是最优GPU策略搜索。',
        '所有负载均为此前真实64条短回答长度，不能外推长序列/大模型质量或稳态。硬件校准/可信标签/最优kernel与通信算法仍待完成。','',
        '## 工程流程','',
        '模型和完整L1长度→同一GRPO/张量IR→独立GPU SM/CTA或晶圆OS成本→独立网络拓扑→原生Astra与资源hook→全事件/生命周期验证→来源和预算门禁→四组对照。',
        'GPU使用节点内switch/节点间共享uplink，native fixture验证竞争而不是无限全互连；不申请GPU、不重新启动数值训练。',
        '新增和修改模块、精度口径/网络假设/来源及复现命令见docs/components/gpu_simulation.md。','',
        '## 全部文件说明','',f'共{len(paths)}文件。','', '| 文件 | 功能 |','| --- | --- |']
    lines += [f'| `{path}` | {FILE_INFO[path]} |' for path in paths]
    text='\n'.join(lines)+'\n';a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'GPU仿真对照与工程说明.md').write_text(text)
    (a.out/'预览.html').write_text('<!doctype html><meta charset="utf-8"><style>body{max-width:1180px;margin:32px auto;padding:0 24px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.8 system-ui}</style><pre>'+html.escape(text)+'</pre>')
    print(f'{len(paths)} files documented')


if __name__=='__main__':main()
