#!/usr/bin/env python3
"""Generate 0.4 execution walkthrough, evidence table and complete inventory."""
import argparse
import html
import json
from pathlib import Path

from write_input_report import FILE_INFO,project_files
from rl_xy.reporting import deterministic_tree_hash

FILE_INFO.update({
    'src/rl_xy/execution_contract.py':'多prompt/group语义、独立组内demean和compact QKV宽度；未知reward不伪造。',
    'src/rl_xy/execution_validation.py':'通用阶段/版本/训练/形状及实际Astra事件检查，独立于旧A/B曲线。',
    'scripts/make_execution_case.py':'实际HF形状+指定完整长度group编译；记录来源、未知数值和显式反事实response cap。',
    'scripts/execution_acceptance.py':'0.4 Docker编排：多组/两轮/重复/带宽/实际模型/旧长度短探针和原生回归。',
    'scripts/write_execution_report.py':'从最终PASS和匹配源码生成本报告及全文件说明。',
    'docker/Dockerfile.execution':'复用0.3原生后端，打包0.4 Python执行层，不安装GPU训练依赖。',
    'tests/test_qwen_execution.py':'Qwen参数/KV金标准、多组/未知reward/绑权/生命周期、RoPE/GQA梯度恒等式和合同变异负例。',
    'docs/components/qwen_execution.md':'Qwen2/GQA执行职责、数据流、验收方式、固定来源和不覆盖的能力。',
    'src/rl_xy/workflow.py':'兼容旧toy；Qwen2 compact QKV/bias/RoPE/GQA、独立group和全局单次更新事件。',
    'src/rl_xy/model.py':'MHA/GQA、QKV bias与绑权后的精确unique/local参数、持久状态和compact KV。',
    'src/rl_xy/types.py':'类型合同与TokenExtent长度对象；长度不假装token值。',
    'src/rl_xy/model_import.py':'实际Qwen2配置/参数检查和受限算子回放能力；正式H门禁仍保持。',
})
for name in ('case.json','astra_system.json','astra_network.yml','astra_remote_memory.json'):
    FILE_INFO['configs/qwen2_group_fixture/'+name]={
        'case.json':'小GQA/绑权/RoPE、两group四异长trajectory手工fixture及明确教学硬件。',
        'astra_system.json':'FIFO、TP ring、roofline关闭的原生系统合同。',
        'astra_network.yml':'TP2教学FullyConnected带宽8B/ns和时延1model ns，不是正式晶圆mesh。',
        'astra_remote_memory.json':'无外部内存扩展，所有容量由原生资源合同检查。'}[name]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--acceptance',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    r=json.loads(a.acceptance.read_text())
    if r['status']!='PASS' or deterministic_tree_hash(root)!=r['source_tree_sha256']:
        raise ValueError('requires actual PASS bound to current source')
    paths=project_files(root)
    unit=(root/next(c['log'] for c in r['commands'] if c['label']=='unit_tests')).read_text()
    unit_summary=next(line for line in unit.splitlines() if line.startswith('Ran '))
    md=['# rl-xy 0.4：Qwen2/GQA与多组同步执行层','',
        '本报告从实际Docker PASS证据生成；日期2026-10-26。唯一活动源码为服务器/path/to/research/rl-xy。','',
        '本轮接通受限Qwen2算子级执行，不是数值RL训练、完整54模块晶圆或硬件校准。旧数据探针显式修改response cap为1，不能将它称作原旧训练负载的性能。','',
        f"镜像：`{r['image']}`；ID：`{r['image_id']}`。",'',
        f"源码树：`{r['source_tree_sha256']}`；单测：{unit_summary}；总断言{len(r['checks'])}项。",'',
        '## 1. 本轮工程状态','',
        '| 部分 | 实际完成 | 仍未完成 |','| --- | --- | --- |',
        '| Qwen2算子 | compact GQA QKV、bias、default RoPE、因果mask、repeat KV与反向归并 | FlashAttention/非default RoPE/滑窗/MoE/量化 |',
        '| 绑权 | 参数/Adam/梯度只分配一份；embedding/LM-head两项贡献合并 | 数值权重训练及与真实框架算子融合对齐 |',
        '| 多group流程 | 逐trajectory prompt；组内demean；全局token分母；单trajectory microbatch共享一次更新 | 多trajectory打包、多minibatch/epoch、PP/DP |',
        '| 长度输入 | TokenExtent保持未知token/reward/logprob；完整选定group可执行 | 全116/30步原长度执行及真实逐轨迹采集 |',
        '| H执行 | 两die/单逻辑核；原生资源与网络fixture回归 | 54模块/256核、模块NoC与多核共享带宽 |',
        '| 科学证据 | 原生Astra issue/callback、资源回放、来源hash、小数值恒等式 | 可信profile/DNN/硬件校准/学习质量 |','',
        '## 2. 本次实际Astra结果','',
        '| 运行 | model ns | 操作数 | DRAM峰值/die B | 来源口径 |','| --- | ---: | ---: | --- | --- |']
    for label,m in r['metrics'].items():
        scope={'legacy':'原A/B金标准回归','actual_qwen_multigroup':'实际24层Qwen形状+手工极短两组输入，教学H',
               'actual_qwen_old_length_probe':'旧统计合成group/prompt，response cap显式改1；假设TP2 H，非原负载性能'}.get(label,'小GQA多group合成fixture')
        peaks={k:v for k,v in m['memory_peak_bytes_by_location'].items() if k.endswith('.dram')}
        md.append(f"| {label} | {m['total_simulated_time_ns']:.3f} | {m['operation_count']} | {peaks} | {scope} |")
    md += ['', '时间来自Astra实际事件，不是GPU观测秒数；算子成本仍是解析OS/向量假设，因此不等于实体芯片耗时。原生collective内部buffer/碎片/逐bank争用未覆盖。','',
        '## 3. 完整流程','',
        '1. 核对0.3冻结证据和活动源码，保留所有旧运行；新建本轮镜像/容器，不请求GPU。',
        '2. 严格读取1.1合同：组完整、同组prompt一致、同步版本、未知值和合法TP切分。',
        '3. 从实际HF配置导入24层/896 hidden/14 query heads/2 KV heads等形状；494032768 unique参数，状态按绑权只计一次。',
        '4. forward生成compact QKV→bias→Q/K RoPE→compact KV保存→临时repeat→因果attention→SwiGLU→LM head。',
        '5. rollout最后token不回灌；按public cap预留，按真实已处理位置materialize，完成即释放；多group teacher-forcing求reference/old-policy形状。',
        '6. 组内advantage、全局loss分母；每trajectory forward/backward；GQA重复头梯度求和、RoPE转置旋转、bias token轴归约和绑权梯度合并。未知reward不产生虚假数值。',
        '7. 一次梯度clip/AdamW形状更新→all-shard barrier→actor版本发布；Reference保持初始版本。',
        '8. OS合法SRAM映射/流量→显式通信端点→Chakra→原生Astra时钟与资源hook；独立重放日志、shape和hash。',
        '9. 两轮峰值/版本、重复事件逐字节一致、带宽敏感性、旧toy不变和原生mesh/resource负例分别验收。','',
        '## 4. 复现','', '```bash','cd /path/to/research/rl-xy',
        'python3 scripts/execution_acceptance.py --run-id <新的唯一ID> --container <新容器名>',
        'PYTHONPATH=src python3 scripts/write_execution_report.py --acceptance acceptance/<ID>/report.json --out runs/execution_delivery/<ID>',
        '```','',f"正式验收：`{a.acceptance}`；旧输入编译说明：`{r['old_compilation']}`；原生fixtures：`{r['integration']}`。",'',
        '## 5. 后续验收门槛','',
        '先补多核映射/模块共享带宽及完整mesh全流程，再将正式H缺参按明确假设或可信来源补齐，分层做敏感性与校准。原长度长组运行还需要控制trace/JSON规模与磁盘占用；不能把cap1短探针外推为完整step吞吐。多minibatch/epoch必须显式定义更新次数、old-policy固定范围和重算，不可直接把轮数当epoch。','',
        '## 6. 全工程文件说明','',f'共{len(paths)}个文件。运行产物按验收report/各manifest索引，不混入活动源码。','',
        '| 文件 | 包含什么、有什么功能 |','| --- | --- |']
    md += [f'| `{path}` | {FILE_INFO[path]} |' for path in paths]
    a.out.mkdir(parents=True,exist_ok=False)
    text='\n'.join(md)+'\n'
    (a.out/'工程说明与完整流程.md').write_text(text)
    (a.out/'流程预览.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><title>rl-xy0.4执行层</title><style>body{max-width:1180px;margin:32px auto;padding:0 24px;background:#fafafa;color:#20242b}pre{white-space:pre-wrap;font:15px/1.8 system-ui;overflow-wrap:anywhere}</style><pre>'+html.escape(text)+'</pre></html>')
    print(f'generated {len(paths)} file descriptions at {a.out}')


if __name__=='__main__':main()
