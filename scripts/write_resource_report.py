#!/usr/bin/env python3
"""Generate the v0.2 Chinese file/flow guide strictly from PASS Docker evidence."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
from write_walkthrough import FILE_INFO,PHASES,project_files


FILE_INFO.update({
    'docker/Dockerfile.mapped':'在固定0.1镜像上接入资源、CPU统计修正和mesh扩展，编译两个analytical后端并冻结patch hash。',
    'docker/astrasim/RlxyResources.hh':'实际Astra时钟的ALLOC/READY/FREE、容量、KV预留、SRAM和排他local-memory准入；错误强制非零退出。',
    'docker/astrasim/workload.patch':'向上游Workload的can_issue、issue、callback插入资源hook，不另建事件时钟。',
    'docker/astrasim/statistics.patch':'允许CPU/控制节点参与回放；以GPU/COMM区间并集计算overlap，避免CPU空隙误计或无符号下溢。',
    'docker/astrasim/RlxyMesh.hh':'新增无wrap二维mesh、XY路由；使用原生Device/Link/Chunk排队并输出实际路径。',
    'docker/astrasim/mesh.patch':'在congestion-aware frontend显式选择mesh2d override；无override保留上游行为。',
    'src/rl_xy/operator_ir.py':'有类型的张量shape/dtype/layout、Matmul方向与KernelSpec及互斥计时合同。',
    'src/rl_xy/mapping.py':'单逻辑核OS合法tile枚举与精确ragged/batch流量；不将非法tile静默塞入SRAM。',
    'src/rl_xy/cost_model.py':'解析inclusive-local成本和带来源/OOD拒绝的精确profile lookup接口；未训练DNN。',
    'src/rl_xy/mapped_costs.py':'在操作图不变的条件下应用typed GEMM和显式向量/状态RMW成本，不生成时间戳。',
    'src/rl_xy/communication_lowering.py':'跨die通信展开端点读/原生网络/端点写，并补反向残差相加；重建rank-local依赖。',
    'src/rl_xy/tensor_plan.py':'完整声明对象、保守tape/反向partial保留、KV预留兑现和每节点可执行资源合同。',
    'src/rl_xy/resource_validation.py':'独立重放C++资源日志并与实际Astra ticks逐节点交叉核验，生成主流程内存峰值。',
    'tests/test_mapping_cost_model.py':'OS手算、ragged独立循环、SRAM合法性/自动重映射、带宽/dtype/profile失配等测试。',
    'tests/test_tensor_resources.py':'tape/梯度shape、延迟READY、容量/生命周期/预留失败、只读Reference及两轮无增长。',
    'scripts/integration_fixtures.py':'Docker内真实执行资源竞争/READY门控/七类C++负例及4rank共享/独立mesh路由反事实。',
    'scripts/resource_acceptance.py':'宿主只编排Docker命令；单轮/两轮/repeat、带宽/SRAM敏感性、OOM、native fixture、完整性与版本冻结。',
    'scripts/write_resource_report.py':'从资源版PASS证据生成本文件与HTML预览；逐文件说明缺项即拒绝。',
    'docs/components/mapped_costs.md':'映射/流量公式、计时合同、analytic/lookup来源与DNN边界。',
    'docs/components/tensor_resources.md':'张量类别、保守pin与真实流量的区别、准入时序、资源合同及未建模buffer。',
    'docs/components/network.md':'mesh适配、原生排队、四rank反事实、YAML override和拓扑适用边界。',
    'docs/资源版工程说明与完整流程.md':'v0.2自动生成的验收、指标、阶段时间、模块及全部文件索引；不替换历史报告。',
    'docs/资源版完整流程预览.html':'v0.2自包含轻量预览，含实际指标和逐阶段数据。',
})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--acceptance',required=True,type=Path)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    def read(path):return json.loads(path.read_text())
    accepted=read(a.acceptance.resolve())
    if accepted['status']!='PASS':raise ValueError('PASS evidence required')
    bundles={n:root/b for n,b in accepted['bundles'].items()}
    metrics={n:read(b/'metrics.json') for n,b in bundles.items()}
    manifests={n:read(b/'manifest.json') for n,b in bundles.items()}
    for n,m in manifests.items():
        if m['status']!='success' or m['software']['rl_xy_source_tree_sha256']!=accepted['source_tree_sha256']:
            raise ValueError('run/source mismatch '+n)
    integration=read(root/accepted['integration']/'report.json')
    comparison=read(root/accepted['integration']/'network/network_comparison.json')
    if integration['status']!='PASS':raise ValueError('native fixtures not accepted')
    # Include report outputs even before their first generation.
    for rel in ('docs/资源版工程说明与完整流程.md','docs/资源版完整流程预览.html'):
        (root/rel).touch(exist_ok=True)
    paths=project_files(root)
    toy=metrics['toy'];software=manifests['toy']['software']
    unit=next(c for c in accepted['commands'] if c['label']=='unit_tests')
    unit_summary=next(l for l in (root/unit['log']).read_text().splitlines() if l.startswith('Ran '))
    md=['# rl-xy 0.2：资源版工程说明与完整流程','',
        '交付日期：2026-10-26（Asia/Shanghai）。这是实际Docker/AstraSim运行证据生成的工程报告，不是目标硬件精度声明。','',
        '## 1. 本次新增内容','',
        '原0.1是算子级控制/通信回放骨架。0.2在算子级在线执行不变的前提下，补充typed shape、SRAM合法OS映射、DRAM流量、本地搬运成本、保守activation tape/反向partial、KV预算、实际资源准入和独立mesh竞争测试。没有把DNN当成缺失硬件标签的替代品。','',
        '远端唯一活动工程：`/path/to/research/rl-xy`；Docker容器内：`/workspace/rl-xy`；运行容器：`'+accepted['container']+'`。无sudo，无GPU数值训练，未提交/推送Git。','',
        '## 2. 实际验收结果','',
        '| 场景 | 总时间 / model ns | 逻辑操作 | rank节点 | 内部检查 |',
        '|---|---:|---:|---:|---:|']
    for n,m in metrics.items():
        checks=read(bundles[n]/'checks.json')['checks']
        md.append(f"| {n} | {m['total_simulated_time_ns']:.3f} | {m['operation_count']} | {m['operation_rank_event_count']} | {sum(c['passed'] for c in checks)}/{len(checks)} |")
    md+=['',f"单测：`{unit_summary}`，OK。总验收{len(accepted['checks'])}条检查全部通过，另有{len(integration['checks'])}条原生资源/网络fixture断言。所有正例命令退出0；OOM命令预期非零，该非零不能被误报为验收失败或成功运行。",'',
        f"验收入口：`{a.acceptance.resolve().relative_to(root)}`。每条命令、实际退出码、日志和SHA-256均在该报告中。",'',
        '原0.1的100264.270ns与当前结果不是优化前后speedup：模型合同、访存、反向产品与成本均改变，不可作性能收益比较。两轮验证版本和容量复用，不强求任意异步rank时间恰好是单轮2倍。','',
        '## 3. 运行身份与复现','', '| 字段 | 本次值 |','|---|---|']
    for k in ('rl_xy_version','container_image_tag','container_image_id','astrasim_commit',
              'astrasim_binary_sha256','rl_xy_source_tree_sha256','python'):
        md.append(f'| {k} | `{software[k]}` |')
    md+=['','AstraSim为固定官方commit加本工程透明扩展，并非未修改的上游二进制；Workload资源hook、CPU统计修正、二维mesh适配均保留源码和hash：','',
         '```text',software['astrasim_local_resource_patches'].strip(),'```','',
         '源码hash覆盖src、configs、tests、scripts、Docker及手写文档；排除4份自引用的0.1/0.2派生MD/HTML、运行目录、缓存和日志。结果manifest另对全部运行文件hash。','',
         '```bash','cd /path/to/research/rl-xy','bash scripts/docker.sh build-mapped rlxy-astrasim:0.2.0',
         'RLXY_CONTAINER=rl-xy-mapped bash scripts/docker.sh create-container rlxy-astrasim:0.2.0',
         'python3 scripts/resource_acceptance.py --container rl-xy-mapped --run-id <新的唯一ID>',
         'python3 scripts/write_resource_report.py --acceptance acceptance/<新的唯一ID>/report.json','```','',
         '基础镜像为已构建的0.1项目镜像（其底层为pjlab/deeplearning:v2.2.3-cuda12.8.0-torch2.7.0）。从零构建0.1参见docker组件说明。已存在但镜像不同的容器会被拒绝，不自动删除旧容器。每次run拒绝覆盖目录。','',
         '## 4. 分层架构与完整执行链','',
         '```text','case W/H/S → workflow typed operator DAG → communication endpoint lowering',
         '          → tensor plan/ALLOC-READY-FREE → SRAM mapper + local costs',
         '          → resource_contract + per-rank Chakra ET → actual AstraSim',
         '          → issue/callback + resource logs → independent replay/checks → hashed bundle','```','',
         '在线不逐tile执行：离线mapper计算合法tile及其流量/工作区，在线算子服务段使用该成本并获取资源。Python不计算“仿真完成时间”替代Astra；所有最终时间与资源快照均来自native issue/callback。','',
         '| 层 | 功能与关键文件 |','|---|---|',
         '| workload | `config.py/model.py/workflow.py`：同步GRPO、TP2、版本与typed matmul方向 |',
         '| mapping/cost | `operator_ir.py/mapping.py/cost_model.py/mapped_costs.py`：OS/SRAM/DRAM及来源 |',
         '| tensor/resource | `tensor_plan.py/RlxyResources.hh/resource_validation.py`：对象、准入、同一时钟核验 |',
         '| network | `communication_lowering.py/astra_adapter/`：端点服务、native collectives/P2P，独立mesh fixture |',
         '| evidence | `run.py/reporting.py/check_result.py/resource_acceptance.py`：失败关闭、归档、不可混淆的指标 |','',
         '## 5. 一个完整同步GRPO轮','',
         '模型2层、hidden8、MHA2heads、vocab16；2die各1logical core、TP2。输入prompt=[1,2]；A回答3token/reward1，B回答2token/reward0；去均值优势+0.5/-0.5，全局loss分母5。只回放shape/依赖/资源，不计算这些张量的数值。','',
         '| 阶段 | 本轮动作 | actual start / ns | actual end / ns | span / ns |','|---|---|---:|---:|---:|']
    for phase,label,description in PHASES:
        t=toy['phase_timing']['r00.'+phase]
        md.append(f"| {label} | {description} | {t['start_ns']:.3f} | {t['end_ns']:.3f} | {t['span_ns']:.3f} |")
    md+=['','阶段起止为该phase所有rank事件的min(issue)/max(callback)，局部1ns控制marker可与另一rank后续事件交叠，不能将phase span直接求和当wall time。完成时间是所有rank最终callback最大值。actor经过两条microbatch累积后更新并发布；Reference保留原版本且禁止写入。','',
         '## 6. 访存、容量和生命周期','',
         '| 项目 | 本轮值 |','|---|---|',
         f"| 声明对象数（跨rank、含持久对象） | {toy['tensor_storage_count']} |",
         f"| local DRAM reads / writes（跨rank；非网络字节） | {toy['local_dram_read_bytes']} / {toy['local_dram_write_bytes']} B |"]
    for loc,peak in toy['memory_peak_bytes_by_location'].items():md.append(f'| {loc} peak allocated | {peak} B |')
    md+=['','这些峰值包含当前模板声明的前向激活/tape、反向产品、KV、trajectory和通信目标输出。保守out-of-place/no-recompute保留不等于最优allocator；`reads`为生命周期pin集合，不等于实际kernel逐一读全部pin对象。实际访问量由kernel成本合同单独计算。','',
         '参数1576 unique、每die808 elements；长期状态12928B，加常驻FP32梯度为16160B/die。每microbatch参数梯度partial3232B/die。旧16352B/die只是旧常驻+KV小计，不能代表本版全轮峰值。','',
         '8×8×8 OS手算：tile4工作区128B，读512B/写128B；tile2工作区32B，读1024B/写128B。更小SRAM会改变映射和流量。inclusive-local-serial时间=算术时间+本地搬运时间；局部DRAM/module/SRAM路径只按瓶颈带宽计一次。','',
         '网络目标ALLOC与READY分开：native到达后仍须endpoint_write callback；KV预留转实际分配不得重复计费。runtime snapshot逐条与Astra日志同tick核验，轮末恢复持久对象，无KV/tape/资源泄漏。','',
         '## 7. 资源与网络反事实','', '| 检查 | PASS | 实际证据 |','|---|---|---|']
    for c in integration['checks']:
        evidence={k:v for k,v in c.items() if k not in ('check','passed')}
        md.append(f"| {c['check']} | {c['passed']} | `{json.dumps(evidence,ensure_ascii=False)}` |")
    md+=['','资源独立lane65ns与共享lease129ns来自原生callback。七种负例直接进入C++，未仅依赖Python预检；错误必须非零退出，避免上游回调异常被捕获后假成功。另检查ALLOC未READY会阻止无数据DAG边的独立lane消费者。','',
         '网络为独立4rank/2×2mesh XY fixture，8192B probe 0→1→3；干扰流延迟512ns，共享1→3或独立2→0。aware采用原生store-and-forward Link队列，unaware不表达争用。','',
         '| 配置 | unaware probe / model ns | aware probe / model ns |','|---|---:|---:|']
    for n in ('solo','disjoint','shared'):
        md.append(f"| {n} | {comparison[n+'_unaware']['probe_receive_service_ticks']/1000:.3f} | {comparison[n+'_aware']['probe_receive_service_ticks']/1000:.3f} |")
    md+=['','aware与unaware的solo差异来自原生模型的逐跳/端到端计时差异，不是拥塞损失；拥塞对比必须在同一backend内比较solo/shared。同样，fixture不是全流程已扩为4rank或大规模晶圆。','',
         '## 8. 全部工程文件说明','',
         f'下表覆盖{len(paths)}个工程文件，不包括外部上游依赖、Git内部、缓存、runs/acceptance/build_logs和本文派生证据。逐文件遗漏会使报告生成失败。','',
         '| 文件 | 内容与功能 |','|---|---|']
    for rel in paths:md.append(f'| [{rel}](../{rel}) | {FILE_INFO[rel]} |')
    md+=['','## 9. 结果包怎么读','',
         '先看 `summary.json/status.json/checks.json`，再看 `manifest.json` 的code/image/binary/input/artifact hash。`operations.jsonl` 是计划，`events.jsonl` 是actual issue/callback；二者不能混用。`resource_contract.json` 定义逐节点读/分配/READY/FREE及资源；`astra_logs/resources.<rank>.jsonl` 是C++实际资源快照。','',
         '`tensor_specs.json/tensor_objects.csv` 为完整声明对象清单；`objects.csv/legacy_memory.csv` 为历史持久/KV交叉账，不能与新 `memory.csv` 相加。`cost_estimates.json` 给shape、映射、流量、时长与来源；`transfers.csv` 的logical payload不是物理网络每跳总字节。','',
         '| bundle | 远端相对目录 |','|---|---|']
    for n,b in accepted['bundles'].items():md.append(f'| {n} | `{b}` |')
    md+=['','## 10. 与实施计划对齐及剩余边界','',
         '| 项 | 当前状态 |','|---|---|',
         '| M1算子合同 | 已接typed matmul方向、dtype、layout/shape和timing合同；非任意框架图导入器 |',
         '| M2参考mapper | 已接OS合法映射、ragged/batch流量、带宽/SRAM敏感性；非经过校准的cycle-level核仿真 |',
         '| M3张量 | 当前模板保守tape/反向产品/KV/端点空间与真实准入已接；不声称tight allocator峰值 |',
         '| M4资源/网络 | local-memory/SRAM/容量原生hook与独立mesh争用验证已接；主流程仍固定2die同步，非细粒度多核/NoC模型 |',
         '| M5预测器 | 精确profile lookup接口和来源/OOD拒绝已实现；CLI主成本使用analytic provider。无真实标签，未训练DNN或宣称预测精度 |',
         '| M6真实输入 | 目前仅严格配置化toy模板与Chakra适配；未完成真实框架trace导入或大模型校准 |','',
         '仍未建模native collective内部buffer、allocator碎片、DMA启动、bank/NoC/VC/细粒度竞争；无目标晶圆硬件校准、误差数据、功耗或数值学习质量结论。下一步需真实profile标签与held-out校准后，才能接DNN并扩到真实RL workload；不要以本验收替代这些证据。','']
    text='\n'.join(md);(root/'docs/资源版工程说明与完整流程.md').write_text(text)
    preview='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>rl-xy 0.2 资源版流程</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;background:#f5f7fa;color:#182b42;font:16px/1.75 system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:28px;border-radius:12px}h1{color:#164a80}</style><h1>rl-xy 0.2：实际验收与完整流程</h1><p>轻量文本预览。完整结构化文件请阅读同目录Markdown；不执行网络脚本。</p><pre>'+html.escape(text)+'</pre></html>'
    (root/'docs/资源版完整流程预览.html').write_text(preview)
    print(f'PASS: documented {len(paths)} files; report and HTML generated')


if __name__=='__main__':main()
