#!/usr/bin/env python3
"""Full file inventory and workflow, generated only from matching actual PASS."""
import argparse
import html
import json
from pathlib import Path
from write_schedule_report import FILE_INFO,project_files
from rl_xy.reporting import deterministic_tree_hash

FILE_INFO.update({
    'configs/hardware/xy_h0_case3_performance.json':'54die Case3来源合同；Fig3几何仅模板参考，不缩计算die或宣称制造布局验证。',
    'configs/hardware/fixture_4die.json':'独立4die/4核native回归来源，不是正式H0。',
    'src/rl_xy/multicore.py':'N/M/batch输出分区、ragged输出/FLOP守恒、私有SRAM和共享DRAM/模块瓶颈，BF16/FP32独立rate。',
    'src/rl_xy/resource_encoding.py':'严格rank顺序的读集合delta和内容寻址workspace，无损还原与篡改拒绝。',
    'src/rl_xy/wafer_graph.py':'完整group轮转分DP、全局token分母、真实TP/DP通信组与梯度SUM、round fence及唯一资源映射。',
    'src/rl_xy/wafer_run.py':'来源绑定的显式H场景、rank分文件、原生mesh与独立完整事件/内存/梯度检查、证据归档。',
    'scripts/private_sram_fixtures.py':'原生同核/异核争用、单核溢出/错误总和/核数负例与raw/compact时钟等价。',
    'scripts/gpu_smoke.py':'专用GPU容器中受限veRL真训练/长度采集；模型/数据/包/源码hash及失败日志。',
    'scripts/gpu_capture_wrapper.py':'只在私有容器适配既有collector，另存未舍入阶段时间和完整解析配置，不激活主机flag。',
    'scripts/prepare_matched_case.py':'真实GPU成功step的全部group导入，保持每条原长度，无cap改动及GPU计时注入。',
    'scripts/compare_gpu_wafer.py':'严格匹配模型/trace/group/长度/算法更新后输出GPU观测与场景预测；失配拒绝比值。',
    'scripts/wafer_acceptance.py':'Docker单测、逐核/旧native、4/54die、原toy、64条真实GPU全step与对比/完整性门禁。',
    'scripts/write_wafer_report.py':'读取实际匹配源码的PASS与比较报告，生成全部文件说明和轻量HTML。',
    'docker/Dockerfile.wafer':'固定0.4.1基础上编译逐核/压缩资源hook，记录新增header hash并打包0.5 Python。',
    'docs/components/wafer_execution.md':'54die/多核/DP语义、几何和假设、GPU隔离/复现、对比口径、资源压缩及未实现边界。',
    'tests/test_wafer_graph.py':'私有映射/守恒、共享带宽、DP不均匀长度分母/组、几何scope、无损压缩与拒绝负例。',
    'tests/test_comparison.py':'纯合成单测验证真实比较的门禁：cap、模型/日志hash、更新次数失配不能输出ratio。',
    'scripts/docker.sh':'无sudo Docker构建/受控容器复用，精确UID/项目/镜像检查；可挂载已核对链接的自有外部证据目录。',
})
for directory in ('wafer4_fixture','wafer54_fixture'):
    for name,description in {'case.json':'明确合成完整group工作负载，原始TP2逻辑模板，hierarchical入口展开DP。',
        'wafer.json':'mesh/DP、来源H路径及每项未校准执行假设，绝不混成论文实测参数。',
        'astra_system.json':'基础FIFO/TP ring模板，运行时生成两维物理metadata及明确通信组。',
        'astra_network.yml':'历史TP2模板元数据；run-wafer从显式scenario生成真实mesh配置。',
        'astra_remote_memory.json':'无外部扩展内存，实际容量仍由native资源hook校验。'}.items():
        FILE_INFO[f'configs/{directory}/{name}']=description


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--acceptance',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=Path(__file__).resolve().parents[1]
    r=json.loads(a.acceptance.read_text())
    if r['status']!='PASS' or r['source_tree_sha256']!=deterministic_tree_hash(root):raise ValueError('actual matching PASS required')
    paths=project_files(root);comparison=json.loads((root/r['comparison']/'comparison.json').read_text())
    unit=(root/next(c['log'] for c in r['commands'] if c['label']=='unit_tests')).read_text()
    summary=next(line for line in unit.splitlines() if line.startswith('Ran '))
    lines=['# rl-xy 0.5：54die多核mesh、真实veRL输入与GPU对照','',
        '**这是完整受支持单step的仿真迁移与预测对照，不是实体晶圆部署、veRL所有功能迁移或实测晶圆加速比。**','',
        f"镜像 `{r['image']}`，ID `{r['image_id']}`。",'',f"源码树 `{r['source_tree_sha256']}`。",'',
        f"{summary}；{len(r['commands'])}条命令符合预期；{len(r['checks'])}项总断言。",'',
        '## 实际执行结果','',
        '| 运行 | model ns | die / 核每die | DP×TP | trajectory | 操作 / rank事件 |',
        '| --- | ---: | --- | --- | ---: | --- |']
    for label,m in r['metrics'].items():
        lines.append(f"| {label} | {m['total_simulated_time_ns']:.3f} | {m.get('dies',2)} / {m.get('cores_per_die',1)} | {m.get('data_parallel',1)}×{m.get('tensor_parallel',2)} | {m.get('trajectory_count',2)} | {m['operation_count']} / {m['operation_rank_event_count']} |")
    lines+=['','## GPU观测与晶圆场景预测','',
        f"比较状态：{comparison['status']}。GPU step观测 `{comparison['gpu_step_observed_s']}`秒；晶圆场景预测 `{comparison['wafer_step_predicted_s']}`秒。",'',
        f"时间比 `{comparison['scenario_time_ratio_NOT_validated_speedup']}` **不是已验证speedup**。",'',
        '单张A100与54die的资源预算不同；GPU是首step真实计时，晶圆是未校准解析成本。没有等面积/功耗/算力、稳态或学习质量结论。',
        '本例真实GPU设置response cap4，全部64条实际长度4，并非事后把旧负载改短；实际prompt长度逐条保留，完整32group回放。奖励全0，不用于学习效果。','',
        '## 完整流程','',
        '1. 保留54die的Case3性能参数；模板几何与实际布局分离，未知执行率独立列为场景假设。',
        '2. 在获授权的空闲GPU上用专用Docker运行真实veRL；只读既有环境/权重，私有flag采集逐条长度及未舍入阶段时间。',
        '3. 检查GPU退出/采集成功及模型/数据/source hash，导入全部原group，不cap、不使用GPU秒数拟合chip成本。',
        '4. 完整group分配DP副本，各副本按全step有效token分母累加，原生DP ALL_REDUCE SUM后clip/Adam/publish，round fence维持同步。',
        '5. 每kernel输出分区到私有核、合法OS tile、共享DRAM/模块服务；原生逐核SRAM准入和张量ALLOC/READY/FREE。',
        '6. 资源合同无损压缩并按rank加载；真实TP/DP通信组进入6×9无wrap XY mesh，原生Link排队。',
        '7. 严格整数tick、完整事件矩阵、依赖、逐副本反向/版本/分母、梯度形状及C++资源日志独立重放；归档manifest。',
        '8. 严格匹配后输出GPU实测与晶圆预测对照；任何截短/部分组/模型/算法/更新失配都拒绝比值。','',
        '## 当前边界','',
        'hierarchical入口当前为全采集批次一次更新/epoch，可多epoch；通用DP多minibatch尾批/空副本、PP、异步区域、packing/checkpointing/MoE等未完成。',
        'NoC为共享带宽服务，不是逐bank/VC模型。FP32/BF16/vector/NoC/SRAM和D2D口径仍含假设；DNN/可信标签/硬件校准未完成。',
        '不能把本例成功当成实体芯片数值veRL后端或正式论文主表完成。','',
        '## 复现','', '```bash','cd /path/to/research/rl-xy',
        'export RLXY_RUNS_ROOT="$(readlink -f runs)"  # 仅runs已经迁到独立自有目录时设置',
        'python3 scripts/wafer_acceptance.py --run-id <新ID> --container <新容器名> --gpu runs/gpu_matched/<已完成GPU_ID>',
        'PYTHONPATH=src python3 scripts/write_wafer_report.py --acceptance acceptance/<ID>/report.json --out runs/wafer_delivery/<ID>',
        '```','',f"实际验收 `{a.acceptance}`；GPU报告 `{r['gpu_report']}`；对比 `{r['comparison']}`。",'',
        'GPU环境隔离/缓存/头文件/collector说明及所有假设见docs/components/wafer_execution.md；失败准备轮保留，不追认成功。','',
        '## 全工程文件说明','',f'共{len(paths)}个文件；运行证据由acceptance/report与各bundle manifest索引。','',
        '| 文件 | 内容与功能 |','| --- | --- |']
    lines += [f'| `{path}` | {FILE_INFO[path]} |' for path in paths]
    text='\n'.join(lines)+'\n';a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'工程说明与完整流程.md').write_text(text)
    (a.out/'流程预览.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>54die与GPU对照</title><style>body{max-width:1180px;margin:32px auto;padding:0 24px;background:#fafafa;color:#20242b}pre{white-space:pre-wrap;font:15px/1.8 system-ui;overflow-wrap:anywhere}</style><pre>'+html.escape(text)+'</pre></html>')
    print(f'{len(paths)} files documented; {a.out}')


if __name__=='__main__':main()
