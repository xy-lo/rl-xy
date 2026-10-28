#!/usr/bin/env python3
"""Publish small result summaries without raw multi-GB traces or credentials."""
import argparse,hashlib,html,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--acceptance',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=read(a.acceptance);assert r['status']=='PASS' and all(x['exit_status']==0 for x in r['commands'])
    a.out.mkdir(parents=True,exist_ok=False)
    summaries={};comparisons={}
    for label,relative in r['bundles'].items():
        b=ROOT/relative;m=read(b/'metrics.json');manifest=read(b/'manifest.json')
        assert manifest['software']['rl_xy_source_tree_sha256']==r['source_tree_sha256']
        summaries[label]=dict(metrics=m,manifest_sha256=sha(b/'manifest.json'),
            summary=read(b/'summary.json'),global_stage_times=read(b/'global_stage_times.json'),
            input_hashes={x.name:sha(x) for x in (b/'inputs').iterdir() if x.is_file() and not x.name.startswith('resources.')},
            native_command=read(b/'astra_command.json'))
        summaries[label]['metrics'].get('gpu_profile',{}).pop('gemm_measurements',None)
    for label,relative in r['comparisons'].items():comparisons[label]=read(ROOT/relative/'comparison.json')
    (a.out/'results.json').write_text(json.dumps(dict(source_tree_sha256=r['source_tree_sha256'],bundles=summaries,comparisons=comparisons),ensure_ascii=False,indent=2)+'\n')
    clean={k:v for k,v in r.items() if k not in ('metrics',)}
    (a.out/'acceptance.json').write_text(json.dumps(clean,ensure_ascii=False,indent=2)+'\n')
    rows=['# 0.7：veRL同步分时共置协议的双平台仿真','',
      '本实验对齐固定 veRL 版本的受限同步协议；所有时间为原生 AstraSim 事件时钟输出。GPU主对照也是仿真，未使用旧单卡冷首步缩放。**结果是条件性体系结构预测，尚不足以写成完整veRL端到端硬件加速结论。**','',
      f'实验源树 SHA256：`{r["source_tree_sha256"]}`；镜像：`{r["image"]}` / `{r["image_id"]}`。Git提交在实验后生成，不能把后生成的commit回填为当时源码身份。','',
      'veRL v0.9.1 / 1876b06d0a3e4e71e06230be10af14492ca8a75b；dense Megatron、PP1、单trajectory训练microbatch、完整batch一次更新、BF16 GEMM/FP32参数梯度累积、无offload。选用naive full-tensor gathered export与独立推理权重，连续rank推理TP组。两侧共同使用完整GRPO组LPT分配和按步打包生成线性算子；并非默认veRL balancing/padding或全部vLLM融合实现。','',
      '| 控制场景 | 训练TP→生成TP | 晶圆活动/分配 | GPU活动/分配 | 晶圆model ms | GPU model ms | GPU/晶圆时间比 |','| --- | --- | --- | --- | ---: | ---: | ---: |']
    for name,c in comparisons.items():
        assert not c['blockers']
        rows.append(f'| {name} | {c["train_tp"]}→{c["rollout_tp"]} | {c["wafer_active"]}/{c["wafer_allocated"]} | {c["gpu_active"]}/{c["gpu_allocated"]} | {c["wafer_model_ms"]:.6f} | {c["gpu_model_ms"]:.6f} | {c["gpu_over_wafer_model_time"]:.6f} |')
    rows+=['','时间比>1只表示当前成本与负载下晶圆模型较快，<1表示GPU模型较快。没有为了结果方向调整硬件常量，不把单次确定性仿真重复当作统计样本。','',
      '0.5B为原32组×2、64条真实短长度回放，prompt70–127/response4。3B和7B是官方结构配置加P512/R32固定长度、32组×2的明确合成控制；不代表这些模型生成了此回答或有相同学习质量。Exact模型配置与来源在configs/models和configs/colocation/sources.json。','',
      'H0保留54die、256核/die、每核1.25MB SRAM、每die64GB DRAM/2TB/s，D2D2TB/s采用显式per-edge/per-direction假设。48A100主对照总BF16峰值高6.209%、容量高11.111%，其它资源不等；7B晶圆52die活动+2die闲置仍按54die计预算。完整资源差异见results.json各comparison.budget。几何scope和执行参数的test_assumption见协议说明。','',
      '| 实现内容 | 本轮检查 |','| --- | --- |',
      '| 全局阶段屏障 | 所有rank原始issue/callback验证rollout结束后才启动后续阶段 |',
      '| 权重重分片 | 非复制参数TP ALL_GATHER、完整export scratch、转换/加载、独立rollout副本显式收费 |',
      '| 训推切换 | resume/sleep控制延时+对象分配释放；训练分片保留，无免费隐含反向复制 |',
      '| 更新与驻留 | 全step token分母、DP SUM、Actor版本链/Reference冻结、SRAM/DRAM峰值与读写释放回放 |',
      '| TP4与闲置设备 | 原生10GPU fixture检查连续TP子组与2个idle rank；不放宽parser |','',
      f'验收共{len(r["commands"])}条命令全部退出0；逐命令argv、退出状态、日志SHA在acceptance.json。大体积原始日志仍保留在服务器runs与本地证据归档，不在GitHub轻量摘要中，公开摘要不能代替原始日志复核。','',
      'A100热cuBLAS BF16 GEMM微基准已另行采集，数值fixture与重复性只覆盖GEMM子模型；若comparison未列GPU_GEMM_measurement_profile_sha256，主表使用解析GPU成本，不能声称主表已按这些标签校准。向量/网络/控制与wafer尚未校准。','',
      '论文可用表述：“在声明硬件预算、受限同步共置协议和解析算子成本下的仿真对比”。不能写“完整veRL已部署到晶圆芯片”“已验证硬件加速比”或“学习效果不下降”。当前缺少GPU完整融合/缓存/调度误差、wafer可信成本、默认分配/padding复现及双方完整并行搜索，正式主结论应在补齐后给出。','',
      '后续3B/7B安排：先在当前固定TP控制下扫P512/2048、R128/512/1024，再扫B64/128/256与n2/4/8；将长度来源和容量失败显式记录。第二层双方分别搜索合法TP（3B1/2，7B1/2/4）、microbatch/accumulation；PP2/4与offload先实现再纳入。第三层做未知vector/SRAM/NoC/host延时敏感性、微基准/集群通信校准与多真实长度样本；最后才扩展稳态数值质量。','',
      '协议与逐模块说明：[同步共置](../../docs/components/colocation.md)。复现：构建完整镜像链后执行`python3 scripts/colocation_acceptance.py --run-id <unique-id> --container <unique-name>`。旧0.6实验保持冻结；0.6与0.7同时变更协议、精度及映射，二者差异不能单独归因于阶段屏障。']
    text='\n'.join(rows)+'\n';(a.out/'实验结果.md').write_text(text)
    (a.out/'预览.html').write_text('<!doctype html><meta charset="utf-8"><title>veRL同步共置仿真</title><style>body{max-width:1120px;margin:40px auto;font:16px/1.7 system-ui}pre{white-space:pre-wrap}</style><pre>'+html.escape(text)+'</pre>')
    print(a.out)
if __name__=='__main__':main()
