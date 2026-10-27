# 48 A100近预算仿真对照

## 当前选择与论文依据

用户接受48张A100的总FP16峰值与显存容量略高于54die晶圆，主线改为两边均仿真。
旧0.5单A100首step实测保留为真实长度/配置/计时采集证据，不再作为主性能分母，不按卡数/峰值比例缩放42.55秒。

- WSC-LLM PDF第11页§5.3：6×8张A100-80GB；GPU14976TFLOPS、3840GB，晶圆约14100TFLOPS、3456GB；400GB/s节点间带宽。GPU不是所有资源都严格相等。
- WATOS PDF第11页§V-C：8张Blackwell Ultra，GPU40000TFLOPS与晶圆39648TFLOPS接近；将GPU DRAM2304GB调整到3920GB。说明论文允许显式资源归一化，但不能把改后规格称为真实SKU。
- 两文的2TB/s带宽句子不自动等于两套系统聚合带宽相等。本文独立声明每GPU2039GB/s、每die2000GB/s再汇总，不混Gb/s、GB/s或双向总量。

本工程采用第一种近预算口径，**不是复现Splitwise/Megatron论文分数**。工作负载仍为受支持的同步GRPO、实际Qwen0.5B与相同64条长度。当前控制TP2，GPU为DP24、晶圆为DP27；不是GPU最优并行搜索。

## 资源与来源

48张A100-SXM80GB分6节点、每节点8卡。按NVIDIA datasheet密集FP16/BF16312TFLOPS（不是稀疏624）、FP32 CUDA19.5TFLOPS、80GB HBM、2039GB/s；108SM、每SM164KiB共享内存来自架构白皮书/调优指南。

主源：

- https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf
- https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
- https://docs.nvidia.com/cuda/ampere-tuning-guide/

`gpu_model.budget_audit`逐项列出FP16/BF16、FP32、vector、HBM容量/聚合带宽、私有SRAM总量。GPU FP16峰值约高6.21%、显存高11.11%，但聚合HBM带宽/SRAM容量低于晶圆；FP32/vector差别还受晶圆未验证假设影响，不能笼统说GPU所有硬件更强。

## 独立GPU成本，不复用Dojo映射

`GPUCostProvider`按矩阵形状枚举32×64、64×128、128×128、256×128 CTA，K工作集32，只在每SM共享内存合法的候选中按声明计算+本地服务成本取最小。避免固定128×128使小矩阵的SM利用率人为偏低；这仍不是cuBLAS全部映射空间。按CTA轮转分配SM，不split K；不把164KiB×108汇成共享池。
精确计算ragged CTA的FLOP、输入重复读取和输出写入，核验FLOP守恒。无跨CTA L2复用、每SM一次一个CTA，尚无寄存器/warp/流水/真实CUDA融合优化。
每SM计算以GPU总峰值÷SM数的有理比例整数上取整，不将总312TFLOPS再乘108。HBM是整GPU共享瓶颈；SM共享带宽256B/ns、launch1us为独立场景假设。
计算/本地服务保守串行，launch只收一次并单列解释在compute_ns；不是cycle-accurate GPU模型，没有硬件校准。

原IR含FP32/混合梯度GEMM。主设置对任何4B输入按FP32 CUDA峰值计算，这是**声明精度的未融合参考**，并非声称PyTorch BF16 backward真的这样执行。
因此额外跑optimistic sensitivity：所有GEMM按BF16 TensorCore峰值计、忽略cast成本、launch为0，刻意偏向GPU。它不是物理FP32吞吐，也不是整个runtime的严格下界（仍存在CTA流量/未融合限制）。

## 原生网络与共享上行

`RlxyGpu.hh`创建GPU端点、每节点一个NVSwitch抽象节点与一个spine。路径：

```text
同节点：GPU -> node switch -> GPU
跨节点：GPU -> node switch -> shared node uplink -> spine -> destination switch -> GPU
```

NVLink600GB/s暂解释为总双向，各方向300GB/s，节点间400GB/s按WSC文字设为每节点每方向共享uplink；两处口径均在profile显式记录。每hop250/1000ns是场景假设，不冒充论文测量。节点内crossbar无内部额外阻塞，spine端口独立；不是无限制GPU全互连。
原生Link/Chunk队列负责竞争，不把每GPU对都赋予独立400GB/s。原生fixture证实同上行争用比独立全双工路径更慢，同节点不走spine。未模拟NIC packet/credit回压，不能据此宣称完整NIC硬件。

## 复用与新边界

只复用模型/GRPO算子图、张量生命周期、TP/DP通信语义、资源日志和验证接口。
通过可选provider注入到mapped_costs/tensor_plan/wafer_graph，不改原wafer默认provider；旧4die clock必须保持1675.970model ns。
GPU仍用原生Astra tick，记录独立GPU patch/header和二进制hash，不能把Python成本求和当网络仿真时间。

结果初步判定要同时看四组：原晶圆vs主GPU、原晶圆vs偏向GPU、较低晶圆FP32/vector rate64场景vs主GPU、低rate晶圆vs偏向GPU。
rate64不是测量或正式H参数；其目的在于揭示512/256不确定值可能带来的结论翻转，不为晶圆胜出调参。
比较门禁要求实际全step/无截短、每条group/长度、模型和算法、更新数、GPU primary预算均通过。输出时间比只叫场景结果，`feasibility_proven=false`，不能仅凭比值放行论文结论。

## 运行

```bash
export RLXY_RUNS_ROOT="$(readlink -f runs)"
python3 scripts/gpu_sim_acceptance.py --run-id <唯一ID> --container <新容器名>
PYTHONPATH=src python3 scripts/write_gpu_sim_report.py --acceptance acceptance/<ID>/report.json --out runs/gpu_delivery/<ID>
```

不申请GPU设备，不再启动数值训练。完整GPU/敏感性过程在CPU侧Docker原生后端运行。最终文档只从匹配源码的实际PASS生成；失败轮保留。
