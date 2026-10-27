# 单 die 映射成本组件

## 职责与状态

`operator_ir.py`、`mapping.py`、`cost_model.py`、`mapped_costs.py` 实现M1/M2。它们不计算数值张量，不预测学习质量。成本来源明确为 `analytic_os_reference_v1_not_hardware_calibrated`，不是DNN或实测晶圆数据。

`run --resource-model` 接入映射成本、张量计划和实际Astra资源准入；`run --local-cost-model mapped` 只接入成本。旧的默认legacy路径为历史回归保留，不改写已有release。新流程请使用 `scripts/docker.sh toy-resource rlxy-astrasim:0.2.0 <ID>`。正式验收在Docker中完成，以新验收报告为准。

## 输入和输出

`TensorSpec` 有身份、shape、dtype字节宽度和layout；`Matmul` 包含M/N/K、batch、操作数/输出/累加精度及forward/dinput/dweight方向；`KernelSpec` 记录具体kernel组和向量/额外访存工作量。Transformer GEMM与attention矩阵乘在workflow调用点显式传shape，不通过FLOPs反推矩阵。

反向线性层将dinput与dweight作为两次串行matmul。attention backward使用四次matmul加显式向量操作；mapped模式修正旧版按forward乘常数估计的FLOPs，旧值保留在metadata便于审计。

`AnalyticalCostProvider.estimate()` 返回本地时长、算术时长、本地搬运时长、DRAM读写、SRAM峰值工作块、映射与来源。`cost_estimates.json` 保存每个compute的spec和估计；操作图不补造AstraSim时间戳。

## OS参考映射

固定单逻辑核，output-stationary，无跨输出tile缓存、无跨算子缓存、无预取或双缓冲。保留输出累加块直到全部K块结束；A/B按每个输出tile重新读取。边界块按实际尺寸计算，batch串行复用工作区。

对M×K乘K×N，tile为m/n/k，工作区为 `m*k*a_bytes + k*n*b_bytes + m*n*accumulator_bytes`；DRAM读为 `batch*(M*K*ceil(N/n)*a_bytes + K*N*ceil(M/m)*b_bytes)`，写为 `batch*M*N*output_bytes`。禁止把整个输入矩阵硬塞入SRAM。

自动候选为受上限约束的1/2/4/8/16/32块以及小维度的实际边界；从合法候选中按访问量及确定性tie-break选取。固定tile放不下立即拒绝；自动映射无合法候选时拒绝，不静默超容量。

## 计时合同

当前仅支持 `inclusive_local_serial`。时间为算术时间＋本地搬运时间，不假定重叠。本地路径以DRAM、module path、SRAM三者瓶颈带宽计费一次，不把路径各段时间重复相加。module path只是合成带宽约束，不声称已模拟NoC packet、bank或路由。

GEMM算术按typed matmul的FLOPs计数；向量算子默认使用声明的“两读一写”流式流量模型，梯度清零、梯度范数与AdamW使用独立读写合同。向量工作块16B；不把整个向量放入SRAM。所有这些都是透明测试假设，不是目标kernel的精确访存测量。

跨die通信由AstraSim计时。`communication_lowering.py` 在native网络节点前后补独立端点DRAM读写；`tensor_plan.py` 和 `RlxyResources.hh` 负责生命周期/容量/资源准入，见对应组件文档。不是逐tile在线模拟或精确硬件存储模型；多核映射、bank/NoC细粒度竞争未实现。

## 查询表与DNN边界

`LookupCostProvider` 按完整kernel/hardware签名精确查找；缺记录或缺label_origin/reference_version时失败，不静默回退。检查时长、SRAM和计费合同，不在本组件生成“实测”标签。未训练DNN；没有可信profile前不应训练一个只拟合旧FLOP公式的模型。

## 验证

`tests/test_mapping_cost_model.py` 包含8×8×8手算640/1152B流量与128/32B工作区、ragged/batch的独立循环对照、固定tile容量失败、自动重映射、dtype/非法shape、带宽敏感性、profile缺失/来源失败和完整toy的shape覆盖/成本敏感性。Docker实跑后还需记录实际命令、image ID、输出与全部语义检查。
