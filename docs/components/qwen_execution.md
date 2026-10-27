# Qwen2/GQA与多组同步执行层

本模块扩展是**算子形状、资源和事件回放**，不加载权重、不做数值训练，不声称正式54模块晶圆性能。`schema_version=1.0`的原A/B fixture保持金标准；`1.1`要求原生资源检查，避免使用旧硬编码KV曲线。

## 文件与边界

| 文件 | 职责 |
| --- | --- |
| types.py / config.py | 可选QKV bias、RoPE、绑权；1.1完整group和逐trajectory prompt；TokenExtent只表示长度，无伪造token值 |
| execution_contract.py | prompt/group解析和组内demean；未知reward保持None，不冒充零奖励 |
| workflow.py | compact QKV→bias→RoPE→显式repeat KV→query-head attention；反向repeat梯度归并→RoPE转置旋转→bias梯度归约→compact QKV backward |
| model.py | QKV宽度 `(Q+2KV)×head_dim`；bias参数；绑权只分配一组参数/优化器/梯度状态 |
| tensor_plan.py | rotated K/原V的compact KV；重复头仅是临时激活，不增大持久KV；保守out-of-place tape与梯度贡献 |
| execution_validation.py | 通用形状/阶段/训练/真实Astra tick检查，独立于旧A/B金标准 |
| resource_validation.py | 从原生回调日志重放容量、READY、lease、释放；绑权两份梯度贡献合并到一份持久梯度 |
| make_execution_case.py | 实际HF配置+标准L1长度→完整指定group的可执行case；不截断、不恢复reward/logprob，不使用GPU时间作芯片cost |

## 明确执行策略

TP=2，每die一个逻辑核，PP=DP=1；全部选定trajectory按公共cap预留KV；轮转发射但单rank计算仍串行；训练每microbatch一条trajectory。默认所有组共享一次全批response-token分母和一次更新；0.4.1可显式指定optimizer minibatch和epoch，每批独立token分母及zero/clip/AdamW形状更新/权重发布，行为策略在所有epoch内冻结。不同group各自demean且划分minibatch后不重算。多轮fixture复用声明输入，仅用于版本/释放回归，不代表策略变化后真实生成的内容。详见[更新语义](../10_OPTIMIZER_SCHEDULE.md)。

绑权LM头与embedding只有一份持久storage；反向分别产生两份梯度贡献，显式相加后累积。推理KV在QKV bias/RoPE后写入，尺寸按KV heads而不是query heads；最后一个生成token不再送入模型。

RoPE采用default完整偶数head_dim，显式table与apply/backward向量工作；不是实测融合kernel延时。GQA采用物化repeat作为保守参考数据流，未实现FlashAttention；attention按dense矩形执行并带因果mask语义，不声称最优三角计算。向量streaming流量仍为声明的保守假设。

## 来源与验证

结构依据固定版本的[官方Transformers Qwen2实现](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2/modeling_qwen2.py)：Q/K/V有bias、O/MLP无bias，Q/K旋转，KV按query/KV比重复。仅核验结构并独立实现cost/IR，未复制其实现，也不将当前版本当成历史训练版本。

测试覆盖小尺寸手算、compact参数与KV、绑权状态唯一性、GQA/RoPE/bias梯度小型数值恒等式、异长group/未知reward、非法合同负例、真实Astra callback资源检查，以及旧toy时间不变。单测不替代硬件校准。

尚未覆盖：54模块/256核执行、TP以外的PP/DP、每microbatch多trajectory打包、随机shuffle/dropout、滑窗/非default RoPE、MoE/量化、数值RL质量、可信profile/DNN/硬件校准。多epoch/minibatch的显式确定顺序已实现；old policy在同批所有更新内固定，不错误地每次更新后重算为新Actor。

## 大模型日志与短探针

首次实际24层结构运行触发上游10MiB日志轮转，旧parser只读log.log，严格检查发现早期事件缺失并拒绝成功。现读取log.N.log到log.log全部连续分段，重复/缺失事件和任意分段critical依然拒绝；manifest包含所有原始分段。失败准备轮不追认成功。

服务器磁盘紧张，旧输入默认验收仅选一个完整group，显式`--probe-response-cap 1`；compilation.json同时保留原长度和执行长度、改动标记、原输入hash。它用于证明输入→实际模型形状→Astra资源执行闭环，不是原旧负载性能。另有实际Qwen形状的手工极短两group/decode例，不冒充旧数据。完整长组/全步运行仍需控制事件和资源合同文件规模，不能从短探针外推。

第二个准备轮又暴露大时间戳下浮点ns相减的精度问题：整数tick配对完整，但绝对ns约十亿时差值无法满足固定1e-9误差。修正为整数callback-issue严格相等、每个ns字段分别等于tick/scale；不放宽整数duration或因果检查，并新增长时间戳正例与篡改ns负例。失败记录保留，不以退出0替代完整门禁。
