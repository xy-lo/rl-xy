# 张量生命周期与实际资源准入

## 输入、输出和职责

`tensor_plan.py` 接收已展开端点搬运的rank-local操作图，生成 `tensor_specs.json` 和 `resource_contract.json`；`resource_validation.py` 独立重放C++实际资源日志。前者在发射前预检，后者在执行后核对；真正的运行时准入由AstraSim进程中的 `docker/astrasim/RlxyResources.hh` 完成，不由Python预造时间戳。

张量记录包含storage/tensor身份、shape、dtype字节、nbytes、rank、类别、round、ALLOC节点、READY节点、FREE节点、persistent。每个逻辑对象独立分配，不做alias、重计算、紧凑复用或碎片化模拟。复制参数与TP分片按已有参数表初始化，权重、master、梯度、Adam m/v分开。

## 当前覆盖的张量

前向GEMM产品、attention score/softmax/AV、norm输出和统计、FFN与残差、LM logits；完整反向dinput/dweight、中间激活梯度、norm/embedding参数梯度、残差分支梯度加法；trajectory、token、reward、advantage、old/ref log-prob、KV和通信目标输出。loss反向包含所有teacher-forcing位置的本地词表梯度，prompt位置按mask处理但仍占完整存储。每个microbatch各die的参数梯度partial总字节应等于常驻梯度缓冲3232B。

当前使用保守的out-of-place策略：无梯度前向中间量保留至该次调用结束，训练前向tape及反向partial保留至该microbatch梯度累积完成。保守pin集合 `reads` 表示这些对象必须已READY且不可提前释放，不表示kernel逐一读取pin集合的全部字节。DRAM流量另由shape/映射或明确向量合同计算；两者不可相加当作实测访问。当前峰值是这套已声明对象/保留策略的峰值，不是最优分配器峰值或完整实体芯片峰值。

## ALLOC、READY、FREE与KV预留

输出在issue时ALLOC并置为未READY，生产者callback后READY；消费者只有读集合均READY才能issue。通信目标空间在native网络节点发射时分配，但直到目标endpoint_write callback才READY。网络到达不等于数据已写入目标DRAM。

KV按公开最大长度预留未兑现字节，QKV产生的增量将预留转为实际分配；已分配字节和剩余预留不重复计算。轨迹完成释放KV并释放相应未兑现预留；轮末回到初始化持久对象集合。KV复制额外读写显式进入局部成本。

## 运行时接口与统一时钟

`workload.patch` 将C++检查接入原生Workload：can_issue查询可用性，issue分配并获取资源，callback置READY/释放。`RLXY_RESOURCE_CONTRACT` 由runner只对子进程设置，无合同则走历史原生路径。每rank输出 `astra_logs/resources.<rank>.jsonl`，记录actual tick、DRAM、SRAM、KV剩余预留和local-memory owner，必须逐条对应Astra日志issue/callback。

每die一个logical core。GEMM/向量/端点搬运使用排他local-memory lease；Astra原生compute lane继续约束计算。当前inclusive串行成本将compute与DMA视为同一服务段，lease覆盖整个段，是保守baseline，不是compute/DMA可重叠引擎。host/control不持有该资源。集成fixture用独立CPU/GPU原生lane验证lease确实阻止重叠；该fixture的CPU标签只为构造独立执行lane，不代表目标晶圆使用CPU处理DMA。

SRAM工作区在issue获取、callback释放；映射工作区不能超过容量。当前严格串行主调度下，单操作SRAM或DRAM需求超容量直接拒绝，不实现等待其他live对象释放后的动态DRAM调度；排他资源或共享SRAM工作区暂时忙则等待callback重新扫描。逐storage维护active readers/writer；写入需等待旧读者/写者结束，FREE时仍有活跃访问直接报错。缺对象、双重分配/释放、只读Reference写入、预留负数也失败关闭。

## 输出与不覆盖项

`tensor_objects.csv`/`tensor_specs.json` 是完整声明对象清单；`objects.csv` 是历史persistent/KV账本，保留用于语义交叉检查。`memory.csv` 在resource模式下来自C++actual tick独立重放，`legacy_memory.csv` 保存旧小计。不要把两套账相加。

native collective内部临时buffer、runtime metadata、allocator碎片、DMA启动/描述符开销、逐bank冲突、NoC包/路由/VC和多核共享带宽未建模。NIC计划容量检查仍只覆盖原P2P packet假设，不将native collective内部buffer冒充已核算。

## 验证入口

`tests/test_tensor_resources.py` 覆盖shape/梯度、READY延迟、KV预留、提前释放、OOM、两轮无增长；`scripts/integration_fixtures.py` 绕过Python预检，要求C++对DRAM/SRAM超容、缺读对象、double-free、Reference写入、active-reader期间FREE和释放后读七类错误直接失败。另用无数据DAG边的独立CPU/GPU lane检查未READY会阻止消费者；对比lease开关检查资源确实改变actual issue ticks。
