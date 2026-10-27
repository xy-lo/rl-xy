# 多minibatch、多epoch：执行语义与限制

组件：`training_schedule.py` 编排；`training_validation.py` 独立检查；
`types/config` 参数；`workflow` 接入；`tensor_plan` 唯一存储与tape；
`resource_validation` 每次访问的梯度形状；`reporting` 更新次数/分母/版本。
这是算子事件与状态移动仿真，不执行loss、Adam权重或数值训练。

schema1.1新增可选策略字段：`training_epochs`正整数，默认1；
`optimizer_minibatch_trajectories`非负整数，默认0表示整批。
旧schema1.0仅允许默认值，旧case和A/B金标准保持不变。

1. 每轮只采集一次rollout、reference/old-policy形状与完整group优势。未知reward仍为null。
2. 按输入trajectory顺序划分优化器minibatch，不随机打乱；最后不足一批不丢弃。
3. epoch重复同一采集批次，不重复rollout；训练microbatch仍为单trajectory，不做padding/packing。
4. 每个优化器minibatch先zero FP32梯度，逐trajectory forward/loss/backward累加；
   分母为该minibatch有效response token总数。组优势在划分前计算，即使组被拆开也不重新demean。
5. 每次累加完后norm/clip→AdamW形状RMW→全shard barrier→Actor版本+1。
   同一轮behavior version始终不变，reference固定初始版本；下一轮行为策略用上一轮最后发布版本。
6. 训练tape在每trajectory梯度累加回调后释放；round payload保留至所有epoch结束后统一cleanup。
   没有声称模拟了真实框架中的activation checkpointing或融合内核。

例：四trajectory A/B/C/D响应长度3/1/2/3，minibatch大小3、epoch2：
每轮依次ABC分母6、D分母3，再ABC、D，共4次优化器更新；两轮共8次。
Actor0→8，采集behavior分别0/4，Reference始终0。
该token-mean口径显式固定，不保证与任意veRL版本默认loss aggregation完全相同。

独立验收从case重建批次、epoch、版本链与分母，检查每次zero、完整反向、clip/Adam/barrier/publish；
原生Astra真实issue/callback和C++资源日志仍独立回放。测试篡改behavior、Actor、清零、优势和分母必须拒绝。

入口：`configs/qwen2_updates_fixture/`；`scripts/schedule_acceptance.py`。
H0源表与这个教学TP2执行fixture是两个独立输入，不能将该fixture时间称为54die芯片时间。
