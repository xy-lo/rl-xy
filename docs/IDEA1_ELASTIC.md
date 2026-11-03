# Idea1：可变推理 TP 与独立训练区域

本扩展将实际推理TP与训练区域解耦；TUNE搜索锁定后，以同版源完成4B/8B/14B两个窗口的完整原生复跑。结果、TP搜索、逐die状态及适用边界见[动态TP实验](../experiments/2026-11-03-idea1-elastic/README.md)。固定推理TP2的冻结对照见[上一版完整系统](IDEA1_COMPLETE.md)，原证据保持不变。

## 资源含义

54 个 die 上保留 27 对 TP2 权重、梯度及分布式优化器 owner。这是状态保存和原生控制的基础单位，不等于推理仍固定有 27 个 TP2 实例。

推理算子实际支持 TP1、TP2、TP4、TP8。TP4 紧凑布局可以是 13 个 TP4 加 1 个 TP2；TP8 紧凑布局可以是 6 个 TP8 加 1 个 TP4 加 1 个 TP2。余下的 die 保留为较小实例，不丢弃。TP1 每对 die 上有两个真正的 TP1 请求引擎；缺少的 BF16 半份权重先通过原生通信复制，输出记录复制完成才报告 EOS。其双 die 控制器使用公共安全点，可能有等待，不声称等价于完全异步的 TP1 硬件实现。

训练区域独立于推理分组，支持 TP2/4/8 区域混合组成训练 DP。一个 die 的顶层属性是 ROLLOUT 或 TRAINING；执行活动和临时借算租约另行记录。区域内推理退出、数据到齐且状态可用后才能开始训练。每轮完成后将梯度显式恢复到原有 TP2 optimizer owner，按全局样本数求和并统一更新一次。

## 分层路径

| 层 | 责任 | 实现 |
|---|---|---|
| 公开入口 | 校验输入身份、切断未来回答信息、规划与执行分离 | `scripts/idea1_layered/run_elastic_system.py` |
| 联合资源选择 | 在已通过原生 TUNE 的推理 TP、物理分组、cap、训练 DP 和放置候选间选择 | `elastic_planner.py` |
| 任务放置 | 重用 TUNE 选定的放置规则，对当前历史重新预测、分散长任务、域内长短互补；可选择按实际推理吞吐独立放置 | `inference_search.py`、`assignment_v2.py` |
| 容量和坐标 | 实际 TP 的 KV/prefill 上界、54 die 精确覆盖、原权重子视图或 TP1 拼接 | `inference_groups.py`、`coordinates.py` |
| 原生执行 | 实际 TP 的全层算子、完整参与者集合、全组开始/完成屏障 | `native/inference_cohorts.py` |
| 未开始回答接管 | 根节点间 prompt 搬运、目标 TP 内部广播、全实例租约、原训练 home 不变 | `native/cohort_pending.py` |
| 在线扩组 | 按已观测上下文和历史条件剩余长度决定 TP2→4 或 TP4→8；支付 KV/元数据搬运 | `native/inference_cohorts.py` |
| 训练协作和更新 | 保留微批转移、原生容量约束、梯度恢复和一次更新 | `native/region_training_help.py`、`native/gradient_protocol.py` |
| 独立审计 | 逐回答 prefill/decode/EOS、KV 头区间、退休依赖、prompt 广播、全 54 rank 实际回包成本复算 | `scripts/audit_idea1_inference.py` |

规划器只接收 prompt 身份、公开 P、上限与独立历史快照。实际回答长度在规划结束之后才用于绑定和原生回放。TUNE 原生结果可用于拟合选择器的模型偏差，但不能修改仿真器算子成本、通信带宽或最终报告时间。

## 在线安全点和代价

扩组只能发生在完整原生计算块结束、没有未提交 prefill/decode、源实例没有待开始回答的安全点。新成员必须空闲且未执行 teacher/reference/backward；可暂借尚未启动准备工作的 DATA_WAIT 区域，借算结束恢复角色。

KV 依照实际 GQA 头区间重分片。原目的 die 的旧 KV 全部读出并退休后，才允许覆盖该缓存；远程发送先于本地压缩。每个旧源都有原生退休门禁，随后还有整个新 TP 的迁移完成屏障。只有头字节量相等或 Python 列表顺序正确，不足以证明原位覆盖安全。早期缺少该协议的开发扩组结果不进入最终候选银行。

迁移成本仍采用既有的流式 pack/传输抽象及有界临时缓冲假设。原生门禁与字节量检查验证该抽象下的依赖，不构成真实 KV 张量、DMA 分块实现或物理分配器的验证。

在线只实现扩大 TP 和全部 EOS 后释放，尚未实现活跃 KV 的任意缩组、TP1 在线扩组或强制抢占已经开始训练的 die。训练 TP/DP 在一轮开始前选择；训练中协作仍按已验收的未开始微批接管协议执行。

通信沿既有 mesh 路由，在原生后端的有向链路队列中竞争，搬运不会穿越繁忙 die 而无代价。调度准入使用距离和预期收益，尚未读取实时链路排队或执行自适应路由。

## 搜索和解释边界

“最优”仅指声明的有限候选及不变成本模型下，TUNE 验证后选出的配置。它不代表所有布局、所有 TP/cap、硬件或真实训练的全局最优。测试窗口不用于重新挑最小时间的配置。最终时间计入一次实测 Mac 规划墙钟代理；服务目录准备和原生仿真器宿主运行时间另列。

TP1 的结果对应当前成对安全点实现，不能用它证明完全异步 TP1 硬件的性能上限。TP2/4/8 的最终选择须使用相同的完整在线策略复核；静态分组、半 cap 和单独改变训练 DP 的运行用于消融。初始较大 TP 把多 die 计算直接归为自身推理，不能仅凭 helper 标签数量判断使用率或收益。

这里运行的是固定公开长度轨迹的事件/成本仿真。未执行 Transformer 数值计算、奖励或学习质量评测，未做实体晶圆计时校准。历史是独立同 prompt 的离线 proxy，不是连续 actor 版本下的预测验证。GPU 同机制调度对照仍需补齐。

## 复跑与便携证据

先按[分层原生环境](IDEA1_LAYERED.md#复跑)准备指定AstraSim和small/14B两个冻结成本工作区；不改变原有187个数学/配置文件。以下命令使用已锁定的14B候选银行，输出目录必须不存在：

```bash
python - <<'PY'
import gzip
from pathlib import Path
root=Path('experiments/2026-11-03-idea1-elastic/assets')
Path('/tmp/idea1-elastic-history-14B.json').write_bytes(gzip.decompress((root/'history-14B.json.gz').read_bytes()))
PY
PYTHONPATH=/tmp/layered-cost-14b/src python scripts/idea1_layered/run_elastic_system.py \
  --model 14B --window test1 --output /tmp/idea1-elastic-14B \
  --history /tmp/idea1-elastic-history-14B.json --allow-offline-history-proxy \
  --catalog experiments/2026-11-03-idea1-elastic/assets/inference-catalog-14B.json \
  --tp-bank experiments/2026-11-03-idea1-elastic/assets/tp-bank-14B.json \
  --fast-cost-validation experiments/2026-11-03-idea1-elastic/assets/cost-backend-validation.json \
  --native-root "$ASTRA_SIM_ROOT"
python scripts/audit_idea1_inference.py /tmp/idea1-elastic-14B/native-run --output /tmp/idea1-elastic-14B/inference-audit.json
```

4B/8B改用small成本工作区及对应assets文件。`--plan-only`只生成计划；`--force-profile`仅用于受控消融，正式六次均未使用。重新准备服务目录可用`prepare_inference_catalog.py`，非节点ragged服务误差检查用`validate_inference_catalog.py`；这些是冻结公式的服务近似校验，不是硬件计时校准。重建目录后SHA身份会变化，不能冒充原锁定资产。

无需原生二进制即可核对公开证据：

```bash
python scripts/check_idea1_elastic.py
```

该检查对六个便携包重新执行实际TP、KV退休、prompt广播、租约互斥、逐rank程序/成本及训练守恒审计；TUNE包提供配置、结果、审计和日志摘要，完整TUNE原生程序日志仍保存在本地实验产物中。原始源冻结和撤回清单一并保留，不能用较早不安全扩组的数字替代最终复跑。
