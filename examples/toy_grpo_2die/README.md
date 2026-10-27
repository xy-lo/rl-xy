# toy_grpo_2die：2-die 同步 GRPO 完整例子

这个例子是 `rl-xy` 的主端到端 fixture。它足够小，可以逐项核对参数、token、版本、KV 和通信；又覆盖一次 RL post-training round 的完整控制流。所有硬件速率和时长均为合成输入，结果只能标记为 `synthetic_fixture`。

## 1. 固定输入

配置文件：`configs/toy_grpo_2die/case.json`。

### 模型

- 2 层 decoder-only toy Transformer；
- hidden size 8；query/KV heads 均为 2；head dim 4；
- FFN size 16；vocab size 16；
- actor/reference 使用 2 B element；master/gradient/Adam 状态使用 4 B element。

### 工作负载

```text
prompt: [1, 2]

trajectory A:
  response: [3, 4, 5]
  finish_reason: fixture_stop
  reward: 1.0

trajectory B:
  response: [6, 7]
  finish_reason: fixture_stop
  reward: 0.0
```

组内 reward mean 为 0.5，所以 demean-only advantage 合同为：

```text
A: +0.5
B: -0.5
```

loss 按全部 response token 做统一均值，分母为 `3 + 2 = 5`。这不是两个 microbatch loss 的简单平均。

### 映射与调度

- 2 个 die，rank0/rank1 各占一个 die；
- `TP=2, PP=1, DP=1`；
- 一个训练 microbatch 只含一条 trajectory；先 A，后 B；
- actor train/inference layout 兼容；
- 不共享 prompt KV；
- KV admission 使用公开最大 response length；
- 调度为每个参与 rank 的本地严格依赖链；collective 与 p2p 节点显式同步两条链，未参与操作的 rank 不生成占位节点。

## 2. 一轮如何执行

### 2.1 输入

host 把 2 个 prompt token 交给 rank0，rank0 再通过 p2p 复制给 rank1。两个 TP rank 从同一个 actor 行为版本 `v0` 开始。

### 2.2 Rollout

A 和 B 分别执行 prompt prefill。prefill 的 logits 分片在 rank0 完成单 owner sampling，采样 token 再复制给 rank1。随后按确定性顺序执行：

```text
A prefill → B prefill → A decode1 → B decode1 → A decode2
```

每层 row-parallel O projection 和 FFN down projection都有 TP sum。最终 token 不再反馈进入模型，所以 A 实际写入 4 个位置的 KV，B 写入 3 个位置的 KV。

### 2.3 Reward、reference 与 old policy

两条 trajectory 的 token payload 回到 host，规则 reward 产生 `1.0/0.0`。随后 reference 和 actor 行为版本分别通过 teacher forcing 计算目标位置 log-prob：

- A：4 个输入位置，3 个目标位置；
- B：3 个输入位置，2 个目标位置。

distributed log-softmax 对 max、sumexp 和 target logit 使用语义独立的 TP reductions。reference 只读，actor 此时仍为 `v0`。

### 2.4 Advantage 与训练

host 计算 `+0.5/-0.5` 并把结果复制给两个 rank。随后执行：

```text
train_A forward → loss → backward → gradient RMW
train_B forward → loss → backward → gradient RMW
```

backward 显式包含 LM head、FFN、attention、QKV、RMSNorm 的反向成本，以及 column-parallel 输入梯度 TP sum。gradient buffer 是跨两个 microbatch 的 read-modify-write 累积。

### 2.5 Clip、optimizer 与发布

先计算本地 gradient square sum，再对一个 FP32 标量做全局 norm reduction，最后执行 clip。shape-only AdamW 读取 gradient/master/`m`/`v` 并写 actor；它只表达形状、依赖、版本和合成成本，不做数值参数更新。

所有 shard 完成后，用小 payload `ALL_REDUCE` 表达 publish barrier，再把 actor `v0` 发布为 `v1`。reference 仍为 `v0`。cleanup 后本轮临时 KV 为零。`publish_actor_version` 和 `release_round_temporaries` 各占 1 ns；这是由 `control_marker_ns` 显式加入的 synthetic control cost，只用于让控制节点可见，不是硬件发布或释放时延。

## 3. 可在运行前手工核对的推导量

以下是配置和公式的确定性期望，不是 AstraSim 实测指标：

| 项目 | 推导值 |
|---|---:|
| unique parameter elements | 1576 |
| local parameter elements / die | 808 |
| actor shard / die | 1616 B |
| reference shard / die | 1616 B |
| master、gradient、Adam m、Adam v 各自 / die | 3232 B |
| 不含 gradient 的长期状态 / die | 12928 B |
| 含 gradient 的 resident state / die | 16160 B |
| KV / processed token / die | 32 B |
| KV reservation / die | 256 B |
| 每轮 KV 累计写入 / die | 224 B |
| KV materialized 曲线 / die | 0→128→192→96→128→0 B |

DRAM 合同为 65536 B/die。验证器必须按每个 die 独立检查 resident state、KV materialization 和 reservation；`kv_view` 不能重复加到物理占用。

## 4. Docker 运行

从项目根目录执行：

```bash
./scripts/docker.sh build rlxy-astrasim:0.1.0
./scripts/docker.sh test rlxy-astrasim:0.1.0
./scripts/docker.sh toy rlxy-astrasim:0.1.0 toy-001
```

输出目录为：

```text
runs/toy_grpo_2die/toy-001/
```

完整性复核：

```bash
python3 scripts/check_result.py runs/toy_grpo_2die/toy-001
```

两轮版本和泄漏检查：

```bash
./scripts/docker.sh two-rounds rlxy-astrasim:0.1.0 toy-2r-001
```

两轮合同应为 `v0 → v1 → v2`；这仍然是版本/生命周期语义验证，不是两次真实训练更新。

## 5. 应检查哪些产物

1. `resolved_config.json` 是否与预期 case 和 rounds 一致；
2. `operations.jsonl` 是否覆盖所有 phase，A/B target positions 和 loss 分母是否正确；
3. 每个 operation 在其参与 rank 上的 Chakra 节点是否与 `events.jsonl` 一一对应，未参与 rank 是否没有多余节点；
4. `astra_command.json` 是否记录实际 argv 和退出码，`manifest.json` 是否记录固定 Astra commit、binary hash、Docker image tag/ID 与源码 tree hash；
5. `astra_stdout.log`、`astra_stderr.log` 和 debug log 是否完整；
6. `objects.csv`、`memory.csv` 是否显示 KV 归零、reference 只读且容量未越界；
7. `checks.json` 是否所有必需项通过；
8. `summary.json` 的每个指标是否带清晰单位与来源。

## 6. 实测结果

> 待正式 Docker run 完成后更新。这里不预填 completion time、通信时间、事件数、吞吐或峰值内存；上节推导量不能替代实测。

更新时至少填写：

| 字段 | 本次值 |
|---|---|
| run bundle | 待运行 |
| Docker image / digest | 待运行 |
| 项目代码版本 | 待运行 |
| AstraSim commit | 待运行核验 |
| resolved config SHA-256 | 待运行 |
| 实际命令与退出码 | 待运行 |
| checks 状态 | 待运行 |
| completion 与单位 | 待运行 |
| compute/communication 口径 | 待运行 |
| 原始日志 | 待运行 |

无论结果如何，结论都必须注明：这是 synthetic fixture，没有实体晶圆硬件校准，也没有真实数值训练或学习质量指标。Astra 原始 `GPU time` 应写为 compute-lane occupancy，而非硬件 GPU 利用率；`congestion_unaware` 不模拟链路争用。若引用 T01 transfer fixture，RECV 可包含 posted wait，5 ns 链路服务时延以 SEND issue→callback 为准。

