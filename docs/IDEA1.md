# Idea1：KV 局部性感知晶圆长尾协作

本目录发布 Qwen3 4B、8B、14B 的 Idea1 仿真实现与结果摘要。机制将 KV 头布局、TP4→TP8 按需跨 die 协作和请求生命周期联合建模；迁移计入 KV/权重、端点本地读写、打包/解包、激活通信和归约。晶圆采用 6×9 物理 die、48 活动 die；GPU 是 A100 本地校准的 48 卡模拟。

[完整结果与口径](../experiments/2026-11-01-idea1/论文实验结果与表述.md)；[方法草稿](../experiments/2026-11-01-idea1/test1-Idea1方法正文草稿.md)。test1 为主表，test2 是冻结方法后不同 prompt 组的辅助验证；每窗 192×8 条轨迹。九个阶段分别模拟后按 f2 同步屏障求和，吞吐是 prompt 与 response 的 token 总量除以 step 时间。

| 模型 | test1 GPU / s | test1 晶圆 / s | test1 Idea1 / s | test2 GPU / s | test2 晶圆 / s | test2 Idea1 / s |
|---|---:|---:|---:|---:|---:|---:|
| 4B | 132.966 | 159.060 | 97.220 | 142.357 | 179.399 | 106.565 |
| 8B | 178.512 | 208.238 | 150.121 | 193.456 | 235.873 | 165.647 |
| 14B | 260.755 | 266.747 | 213.094 | 262.669 | 273.636 | 214.421 |

模型采用理想化 GQA 局部 KV 复用假设，具体条件与开销见结果报告；当前是逻辑状态和事件仿真，不是实机张量执行。原 baseline 与 Idea1 的 KV 访问成本模型不同，14B 的 rollout TP/cap 也不同，整体差异不能全部归因于调度；晶圆成本未经过实机校准。test1 曾用于开发，test2 仅提供同数据源不同 prompt 组的独立窗口。模拟证据和实际硬件测量应明确区分。

## 代码和数据

- `scripts/idea1/locality_gqa_4b_v3.py`：融合方法入口；`locality_full4b_v3.py` 调度运行时。
- `locality_runtime_v3.py` 和 `attention_tp8_program.py`：局部性决策、KV 头布局、迁移及 TP8 attention 事件；其余同目录文件为执行依赖。
- `experiments/2026-11-01-idea1/`：test1/test2 结果、事件审计摘要、迁移复用凭证和历史结果哈希。原始日志与完整事件未随本仓库发布；导出的 JSON 将历史本机绝对路径替换为 `<historical-workspace>`，因此导出文件自身哈希与原始结果文件不同，记录的原始哈希不变。
- 历史运行身份保留在产物中。发布版只迁移了 Python 模块路径、增加了 test2 选项；它是新源码身份，不假装是历史运行时的字节副本，尚未用本发布版重跑全部结果。

## 准备原生后端

参考 [f2 接手文档](F2_HANDOFF.md) 配置构建依赖和 Python 环境。Idea1 需要带在线控制器 RPC、gate 和点对点事件的 ASTRA 扩展，不可直接把普通 f2 后端当作 Idea1 后端。`astra-sequence.patch` 是相对于上游固定 revision 的**完整补丁**，不要再叠加 f2 或其他旧补丁：

```bash
mkdir -p native
git clone https://github.com/astra-sim/astra-sim.git native/astra-sim
git -C native/astra-sim checkout 4ce9ecfe746a3ee5fe47559a11d91644d641fa7f
git -C native/astra-sim submodule update --init --recursive
git -C native/astra-sim apply "$PWD/experiments/2026-11-01-idea1/astra-source/astra-sequence.patch"
cp experiments/2026-11-01-idea1/astra-source/Rlxy*.hh native/astra-sim/astra-sim/workload/
cp experiments/2026-11-01-idea1/astra-source/packet_fixture.cc native/astra-sim/build/astra_analytical/
cmake -S native/astra-sim/build/astra_analytical -B native/astra-sim/build-dynarl -DCMAKE_BUILD_TYPE=Release
cmake --build native/astra-sim/build-dynarl -j
```

准备编译依赖的过程见 f2 文档。本次发布已检验补丁与历史原生源码相符，未在干净机器上重新构建原生二进制。运行示例（输出和缓存目录需可写，输出目录不能预先存在）：

```bash
python -m pip install -e '.[f2]'
python scripts/idea1/locality_gqa_4b_v3.py --mode ideal-gqa --model 4B --cap 128 --train-tp 4 --train-mb 8 --helper --priority --safe-points --interval 16 --window test1 --native-root "$PWD/native/astra-sim" --cache-root "$PWD/runs/idea1-cache-4b" --output "$PWD/runs/idea1-4b-test1"
```

8B 使用 `--cap 64 --train-tp 4 --train-mb 4`；14B 使用 `--cap 64 --train-tp 8 --train-mb 8`。设置 `--window test2` 复放辅助窗口。记录新运行的源文件、数据、配置和二进制哈希；实际运行会产生新身份，不可覆盖历史数据。
