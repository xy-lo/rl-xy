# 构建、复现与检查

> 本页的原生后端和旧策略用于历史实验；f2 的补丁、参数及回放入口见 [f2 接手说明](F2_HANDOFF.md)。

当前发布包含可直接核对的三档长度输入、24份结果和图表；完整原生日志与编译依赖留在独立实验归档。下面区分记录复算、新原生运行和历史兼容。

## 1. 纯Python记录复算

Python≥3.9；基础安装仅需protobuf依赖。无需模型权重、GPU、tokenizer或完整回答文本。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/check_public_results.py
.venv/bin/python scripts/check_qwen3_results.py
.venv/bin/python scripts/check_docs.py
```

单测的Docker环境专用skip不算原生验收。Qwen3检查器验证公开输入manifest、配对轨迹、24个结果、216阶段与有效吞吐，没有读取未发布的原生事件。重新绘图需要可选依赖，默认写入新目录：

```bash
.venv/bin/python -m pip install -e '.[figures]'
.venv/bin/python scripts/plot_qwen3_baselines.py --output runs/qwen3-figures
```

## 2. Qwen3原生后端

固定AstraSim commit为`4ce9ecfe746a3ee5fe47559a11d91644d641fa7f`。历史实际运行使用Darwin arm64、同编译器源码构建的Abseil20240722.0/Protobuf29.0以及Python protobuf5.29.1；没有本次Qwen3 Docker构建验收记录。

从干净固定commit和对应递归submodule构建时，应用 [完整原生修改](../docker/astrasim/qwen3-native.patch)，并将`docker/astrasim/`的`RlxyResources.hh`、`RlxyMesh.hh`、`RlxyGpu.hh`、`RlxySequence.hh`复制到ASTRA的`astra-sim/workload/`。递归submodule固定身份见 [native-submodules.txt](../experiments/2026-10-29-qwen3/source/native-submodules.txt)。完整patch已经包含历史修改，不应再叠加旧patch链。

已具备与历史相同的`native/deps` CMake/Boost环境时，`scripts/rebuild_qwen3_native_dependencies.py`可构建源码Abseil/Protobuf、生成Chakra schema并编译。它是有前提的构建辅助，不是自动安装所有系统依赖的bootstrap。已有按上述commit/header/patch构建的后端，也可通过环境指定：

```bash
export ASTRA_SIM_ROOT=/path/to/pinned-patched-astra-sim
export ASTRA_SIM_BIN="$ASTRA_SIM_ROOT/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
export PYTHONPATH="$PWD/src:$ASTRA_SIM_ROOT"
.venv/bin/python scripts/check_qwen3_compaction.py --output runs/qwen3-compaction-new
```

小例验证GPU和晶圆两端compact/expanded的逐rank完成时间；它不替代正式24案例重跑。输出目录必须全新，原始失败保留。`docker/Dockerfile.qwen3`是0.10.0容器构建适配，继承0.9.5；本次未将其声明为已验证Qwen3容器。

## 3. 单候选与冻结策略配对回放

在原生后端已配置的环境中，下列命令产生新源码/二进制身份与新证据，运行成本可能很高。原公开结果不会被覆盖。

```bash
.venv/bin/python scripts/qwen3_baselines.py candidate   --model 4B --platform gpu --window test1   --policy experiments/2026-10-29-qwen3/policies/4B-gpu.json   --output runs/qwen3-new/4B-test1-gpu

.venv/bin/python scripts/replay_qwen3_policies.py   --models 4B 8B 14B --windows test1 test2 test3 test4   --output runs/qwen3-paired-new
```

配对入口按已公开策略回放，**不声称重复了历史离线调优搜索**。`--resume`只复用源树、输入、平台、策略和二进制均一致的PASS检查点。候选原生超时仍为900s/调用；宿主耗时与设备时间分开记录。默认generation缓存位于`runs/qwen3/generation_cache`，可用`RLXY_QWEN3_CACHE_ROOT`指定专属输出位置；不会写入只读实验输入。

`qwen3_baselines.py screen`保留历史解析筛选，可显式选择`--inputs`和`--output`；它的C≤64网格不是充分GPU最优证明。完整新实验应先处理当前已知成本与搜索限制。

## 4. 输入重建与历史兼容

直接回放只需要已发布长度。若重新从源数据构造窗口，先准备独立staging目录、原manifest、固定官方config/tokenizer，安装`.[public-trace]`，再向`prepare_qwen3_public_inputs.py`显式传入`--inputs`、`--lengths-csv`、`--ds1000`。长度审计CSV和原tokenizer未作为当前最小回放包重复分发；其SHA与来源仍保留。

旧0.1→0.9.5的Docker构建链继续由`bash scripts/build_images.sh`维护，`make test`沿用该历史镜像配置。旧3B输入和轻量证据保留，但应按其原版本复现，不能用新commit冒称原镜像或原始运行。
