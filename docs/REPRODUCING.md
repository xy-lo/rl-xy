# 构建与复现0.6

## 环境

Linux、可直接调用的Docker、Git、Python≥3.10。仿真运行在CPU容器中，不需要GPU或宿主sudo。完整流程会产生大量中间文件，建议为独立runs目录预留50GB以上；并行仿真需要足够主机内存。完整验证不等同于硬件校准。

```bash
git clone https://github.com/xy-lo/rl-xy.git
cd rl-xy
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
bash scripts/build_images.sh
```

构建链依次为0.1基础→0.2 mapped→0.3 inputs→0.4 execution→0.4.1 schedule→0.5 wafer→0.6 GPU。上游AstraSim固定commit为`4ce9ecfe746a3ee5fe47559a11d91644d641fa7f`，C++扩展和补丁在`docker/astrasim/`。各tag是本地分层依赖，不是声称Docker Hub已发布这些镜像。

可以用`RLXY_BASE_IMAGE`指定可获取的兼容基础镜像；历史验收使用的基础镜像记录在实验来源中。代理只通过调用环境传入，不把认证URL写入脚本。当前文档改版仅验证脚本语法及既有镜像下的回归；没有声称在另一台干净主机重新完成了全部镜像构建。

## 小例验收

```bash
bash scripts/docker.sh test rlxy-astrasim:0.6.0
export RLXY_CONTAINER=rl-xy-quickcheck
bash scripts/docker.sh create-container rlxy-astrasim:0.6.0
IMAGE_ID="$(docker inspect --format '{{.Image}}' "$RLXY_CONTAINER")"
docker exec --env RLXY_IMAGE=rlxy-astrasim:0.6.0 --env "RLXY_IMAGE_ID=$IMAGE_ID" \
  "$RLXY_CONTAINER" python3 -m rl_xy.cli run-wafer \
  --case configs/wafer4_fixture/case.json \
  --scenario configs/wafer4_fixture/wafer.json --out runs/quickcheck-wafer4
docker exec "$RLXY_CONTAINER" python3 scripts/check_result.py runs/quickcheck-wafer4
```

输出目录必须是新目录。旧wafer4金标准为1675.970model ns；它是合成回归小例，不是Qwen主实验。

## 相同64轨迹完整对照

```bash
bash scripts/reproduce_v06.sh my-unique-v06-run
```

该脚本先重建54die默认计算率control，再调用`gpu_sim_acceptance.py`运行主GPU、偏向GPU、晶圆低计算率与四组比较，同时完成单测/小例/native fixture。最终报告位于`acceptance/<ID>/report.json`；各运行目录及比较目录由报告索引。

公开的`experiments/2026-10-27-v0.6/inputs/`保留原实验case、模型配置和长度编译来源，逐文件SHA见`input_hashes.json`。只含结构、长度与组ID，没有原始prompt文本、真实reward/logprob或模型权重。case中的`synthetic_fixture`是执行模板的证据标签；`compilation.json`单独标记其长度来源为framework_capture。未知数值保持null，不把长度回放称为数值训练。

新运行记录新的源码树和镜像身份。整理版README/脚本变化使源码树不同于历史实验；历史数字对应的源hash保存在`provenance.json`。不能把新commit SHA回填为旧实验源码身份。完整原始日志包未随Git上传，因此仅clone不能独立重放历史的所有原始事件；可以执行上述流程产生新的完整证据。

## 外置runs与可选历史采集

若runs是指向自己专用数据盘目录的符号链接，运行前设置`RLXY_RUNS_ROOT`为该链接解析后的绝对目录；`docker.sh`会核验所有权及精确挂载。普通目录无需设置。不要自动删除既有数据或重用已完成run ID。

`gpu_smoke.py`和`gpu_capture_wrapper.py`是历史真实veRL采集适配器，依赖外部veRL环境与collector，不是GPU仿真的必要依赖。分别使用`RLXY_LEGACY_ROOT`及`RLXY_LEGACY_HOOK`配置外部位置；collector flag仅允许存在于隔离容器。未在本轮重新验收数值训练。

`remote_session.py`仅为可选SSH辅助，须从自己的可信配置设置`RLXY_SSH_USER`、`RLXY_SSH_FINGERPRINT`、`RLXY_SSH_HOST_ALIAS`，可选`RLXY_SSH_SOCKET`。仓库不含任何预设服务器身份或密码。
