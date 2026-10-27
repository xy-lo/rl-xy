# Docker 执行环境

## 0.2资源版

`docker/Dockerfile.mapped` 以现有 `rlxy-astrasim:0.1.0` 为base，接入Workload资源hook、CPU混合trace统计修正与显式mesh2d适配，再编译aware/unaware两个后端。`RLXY_MAPPED_BASE_IMAGE` 可指定相同基础项目镜像的稳定引用；不能直接换成缺少Astra依赖的pjlab裸镜像。完整项目不保存任何代理口令。

```bash
bash scripts/docker.sh build-mapped rlxy-astrasim:0.2.0
RLXY_CONTAINER=rl-xy-mapped bash scripts/docker.sh create-container rlxy-astrasim:0.2.0
RLXY_CONTAINER=rl-xy-mapped bash scripts/docker.sh toy-resource rlxy-astrasim:0.2.0 <唯一ID>
RLXY_CONTAINER=rl-xy-mapped bash scripts/docker.sh two-rounds-resource rlxy-astrasim:0.2.0 <另一个ID>
python3 scripts/resource_acceptance.py --container rl-xy-mapped --run-id <验收ID>
```

镜像tag可能移动，结果必须以container实际image ID为准。源码bind mount为唯一活动目录；镜像也含src fallback以支持审计，但运行以bind源码hash为准。新增patch/header hash在 `/opt/astra-sim/.rlxy-resource-patch`；只读Reference/容量等错误直接退出2，防止上游捕获回调异常后仍退出0。

下文是0.1基础镜像构建与容器接口；两个版本都不需要sudo或GPU。

## 职责

Docker 组件提供唯一正式、可复现的构建和运行环境：拉取固定 official AstraSim commit、递归初始化其子模块、构建 analytical congestion-unaware binary、安装固定 Python 依赖和 `rl-xy` 包，并以同一镜像执行测试与案例。该 backend 不模拟共享链路的多流争用或拥塞排队，容器复现性不会改变这一模型边界。

入口：`docker/Dockerfile`、`scripts/docker.sh`、`compose.yaml`、`Makefile`。

## 输入

- 项目源码和配置；
- 可访问互联网或已有构建缓存；
- 当前用户可直接访问的 Docker daemon；
- build arg `ASTRA_SIM_COMMIT`，脚本固定为 `4ce9ecfe746a3ee5fe47559a11d91644d641fa7f`；
- 可选 build arg `BASE_IMAGE`（脚本通过 `RLXY_BASE_IMAGE` 传入），默认 `ubuntu:22.04`；
- 可选 image 名、run ID 和 `RLXY_CONTAINER` 持久容器名。

## 构建输出

默认镜像：`rlxy-astrasim:0.1.0`。镜像内包含：

- 默认 Ubuntu 22.04，或显式选定的基础镜像，与官方文档需要的 C++/CMake/Boost/Protobuf/OpenMPI 工具链；
- `/opt/astra-sim` 下固定 commit 的源码和子模块；
- `AstraSim_Analytical_Congestion_Unaware` 可执行文件；
- 与固定上游一致的 Protobuf `29.0` C++ toolchain，以及 `/opt/rlxy-venv` 中的 Python protobuf runtime `5.29.0` 和 `rl-xy` 包；
- `ASTRA_SIM_ROOT`、`ASTRA_SIM_BIN` 和 `PYTHONPATH`。

运行时把项目根绑定到 `/workspace/rl-xy`，因此结果直接写回宿主机 `runs/`。容器以宿主用户 UID/GID 执行，避免产生 root-owned run 文件。

运行脚本在启动前用 `docker image inspect` 解析不可变 image ID，并连同 image tag 传入容器；reporter 把二者写入 manifest。缺任一项都不能生成 success manifest。`compose.yaml` 仅提供构建/交互声明，并默认使用无特权的 `65534:65534`，避免误生成 root-owned 文件；交互使用时需显式传 `RLXY_UID=$(id -u)` 与 `RLXY_GID=$(id -g)`。需要正式证据包时仍使用 `scripts/docker.sh`，由脚本自动补齐 image provenance。

## 支持动作

```bash
./scripts/docker.sh build [image]
./scripts/docker.sh create-container [image]
./scripts/docker.sh test [image]
./scripts/docker.sh transfer [image] [run-id]
./scripts/docker.sh toy [image] [run-id]
./scripts/docker.sh two-rounds [image] [run-id]
./scripts/docker.sh all [image] [run-id]
```

- `build`：从 Dockerfile 构建并验证 AstraSim binary 存在；
- `create-container`：创建默认名为 `rl-xy` 的持久容器；同名容器存在时验证项目绑定、宿主 UID/GID 和不可变 image ID 后复用，停止状态则启动；不覆盖或删除已有容器；
- `test`：容器内运行单元/负向测试；
- `transfer`：运行可手算传输 fixture；
- `toy`：运行单轮主例子；
- `two-rounds`：覆盖版本连续性和资源释放；
- `all`：按 test → transfer → toy → two-rounds 顺序执行。

## 使用已有基础镜像创建项目容器

用户指定的候选本地基础镜像是 `pjlab/deeplearning:v2.2.3-cuda12.8.0-torch2.7.0`，实际是否存在需以该服务器的 `docker images` 为准。它提供通用工具链；项目仍需通过 Dockerfile 构建固定版本 AstraSim 和独立 Python 环境。构建依赖时使用 root，运行项目时仍使用宿主 UID/GID。独立 venv 避免覆盖基础镜像的 Torch/Python 包，也支持启用 PEP 668 的发行版。

```bash
docker images
RLXY_BASE_IMAGE=pjlab/deeplearning:v2.2.3-cuda12.8.0-torch2.7.0 \
  ./scripts/docker.sh build rlxy-astrasim:0.1.0
RLXY_CONTAINER=rl-xy ./scripts/docker.sh create-container rlxy-astrasim:0.1.0
RLXY_CONTAINER=rl-xy ./scripts/docker.sh all rlxy-astrasim:0.1.0 my-run
```

不设置 `RLXY_CONTAINER` 时，`test`、`transfer`、`toy`、`two-rounds` 和 `all` 保留每次 `docker run --rm` 的执行方式。设置后，这些动作通过 `docker exec` 在指定容器内运行，并在需要时启动该容器。持久容器以 `/bin/bash -c 'exec sleep infinity'` 保持运行，绑定本项目到 `/workspace/rl-xy`，配置 8 GiB shared memory。脚本不请求 GPU、HF token 或代理凭据，也不触碰其它容器。

需要代理时，由调用环境临时提供 `HTTP_PROXY` / `HTTPS_PROXY`（或小写对应变量）。build 动作通过 Docker 预定义 build args 按变量名传递，脚本和 Dockerfile 不保存认证值，也不把代理写成镜像的 `ENV`。不要把认证值写入本仓库或提交日志。

持久容器执行时，脚本读取该容器的 `.Image` 和 `.Config.Image`，分别写入 manifest 的 image ID 和创建时 image 引用；即使宿主机同名 tag 后来发生变化，也不会把新 tag 对应的错误 image ID 写进结果。执行前验证项目目录可写绑定、宿主 UID/GID，以及 image 中的 pinned AstraSim commit label。更换镜像后应使用新的 `RLXY_CONTAINER` 名创建容器；脚本会拒绝把同名旧容器替换成新镜像。

## 安全与权限边界

脚本直接调用 `docker`，不嵌入 `sudo`、密码、socket chmod、用户组修改或 daemon 启停命令。用户要求“无需 sudo”意味着宿主环境应已提供 Docker 访问能力；若当前 SSH 身份没有权限，脚本必须失败并报告环境问题，不能偷偷切换身份或放宽 socket 权限。

镜像构建过程中只使用 official AstraSim 仓库固定 commit。不会使用参考仓库中的既有容器名、宿主绝对路径、未锁版本 binary 或凭据。

## 不做什么

- 不支持把宿主机直跑结果标为正式结果；
- 不自动使用 `latest` AstraSim；
- 不在失败时切换到来源不明的预建镜像；
- 不需要 GPU，也不占用真实模型训练资源；
- 不自动删除 image、build cache 或 `runs/`；
- 不在镜像或文档中保存 SSH/Docker 密码。

## 失败条件

- Docker CLI/daemon 不可用或当前用户没有访问权限；
- 基础镜像、AstraSim commit 或子模块无法获取；
- 编译依赖安装/analytical build 失败；
- 固定 commit 校验不一致；
- binary 不可执行；
- bind mount 不可写或 run 目录冲突；
- 容器内测试、fixture、AstraSim 或 validation 任一失败。

失败后保留可获得的 build/run 日志，并修复原因后使用新的 run ID 重跑；不要手改 summary。

## 本服务器实际构建说明

2026-10-26 执行用户具有 Docker 访问权限。项目直接使用已存在的 pjlab/deeplearning:v2.2.3-cuda12.8.0-torch2.7.0 作为基础镜像，其实际系统为 Ubuntu24.04、Python3.12。依赖在 /opt/rlxy-venv 内隔离安装。

构建采用 Docker 内置 Dockerfile 前端，无需额外从 Docker Hub 下载 docker/dockerfile 前端；使用 host 网络与调用环境的临时代理。AstraSim、Abseil、Protobuf 编译均在 Docker 数据盘内完成，工程目录只保留源码/配置和小体量结果。

