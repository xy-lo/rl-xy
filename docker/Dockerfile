ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

USER root

ARG DEBIAN_FRONTEND=noninteractive
ARG ASTRA_SIM_COMMIT=4ce9ecfe746a3ee5fe47559a11d91644d641fa7f
ARG ABSL_VERSION=20240722.0
ARG PROTOBUF_VERSION=29.0

LABEL org.opencontainers.image.title="rl-xy AstraSim MVP"
LABEL org.opencontainers.image.description="Auditable two-die synchronous GRPO simulation on the official AstraSim analytical backend"
LABEL org.opencontainers.image.source="https://github.com/astra-sim/astra-sim"
LABEL io.rlxy.astrasim.commit="${ASTRA_SIM_COMMIT}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates coreutils git wget tar \
       gcc g++ make cmake \
       libboost-dev libboost-program-options-dev \
       openmpi-bin libopenmpi-dev \
       python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Keep project Python dependencies isolated from a supplied base image's Torch
# environment and from distributions enforcing PEP 668.
RUN /usr/bin/python3 -m venv /opt/rlxy-venv
ENV PATH=/opt/rlxy-venv/bin:${PATH}

# Match the pinned upstream Docker toolchain: current AstraSim/Chakra uses
# Protobuf 29 (Python runtime 5.29) and its Abseil dependency.  Ubuntu 22.04's
# Protobuf is too old for a version-consistent C++/Python ET schema.
RUN wget --https-only --secure-protocol=TLSv1_2 \
      "https://github.com/abseil/abseil-cpp/releases/download/${ABSL_VERSION}/abseil-cpp-${ABSL_VERSION}.tar.gz" \
      -O /tmp/abseil.tar.gz \
    && mkdir -p "/opt/abseil-cpp-${ABSL_VERSION}/build" \
    && tar -xzf /tmp/abseil.tar.gz -C /opt \
    && cd "/opt/abseil-cpp-${ABSL_VERSION}/build" \
    && cmake .. \
       -DCMAKE_CXX_STANDARD=14 \
       -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_INSTALL_PREFIX="/opt/abseil-cpp-${ABSL_VERSION}/install" \
    && cmake --build . --target install --config Release --parallel 16 \
    && rm -f /tmp/abseil.tar.gz

ENV absl_DIR=/opt/abseil-cpp-20240722.0/install

RUN wget --https-only --secure-protocol=TLSv1_2 \
      "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-${PROTOBUF_VERSION}.tar.gz" \
      -O /tmp/protobuf.tar.gz \
    && mkdir -p "/opt/protobuf-${PROTOBUF_VERSION}/build" \
    && tar -xzf /tmp/protobuf.tar.gz -C /opt \
    && cd "/opt/protobuf-${PROTOBUF_VERSION}/build" \
    && cmake .. \
       -DCMAKE_CXX_STANDARD=14 \
       -DCMAKE_BUILD_TYPE=Release \
       -Dprotobuf_BUILD_TESTS=OFF \
       -Dprotobuf_ABSL_PROVIDER=package \
       -DCMAKE_INSTALL_PREFIX="/opt/protobuf-${PROTOBUF_VERSION}/install" \
    && cmake --build . --target install --config Release --parallel 16 \
    && rm -f /tmp/protobuf.tar.gz

ENV PATH=/opt/protobuf-29.0/install/bin:${PATH}
ENV protobuf_DIR=/opt/protobuf-29.0/install
ENV LD_LIBRARY_PATH=/opt/protobuf-29.0/install/lib:/opt/abseil-cpp-20240722.0/install/lib
ENV PROTOBUF_FROM_SOURCE=True

# Follow AstraSim's documented bare-metal dependency path, but isolate it in
# the image and pin the exact upstream commit used by every result manifest.
RUN git init /opt/astra-sim \
    && cd /opt/astra-sim \
    && git remote add origin https://github.com/astra-sim/astra-sim.git \
    && git fetch --depth 1 origin "${ASTRA_SIM_COMMIT}" \
    && git checkout --detach FETCH_HEAD \
    && git submodule update --init --recursive --depth 1 -- \
       extern/graph_frontend/chakra \
       extern/helper/fmt \
       extern/helper/spdlog \
       extern/network_backend/analytical \
       extern/remote_memory_backend/analytical \
    && test "$(git rev-parse HEAD)" = "${ASTRA_SIM_COMMIT}" \
    && printf '%s\n' "${ASTRA_SIM_COMMIT}" > /opt/astra-sim/.rlxy-pinned-commit \
    && git config --system --add safe.directory /opt/astra-sim

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir protobuf==5.29.0

RUN cd /opt/astra-sim \
    && ./build/astra_analytical/build.sh -t congestion_unaware \
    && test -x /opt/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware

ENV ASTRA_SIM_ROOT=/opt/astra-sim
ENV ASTRA_SIM_BIN=/opt/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware
ENV PYTHONPATH=/workspace/rl-xy/src:/opt/astra-sim

WORKDIR /workspace/rl-xy
COPY pyproject.toml README.md /tmp/rlxy-package/
COPY src /tmp/rlxy-package/src
RUN python3 -m pip install --no-cache-dir /tmp/rlxy-package \
    && rm -rf /tmp/rlxy-package

ENTRYPOINT ["python3", "-m", "rl_xy.cli"]
CMD ["--help"]

