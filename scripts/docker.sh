#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-help}"
IMAGE="${2:-rlxy-astrasim:0.1.0}"
RUN_ID="${3:-$(date -u +%Y%m%dT%H%M%SZ)}"
CONTAINER="${RLXY_CONTAINER:-rl-xy}"
ASTRA_SIM_COMMIT="4ce9ecfe746a3ee5fe47559a11d91644d641fa7f"
ARTIFACT_MOUNTS=()
if [[ -n "${RLXY_RUNS_ROOT:-}" ]]; then
  if [[ "${RLXY_RUNS_ROOT}" != /* || ! -d "${RLXY_RUNS_ROOT}" || ! -O "${RLXY_RUNS_ROOT}" ]]; then
    printf 'ERROR: RLXY_RUNS_ROOT must be an existing absolute directory owned by this UID.\n' >&2
    exit 4
  fi
  RUNS_REAL="$(cd "${RLXY_RUNS_ROOT}" && pwd -P)"
  if [[ "${RUNS_REAL}" == / || "${RUNS_REAL}" == "${ROOT_DIR}" ||
        "$(cd "${ROOT_DIR}/runs" && pwd -P)" != "${RUNS_REAL}" ]]; then
    printf 'ERROR: project runs must already resolve to the dedicated RLXY_RUNS_ROOT directory.\n' >&2
    exit 4
  fi
  # Mount at the same absolute path: Path.resolve() and archived provenance
  # then agree inside Docker and on the host, including the runs symlink.
  ARTIFACT_MOUNTS=(--volume "${RUNS_REAL}:${RUNS_REAL}")
fi

docker_ready() {
  if ! docker info >/dev/null 2>&1; then
    printf '%s\n' \
      "ERROR: current user cannot access the Docker daemon." \
      "This project never invokes sudo or changes docker.sock permissions." >&2
    exit 3
  fi
}

validate_named_container() {
  local mount_info container_user image_commit
  mount_info="$(docker container inspect --format '{{range .Mounts}}{{if eq .Destination "/workspace/rl-xy"}}{{.Type}}|{{.Source}}|{{.RW}}{{end}}{{end}}' "${CONTAINER}")"
  if [[ "${mount_info}" != "bind|${ROOT_DIR}|true" ]]; then
    printf 'ERROR: container %s must bind %s read-write at /workspace/rl-xy.\n' "${CONTAINER}" "${ROOT_DIR}" >&2
    exit 4
  fi
  if [[ -n "${RLXY_RUNS_ROOT:-}" ]]; then
    local runs_mount
    runs_mount="$(docker container inspect --format "{{range .Mounts}}{{if eq .Destination \"${RUNS_REAL}\"}}{{.Type}}|{{.Source}}|{{.RW}}{{end}}{{end}}" "${CONTAINER}")"
    if [[ "${runs_mount}" != "bind|${RUNS_REAL}|true" ]]; then
      printf 'ERROR: container %s lacks the exact dedicated artifact bind mount.\n' "${CONTAINER}" >&2
      exit 4
    fi
  fi
  container_user="$(docker container inspect --format '{{.Config.User}}' "${CONTAINER}")"
  if [[ "${container_user}" != "$(id -u):$(id -g)" ]]; then
    printf 'ERROR: container %s must use the current host UID:GID.\n' "${CONTAINER}" >&2
    exit 4
  fi
  # Read provenance from the container, even when a tag has been moved since
  # creation. Never substitute the current host tag's image ID for this value.
  NAMED_IMAGE_ID="$(docker container inspect --format '{{.Image}}' "${CONTAINER}")"
  NAMED_IMAGE="$(docker container inspect --format '{{.Config.Image}}' "${CONTAINER}")"
  image_commit="$(docker image inspect --format '{{index .Config.Labels "io.rlxy.astrasim.commit"}}' "${NAMED_IMAGE_ID}")"
  if [[ "${image_commit}" != "${ASTRA_SIM_COMMIT}" ]]; then
    printf 'ERROR: container %s image does not declare the pinned AstraSim commit.\n' "${CONTAINER}" >&2
    exit 4
  fi
}

start_named_container() {
  if [[ "$(docker container inspect --format '{{.State.Running}}' "${CONTAINER}")" != "true" ]]; then
    docker start "${CONTAINER}" >/dev/null
  fi
}

create_container() {
  local requested_image_id requested_commit
  requested_image_id="$(docker image inspect --format '{{.Id}}' "${IMAGE}")"
  requested_commit="$(docker image inspect --format '{{index .Config.Labels "io.rlxy.astrasim.commit"}}' "${requested_image_id}")"
  if [[ "${requested_commit}" != "${ASTRA_SIM_COMMIT}" ]]; then
    printf 'ERROR: build the project image before creating a container.\n' >&2
    exit 4
  fi
  if docker container inspect "${CONTAINER}" >/dev/null 2>&1; then
    validate_named_container
    if [[ "${NAMED_IMAGE_ID}" != "${requested_image_id}" ]]; then
      printf 'ERROR: existing container %s uses a different image; choose another RLXY_CONTAINER name.\n' "${CONTAINER}" >&2
      exit 4
    fi
    start_named_container
    printf 'Reusing container %s (%s).\n' "${CONTAINER}" "${NAMED_IMAGE_ID}"
    return
  fi
  docker create \
    --name "${CONTAINER}" \
    --user "$(id -u):$(id -g)" \
    --volume "${ROOT_DIR}:/workspace/rl-xy" \
    "${ARTIFACT_MOUNTS[@]}" \
    --workdir /workspace/rl-xy \
    --env PYTHONPATH=/workspace/rl-xy/src:/opt/astra-sim \
    --env ASTRA_SIM_ROOT=/opt/astra-sim \
    --env ASTRA_SIM_BIN=/opt/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware \
    --shm-size=8g \
    --entrypoint /bin/bash \
    "${IMAGE}" -c 'exec sleep infinity' >/dev/null
  validate_named_container
  if [[ "${NAMED_IMAGE_ID}" != "${requested_image_id}" ]]; then
    printf 'ERROR: image tag changed during container creation; inspect container %s before reuse.\n' "${CONTAINER}" >&2
    exit 4
  fi
  start_named_container
  printf 'Created container %s (%s).\n' "${CONTAINER}" "${NAMED_IMAGE_ID}"
}

docker_python() {
  local image_id
  if [[ -n "${RLXY_CONTAINER:-}" ]]; then
    validate_named_container
    start_named_container
    docker exec \
      --user "$(id -u):$(id -g)" \
      --workdir /workspace/rl-xy \
      --env PYTHONPATH=/workspace/rl-xy/src:/opt/astra-sim \
      --env ASTRA_SIM_ROOT=/opt/astra-sim \
      --env ASTRA_SIM_BIN=/opt/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware \
      --env "RLXY_IMAGE=${NAMED_IMAGE}" \
      --env "RLXY_IMAGE_ID=${NAMED_IMAGE_ID}" \
      "${CONTAINER}" python3 "$@"
    return
  fi
  image_id="$(docker image inspect --format '{{.Id}}' "${IMAGE}")"
  docker run --rm \
    --user "$(id -u):$(id -g)" \
    --volume "${ROOT_DIR}:/workspace/rl-xy" \
    "${ARTIFACT_MOUNTS[@]}" \
    --workdir /workspace/rl-xy \
    --env PYTHONPATH=/workspace/rl-xy/src:/opt/astra-sim \
    --env ASTRA_SIM_ROOT=/opt/astra-sim \
    --env ASTRA_SIM_BIN=/opt/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware \
    --env "RLXY_IMAGE=${IMAGE}" \
    --env "RLXY_IMAGE_ID=${image_id}" \
    --shm-size=8g \
    --entrypoint python3 \
    "${image_id}" "$@"
}

docker_run() {
  docker_python -m rl_xy.cli "$@"
}

case "${ACTION}" in
  build)
    docker_ready
    # Docker's predefined proxy args read values from the calling environment.
    # Keep credentials out of this script and out of explicit CLI arguments.
    docker build \
      --network host \
      --build-arg "BASE_IMAGE=${RLXY_BASE_IMAGE:-ubuntu:22.04}" \
      --build-arg HTTP_PROXY \
      --build-arg HTTPS_PROXY \
      --build-arg http_proxy \
      --build-arg https_proxy \
      --build-arg NO_PROXY \
      --build-arg no_proxy \
      --file "${ROOT_DIR}/docker/Dockerfile" \
      --build-arg "ASTRA_SIM_COMMIT=${ASTRA_SIM_COMMIT}" \
      --tag "${IMAGE}" \
      "${ROOT_DIR}"
    ;;
  create-container)
    docker_ready
    create_container
    ;;
  build-mapped)
    docker_ready
    docker build --network host --file "${ROOT_DIR}/docker/Dockerfile.mapped" \
      --build-arg "RLXY_BASE_IMAGE=${RLXY_MAPPED_BASE_IMAGE:-rlxy-astrasim:0.1.0}" \
      --tag "${IMAGE}" "${ROOT_DIR}"
    ;;
  test)
    docker_ready
    docker_python -m unittest discover -s tests -v
    ;;
  transfer)
    docker_ready
    docker_run fixture-transfer \
      --case configs/fixture_transfer/case.json \
      --out "runs/fixture_transfer_21ns/${RUN_ID}"
    ;;
  toy)
    docker_ready
    docker_run run \
      --case configs/toy_grpo_2die/case.json \
      --out "runs/toy_grpo_2die/${RUN_ID}"
    ;;
  toy-resource)
    docker_ready
    docker_run run --case configs/toy_grpo_2die/case.json \
      --out "runs/toy_grpo_2die_resource/${RUN_ID}" --resource-model
    ;;
  two-rounds-resource)
    docker_ready
    docker_run run --case configs/toy_grpo_2die/case.json --rounds 2 \
      --out "runs/toy_grpo_2die_resource_2rounds/${RUN_ID}" --resource-model
    ;;
  two-rounds)
    docker_ready
    docker_run run \
      --case configs/toy_grpo_2die/case.json \
      --rounds 2 \
      --out "runs/toy_grpo_2die_2rounds/${RUN_ID}"
    ;;
  all)
    "${BASH_SOURCE[0]}" test "${IMAGE}"
    "${BASH_SOURCE[0]}" transfer "${IMAGE}" "${RUN_ID}-transfer"
    "${BASH_SOURCE[0]}" toy "${IMAGE}" "${RUN_ID}-toy"
    "${BASH_SOURCE[0]}" two-rounds "${IMAGE}" "${RUN_ID}-two-rounds"
    ;;
  help|*)
    printf '%s\n' \
      "Usage: scripts/docker.sh {build|build-mapped|create-container|test|transfer|toy-resource|two-rounds-resource|toy|two-rounds|all} [image] [run-id]" \
      "RLXY_BASE_IMAGE optionally selects a build base image." \
      "RLXY_CONTAINER selects a persistent container (create-container default: rl-xy)." \
      "RLXY_RUNS_ROOT optionally mounts an owned external runs directory already linked from the project." \
      "Docker is the supported execution path. No sudo is embedded in this script."
    ;;
esac

