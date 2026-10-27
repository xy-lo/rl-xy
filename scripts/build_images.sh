#!/usr/bin/env bash
# Build the complete native-backend image chain from a clean Docker host.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
bash scripts/docker.sh build rlxy-astrasim:0.1.0
for stage in mapped:0.2.0 inputs:0.3.0 execution:0.4.0 schedule:0.4.1 wafer:0.5.0 gpu:0.6.0; do
  docker build --network host --file "docker/Dockerfile.${stage%%:*}" --tag "rlxy-astrasim:${stage#*:}" .
done
