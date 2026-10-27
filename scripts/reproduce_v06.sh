#!/usr/bin/env bash
# CPU/Docker only; preserves existing runs and requires a unique ID.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_ID="${1:-reproduce-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  printf 'Use a unique run ID containing letters, digits, _ or -.\n' >&2
  exit 2
fi
export RLXY_CONTAINER="${RLXY_CONTAINER:-rl-xy-${RUN_ID}}"
CASE="experiments/2026-10-27-v0.6/inputs/case.json"
CONTROL="runs/reproductions/${RUN_ID}-wafer-control"
test ! -e "$CONTROL"
test ! -e "acceptance/${RUN_ID}"
bash scripts/docker.sh create-container rlxy-astrasim:0.6.0
IMAGE_ID="$(docker inspect --format '{{.Image}}' "$RLXY_CONTAINER")"
docker exec --env RLXY_IMAGE=rlxy-astrasim:0.6.0 --env "RLXY_IMAGE_ID=$IMAGE_ID" \
  "$RLXY_CONTAINER" python3 -m rl_xy.cli run-wafer \
  --case "$CASE" --scenario configs/wafer54_fixture/wafer.json --out "$CONTROL"
python3 scripts/gpu_sim_acceptance.py --run-id "$RUN_ID" \
  --container "$RLXY_CONTAINER" --wafer "$CONTROL" --case "$CASE"
printf 'Acceptance report: acceptance/%s/report.json\n' "$RUN_ID"
