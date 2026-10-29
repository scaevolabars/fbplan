#!/usr/bin/env bash
# AWR sharpness at 2x (docs/design.md §8): the baseline pi_h with high_alpha 0.2, and B clean with
# one temperature, shared by all seeds, that gives alpha * A the spread of that baseline.
# Runs on a GPU instance after the B1 runs (it needs approach_b/medium/value-single/seed0-4).
#
#   bash scripts/gpu/awr2x.sh [env]          default antmaze-medium-navigate-v0
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
# A B run holds about 2.5 GB of GPU memory; the platform allocator returns what it frees.
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
env_name=${1:-antmaze-medium-navigate-v0}
maze=$(echo "$env_name" | cut -d- -f2)
seeds="0 1 2 3 4"
cache=$HOME/.cache/fbplan
calibration=runs/train_b/$maze/clean-awr2x/calibration.json
log() { echo "$(date -u +%H:%M:%S) $*"; }

# Critics trained by earlier code store rho_max; export renames it (idempotent).
uv run python -m fbplan.learned.checkpoint export "$cache"/approach_b/"$maze"/value-single/seed*

log "baseline pi_h, high_alpha 0.2"
bash scripts/gpu/run_seeds.sh "runs/train_high/$maze/awr2x" "$seeds" fbplan.train_high \
  --env "$env_name" --name awr2x --high-alpha 0.2 || log "some baseline runs failed"

log "one shared AWR temperature for B clean"
mkdir -p "$(dirname "$calibration")"
uv run python -m fbplan.learned.train_b calibrate --env "$env_name" --upstream-alpha 0.2 \
  --out "$calibration"

log "B clean at the shared temperature"
bash scripts/gpu/run_seeds.sh "runs/train_b/$maze/clean-awr2x" "$seeds" fbplan.learned.train_b \
  policy --env "$env_name" --variant clean --alpha-from "$calibration" --name clean-awr2x ||
  log "some B runs failed"
log "all done"
