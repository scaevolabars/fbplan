#!/usr/bin/env bash
# Attribution control (h-learn's H1, docs/hypotheses/h-learn.md): B clean with V_1 where the
# targets use V_H, at the shared 2x temperature of B clean 2x and on the same batches. So it
# separates the learned critic from the AWR temperature. Runs after scripts/gpu/awr2x.sh.
#
#   bash scripts/gpu/awr2x_v1.sh [env]          default antmaze-medium-navigate-v0
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
env_name=${1:-antmaze-medium-navigate-v0}
maze=$(echo "$env_name" | cut -d- -f2)
calibration=runs/train_b/$maze/clean-awr2x/calibration.json
[[ -f $calibration ]] || { echo "missing $calibration: run scripts/gpu/awr2x.sh first" >&2; exit 1; }

echo "$(date -u +%H:%M:%S) B clean, V_1 critic, shared 2x temperature"
bash scripts/gpu/run_seeds.sh "runs/train_b/$maze/clean-v1-awr2x" "0 1 2 3 4" \
  fbplan.learned.train_b policy --env "$env_name" --variant clean --critic v_1 \
  --alpha-from "$calibration" --name clean-v1-awr2x
echo "$(date -u +%H:%M:%S) all done"
