#!/usr/bin/env bash
# T2 (docs/design.md §8, wave 1; H2 of docs/hypotheses/h-learn.md): the baseline pi_h with the
# log-ratio advantage for goal latents (temperature 2.45) and upstream's proxy advantage for
# Gaussian latents (0.1).
# Upstream recipe otherwise; the models go to baselines/<maze>/logratio/seed<k>.
#
#   bash scripts/gpu/logratio.sh [env]          default antmaze-medium-navigate-v0
#   FIRST_CORE=10 bash scripts/gpu/logratio.sh  next to another group of 5 runs (scripts/gpu/run_seeds.sh)
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
env_name=${1:-antmaze-medium-navigate-v0}
maze=$(echo "$env_name" | cut -d- -f2)

echo "$(date -u +%H:%M:%S) baseline pi_h, log-ratio advantage"
bash scripts/gpu/run_seeds.sh "runs/train_high/$maze/logratio" "0 1 2 3 4" fbplan.train_high \
  --env "$env_name" --name logratio --advantage logratio --goal-alpha 2.45
echo "$(date -u +%H:%M:%S) all done"
