#!/usr/bin/env bash
# Retrain the baseline pi_h for several seeds of one env, in parallel on one GPU.
# Runs on the instance (scripts/gpu/vast.sh train starts it in tmux).
#
#   bash scripts/gpu/train_baselines.sh antmaze-medium-navigate-v0 [seed ...]   default 0-4
set -euo pipefail
cd "$(dirname "$0")/../.."

env_name=$1
shift
seeds=${*:-0 1 2 3 4}
maze=$(echo "$env_name" | cut -d- -f2)
exec bash scripts/gpu/run_seeds.sh "runs/train_high/$maze" "$seeds" fbplan.train_high --env "$env_name"
