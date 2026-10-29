#!/usr/bin/env bash
# Retrain the baseline pi_h for several seeds of one env, in parallel on one GPU.
# Runs on the instance (scripts/gpu/vast.sh train starts it in tmux).
#
#   bash scripts/gpu/train_baselines.sh antmaze-medium-navigate-v0 [seed ...]   default 0-4
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
# Each run needs well under 1 GB; without this, the first process takes 75 % of GPU memory.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

env_name=$1
shift
seeds=("$@")
[[ ${#seeds[@]} -gt 0 ]] || seeds=(0 1 2 3 4)
maze=$(echo "$env_name" | cut -d- -f2)

# XLA sizes its thread pools to the CPUs a process may use. On a host with 80 cores each run
# started about 500 threads, and a 5th run hit the container's limit (pids.max = 2304).
# Two cores per run are enough: the training loop runs on the GPU inside lax.scan.
pids=()
for i in "${!seeds[@]}"; do
  seed=${seeds[$i]}
  run_dir=runs/train_high/$maze/seed$seed
  mkdir -p "$run_dir"
  taskset -c "$((2 * i)),$((2 * i + 1))" \
    uv run python -m fbplan.train_high --env "$env_name" --seed "$seed" --run-dir "$run_dir" \
    >"$run_dir/stdout.log" 2>&1 &
  pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=$((failed + 1))
done
echo "$env_name: ${#pids[@]} runs, $failed failed" | tee -a "runs/train_high/$maze/launcher.log"
exit $((failed > 0))
