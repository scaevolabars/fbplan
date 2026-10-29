#!/usr/bin/env bash
# Run one training module for several seeds in parallel on one GPU. Runs on the instance.
# Each run gets `--seed k --run-dir <run-root>/seed<k>` and logs to <run-root>/seed<k>/stdout.log.
#
#   bash scripts/gpu/run_seeds.sh <run-root> "<seeds>" <module> [args ...]
#   bash scripts/gpu/run_seeds.sh runs/train_b/medium/value-single "0 1 2 3 4" \
#     fbplan.learned.train_b value --env antmaze-medium-navigate-v0 --latents single
# FIRST_CORE (default 0): the first CPU core of this group, so that two groups can run at once.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
# Each run needs about 1 GB; without this, the first process takes 75 % of GPU memory.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

run_root=$1
read -r -a seeds <<<"$2"
module=$3
shift 3

# XLA sizes its thread pools to the CPUs a process may use. On a host with 80 cores each run
# started about 500 threads, and a 5th run hit the container's limit (pids.max = 2304).
# Two cores per run are enough: the training loop runs on the GPU inside lax.scan.
first_core=${FIRST_CORE:-0}
pids=()
for i in "${!seeds[@]}"; do
  seed=${seeds[$i]}
  run_dir=$run_root/seed$seed
  mkdir -p "$run_dir"
  taskset -c "$((first_core + 2 * i)),$((first_core + 2 * i + 1))" \
    uv run python -m "$module" "$@" --seed "$seed" --run-dir "$run_dir" >"$run_dir/stdout.log" 2>&1 &
  pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=$((failed + 1))
done
echo "$module $*: ${#pids[@]} runs, $failed failed" | tee -a "$run_root/launcher.log"
exit $((failed > 0))
