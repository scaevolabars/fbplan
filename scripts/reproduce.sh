#!/usr/bin/env bash
# Reproduce the tables and figures of REPORT.md: checkpoints, evaluations, tables, figures.
#
#   scripts/reproduce.sh [--smoke] [--workers N] [stage ...]
#
# Stages run in this order; the default is "checkpoints eval tables figures":
#   checkpoints  download the frozen checkpoints of the four mazes (73 MB, hash-checked) and
#                every published trained checkpoint
#   train        retrain pi_h and train approach B, 5 seeds each; needs a Linux machine with a
#                GPU (about 1 h per job on an RTX 3090), so it is not in the default
#   eval         evaluate every run of RUNS below: one CSV of episodes per run in
#                results/<maze>/, the committed files (git diff shows any change)
#   tables       success with 95 % CIs, paired tests, failure classes and the numbers of the
#                report's time-to-goal section: results/<maze>/tables.md
#   figures      success, calibration, held-out goal and route figures in results/figures/;
#                the tabular study in docs/figures/theory/
#
# --smoke runs every stage, train included, on CPU, in about 10 minutes on an idle 12-core
# laptop: medium only, the first run of each result group (a CSV in results/<maze>/ or a
# subfolder of it), one eval seed, two episodes per task, no parity run, 10 training steps. It
# starts from an empty runs/smoke/ and writes only there: results/, docs/figures/ and
# ~/.cache/fbplan stay unchanged. After each evaluation it prints how many episodes equal the
# committed ones (resets are seeded, so on the same machine all of them should; another CPU can
# change the chaotic dynamics).
# --workers N sets the number of evaluation processes (default 4, smoke 1).
# OGBench datasets download to ~/.ogbench/data on first use (medium train and val: 270 MB).
set -euo pipefail
cd "$(dirname "$0")/.."

# Evaluation runs, one per line: maze, CSV below results/<maze>/, number of eval seeds
# (counted from 0), method (fbplan/controllers/__init__.py), more fbplan.evaluate arguments.
RUNS='
medium baseline.csv                    5 baseline
medium baseline_raw.csv                5 baseline_raw
medium nonhier.csv                     5 nonhier
medium oracle.csv                      5 oracle
medium planner/l0.csv                  5 planner      --opt landmarks.seed=0
medium planner/l1.csv                  5 planner      --opt landmarks.seed=1
medium planner/l2.csv                  5 planner      --opt landmarks.seed=2
medium planner_high/l0.csv             5 planner_high --opt landmarks.seed=0
medium planner_high/l1.csv             5 planner_high --opt landmarks.seed=1
medium planner_high/l2.csv             5 planner_high --opt landmarks.seed=2
medium planner_region/l0.csv           5 planner      --opt landmarks.latent=region landmarks.seed=0
medium planner_region/l1.csv           5 planner      --opt landmarks.latent=region landmarks.seed=1
medium planner_region/l2.csv           5 planner      --opt landmarks.latent=region landmarks.seed=2
medium oracle_sweep/state_l1.csv       1 oracle       --opt waypoint=state lookahead=1
medium oracle_sweep/state_l2.csv       1 oracle       --opt waypoint=state lookahead=2
medium oracle_sweep/state_l3.csv       1 oracle       --opt waypoint=state lookahead=3
medium oracle_sweep/region_r0.5_l1.csv 1 oracle       --opt waypoint=region radius=0.5 lookahead=1
medium oracle_sweep/region_r0.5_l2.csv 1 oracle       --opt waypoint=region radius=0.5 lookahead=2
medium oracle_sweep/region_r0.5_l3.csv 1 oracle       --opt waypoint=region radius=0.5 lookahead=3
medium oracle_sweep/region_r1.0_l1.csv 1 oracle       --opt waypoint=region radius=1.0 lookahead=1
medium oracle_sweep/region_r1.0_l2.csv 1 oracle       --opt waypoint=region radius=1.0 lookahead=2
medium oracle_sweep/region_r1.0_l3.csv 1 oracle       --opt waypoint=region radius=1.0 lookahead=3
medium oracle_sweep/region_r1.5_l1.csv 1 oracle       --opt waypoint=region radius=1.5 lookahead=1
medium oracle_sweep/region_r1.5_l2.csv 1 oracle       --opt waypoint=region radius=1.5 lookahead=2
medium oracle_sweep/region_r1.5_l3.csv 1 oracle       --opt waypoint=region radius=1.5 lookahead=3
'
STAGES='checkpoints train eval tables figures'
SEEDS='0 1 2 3 4' # training seeds

usage() {
  sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'
}

die() {
  echo "error: $*" >&2
  exit 1
}

# Print a command on stderr, then run it. Stdout carries only the command's output.
run() {
  echo "+ $*" >&2
  "$@"
}

py() {
  run uv run --locked python "$@"
}

env_id() {
  echo "antmaze-$1-navigate-v0"
}

smoke=false
workers=
selected=
while [ $# -gt 0 ]; do
  case $1 in
    --smoke) smoke=true ;;
    --workers)
      [ $# -ge 2 ] || die "--workers needs a number"
      workers=$2
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    -*) die "unknown option $1 (see --help)" ;;
    *)
      case " $STAGES " in
        *" $1 "*) selected="$selected $1 " ;;
        *) die "unknown stage $1 (stages: $STAGES)" ;;
      esac
      ;;
  esac
  shift
done

if $smoke; then
  root=runs/smoke/results
  theory_dir=runs/smoke/theory
  report_dir=runs/smoke/report
  episodes=2
  workers=${workers:-1}
  study_args=--quick
  train_steps=10
  # Trained models of the smoke run must not replace the real ones in ~/.cache/fbplan.
  export FBPLAN_BASELINE_DIR=$PWD/runs/smoke/models/baselines
  export FBPLAN_APPROACH_B_DIR=$PWD/runs/smoke/models/approach_b
  if [ -z "$selected" ]; then
    selected=" $STAGES "
    rm -rf runs/smoke
  fi
else
  root=results
  theory_dir=docs/figures/theory
  report_dir=docs/figures/report
  episodes=50
  workers=${workers:-4}
  study_args=
  [ -n "$selected" ] || selected=' checkpoints eval tables figures '
fi

stage_checkpoints() {
  if $smoke; then
    py -m fbplan.frozen "$(env_id medium)"
  else
    py -m fbplan.frozen --trained
  fi
}

stage_train() {
  local env
  env=$(env_id medium)
  if $smoke; then
    local steps=(--seed 0 --steps "$train_steps" --log-every $((train_steps / 2)))
    py -m fbplan.train_high --env "$env" "${steps[@]}" --run-dir runs/smoke/train/high
    py -m fbplan.learned.train_b value --env "$env" "${steps[@]}" \
      --run-dir runs/smoke/train/value-single
    py -m fbplan.learned.train_b policy --env "$env" --variant clean "${steps[@]}" \
      --run-dir runs/smoke/train/clean
    return
  fi
  uv run --locked python -c 'import jax, sys; sys.exit(jax.default_backend() != "gpu")' ||
    die "train needs a GPU; --smoke checks the training code on CPU"
  run bash scripts/gpu/run_seeds.sh runs/train_high/medium "$SEEDS" fbplan.train_high --env "$env"
  run bash scripts/gpu/run_seeds.sh runs/train_b/medium/value-single "$SEEDS" \
    fbplan.learned.train_b value --env "$env" --latents single
  run bash scripts/gpu/run_seeds.sh runs/train_b/medium/clean "$SEEDS" \
    fbplan.learned.train_b policy --env "$env" --variant clean
}

stage_eval() {
  local maze csv n_seeds method args group done_groups=' '
  # fd 3: the commands in the loop must not read the run table from stdin.
  while read -r maze csv n_seeds method args <&3; do
    [ -n "$maze" ] || continue
    if $smoke; then
      group=$maze/${csv%%/*} # baseline.csv, or a folder such as planner
      case $done_groups in *" $group "*) continue ;; esac
      done_groups="$done_groups$group "
      n_seeds=1
    fi
    # shellcheck disable=SC2086 # $args holds several arguments
    py -m fbplan.evaluate --method "$method" --env "$(env_id "$maze")" \
      --seeds $(seq 0 $((n_seeds - 1))) --episodes "$episodes" --workers "$workers" \
      --out "$root/$maze/$csv" $args
    if $smoke; then compare_with_committed "results/$maze/$csv" "$root/$maze/$csv"; fi
  done 3<<<"$RUNS"
  # Upstream's own evaluation loop, for the parity check; its resets are not seeded. It always
  # runs 50 episodes per task and repetition, which is too slow for --smoke.
  $smoke || py analysis/parity.py --ours "$root/medium/baseline.csv" --reps 5 \
    --workers "$workers" --out "$root/medium/upstream.csv"
}

# Print how many episodes of a smoke CSV have the success and step count of the committed
# episode with the same (task, eval seed, episode).
compare_with_committed() {
  awk -F, 'FNR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
    { key = $col["task_id"] "," $col["eval_seed"] "," $col["episode"]
      value = $col["success"] "," $col["steps"] }
    NR == FNR { committed[key] = value; next }
    { n++; same += committed[key] == value }
    END { printf "%s: %d of %d episodes equal the committed ones\n", FILENAME, same, n }' "$1" "$2"
}

# Failure classes of one run without controller options, as a text block in Markdown.
# analysis/failures.py needs at least one failed episode in the CSV.
failures() {
  local csv=$root/$1/$2.csv
  printf '\n```text\n'
  if grep -q ',False,' "$csv"; then
    py analysis/failures.py "$csv" --method "$2" --workers "$workers"
  else
    echo "$2: no failed episode in $csv"
  fi
  printf '```\n'
}

# The CSVs of the main table and the success figure: planner rows pool their landmark sets.
# upstream.csv is left out: its episodes do not pair with ours (analysis/parity.py).
main_runs() {
  local dir=$root/$1
  echo "$dir"/{baseline,baseline_raw,nonhier,oracle}.csv "$dir"/planner*/*.csv
}

stage_tables() {
  local dir=$root/medium
  {
    # shellcheck disable=SC2046 # main_runs prints one path per word
    py analysis/tables.py $(main_runs medium) --label-root "$dir" --cluster config_hash
    py analysis/tables.py "$dir"/oracle_sweep/*.csv --label-root "$dir"
    failures medium baseline
    failures medium baseline_raw
    failures medium nonhier
    failures medium oracle
    # Time to goal (report §8.4): the figure, and on stdout the numbers that the report quotes.
    printf '\n```text\n'
    py analysis/report_figures.py --results "$dir" --out-dir "$report_dir"
    printf '```\n'
  } >"$dir/tables.md"
  echo "tables: $dir/tables.md"
}

stage_figures() {
  local figures=$root/figures maze agg
  # shellcheck disable=SC2046 # main_runs prints one path per word
  py analysis/plots.py $(main_runs medium) --label-root "$root/medium" --cluster config_hash \
    --out-dir "$figures"
  if $smoke; then
    py analysis/calibration.py --env "$(env_id medium)" --n 50 --out-dir "$figures" \
      --table-dir "$root/calibration"
  else
    for maze in medium large; do
      for agg in min mean; do
        py analysis/calibration.py --env "$(env_id "$maze")" --head-agg "$agg" \
          --out-dir "$figures" --table-dir "$root/calibration"
      done
    done
    # It always writes the goal files to results/heldout_goals/, so --smoke skips it.
    py analysis/heldout_goals.py --fig-dir "$figures"
  fi
  py analysis/planner_routes.py --env "$(env_id medium)" --out-dir "$figures"
  py -m fbplan.tabular.run_study --workers "$workers" --out "$theory_dir" $study_args
  py -m fbplan.tabular.study_plots --dir "$theory_dir"
  py -m fbplan.tabular.time_limit --out "$theory_dir"
}

# third_party/ssm is a git submodule: a clone without --recurse-submodules leaves it empty.
[ -e third_party/ssm/agents ] || run git submodule update --init

for stage in $STAGES; do
  case $selected in
    *" $stage "*)
      printf '\n==> %s\n' "$stage"
      start=$SECONDS
      "stage_$stage"
      echo "($stage: $((SECONDS - start)) s)"
      ;;
  esac
done
printf '\nDone in %d min.\n' $((SECONDS / 60))
