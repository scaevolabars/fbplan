#!/usr/bin/env bash
# Reproduce the results of REPORT.md: checkpoints, training, evaluations, tables, figures.
#
#   scripts/reproduce.sh [--smoke] [--workers N] [stage ...]
#
# Stages run in this order; the default is "checkpoints eval tables figures":
#   checkpoints  download the frozen checkpoints of the four mazes (73 MB, hash-checked), every
#                published trained checkpoint, and the medium datasets (270 MB)
#   train        train every pi_h family that the results use, 5 seeds each: the retrained
#                baseline with its 2x and log-ratio variants, and approach B (critics, then
#                policies); needs a Linux machine with a GPU (about 1 h per group of 5 seeds on
#                an RTX 3090), so it is not in the default
#   eval         every run of RUNS below, the failure classes and the parity check: the
#                committed CSVs in results/ (git diff shows any change)
#   tables       every table of the report, from the CSVs in results/, into
#                runs/reproduce/tables.md; it rewrites the held-out seed selection and the
#                Mac-vs-Linux check in results/
#   figures      success per maze and time to goal in docs/figures/report/, calibration and
#                held-out goals in results/figures/, the tabular study in docs/figures/theory/,
#                and the critic check of approach B (when the train stage has run)
#
# --smoke checks every stage on CPU in about 10 minutes on an idle 12-core laptop: medium only,
# the first run of each result folder below results/medium/, eval seed 0, two episodes per
# task, 10 training steps, ensembles of one member, and only the cheap tables. It starts from an
# empty runs/smoke/ and writes only there: results/, docs/figures/ and ~/.cache/fbplan stay
# unchanged; a run of a trained model evaluates the model that the smoke run trained. After each
# evaluation it prints how many episodes and which config hash equal the committed run: on a Mac
# the one in results/macos/ if it exists, else the canonical Linux file, where the chaotic
# dynamics give other episodes on another platform. The committed approach A runs come from code
# commit 84a8127, so their config hash differs at a later commit (README, "Code versions").
# --workers N sets the number of evaluation processes (default 4, smoke 1).
# Other OGBench datasets download to ~/.ogbench/data on first use.
set -euo pipefail
cd "$(dirname "$0")/.."
# The committed numbers come from CPU runs. A GPU multiplies float32 in TF32, which changes the
# chaotic episodes, so every stage runs on CPU except the GPU training.
export JAX_PLATFORMS=cpu

# Controllers without an alias in fbplan/controllers/__init__.py.
ROUTE=fbplan.controllers.planner:make_recorded_route
TRAP=fbplan.controllers.trap_aware:make_trap_aware

# Evaluation runs, one per line: CSV below results/, maze, eval seeds, episodes per task and
# eval seed, method, more fbplan.evaluate arguments. {a..b} stands for one line per value.
# A run on the 5 eval tasks runs one process per eval seed and joins them in seed order, as the
# committed CSVs were made; a seed that an earlier line with the same command ran is reused, so
# results/medium_10seeds/ costs only its eval seeds 5-9. A run on the held-out goals (task ids
# 101-110) is one process.
RUNS="
# Medium, 5 eval seeds: the reference rows and the pi_h families, one CSV per training seed
medium/baseline.csv                               medium 0-4 50 baseline
medium/baseline_raw.csv                           medium 0-4 50 baseline_raw
medium/nonhier.csv                                medium 0-4 50 nonhier
medium/oracle.csv                                 medium 0-4 50 oracle
medium/baseline_retrained/seed{0..4}.csv          medium 0-4 50 baseline --checkpoint retrained:{0..4}
medium/baseline_retrained_awr2x/seed{0..4}.csv    medium 0-4 50 baseline --checkpoint retrained:awr2x:{0..4}
medium/baseline_retrained_logratio/seed{0..4}.csv medium 0-4 50 baseline --checkpoint retrained:logratio:{0..4}
medium/b_clean/seed{0..4}.csv                     medium 0-4 50 baseline --checkpoint b:clean:{0..4}
medium/b_wide/seed{0..4}.csv                      medium 0-4 50 baseline --checkpoint b:wide:{0..4}
medium/b_region/seed{0..4}.csv                    medium 0-4 50 baseline --checkpoint b:region:{0..4}
medium/b_clean_awr2x/seed{0..4}.csv               medium 0-4 50 baseline --checkpoint b:clean-awr2x:{0..4}
# Medium, approach A on landmark sets 0-2, and the oracle sweep on eval seed 0
medium/planner/l{0..2}.csv                        medium 0-4 50 planner --opt landmarks.seed={0..2}
medium/planner_high/l{0..2}.csv                   medium 0-4 50 planner_high --opt landmarks.seed={0..2}
medium/planner_region/l{0..2}.csv                 medium 0-4 50 planner --opt landmarks.latent=region landmarks.seed={0..2}
medium/oracle_sweep/state_l{1..3}.csv             medium 0   50 oracle --opt waypoint=state lookahead={1..3}
medium/oracle_sweep/region_r0.5_l{1..3}.csv       medium 0   50 oracle --opt waypoint=region radius=0.5 lookahead={1..3}
medium/oracle_sweep/region_r1.0_l{1..3}.csv       medium 0   50 oracle --opt waypoint=region radius=1.0 lookahead={1..3}
medium/oracle_sweep/region_r1.5_l{1..3}.csv       medium 0   50 oracle --opt waypoint=region radius=1.5 lookahead={1..3}
# Medium, recorded route (H1): the grid on the held-out goals, then the selected config, and
# its replication on the new eval seeds 10-19; its FB arm (H2): the held-out check
medium/planner_ablations/h1_eval/window_k150.csv  medium 0-9 50 $ROUTE --opt executor=window max_steps=150
medium/planner_ablations/h1_heldout/point_k60.csv   medium 0-2 20 $ROUTE --heldout --opt executor=point max_steps=60
medium/planner_ablations/h1_heldout/point_k150.csv  medium 0-2 20 $ROUTE --heldout --opt executor=point max_steps=150
medium/planner_ablations/h1_heldout/window_k60.csv  medium 0-2 20 $ROUTE --heldout --opt executor=window max_steps=60
medium/planner_ablations/h1_heldout/window_k150.csv medium 0-2 20 $ROUTE --heldout --opt executor=window max_steps=150
medium/planner_ablations/h1_heldout/high_k60.csv    medium 0-2 20 $ROUTE --heldout --opt executor=high max_steps=60
medium/planner_ablations/h1_heldout/high_k150.csv   medium 0-2 20 $ROUTE --heldout --opt executor=high max_steps=150
medium/planner_ablations/h1_heldout/oracle.csv      medium 0-2 20 oracle --heldout
medium/planner_ablations/h1_heldout/baseline.csv    medium 0-2 20 baseline --heldout
medium/planner_ablations/h2_heldout/window_k150.csv medium 0-2 20 $ROUTE --heldout --opt executor=window max_steps=150 edge_weight=fb
medium_seeds10-19/h1_window_k150.csv              medium 10-19 50 $ROUTE --opt executor=window max_steps=150
medium_seeds10-19/baseline.csv                    medium 10-19 50 baseline
medium_seeds10-19/baseline_raw.csv                medium 10-19 50 baseline_raw
# Medium, approach A ablations on landmark set 0: one change of the defaults per run
medium/planner_ablations/no_xy_cap/l0.csv         medium 0-4 50 planner --opt landmarks.seed=0 max_edge_xy=None
medium/planner_ablations/reached_1.5_no_lookahead/l0.csv medium 0-4 50 planner --opt landmarks.seed=0 reached_xy=1.5 lookahead=False
medium/planner_ablations/no_lookahead/l0.csv      medium 0-4 50 planner --opt landmarks.seed=0 lookahead=False
medium/planner_ablations/edge_reward_exact/l0.csv medium 0-4 50 planner --opt landmarks.seed=0 options.edge_reward=exact
medium/planner_ablations/edge_reward_paper/l0.csv medium 0-4 50 planner --opt landmarks.seed=0 options.edge_reward=paper
medium/planner_ablations/head_agg_mean/l0.csv     medium 0-4 50 planner --opt landmarks.seed=0 options.head_agg=mean
medium/planner_ablations/two_way/l0.csv           medium 0-4 50 planner --opt landmarks.seed=0 two_way=True
medium/planner_ablations/replan_every_step/l0.csv medium 0-4 50 planner --opt landmarks.seed=0 replan_every_step=True
medium/planner_ablations/route_following/l0.csv   medium 0-4 50 planner --opt landmarks.seed=0 switch_steps=1e999
medium/planner_ablations/high_act_ahead8/l0.csv   medium 0-4 50 planner_high --opt landmarks.seed=0 act_ahead_xy=8.0
# Medium, held-out goals: every seed of the pi_h families (seed selection), and the ensembles
medium/heldout/retrained/seed{0..4}.csv           medium 0-2 20 baseline --heldout --checkpoint retrained:{0..4}
medium/heldout/retrained_awr2x/seed{0..4}.csv     medium 0-2 20 baseline --heldout --checkpoint retrained:awr2x:{0..4}
medium/heldout/retrained_logratio/seed{0..4}.csv  medium 0-2 20 baseline --heldout --checkpoint retrained:logratio:{0..4}
medium/heldout/b_clean/seed{0..4}.csv             medium 0-2 20 baseline --heldout --checkpoint b:clean:{0..4}
medium/heldout/b_wide/seed{0..4}.csv              medium 0-2 20 baseline --heldout --checkpoint b:wide:{0..4}
medium/heldout/b_region/seed{0..4}.csv            medium 0-2 20 baseline --heldout --checkpoint b:region:{0..4}
medium/heldout/b_clean_awr2x/seed{0..4}.csv       medium 0-2 20 baseline --heldout --checkpoint b:clean-awr2x:{0..4}
medium/heldout_ensemble/t1_retrained.csv          medium 0-2 20 ensemble --heldout --opt family=retrained seeds=5
medium/heldout_ensemble/t1_b_clean.csv            medium 0-2 20 ensemble --heldout --opt family=b:clean seeds=5
medium/heldout_ensemble/t1_b_region.csv           medium 0-2 20 ensemble --heldout --opt family=b:region seeds=5
medium/heldout_ensemble/t4_retrained.csv          medium 0-2 20 ensemble --heldout --opt family=retrained seeds=5 z_input=raw
medium/heldout_ensemble/t4_b_clean.csv            medium 0-2 20 ensemble --heldout --opt family=b:clean seeds=5 z_input=raw
medium/heldout_ensemble/t4_b_region.csv           medium 0-2 20 ensemble --heldout --opt family=b:region seeds=5 z_input=raw
# Medium, 10 eval seeds (the final protocol): seeds 0-4 come from the runs above
medium_10seeds/baseline.csv                       medium 0-9 50 baseline
medium_10seeds/baseline_raw.csv                   medium 0-9 50 baseline_raw
medium_10seeds/oracle.csv                         medium 0-9 50 oracle
medium_10seeds/baseline_retrained/seed{0..4}.csv  medium 0-9 50 baseline --checkpoint retrained:{0..4}
medium_10seeds/baseline_retrained_awr2x/seed{0..4}.csv    medium 0-9 50 baseline --checkpoint retrained:awr2x:{0..4}
medium_10seeds/baseline_retrained_logratio/seed{0..4}.csv medium 0-9 50 baseline --checkpoint retrained:logratio:{0..4}
medium_10seeds/b_clean/seed{0..4}.csv             medium 0-9 50 baseline --checkpoint b:clean:{0..4}
medium_10seeds/b_wide/seed{0..4}.csv              medium 0-9 50 baseline --checkpoint b:wide:{0..4}
medium_10seeds/b_region/seed{0..4}.csv            medium 0-9 50 baseline --checkpoint b:region:{0..4}
medium_10seeds/b_clean_awr2x/seed{0..4}.csv       medium 0-9 50 baseline --checkpoint b:clean-awr2x:{0..4}
medium_10seeds/ensemble_retrained.csv             medium 0-9 50 ensemble --opt family=retrained seeds=5
medium_10seeds/ensemble_b_clean.csv               medium 0-9 50 ensemble --opt family=b:clean seeds=5
medium_10seeds/ensemble_b_region.csv              medium 0-9 50 ensemble --opt family=b:region seeds=5
medium_10seeds/ensemble_raw_retrained.csv         medium 0-9 50 ensemble --opt family=retrained seeds=5 z_input=raw
medium_10seeds/ensemble_raw_b_clean.csv           medium 0-9 50 ensemble --opt family=b:clean seeds=5 z_input=raw
medium_10seeds/ensemble_raw_b_region.csv          medium 0-9 50 ensemble --opt family=b:region seeds=5 z_input=raw
# Large, 5 eval seeds: the reference rows, approach A and its guard ablations
large/baseline.csv                                large 0-4 50 baseline
large/baseline_raw.csv                            large 0-4 50 baseline_raw
large/oracle.csv                                  large 0-4 50 oracle
large/planner/l{0..2}.csv                         large 0-4 50 planner --opt landmarks.seed={0..2}
large/planner_high/l{0..2}.csv                    large 0-4 50 planner_high --opt landmarks.seed={0..2}
large/planner_region/l{0..2}.csv                  large 0-4 50 planner --opt landmarks.latent=region landmarks.seed={0..2}
large/planner_ablations/no_xy_cap/l0.csv          large 0-4 50 planner --opt landmarks.seed=0 max_edge_xy=None
large/planner_ablations/reached_1.5_no_lookahead/l0.csv large 0-4 50 planner --opt landmarks.seed=0 reached_xy=1.5 lookahead=False
large/planner_ablations/no_lookahead/l0.csv       large 0-4 50 planner --opt landmarks.seed=0 lookahead=False
# Large, 10 eval seeds and the recorded route (H1): held-out grid, selected config, and its
# replication on the new eval seeds 10-19; its FB arm (H2): the held-out check
large_10seeds/baseline.csv                        large 0-9 50 baseline
large_10seeds/baseline_raw.csv                    large 0-9 50 baseline_raw
large_10seeds/oracle.csv                          large 0-9 50 oracle
large/planner_ablations/h1_heldout/point_k60.csv    large 0-2 20 $ROUTE --heldout --opt executor=point max_steps=60
large/planner_ablations/h1_heldout/point_k150.csv   large 0-2 20 $ROUTE --heldout --opt executor=point max_steps=150
large/planner_ablations/h1_heldout/window_k60.csv   large 0-2 20 $ROUTE --heldout --opt executor=window max_steps=60
large/planner_ablations/h1_heldout/window_k150.csv  large 0-2 20 $ROUTE --heldout --opt executor=window max_steps=150
large/planner_ablations/h1_heldout/high_k60.csv     large 0-2 20 $ROUTE --heldout --opt executor=high max_steps=60
large/planner_ablations/h1_heldout/high_k150.csv    large 0-2 20 $ROUTE --heldout --opt executor=high max_steps=150
large/planner_ablations/h1_heldout/oracle.csv       large 0-2 20 oracle --heldout
large/planner_ablations/h1_heldout/baseline.csv     large 0-2 20 baseline --heldout
large/planner_ablations/h2_heldout/high_k150.csv    large 0-2 20 $ROUTE --heldout --opt executor=high max_steps=150 edge_weight=fb
large/planner_ablations/h1_eval/high_k150.csv     large 0-9 50 $ROUTE --opt executor=high max_steps=150
large_seeds10-19/h1_high_k150.csv                 large 10-19 50 $ROUTE --opt executor=high max_steps=150
large_seeds10-19/baseline.csv                     large 10-19 50 baseline
large_seeds10-19/baseline_raw.csv                 large 10-19 50 baseline_raw
# Teleport, 5 eval seeds
teleport/baseline.csv                             teleport 0-4 50 baseline
teleport/baseline_raw.csv                         teleport 0-4 50 baseline_raw
teleport/nonhier.csv                              teleport 0-4 50 nonhier
teleport/planner/l{0..2}.csv                      teleport 0-4 50 planner --opt landmarks.seed={0..2}
teleport/planner_high/l{0..2}.csv                 teleport 0-4 50 planner_high --opt landmarks.seed={0..2}
teleport/oracle_teleport.csv                      teleport 0-4 50 oracle_teleport
teleport/oracle_teleport_pi_h.csv                 teleport 0-4 50 oracle_teleport --opt follow=pi_h
# Teleport, trap-aware control (T3): the grid on the held-out goals, then the selected config
teleport/heldout/baseline.csv                     teleport 0-2 20 baseline --heldout
teleport/heldout/baseline_raw.csv                 teleport 0-2 20 baseline_raw --heldout
teleport/heldout/trap_aware/w0.5_r1.5.csv         teleport 0-2 20 $TRAP --heldout --opt weight=0.5 trap_radius=1.5
teleport/heldout/trap_aware/w0.5_r2.5.csv         teleport 0-2 20 $TRAP --heldout --opt weight=0.5 trap_radius=2.5
teleport/heldout/trap_aware/w1.0_r1.5.csv         teleport 0-2 20 $TRAP --heldout --opt weight=1.0 trap_radius=1.5
teleport/heldout/trap_aware/w1.0_r2.5.csv         teleport 0-2 20 $TRAP --heldout --opt weight=1.0 trap_radius=2.5
teleport/heldout/trap_aware/w2.0_r1.5.csv         teleport 0-2 20 $TRAP --heldout --opt weight=2.0 trap_radius=1.5
teleport/heldout/trap_aware/w2.0_r2.5.csv         teleport 0-2 20 $TRAP --heldout --opt weight=2.0 trap_radius=2.5
teleport/heldout/trap_aware/w0.5_ungated.csv      teleport 0-2 20 $TRAP --heldout --opt weight=0.5 gate=none
teleport/heldout/trap_aware/w1.0_ungated.csv      teleport 0-2 20 $TRAP --heldout --opt weight=1.0 gate=none
teleport/heldout/trap_aware/w2.0_ungated.csv      teleport 0-2 20 $TRAP --heldout --opt weight=2.0 gate=none
teleport_10seeds/baseline.csv                     teleport 0-9 50 baseline
teleport_10seeds/baseline_raw.csv                 teleport 0-9 50 baseline_raw
teleport_10seeds/trap_aware.csv                   teleport 0-9 50 $TRAP --opt weight=0.5 trap_radius=1.5
# Giant, 5 eval seeds
giant/baseline.csv                                giant 0-4 50 baseline
giant/nonhier.csv                                 giant 0-4 50 nonhier
giant/oracle.csv                                  giant 0-4 50 oracle
giant/planner/l{0..2}.csv                         giant 0-4 50 planner --opt landmarks.seed={0..2}
giant/planner_high/l{0..2}.csv                    giant 0-4 50 planner_high --opt landmarks.seed={0..2}
"

# Failure classes (analysis/failures.py replays the episodes of a run): CSV below results/,
# method, the run's options. The output is <maze>/failures/<rest of the path, / as _>.
FAILURES='
medium/baseline.csv               baseline
medium/baseline_raw.csv           baseline_raw
medium/nonhier.csv                nonhier
medium/oracle.csv                 oracle
medium/planner/l{0..2}.csv        planner      --opt landmarks.seed={0..2}
medium/planner_high/l{0..2}.csv   planner_high --opt landmarks.seed={0..2}
medium/planner_region/l{0..2}.csv planner      --opt landmarks.latent=region landmarks.seed={0..2}
large/baseline.csv                baseline
large/oracle.csv                  oracle
giant/oracle.csv                  oracle
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
  work=runs/smoke # scratch files, logs, tables.md
  root=$work/results
  report_dir=$work/report
  figure_dir=$work/figures
  theory_dir=$work/theory
  episodes=2
  workers=${workers:-1}
  parity_reps=1
  study_args=--quick
  # Trained models of the smoke run must not replace the real ones in ~/.cache/fbplan.
  export FBPLAN_BASELINE_DIR=$PWD/$work/models/baselines
  export FBPLAN_APPROACH_B_DIR=$PWD/$work/models/approach_b
  if [ -z "$selected" ]; then
    selected=" $STAGES "
    rm -rf "$work"
  fi
else
  work=runs/reproduce
  root=results
  report_dir=docs/figures/report
  figure_dir=results/figures
  theory_dir=docs/figures/theory
  episodes=
  workers=${workers:-4}
  parity_reps=5
  study_args=
  [ -n "$selected" ] || selected=' checkpoints eval tables figures '
fi

# Print the lines of a table ($1) without comments; a line with {a..b} once per value.
table_lines() {
  local line range pattern k
  while IFS= read -r line; do
    case $line in '' | '#'*) continue ;; esac
    case $line in
      *'{'*'..'*'}'*)
        range=${line#*\{}
        range=${range%%\}*}
        pattern="{$range}"
        for k in $(seq "${range%..*}" "${range#*..}"); do
          echo "${line//"$pattern"/$k}"
        done
        ;;
      *) echo "$line" ;;
    esac
  done <<<"$1"
}

stage_checkpoints() {
  if $smoke; then
    py -m fbplan.frozen "$(env_id medium)" --datasets "$(env_id medium)"
  else
    py -m fbplan.frozen --trained --datasets "$(env_id medium)"
  fi
}

# Train one job ($2...: module and arguments) for every training seed; $1 is the folder of its
# logs below runs/. On a GPU the seeds run in parallel; --smoke runs seed 0 for a few steps.
train() {
  local logs=$1
  shift
  if $smoke; then
    py -m "$@" --env "$(env_id medium)" --seed 0 --steps 10 --log-every 5 \
      --run-dir "$work/$logs/seed0"
  else
    run bash scripts/gpu/run_seeds.sh "runs/$logs" "$SEEDS" "$@" --env "$(env_id medium)"
  fi
}

stage_train() {
  local logs=runs calibration_args=()
  if $smoke; then
    logs=$work
    calibration_args=(--seeds 0 --batches 2)
  else
    unset JAX_PLATFORMS
    uv run --locked python -c 'import jax, sys; sys.exit(jax.default_backend() != "gpu")' ||
      die "train needs a GPU; --smoke checks the training code on CPU"
  fi
  local calibration=$logs/train_b/medium/clean-awr2x/calibration.json
  # The baseline pi_h (retrained:<k>), at 2x AWR sharpness, and with the log-ratio advantage.
  train train_high/medium fbplan.train_high
  train train_high/medium/awr2x fbplan.train_high --name awr2x --high-alpha 0.2
  train train_high/medium/logratio fbplan.train_high --name logratio --advantage logratio \
    --goal-alpha 2.45
  # Approach B: the critics V_H first (one per seed), then the policies that use them.
  train train_b/medium/value-single fbplan.learned.train_b value --latents single
  train train_b/medium/value-region fbplan.learned.train_b value --latents region
  train train_b/medium/value-single-expectile0.7 fbplan.learned.train_b value --latents single \
    --expectile 0.7 --name value-single-expectile0.7
  train train_b/medium/clean fbplan.learned.train_b policy --variant clean
  train train_b/medium/wide fbplan.learned.train_b policy --variant wide
  train train_b/medium/region fbplan.learned.train_b policy --variant region
  # B clean at 2x: one AWR temperature for all seeds, matched to the baseline at 2x. It runs on
  # the GPU because the CPU and TF32 give different spreads.
  mkdir -p "$(dirname "$calibration")"
  py -m fbplan.learned.train_b calibrate --env "$(env_id medium)" --upstream-alpha 0.2 \
    ${calibration_args[@]+"${calibration_args[@]}"} --out "$calibration"
  train train_b/medium/clean-awr2x fbplan.learned.train_b policy --variant clean \
    --alpha-from "$calibration" --name clean-awr2x
  export JAX_PLATFORMS=cpu
}

# Evaluate one run of RUNS: $1 CSV below results/, $2 maze, $3 eval seeds (a or a-b),
# $4 episodes, $5 method, then more fbplan.evaluate arguments.
evaluate() {
  local csv=$1 maze=$2 first=${3%-*} last=${3#*-} n=$4 method=$5 key s part
  shift 5
  if $smoke; then
    last=$first
    n=$episodes
    set -- "${@/seeds=5/seeds=1}" # an ensemble of the one seed that the smoke run trained
  fi
  local command=(py -m fbplan.evaluate --method "$method" --env "$(env_id "$maze")"
    --episodes "$n" --workers "$workers" "$@")
  case " $* " in
    *" --heldout "*)
      # shellcheck disable=SC2046 # seq prints one seed per word
      "${command[@]}" --seeds $(seq "$first" "$last") --out "$root/$csv"
      return
      ;;
  esac
  key=$(printf '%s_' "$maze" "$n" "$method" "$@" | tr -c 'A-Za-z0-9.=-' '_')
  local parts=()
  for s in $(seq "$first" "$last"); do
    part=$work/eval/$key/seed$s.csv
    [ -f "$part" ] || "${command[@]}" --seeds "$s" --out "$part"
    parts+=("$part")
  done
  mkdir -p "$(dirname "$root/$csv")"
  {
    head -n 1 "${parts[0]}"
    for part in "${parts[@]}"; do tail -n +2 "$part"; done
  } >"$root/$csv"
}

# Print how many episodes of a smoke CSV have the success and step count of the committed
# episode with the same (task, eval seed, episode), and whether the config hashes are equal.
# $1 is the CSV path below results/, $2 the run's fbplan.evaluate arguments.
compare_with_committed() {
  local committed=results/$1 note=
  case " $2 " in
    *" --checkpoint "* | *" family="*) note=" (a model that the smoke run trained)" ;;
  esac
  if [ "$(uname)" = Darwin ]; then
    if [ -f "results/macos/$1" ]; then
      committed=results/macos/$1
    elif [ -z "$note" ]; then
      note=" (a Linux run: other episodes on a Mac)"
    fi
  fi
  if [ ! -f "$committed" ]; then
    echo "$1: no committed CSV to compare with"
    return
  fi
  awk -F, -v note="$note" '
    FNR == 1 { delete col; for (i = 1; i <= NF; i++) col[$i] = i; next }
    { key = $col["task_id"] "," $col["eval_seed"] "," $col["episode"]
      value = $col["success"] "," $col["steps"] }
    NR == FNR { committed[key] = value; if (hash == "") hash = $col["config_hash"] ""; next }
    { n++; same += committed[key] == value; if (ours == "") ours = $col["config_hash"] "" }
    END { printf "%s: %d of %d episodes equal %s%s; config hash %s\n", FILENAME, same, n,
          ARGV[1], note, (ours == hash ? "equal" : "differs: " ours " against " hash) }' \
    "$committed" "$root/$1"
}

stage_eval() {
  local csv maze seeds n method args top rest group done_groups=' ' out
  rm -rf "$work/eval"
  # fd 3: the commands in the loop must not read the run table from stdin.
  while read -r csv maze seeds n method args <&3; do
    if $smoke; then
      top=${csv%%/*}
      [ "$top" = medium ] || continue
      rest=${csv#*/}
      group=$top/${rest%%/*} # medium/baseline.csv, or a folder such as medium/planner
      case $done_groups in *" $group "*) continue ;; esac
      done_groups="$done_groups$group "
    fi
    # shellcheck disable=SC2086 # $args holds several arguments
    evaluate "$csv" "$maze" "$seeds" "$n" "$method" $args
    if $smoke; then compare_with_committed "$csv" "$args"; fi
  done 3<<<"$(table_lines "$RUNS")"

  while read -r csv method args <&3; do
    [ -f "$root/$csv" ] || continue # --smoke ran only the first run of each folder
    if $smoke; then case " $args " in *" --opt "*) continue ;; esac; fi
    rest=${csv#*/}
    out=$root/${csv%%/*}/failures/$(echo "$rest" | tr / _)
    # shellcheck disable=SC2086 # $args holds several arguments
    py analysis/failures.py "$root/$csv" --method "$method" $args --workers "$workers" \
      --out "$out"
  done 3<<<"$(table_lines "$FAILURES")"

  # Upstream's own evaluation loop, for the parity check; its resets are not seeded.
  py analysis/parity.py --ours "$root/medium/baseline.csv" --reps "$parity_reps" \
    --episodes "${episodes:-50}" --workers "$workers" --out "$root/medium/upstream.csv"
}

# The CSVs of the main table and the success figure of a maze: the reference rows, and the
# approach A rows, which pool their landmark sets. upstream.csv is left out: its episodes do not
# pair with ours (analysis/parity.py).
main_runs() {
  local dir=$root/$1 name
  for name in baseline_raw baseline nonhier oracle oracle_teleport oracle_teleport_pi_h; do
    if [ -f "$dir/$name.csv" ]; then echo "$dir/$name.csv"; fi
  done
  echo "$dir"/planner*/*.csv
}

heading() {
  printf '\n# %s\n\n' "$*"
}

# Every table of the report, in the order of its sections, from the CSVs in results/.
report_tables() {
  local m=results/medium t=results/medium_10seeds l=results/large dir family
  heading "Medium, 5 eval seeds: reference rows and approach A (report 7.1)"
  # shellcheck disable=SC2046 # main_runs prints one path per word
  py analysis/tables.py $(main_runs medium) --label-root $m --cluster config_hash
  heading "Medium, 5 eval seeds: the pi_h families pooled over training seeds (7.1, 8.5)"
  py analysis/tables.py $m/baseline.csv $m/baseline_raw.csv $m/baseline_retrained*/*.csv \
    $m/b_*/*.csv --label-root $m --cluster checkpoint_seed \
    --reference baseline baseline_raw baseline_retrained baseline_retrained_awr2x
  heading "Parity: upstream's evaluation loop against our baseline, unpaired (6)"
  py analysis/tables.py $m/baseline.csv $m/baseline_raw.csv $m/upstream.csv
  heading "Medium oracle sweep, eval seed 0 (8.2)"
  py analysis/tables.py $m/oracle_sweep/*.csv --label-root $m
  for dir in $m $l; do
    heading "Approach A ablations, landmark set 0, $dir (4, 8.4)"
    py analysis/tables.py "$dir"/planner/l0.csv "$dir"/planner_high/l0.csv \
      "$dir"/planner_ablations/*/l0.csv "$dir"/baseline.csv "$dir"/baseline_raw.csv \
      "$dir"/oracle.csv --label-root "$dir" \
      --reference planner/l0 planner_high/l0 baseline baseline_raw oracle
  done
  heading "Medium, 10 eval seeds: the final protocol, T1, T2, T4 (7.1, 8.6)"
  py analysis/tables.py $t/*.csv $t/*/*.csv --label-root $t --cluster checkpoint_seed \
    --reference baseline baseline_raw baseline_retrained ensemble_retrained ensemble_raw_retrained
  heading "Medium, 10 eval seeds: single training seeds, seed gaps and held-out selection (7.1)"
  py analysis/tables.py $t/*.csv $t/*/*.csv --label-root $t --reference b_clean/seed1 \
    b_clean/seed3 baseline_retrained/seed2 baseline_retrained/seed3 baseline_retrained/seed0
  for family in baseline_retrained:retrained b_clean:b_clean b_region:b_region; do
    heading "Ensemble of ${family%:*} against its own training seeds (8.6)"
    py analysis/tables.py $t/"${family%:*}"/*.csv $t/ensemble_"${family#*:}".csv \
      $t/ensemble_raw_"${family#*:}".csv --label-root $t --cluster checkpoint_seed \
      --reference "${family%:*}"
  done
  heading "AWR at 2x, 10 eval seeds (8.5)"
  py analysis/tables.py $t/baseline*.csv $t/baseline_retrained*/*.csv $t/b_clean*/*.csv \
    --label-root $t --cluster checkpoint_seed \
    --reference baseline_retrained_awr2x baseline_raw baseline_retrained b_clean
  py analysis/tables.py $t/baseline*.csv $t/baseline_retrained*/*.csv $t/b_clean*/*.csv \
    --label-root $t --reference b_clean_awr2x/seed1 baseline_retrained_awr2x/seed1 \
    baseline_retrained_awr2x/seed3 baseline_retrained_awr2x/seed4
  heading "Ensembles on the medium held-out goals (8.6)"
  py analysis/tables.py $m/heldout_ensemble/*.csv --label-root $m/heldout_ensemble \
    --reference t1_retrained t4_retrained
  heading "Recorded route (H1) on medium: held-out grid, eval seeds 0-9, 10-19 and 0-19 (7.1, 8.7)"
  py analysis/tables.py $m/planner_ablations/h1_heldout/*.csv \
    --label-root $m/planner_ablations/h1_heldout --reference baseline oracle
  py analysis/tables.py $t/baseline.csv $t/baseline_raw.csv $t/oracle.csv \
    $m/planner_ablations/h1_eval/window_k150.csv --reference baseline baseline_raw oracle
  py analysis/tables.py results/medium_seeds10-19/*.csv --reference baseline baseline_raw
  py analysis/tables.py $t/baseline.csv $t/baseline_raw.csv \
    $m/planner_ablations/h1_eval/window_k150.csv results/medium_seeds10-19/*.csv \
    --reference baseline baseline_raw
  heading "Large, 5 eval seeds (7.2)"
  # shellcheck disable=SC2046 # main_runs prints one path per word
  py analysis/tables.py $(main_runs large) --label-root $l --cluster config_hash
  heading "Recorded route (H1) on large: held-out grid, eval seeds 0-9, 10-19 and 0-19 (7.2, 8.7)"
  py analysis/tables.py $l/planner_ablations/h1_heldout/*.csv \
    --label-root $l/planner_ablations/h1_heldout --reference baseline oracle
  py analysis/tables.py results/large_10seeds/*.csv $l/planner_ablations/h1_eval/high_k150.csv \
    --reference baseline baseline_raw oracle
  py analysis/tables.py results/large_seeds10-19/*.csv --reference baseline baseline_raw
  py analysis/tables.py results/large_10seeds/baseline.csv results/large_10seeds/baseline_raw.csv \
    $l/planner_ablations/h1_eval/high_k150.csv results/large_seeds10-19/*.csv \
    --reference baseline baseline_raw
  heading "FB arm of the recorded route (H2): held-out check on medium and large (8.7)"
  py analysis/tables.py $m/planner_ablations/h2_heldout/*.csv $m/planner_ablations/h1_heldout/*.csv \
    --label-root $m/planner_ablations --reference h1_heldout/baseline h1_heldout/window_k150
  py analysis/tables.py $l/planner_ablations/h2_heldout/*.csv $l/planner_ablations/h1_heldout/*.csv \
    --label-root $l/planner_ablations --reference h1_heldout/baseline h1_heldout/high_k150
  heading "Teleport, 5 eval seeds (7.3)"
  # shellcheck disable=SC2046 # main_runs prints one path per word
  py analysis/tables.py $(main_runs teleport) --label-root results/teleport --cluster config_hash
  heading "Trap-aware control (T3) on teleport: held-out grid, then eval seeds 0-9 (8.6)"
  py analysis/tables.py results/teleport/heldout/*.csv results/teleport/heldout/trap_aware/*.csv \
    --label-root results/teleport/heldout --reference baseline
  py analysis/tables.py results/teleport_10seeds/*.csv --label-root results/teleport_10seeds \
    --reference baseline baseline_raw
  heading "Giant, 5 eval seeds (7.3)"
  # shellcheck disable=SC2046 # main_runs prints one path per word
  py analysis/tables.py $(main_runs giant) --label-root results/giant --cluster config_hash
}

# Print whether a file that --smoke rebuilt ($1, below results/) equals the committed one.
compare_file() {
  if cmp -s "$root/$1" "results/$1"; then
    echo "$1: equal to the committed file"
  else
    echo "$1: differs from the committed file"
  fi
}

# Write a table that the report quotes ($1, below results/) from the committed CSVs. --smoke
# writes it below runs/smoke/ and compares it with the committed file.
derived_csv() {
  local csv=$1
  shift
  py "$@" --out "$root/$csv"
  if $smoke; then compare_file "$csv"; fi
}

stage_tables() {
  mkdir -p "$work"
  derived_csv medium/heldout_selection.csv analysis/heldout_selection.py --results results/medium
  derived_csv medium/heldout_selection_awr2x.csv analysis/heldout_selection.py \
    --results results/medium --reference retrained_awr2x
  derived_csv platform_check.csv analysis/platform_check.py --linux results --macos results/macos
  if $smoke; then
    # The smoke CSVs; the tables of the report take minutes each on the committed CSVs.
    # shellcheck disable=SC2046 # main_runs prints one path per word
    py analysis/tables.py $(main_runs medium) --label-root "$root/medium" \
      --cluster config_hash >"$work/tables.md"
  else
    report_tables >"$work/tables.md"
  fi
  echo "tables: $work/tables.md"
}

# The offline check of the approach B critics (report 8.5); it needs the trained critics.
critic_check() {
  local dir=${FBPLAN_APPROACH_B_DIR:-$HOME/.cache/fbplan/approach_b}/medium value
  local args=(--seeds 0 1 2 3 4)
  if $smoke; then args=(--seeds 0 --n 50); fi
  for value in value-single value-single-expectile0.7; do
    if [ ! -d "$dir/$value/seed0" ]; then
      echo "critic check of $value skipped: no trained critic in $dir (stage train)"
      continue
    fi
    py -m fbplan.learned.diagnostics --env "$(env_id medium)" --value "$value" "${args[@]}" \
      --out-dir "$work/critic_check"
  done
}

stage_figures() {
  local maze agg
  if $smoke; then
    # shellcheck disable=SC2046 # main_runs prints one path per word
    py analysis/plots.py $(main_runs medium) --label-root "$root/medium" --cluster config_hash \
      --out-dir "$report_dir"
    py analysis/calibration.py --env "$(env_id medium)" --n 50 --fig-dir "$figure_dir" \
      --table-dir "$root/calibration"
    py analysis/heldout_goals.py --envs "$(env_id medium)" --fig-dir "$figure_dir" \
      --table-dir "$root/heldout_goals"
    compare_file "heldout_goals/$(env_id medium).csv"
  else
    for maze in medium large giant teleport; do
      # shellcheck disable=SC2046 # main_runs prints one path per word
      py analysis/plots.py $(main_runs "$maze") --label-root "$root/$maze" \
        --cluster config_hash --out-dir "$report_dir"
    done
    for maze in medium large; do
      for agg in min mean; do
        py analysis/calibration.py --env "$(env_id "$maze")" --head-agg "$agg" \
          --fig-dir "$figure_dir" --table-dir "$root/calibration"
      done
    done
    for maze in giant teleport; do
      py analysis/calibration.py --env "$(env_id "$maze")" --fig-dir "$theory_dir" \
        --table-dir "$theory_dir"
    done
    py analysis/heldout_goals.py --fig-dir "$figure_dir" --table-dir "$root/heldout_goals"
  fi
  # Time to goal (report 8.4): the figure, and on stdout the numbers that the report quotes.
  py analysis/report_figures.py "$root/medium" --out-dir "$report_dir"
  # shellcheck disable=SC2086 # $study_args is empty or one flag
  py -m fbplan.tabular.run_study --workers "$workers" --out-dir "$theory_dir" $study_args
  py -m fbplan.tabular.study_plots "$theory_dir"
  py -m fbplan.tabular.time_limit --out-dir "$theory_dir"
  critic_check
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
