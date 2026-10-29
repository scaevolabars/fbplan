# Status: d-eval

Last update: 2026-10-29, 20:00. Branch merged with `main` at 8dac124.

Platform decision (user, through the coordinator, 2026-10-29 19:50): all final numbers come
from one Linux CPU instance that d-eval operates (queue below). The Mac results moved to
`results/macos/` (a reproducibility check); `results/<env>/` will hold the Linux CSVs.

## Done

- `fbplan/evaluate.py`: episode loop, `make_tasks`, `run_episodes`, `run_parallel` (spawn
  workers), `write_csv`, CLI `python -m fbplan.evaluate --method <alias|module:attr>
  [--opt key=value ...] [--checkpoint ...] [--heldout]` (default: 5 eval seeds). Public for
  other components: `reset_seed`, `reset_env`, `run_episode`, `RecordingController`,
  `parse_options`, `build_controller`, `task_latent`, `goal_latent`.
- Checkpoints: `--checkpoint provided | retrained:<seed> | b:<variant>:<seed>`; every CSV row
  records `checkpoint_kind`, `checkpoint_variant`, `checkpoint_seed`; one CSV holds one
  checkpoint (coordinator, 2026-10-29). `tables.py --cluster checkpoint_seed` pools seeds.
- Held-out goals (accepted ids 101..110): `results/heldout_goals/<env>.csv` for medium, large,
  giant. Rule: goal cell ≥ 2 maze moves from every eval goal, ≥ 50 validation states within the
  goal radius, a start cell at eval-task distance (seeded draws). A test checks the files.
- `fbplan/maze.py` (new, X7): `MazeGeometry`, `Cell`, `draw_walls`. `controllers/oracle.py`
  imports them, so `from fbplan.controllers.oracle import MazeGeometry` still works.
- `fbplan/stats.py`: bootstrap CIs, paired (McNemar) and unpaired comparisons, `load_results`,
  `labelled_results` (one load-and-label helper for tables and figures), `row_labels`,
  `pool_label`, `markdown_table`.
- `fbplan/controllers/`: baseline, `baseline_raw`, non-hier, oracle; `PRIVILEGED` (X16).
- `analysis/`: `tables.py` (paired against baseline and baseline_raw, unpaired for upstream,
  runs with other episodes listed apart), `plots.py`, `parity.py` (`--episodes`),
  `failures.py` and `calibration.py` (both also write CSV), `heldout_goals.py`.
- Tests: 354 pass in the merged tree, ruff clean. `tests/test_analysis.py` (new) tests the
  analysis scripts; it loads them by path.
- Refactor check (coordinator rule): eval seed 0 of `results/medium/baseline.csv` and
  `oracle.csv` rerun on the Mac after the fixes: 250 of 250 records identical for both.

## Mac results on medium (`results/macos/medium/`, 5 eval seeds × 5 tasks × 50 episodes)

Planner rows pool 3 landmark sets (`landmarks.seed` 0, 1, 2, each crossed with all eval seeds;
cluster bootstrap); retrained rows pool 5 training seeds. Paired = same 1,250 episodes.

| method | all [95 % CI] | vs baseline (pp) | vs baseline_raw (pp) |
|---|---|---|---|
| baseline (provided π_h, normalized z_r) | 77.8 [75.6, 80.1] | – | −2.6 [−5.8, +0.5], p 0.11 |
| baseline_raw (reference row, raw z_r) | 80.5 [78.3, 82.6] | +2.6 [−0.5, +5.8], p 0.11 | – |
| baseline_retrained (5 training seeds) | 78.3 [76.9, 79.8] | +0.5 (−1.0 to +2.7) | −2.2 (−3.6 to +0.1) |
| non-hier | 64.8 [62.7, 66.8] | −13.0, p 3e-14 | −15.7, p 9e-21 |
| oracle (privileged) | 84.0 [82.0, 86.0] | +6.2 [+3.1, +9.2], p 1e-4 | +3.5 [+0.5, +6.5], p 0.023 |
| planner | 70.1 [67.8, 72.3] | −7.7 (−9.9 to −6.5) | −10.4 (−12.6 to −9.1) |
| planner_high | 73.2 [70.4, 75.6] | −4.6 (−7.5 to −3.0) | −7.3 (−10.2 to −5.7) |
| planner_region | 65.6 [61.0, 71.5] | −12.3 (−16.6 to −6.0) | −14.9 (−19.3 to −8.6) |
| upstream `evaluate` (unpaired) | 80.0 [77.8, 82.2] | +2.2 [−1.0, +5.3] unpaired | −0.5 [−3.5, +2.6] unpaired |

Ranges in brackets with "to" are the spread over landmark sets or training seeds; each
per-run paired result is in `analysis/tables.py` output.

- Retrained seeds 0–4: 77.8, 80.6, 76.9, 77.7, 78.6 %; none differs from the provided π_h
  (paired p ≥ 0.09). The provided π_h is a typical seed.
- With baseline_raw as the reference (design.md §8: a gain counts only if it also holds there),
  the oracle gain shrinks to +3.5 pp and still holds (p 0.023); no planner variant gains.

## Mac failure classes on medium (`results/macos/medium/failures/`, share of failures)

| run | failures | flipped | other | stuck |
|---|---|---|---|---|
| baseline | 277 | 71.5 % | 26.0 % | 2.5 % |
| baseline_raw | 244 | 71.7 % | 26.6 % | 1.6 % |
| oracle | 200 | 71.0 % | 28.0 % | 1.0 % |
| non-hier | 440 | 40.2 % | 48.4 % | 11.4 % |
| planner l0 / l1 | 358 / 401 | 51.1 / 52.4 % | 47.8 / 46.6 % | 1.1 / 1.0 % |
| planner_high l0 / l1 | 318 / 315 | 51.6 / 55.6 % | 46.9 / 43.2 % | 1.6 / 1.3 % |
| planner_region l0 / l1 | 485 / 352 | 34.4 / 45.2 % | 64.3 / 53.4 % | 1.2 / 1.4 % |

Set 2: planner 362 failures (53.0 % flipped), planner_high 371 (48.0 %); planner_region set 2
was stopped by the platform decision. The planner adds "other" failures (still moving at the
time limit: slow or on a wrong route), not flips. The Linux queue recomputes all of them.

## Linux evaluation instance (queue)

Instance 58279543 (vast.ai, label `fbplan-gpu-eval`): AMD EPYC 7662 (Zen 2), 128 effective
cores, 1 TB RAM, Belgium, $0.80/h, created 19:53. Code: commit 84a8127 (pushed with
`scripts/gpu/vast.sh push`; it includes c-planner's tie-break fix 06ea0a8 and `switch_steps`).
JAX on the CPU. A queue runner (`runs/queue/runner.sh` on the instance) takes job files in
name order; an eval job runs eval seeds 0–4 as 5 processes of 25 workers each, pinned to a
fifth of the cores, and joins them into one CSV (records equal to one `--seeds 0 1 2 3 4` run).

Queue, in the coordinator's priority:
- (a) approach B on medium: `--checkpoint b:<variant>:<k>`, k 0–4, queued as soon as e-learner's
  export of a variant is complete (clean, then wide, then region); references baseline,
  baseline_raw and the 5 retrained baselines.
- (b) large: baseline, baseline_raw, oracle, planner / planner_high / planner_region × landmark
  sets 0–2; failure classes of baseline and oracle.
- (c) medium main results again: non-hier, oracle, the 9 planner runs, parity against upstream,
  failure classes of all of them, the oracle sweep (eval seed 0).
- (d) c-planner's ablations and b-theory's giant and teleport runs, from their exact commands
  (send them to me; I add them to the queue).

vast.ai log (team account, label `fbplan-gpu-eval`):

| instance | host | $/h | from | to | cost | note |
|---|---|---|---|---|---|---|
| 58258733 | RTX A2000, Denmark | 0.154 | 16:57 | 17:07 | ~0 | host GPU error at start |
| 58259579 | Xeon Phi 7210, Texas | 0.333 | 17:08 | 19:07 | ~0.66 | 56 steps/s per worker (Mac 476) |
| 58272696 | EPYC 7V13, Texas | 0.802 | 19:08 | 19:36 | ~0.37 | image never loaded, no SSH |
| 58279543 | EPYC 7662, Belgium | 0.80 | 19:53 | running | | evaluation queue |

Selection rule now (coordinator): CPU model AMD EPYC Zen 2/3/4 or Xeon Gold/Platinum, ≥ 64
effective cores, ≥ 128 GB RAM, reliability ≥ 0.99, ≤ $1/h; destroy after 15 min without SSH.

## Mac results on large (`results/macos/large/`, reproducibility check only)

baseline 38.0 %, oracle 41.4 %, baseline_raw 38.1 %, planner set 0 32.4 %, set 1 24.1 % (5 eval
seeds each; planner code of commit 0de6e0a, before 06ea0a8 and `switch_steps`).

## Planner code behind each planner CSV

- `results/macos/medium/planner*/l<s>.csv`, `results/macos/large/planner/l<0,1>.csv`: planner
  code of eb5952b (= 0de6e0a): no tie-break fix, no `switch_steps`.
- All Linux planner CSVs (`results/<env>/planner*/`): commit 84a8127, with the tie-break fix
  06ea0a8 and `switch_steps` (default off).

## FB distance calibration (`results/calibration/`, 400 val states, F heads min)

Wormholes by walking speed (`walk_speed_wormholes`, X13): pairs the planner trusts
(`graph.trusted_edges`, X10) whose FB distance is shorter than the geodesic walk at top speed.
The share now divides by the trusted pairs (raw ρ in [ρ_min, 1)); before, the divisor also
counted the pairs with raw ρ ≥ 1. f-report: the medium shares are now 5.5 / 12.0 / 27.2 %
(were 5.4 / 11.6 / 24.7 %); the counts did not change.

| env | ρ_min | trusted pairs | raw ρ ≥ 1 | wormholes | share | within xy 6 |
|---|---|---|---|---|---|---|
| medium | 0.3 | 61,082 | 1,118 | 3,385 | 5.5 % | 1,965 |
| medium | 0.7 | 11,028 | 1,118 | 2,997 | 27.2 % | 1,965 |
| large | 0.3 | 37,189 | 1,170 | 3,992 | 10.7 % | 1,420 |
| large | 0.7 | 7,173 | 1,170 | 2,707 | 37.7 % | 1,420 |

The wormholes left by the xy cap cross no wall: short pairs where FB is overconfident.

## Review (g-release, first pass)

- D3 fixed: extra methods cycle through the palette. D4 fixed: `stats.labelled_results` for
  tables and figures. D5 fixed: parity docstring (reruns do not repeat), `make_tasks`,
  `--episodes`. D6 fixed: oracle docstring numbers from the committed sweep, "reference, not an
  upper bound". D7: `failures.py` uses `build_controller` and `RecordingController`; trust rule
  from `graph.trusted_edges`; one `draw_walls`; `region_latents` waits for X4.
- D8 fixed: tests build envs with `frozen.make_env`; the medium map comes from the env.
- D9 fixed: no failure gives a table of zeros; several envs, configs or checkpoints give a
  clear error; `--workers` defaults to 10 in both scripts. D10 fixed: help lists all aliases,
  `--env` choices, default 5 eval seeds, reset comment corrected. D11 fixed: calibration `main`
  split; table built as a DataFrame. D12: `RawBaselineController` takes `FrozenFBModel`.
  Rejected: removing `MazeGeometry.path`, because b-theory's `tests/test_oracle_teleport.py`
  and my oracle tests use it as the reference shortest path.
- D13 fixed: hand value 93/128. D14 fixed: a missing reference prints a message; runs with
  other episodes are listed, not paired. D15 fixed: `tests/test_analysis.py` (9 tests; 6 fail
  on the old scripts). D16 done: `chash` parameter, typed `make_oracle`, one base class for the
  3 baseline controllers, the raw-baseline test checks 4 observations.
- X7, X13, X16, X10 done. X5: my side done (public helpers above). X4: waits for c-planner's
  shared `region_latents` and batched-B helper.
- Also fixed: `failure_table` dropped tasks without failures.

## Commands for the CSVs in results/

Run from the repository root; `--workers` does not change results. The Mac CSVs in
`results/macos/` came from the same commands with `results/medium` in their paths; the Linux
CSVs come from the same commands on the instance (one process per eval seed, then joined).

- `results/medium/{baseline,baseline_raw,nonhier,oracle}.csv`:
  `uv run python -m fbplan.evaluate --method <name> --env antmaze-medium-navigate-v0
  --seeds 0 1 2 3 4 --episodes 50 --workers 10 --out results/medium/<name>.csv`
- `results/medium/upstream.csv`: `uv run python analysis/parity.py --ours
  results/medium/baseline.csv --reps 5 --episodes 50 --out results/medium/upstream.csv`
  (upstream resets are unseeded: a rerun gives other episodes).
- `results/medium/oracle_sweep/state_l<k>.csv`: `... --method oracle --seeds 0
  --opt waypoint=state lookahead=<k> --out ...`; `region_r<r>_l<k>.csv`: `--opt
  waypoint=region radius=<r> lookahead=<k>` (k 1–3, r 0.5, 1.0, 1.5).
- `results/medium/planner/l<s>.csv`: `... --method planner --opt landmarks.seed=<s>`;
  `planner_high/l<s>.csv`: `--method planner_high --opt landmarks.seed=<s>`;
  `planner_region/l<s>.csv`: `--method planner --opt landmarks.latent=region
  landmarks.seed=<s>` (s 0–2; all with `--seeds 0 1 2 3 4 --episodes 50`). These ran before
  c-planner added `switch_steps`; the default keeps the behavior but changes the config hash.
- `results/medium/baseline_retrained/seed<k>.csv`: `... --method baseline --checkpoint
  retrained:<k> --seeds 0 1 2 3 4 --episodes 50` (k 0–4).
- `results/medium/failures/<run>.csv`: `uv run python analysis/failures.py
  results/medium/<run>.csv --method <name> [--opt <the run's options>] --out
  results/medium/failures/<run>.csv`; the planner ones ran with the code of commit 0de6e0a
  (before `switch_steps`), because the replay checks the config hash.
- `results/calibration/wormholes_<env>_<agg>.csv` and the figures:
  `uv run python analysis/calibration.py --env <env> --head-agg <min|mean>`.
- `results/heldout_goals/<env>.csv`: `uv run python analysis/heldout_goals.py`.
- `results/figures/success_antmaze-medium-navigate-v0.png`: `uv run python analysis/plots.py
  results/medium/{baseline,baseline_raw,nonhier,oracle}.csv results/medium/planner*/*.csv
  --label-root results/medium --cluster config_hash`.
- Large (follows): the same evaluate commands with `--env antmaze-large-navigate-v0`, run on
  the vast.ai host as 5 processes per config (one per eval seed), joined into one CSV.

## Next

1. Run the queue; pull the Linux CSVs into `results/<env>/`; tables against baseline,
   baseline_raw and the pooled retrained baselines; report approach B clean as soon as done.
2. Mac-vs-Linux agreement (unpaired) for every config run on both.
3. Destroy the instance when the queue has been empty for 15 minutes.
4. X4 after c-planner exposes the shared helpers.

## Blockers

- None.

## Requests

- f-report, g-release: new paths. The Mac results are in `results/macos/{medium,large,figures}/`.
  `results/medium/`, `results/large/` and the figures will hold the Linux results (same file
  names). REPORT numbers and `scripts/reproduce.sh` paths change with them; the medium oracle
  sweep, parity and failure classes are recomputed on Linux too. The calibration and held-out
  goal files stay where they are (Mac; they do not depend on MuJoCo).
- c-planner, b-theory: send me the exact commands of your ablations and giant/teleport runs;
  they run on the instance after (a)–(c).
- g-release: `pythonpath = ["."]` in the pytest config would let `tests/test_analysis.py`
  import the scripts normally instead of loading them by path.
- e-learner, b-theory: `MazeGeometry` now lives in `fbplan/maze.py`; your imports from
  `fbplan.controllers.oracle` still work.
- f-report: the wormhole shares changed definition (see Calibration).
