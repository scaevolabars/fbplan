# Status: d-eval

Last update: 2026-10-30, 00:10. Branch merged with `main` at 3e36461.

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
- (d) from the command lists in docs/status/c-planner.md and b-theory.md (main 3e36461), in
  this order: c-planner's 12 ablations (9 medium, 3 large, landmark set 0), b-theory's 10
  teleport runs, then the 9 giant runs and the giant oracle failure classes. Same code (84a8127)
  as (a)–(c), so the ablations pair with the main planner runs.
- 85 jobs queued at 20:12. Speed: a medium baseline-type config takes about 55 s.
- Added (coordinator, after B and before giant): c-planner's act-ahead run
  (`planner_high --opt landmarks.seed=0 act_ahead_xy=8.0` →
  `results/medium/planner_ablations/high_act_ahead8/l0.csv`). `act_ahead_xy` exists only on main,
  so this job runs from a second code tree on the instance (`~/fbplan_main`, main 8a202e6), with
  a check job before it: planner_high l0 with default options from that tree. If its records
  equal `results/medium/planner_high/l0.csv` (84a8127), the act-ahead run pairs with the main
  run; otherwise it pairs with the check run, which then goes to results/ as its default.

First Linux numbers (medium, 5 eval seeds): baseline 77.0 %, baseline_raw 80.8 %, retrained
seeds 0–4: 79.1, 79.7, 77.5, 76.6, 78.4 %. Mac: 77.8, 80.5 and 77.8, 80.6, 76.9, 77.7, 78.6 %.

vast.ai log (team account, label `fbplan-gpu-eval`):

| instance | host | $/h | from | to | cost | note |
|---|---|---|---|---|---|---|
| 58258733 | RTX A2000, Denmark | 0.154 | 16:57 | 17:07 | ~0 | host GPU error at start |
| 58259579 | Xeon Phi 7210, Texas | 0.333 | 17:08 | 19:07 | ~0.66 | 56 steps/s per worker (Mac 476) |
| 58272696 | EPYC 7V13, Texas | 0.802 | 19:08 | 19:36 | ~0.37 | image never loaded, no SSH |
| 58279543 | EPYC 7662, Belgium | 0.80 | 19:53 | running | ~3.3 at 00:00 | evaluation queue |

Mistake: the queue was empty from 22:56 to 00:00 (waiting for B region), and my destroy watcher
did not fire, because its emptiness test used `ls` on two folders, which prints their names
even when they are empty. About 50 idle minutes were billed (~$0.70). The watcher now calls
`runs/queue/idle_check.sh` (tested on the instance: busy → 1, empty → 0). Total vast.ai so
far: about $4.3 (failed hosts ~$1.0 + this instance ~$3.3).

Selection rule now (coordinator): CPU model AMD EPYC Zen 2/3/4 or Xeon Gold/Platinum, ≥ 64
effective cores, ≥ 128 GB RAM, reliability ≥ 0.99, ≤ $1/h; destroy after 15 min without SSH.

## Approach B clean (Linux, `results/medium/b_clean/seed<k>.csv`, B decision gate)

`--method baseline --checkpoint b:clean:<k>` (k 0–4; e-learner's export, commit e6b99d1), 5 eval
seeds × 5 tasks × 50 episodes each, same 1,250 episodes as every medium run. Pooled rows: cluster
bootstrap over training seeds (both sides) and episodes.

| method | success [95 % CI] | vs baseline_retrained (pp) | vs baseline (pp) | vs baseline_raw (pp) |
|---|---|---|---|---|
| B clean (5 seeds) | 79.3 [76.4, 82.2] | +1.1 [−2.4, +4.5] | +2.3 [−1.5, +6.1] | −1.5 [−5.1, +2.1] |
| baseline_retrained (5 seeds) | 78.3 [76.9, 79.7] | – | +1.2 [−1.5, +4.1] | −2.5 [−5.2, +0.2] |
| baseline | 77.0 [74.7, 79.3] | −1.2 [−4.1, +1.5] | – | −3.8 [−6.8, −0.7] |
| baseline_raw | 80.8 [78.6, 82.9] | +2.5 [−0.2, +5.2] | +3.8 [+0.7, +6.8] | – |

- B clean seeds 0–4: 81.9, 75.5, 83.0, 75.8, 80.5 % (paired against the baseline: +4.9, −1.5,
  +5.9, −1.2, +3.4 pp; seeds 0, 2, 4 significant). The retrained baselines spread less (76.6–79.7 %).
- No significant gain over the retrained baselines (+1.1 pp, CI contains 0), and none against
  baseline_raw (−1.5 pp). The spread between B training seeds is the largest effect.
- B wide (`results/medium/b_wide/`, `b:wide:<k>`): seeds 0–4 81.0, 77.0, 82.2, 72.3, 79.6 %,
  pooled 78.4 %; vs baseline_retrained +0.2 [−3.8, +3.7], vs baseline +1.4 [−2.7, +5.3], vs
  baseline_raw −2.4 [−6.4, +1.4]. Seeds 1 and 3 are low again, as in B clean (e-learner's
  temperature explanation predicts this pattern).
- B region (`results/medium/b_region/`, `b:region:<k>`): seeds 0–4 80.6, 79.8, 80.8, 77.1,
  81.1 %, pooled 79.9 [78.1, 81.3]; vs baseline_retrained +1.6 [−0.9, +4.1], vs baseline +2.8
  [−0.1, +5.7], vs baseline_raw −0.9 [−3.8, +1.9]. Its seeds spread less than clean and wide.
- Summary: no B variant gains over the retrained baselines or baseline_raw (pooled).

## Held-out seed selection (design.md §8; `results/medium/heldout/`, `heldout_selection.csv`)

Every seed on the 10 medium held-out goals (`--heldout --seeds 0 1 2 --episodes 20`, 600
episodes per seed), selection by `analysis/heldout_selection.py` (highest held-out success, ties
to the lower id), then the selected seeds on the 5 eval tasks:

| family | held-out success, seeds 0–4 (%) | selected | eval success | vs selected retrained (pp) | vs baseline_raw (pp) |
|---|---|---|---|---|---|
| retrained | 87.8, 86.3, 83.7, 87.7, 86.0 | seed 0 | 79.1 | – | −1.7 [−4.7, +1.4], p 0.31 |
| B clean | 87.0, 86.0, 87.7, 85.3, 82.2 | seed 2 | 83.0 | +3.8 [+0.9, +6.9], p 0.014 | +2.2 [−1.0, +5.1], p 0.18 |
| B wide | 86.5, 84.0, 84.8, 82.0, 83.3 | seed 0 | 81.0 | +1.8 [−1.2, +4.9], p 0.25 | +0.2 [−2.9, +3.1], p 0.96 |
| B region | 86.0, 87.8, 86.3, 86.5, 85.7 | seed 1 | 79.8 | +0.6 [−2.4, +3.7], p 0.72 | −1.0 [−4.1, +2.0], p 0.53 |

- The selected B clean seed beats the selected retrained baseline, but not baseline_raw, so by
  the rule of §8 it is no gain. Held-out differences between seeds are small (one seed: 600
  episodes, standard error about 1.4 points), so the selection itself is noisy: seeds 0 and 3
  of the retrained baseline differ by 0.1 point.

## Giant (Linux, `results/giant/`, 5 eval seeds × 5 tasks × 50 episodes)

baseline 0.5 %, non-hier 0.5 %, oracle (privileged) 2.5 % (+2.0 [+1.1, +3.0] pp), planner
(sets 0–2 pooled) 0.4 %, planner_high 0.8 %. No headroom, as b-theory's preview showed. Oracle
failures (`results/giant/failures/oracle.csv`): 77.6 % other, 21.7 % flipped.

## Medium (Linux, `results/medium/`, 5 eval seeds × 5 tasks × 50 episodes)

Planner rows pool landmark sets 0–2 (cluster config_hash); retrained rows pool 5 training seeds
(cluster checkpoint_seed). Paired on the same 1,250 episodes.

| method | success [95 % CI] | vs baseline (pp) | vs baseline_raw (pp) |
|---|---|---|---|
| baseline | 77.0 [74.7, 79.3] | – | −3.8 [−6.8, −0.7] |
| baseline_raw | 80.8 [78.6, 82.9] | +3.8 [+0.7, +6.8] | – |
| baseline_retrained (5 seeds) | 78.3 [76.9, 79.7] | +1.2 [−1.5, +4.1] | −2.5 [−5.2, +0.2] |
| non-hier | 65.4 [63.4, 67.5] | −11.6 [−14.7, −8.5] | −15.4 [−18.3, −12.4] |
| oracle (privileged) | 81.3 [79.1, 83.4] | +4.2 [+1.1, +7.5] | +0.5 [−2.6, +3.5] |
| planner | 68.4 [66.2, 70.5] | −8.6 [−12.0, −5.3] | −12.4 [−15.8, −9.2] |
| planner_high | 72.5 [70.7, 74.3] | −4.5 [−7.7, −1.4] | −8.3 [−11.3, −5.2] |
| planner_region | 64.9 [59.0, 71.9] | −12.1 [−18.8, −5.1] | −15.9 [−22.5, −8.9] |

- On Linux, baseline_raw beats the normalized baseline (+3.8 pp, CI excludes 0; Mac: +2.6, p 0.11).
- The oracle gains against the baseline but not against baseline_raw (+0.5 pp): by the rule of
  design.md §8 even the privileged oracle has no gain on medium.
- Per landmark set: planner 68.8 / 69.9 / 66.5 %, planner_high 73.8 / 71.4 / 72.2 %,
  planner_region 58.7 / 72.2 / 63.8 %. Retrained seeds 0–4: 79.1, 79.7, 77.5, 76.6, 78.4 %.
- Large failure classes (Linux, `runs/linux/large/failures/`): baseline 72.0 % other, 27.7 %
  flipped; oracle 70.4 % other, 29.6 % flipped. Unlike medium, large failures are mostly not flips.

## Medium analyses (Linux)

- Parity: upstream `evaluate` − our baseline = +0.9 pp [−2.3, +4.1] (unpaired, 2,500 episodes,
  `results/medium/upstream.csv`); upstream − baseline_raw = −2.9 [−6.0, +0.2]. Holds.
- Failure classes (`results/medium/failures/`, share of failures):

  | run | failures | flipped | other | stuck |
  |---|---|---|---|---|
  | baseline | 287 | 70.4 % | 27.5 % | 2.1 % |
  | baseline_raw | 240 | 74.6 % | 25.0 % | 0.4 % |
  | oracle | 234 | 75.2 % | 23.5 % | 1.3 % |
  | non-hier | 432 | 35.9 % | 52.1 % | 12.0 % |
  | planner l0 / l1 / l2 | 390 / 376 / 419 | 53.8 / 56.1 / 49.9 % | 44.4 / 42.3 / 49.6 % | ≤ 1.8 % |
  | planner_high l0 / l1 / l2 | 327 / 357 / 347 | 52.9 / 51.8 / 49.3 % | 45.9 / 47.3 / 48.7 % | ≤ 2.0 % |
  | planner_region l0 / l1 / l2 | 516 / 348 / 452 | 34.5 / 45.1 / 40.3 % | 64.1 / 52.9 / 58.8 % | ≤ 2.0 % |

  The planner variants add "other" failures (slow or wrong routes), not flips.
- Oracle ablation, eval seed 0 (`results/medium/oracle_sweep/`; the baseline scores 72.8 % on
  these 250 episodes):

  | waypoint latent | lookahead 1 | 2 | 3 |
  |---|---|---|---|
  | single state | 50.0 | 57.2 | 63.2 |
  | region, radius 0.5 | 78.4 | 80.4 | 82.4 |
  | region, radius 1.0 | 81.6 | 76.8 | 79.6 |
  | region, radius 1.5 | 76.8 | 80.8 | 77.2 |

  CIs are about ±5 pp: region latents beat single-state ones; no region config stands out.
- Mac-vs-Linux check (`analysis/platform_check.py`, `results/platform_check.csv`, 36 runs on
  both): unpaired Linux − Mac differences lie between −6.0 and +5.2 pp; 1 of 36 CIs excludes 0
  (medium planner set 2, −4.6 pp; about 2 are expected at 95 %, and the planner code differs:
  the Mac runs predate the tie-break fix). Only 55–75 % of episodes have the same outcome and
  few the same step count: small float differences grow along a trajectory, so tables never
  mix platforms.

## Large (Linux, `results/large/`, 5 eval seeds × 5 tasks × 50 episodes)

Planner rows pool landmark sets 0–2 (bootstrap over config_hash and episodes).

| method | success [95 % CI] | vs baseline (pp) | vs baseline_raw (pp) |
|---|---|---|---|
| baseline | 37.4 [35.0, 39.9] | – | +2.3 [−1.0, +5.6] |
| baseline_raw | 35.1 [32.8, 37.4] | −2.3 [−5.6, +1.0] | – |
| oracle (privileged) | 39.4 [36.7, 42.0] | +1.9 [−1.8, +5.6] | +4.2 [+0.7, +7.7] |
| planner | 28.1 [24.7, 32.1] | −9.3 [−13.8, −4.5] | −7.0 [−11.4, −2.2] |
| planner_high | 29.5 [27.4, 31.4] | −8.0 [−11.2, −4.6] | −5.7 [−9.0, −2.4] |
| planner_region | 26.4 [24.7, 28.2] | −11.1 [−14.2, −7.8] | −8.7 [−11.8, −5.7] |

- Per landmark set: planner 32.2 / 24.8 / 27.4 %, planner_high 30.9 / 27.8 / 29.7 %,
  planner_region 27.8 / 25.8 / 25.6 %.
- Even the privileged oracle does not beat the baseline on large (task 3: 54.8 against 78.8 %).
- Failure classes of baseline and oracle: requeued (the first try hit the container's thread
  limit: 122 unpinned workers; replay and parity jobs now run pinned to 24 cores).

## Teleport (Linux, `results/teleport/`, 5 eval seeds × 5 tasks × 50 episodes)

Moved ahead of large by the coordinator (best lead for a gain). Planner rows pool landmark sets
0–2 (bootstrap over config_hash and episodes). Trap = failed episode whose final distance to the
goal is within 2 xy units of the trap centre's (cell (1, 7); b-theory's rule, theory.md §11.3).

| method | success [95 % CI] | vs baseline (pp) | vs baseline_raw (pp) | trapped | other failures |
|---|---|---|---|---|---|
| baseline | 27.7 [25.3, 30.2] | – | +3.5 [+1.0, +6.2] | 54.7 % | 17.6 % |
| baseline_raw | 24.2 [21.8, 26.6] | −3.5 [−6.2, −1.0] | – | 57.4 % | 18.4 % |
| planner | 18.6 [16.5, 21.0] | −9.1 [−12.2, −6.0] | −5.6 [−8.6, −2.5] | 64.2 % | 17.3 % |
| planner_high | 24.1 [21.5, 26.4] | −3.6 [−6.7, −0.5] | −0.1 [−3.3, +3.1] | 53.9 % | 22.1 % |

- No gain on teleport: the planner ends in the trap more often than the baseline (64 % against
  55 %), although its xy cap removes the jump edges; planner_high equals baseline_raw.
- Per landmark set: planner 17.3 / 17.4 / 21.0 %, planner_high 24.7 / 21.5 / 26.0 %.
- Teleport nonhier and the teleport-aware oracle (both follow modes) run later in the queue (d).

## Planner ablations (Linux, `results/<env>/planner_ablations/<name>/l0.csv`, landmark set 0)

c-planner's commands; paired on the same 1,250 episodes with the default run of the same code
(84a8127) and the baselines. Medium:

| run | success | vs planner l0 (pp) | vs baseline (pp) | vs baseline_raw (pp) |
|---|---|---|---|---|
| planner l0 (defaults) | 68.8 | – | −8.2 | −12.0 |
| no_xy_cap | 69.5 | +0.7 [−2.7, +4.1] | −7.5 | −11.3 |
| reached_1.5_no_lookahead | 51.3 | −17.5 [−21.2, −13.7] | −25.8 | −29.5 |
| no_lookahead | 65.4 | −3.4 [−7.0, +0.1] | −11.7 | −15.4 |
| edge_reward_exact | 39.5 | −29.3 [−32.6, −25.9] | −37.5 | −41.3 |
| edge_reward_paper | 60.6 | −8.2 [−11.4, −4.9] | −16.4 | −20.2 |
| head_agg_mean | 67.5 | −1.3 [−4.7, +2.2] | −9.5 | −13.3 |
| two_way | 71.8 | +3.0 [−0.4, +6.5] | −5.2 | −9.0 |
| replan_every_step | 69.7 | +0.9 [−2.7, +4.5] | −7.4 | −11.1 |
| route_following | 70.1 | +1.3 [−2.1, +4.6] | −7.0 | −10.7 |

Large (vs planner l0, 32.2 %): no_xy_cap 27.1 % (−5.1 [−8.6, −1.8]), reached_1.5_no_lookahead
22.7 % (−9.5 [−12.8, −6.2]), no_lookahead 21.7 % (−10.6 [−13.8, −7.2]).

Act-ahead (`planner_high --opt landmarks.seed=0 act_ahead_xy=8.0`, code main 8a202e6 from a
second tree on the instance): 77.0 %; vs planner_high l0 +3.2 [−0.3, +6.6] (p 0.071), vs baseline
+0.0 [−3.2, +3.4], vs baseline_raw −3.8 [−6.9, −0.6] (p 0.022). The check run of planner_high l0
with the new code repeats `results/medium/planner_high/l0.csv` exactly (1,250 of 1,250 records),
so the pairing with the committed run is valid.

Teleport, more rows (vs baseline 27.7 %): nonhier 19.0 % (−8.6 [−11.2, −6.1]), teleport-aware
oracle with π_l 5.6 % (−22.1), with π_h 21.4 % (−6.3 [−9.2, −3.5]).

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
- `results/medium/b_clean/seed<k>.csv`: `... --method baseline --checkpoint b:clean:<k>
  --seeds 0 1 2 3 4 --episodes 50` (k 0–4).
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

## AWR sharpness at 2× (design.md §8, coordinator 2026-10-29 23:00)

Waiting for e-learner's export of the baseline at `high_alpha = 0.2`
(`baselines/medium/awr2x/seed<k>`, `--checkpoint retrained:awr2x:<k>`) and B clean at matched 2×
sharpness (`approach_b/medium/clean-awr2x/seed<k>`, `--checkpoint b:clean-awr2x:<k>`), 5 seeds
each; watchers wake me when each family is exported, and the jobs are written. CSVs will go to
`results/medium/baseline_retrained_awr2x/` and `results/medium/b_clean_awr2x/`; the held-out
selection runs with `--reference retrained` and `--reference retrained_awr2x`. Then: all 5 seeds × 5 eval seeds on medium, pooled rows paired against
each other, the original retrained baselines and baseline_raw, and the held-out selection rule
for both families. `--checkpoint retrained:<variant>:<seed>` (new) loads a variant folder of the
retrained baselines (`baselines/medium/<variant>/seed<k>`); B uses `b:<variant>:<seed>`. These
jobs need my branch's code on the instance (a separate tree, checked against a committed run).
The instance stays up; it is destroyed after 30 idle minutes and rented again when the
checkpoints arrive.

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
- f-report: the wormhole shares changed definition (see Calibration), and the pinch fix below
  adds 20 wormholes on medium (ρ_min 0.3: 3,405, share 5.6 %); large did not change.
- b-theory: done. `MazeGeometry.geodesic_distances` no longer passes between walls that touch at
  a corner (tests). `calibration.py` plots only walkable pairs and counts trusted pairs that cannot
  be walked (`unreachable_on_foot`), so it runs on teleport. `failures.py` has a `trapped` class.
