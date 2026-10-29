# Status: d-eval

Last update: 2026-10-29. Branch merged with `main` (a-frozen, b-theory; MuJoCo 3.1.6).

## Done

- `fbplan/evaluate.py`: episode loop, `make_tasks`, `run_episodes` (one process),
  `run_parallel` (spawn workers; a job = 10 episodes of one (task, eval_seed)), `write_csv`, CLI
  `python -m fbplan.evaluate --method <alias|module:attr> [--opt key=value ...]`.
  Reset seed = `SeedSequence([eval_seed, task_id, episode])`, then the reset recipe of design.md §5.
  The loop checks that the env goal equals `TaskSpec.goal_xy` (no goal noise).
- `fbplan/stats.py`: stratified bootstrap CI (strata = env × task, optional cluster level for
  retrained checkpoints), paired comparison (stratified bootstrap + exact McNemar), unpaired
  comparison (parity), per-task tables.
- `fbplan/controllers/`: `BaselineController`, `NonHierController`, `OracleWaypointController`
  (privileged; region-mean waypoint latents by default), `MazeGeometry` (BFS, exact geodesics).
  Aliases: baseline, nonhier, oracle, planner, planner_high (c-planner).
- `analysis/`: `tables.py`, `plots.py`, `parity.py`, `failures.py`, `calibration.py`
  (`calibration.py` runs after c-planner's `option_model` merges).
- Tests: 155 pass (all components), ruff clean. Real-model smoke tests for the 3 controllers and
  the CLI skip when the checkpoint or the val split is absent.
- Results: `results/medium/{baseline,nonhier,oracle,upstream}.csv` (5 eval seeds) and
  `results/medium/oracle_sweep/` (eval seed 0). All on MuJoCo 3.1.6.

## Results on medium (5 eval seeds × 5 tasks × 50 episodes = 1,250 episodes per method)

| method | task 1 | task 2 | task 3 | task 4 | task 5 | all [95 % CI] |
|---|---|---|---|---|---|---|
| baseline (provided π_h, normalized z_r) | 81.6 | 84.4 | 76.0 | 62.8 | 84.4 | 77.8 [75.6, 80.1] |
| upstream `evaluate` (raw z_r, unpaired) | | | | | | 80.0 [77.8, 82.2] |
| non-hier (π_l with z_r) | 76.4 | 85.6 | 74.8 | 6.4 | 80.8 | 64.8 [62.7, 66.8] |
| oracle (privileged; region r 0.5, lookahead 2) | 81.6 | 88.4 | 88.8 | 74.8 | 86.4 | 84.0 [82.0, 86.0] |

- Paired against the baseline: non-hier −13.0 pp [−16.2, −10.0], p = 3e-14;
  oracle +6.2 pp [+3.1, +9.2], p = 1e-4 (McNemar).
- Parity: ours − upstream = −2.2 pp, 95 % CI [−5.3, +1.0] (unpaired, 2,500 episodes). Holds.
  The upstream row is also the raw-`z_r` path of design.md §8.
- Deterministic: 5 and 10 workers give identical records; `failures.py` replays every failed
  episode and gets the same outcome and step count.
- Throughput: 800–1,250 steps/s per worker; 7,000–9,900 steps/s total with 10 workers on the
  12-core Mac. One method × 5 eval seeds on medium takes 65–95 s.

## Failure classes (`analysis/failures.py`, same 1,250 episodes)

| method | failures | flipped | other (moving at the time limit) | stuck |
|---|---|---|---|---|
| baseline | 277 | 198 (71 %) | 72 | 7 |
| oracle | 200 | 142 (71 %) | 56 | 2 |
| non-hier | 440 | 177 (40 %) | 213 | 50 |

Flips end 11–16 % of all episodes. Non-hier fails differently: on task 4 (a long detour) it gets
lost or stuck (187 of 250 episodes).

## Oracle ablation (eval seed 0, 250 episodes per config)

| waypoint latent | lookahead 1 | 2 | 3 |
|---|---|---|---|
| single state (B of the nearest val state) | 44.8 | 56.4 | 65.2 |
| region, radius 0.5 | 80.8 | **86.4** | 78.0 |
| region, radius 1.0 | 87.2 | 76.0 | 74.8 |
| region, radius 1.5 | 80.8 | 80.0 | 80.0 |

CIs are about ±5 pp; no region config is clearly best. The default (r 0.5 = goal radius of the
task reward, lookahead 2) stays. Single-state latents are clearly worse: `B(w)` carries the pose
and the velocity of `w`.

## FB distance calibration (medium, 400 val states, 159,600 ordered pairs)

Measured with c-planner's `OptionModel` in a scratch copy (independent of the MuJoCo version).
Wormhole = trusted pair whose FB distance `−log ρ / −log γ` is shorter than the geodesic walk at
the ant's top speed (p99 step length 0.268; median 0.138).

| F heads | ρ_min | trusted pairs | of which ρ = 1 (clipped) | wormholes with ρ < 1 | share of trusted |
|---|---|---|---|---|---|
| min | 0.3 | 62,200 | 1,118 | 3,391 | 5.5 % |
| min | 0.5 | 28,962 | 1,118 | 3,359 | 11.6 % |
| min | 0.7 | 12,146 | 1,118 | 3,003 | 24.7 % |
| mean | 0.3 | 67,397 | 1,569 | 4,296 | 6.4 % |
| mean | 0.7 | 13,799 | 1,569 | 3,702 | 26.8 % |

## Next

- After c-planner merges: run `analysis/calibration.py` from the branch, commit its figures;
  evaluate `planner` and `planner_high` with 5 eval seeds, paired against the baseline.
- Large: the same runs when needed (about 2–3 min per method).
- Retrained baselines (e-learner): `summary_table(..., cluster="config_hash")` pools the seeds.

## Blockers

- None.

## Requests to the coordinator

- design.md §5: the oracle is now +6.2 pp over the baseline (5 eval seeds, paired, p = 1e-4),
  not "about 4 points (one eval seed)". Flips are 71 % of baseline failures on MuJoCo 3.1.6.
- design.md §8 (raw vs normalized `z_r`): the upstream row above is the raw path, unpaired. If you
  want a paired raw-`z_r` reference row, I can add a controller that calls upstream
  `sample_actions` through `FrozenFBModel.agent`.
- API note: `run_episodes(controller, tasks, eval_seeds, n_episodes)` takes `TaskSpec`s instead of
  `(env_name, task_ids)`, because `z_r` comes from the model. `make_tasks(env_name, task_ids,
  model.infer_task_latent)` builds them.
- New file outside the brief list: `tests/test_eval_controllers.py` (tests of my controllers).
- `analysis/parity.py` uses a-frozen's private `_upstream`; I switch when a public name exists.
