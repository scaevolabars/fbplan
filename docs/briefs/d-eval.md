# Brief: d-eval (evaluation, statistics, diagnostics)

Read `docs/design.md` first, especially §5.

## Goal

One trustworthy evaluation path for every method, plus the statistics and diagnostics that the
report needs.

## Owns

`fbplan/evaluate.py`, `fbplan/stats.py`, `fbplan/controllers/__init__.py`,
`fbplan/controllers/baseline.py`, `fbplan/controllers/oracle.py`, `analysis/`,
`tests/test_evaluate.py`, `tests/test_stats.py`, `docs/status/d-eval.md`.

## Modules

- `evaluate.py`: `run_episodes(controller, env_name, task_ids, eval_seeds, n_episodes) -> list[EpisodeRecord]`
  and a CLI that writes CSV to `runs/`. Derive each reset seed from `(eval_seed, task_id, episode)`
  so all methods see the same start states. Allow parallel workers over (task, seed).
- `controllers/baseline.py`: `BaselineController` (upstream behaviour: `π_h` every step, then `π_l`),
  `NonHierController` (`π_l` with `z_r` directly). These are also the parity reference.
- `controllers/oracle.py`: `OracleWaypointController`: shortest path on the true maze map
  (`env.unwrapped` maze map and cell size), waypoints = nearest dataset observation in xy to each
  cell centre, `z = B(waypoint)`, act with `π_l`. It may read `TaskSpec.goal_xy`. Label it privileged.
- `stats.py`: mean success with 95 % stratified bootstrap CI (strata = tasks), paired comparison
  of two methods on the same episodes, tables per task and overall.
- `analysis/`: scripts (not notebooks) for result tables and plots from `results/*.csv`, and the
  FB distance calibration plot (`−log ρ / −log γ` vs. maze geodesic distance, wormhole count).

## Parity check (first)

`BaselineController` in our loop must reproduce upstream `utils/evaluation.evaluate` on medium.
Upstream does not seed resets, so compare statistically (same N, CI overlap) and document it.

## Before a-frozen merges

Build and test the loop, stats and oracle path with a stub controller and the real env
(`ogbench` env creation does not need checkpoints).

## Done when

Parity documented, stats tested, baseline + non-hier + oracle numbers on medium (1 seed smoke)
in the status file, throughput (steps/s) measured.

## Stop and ask the user

Before full runs that take more than about an hour.
