# Status: d-eval

## Done

- `fbplan/evaluate.py`: episode loop, `make_tasks`, `run_episodes` (one process),
  `run_parallel` (spawn workers; a job = 10 episodes of one (task, eval_seed)), `write_csv`, CLI
  `python -m fbplan.evaluate --method <alias|module:attr> [--opt key=value ...]`.
  Reset seed = `SeedSequence([eval_seed, task_id, episode])`. The loop seeds the env, the
  action-space RNG and the global NumPy RNG: OGBench draws the start noise from the global RNG,
  so an env seed alone does not fix the start state (checked on medium).
- `fbplan/stats.py`: stratified bootstrap CI (strata = env × task, optional cluster level for
  retrained checkpoints), paired comparison (stratified bootstrap + exact McNemar), unpaired
  comparison (for the parity check), per-task tables.
- `fbplan/controllers/`: `BaselineController`, `NonHierController`, `OracleWaypointController`
  (privileged; waypoints = val-split states nearest to cell centres), `MazeGeometry` with BFS and
  exact geodesic distances (for the calibration plot). Factory lookup by alias or module path.
- `analysis/tables.py` (Markdown tables), `analysis/plots.py` (success figure per env).

## Preliminary numbers (medium, eval seed 0, 5 tasks × 50 episodes)

Run with a-frozen's uncommitted `frozen.py` in a scratch copy. To repeat after the merge.

| method | success % [95 % CI] |
|---|---|
| baseline (provided π_h) | 82.4 [77.6, 86.8] |
| upstream `evaluate` (parity) | 78.8 [74.0, 83.6] |
| non-hier (π_l with z_r) | 64.0 [59.2, 68.4] |
| oracle, single-state waypoints, lookahead 1/2/3 | 48.4 / 58.4 / 65.6 |
| oracle, region waypoints r 0.5, lookahead 1/2/3 | 79.2 / **86.8** / 82.0 |
| oracle, region r 1.0 / 1.5 (lookahead 1–3) | 81.2–82.8 / 80.4–86.4 |

- Parity: ours − upstream = +3.6 pp, 95 % CI [−3.2, +10.4] (unpaired, 500 episodes). Holds.
- Deterministic: 5 and 10 workers give identical records; replays reproduce every failure.
- Throughput: 700–900 steps/s per worker, 2,300–5,200 steps/s total (10 workers, 12-core Mac,
  ~20 s start-up). One full medium seed of one method takes 30–60 s.

## Findings (important for c-planner and the report)

1. `z = B(w)` of one dataset state is a poor intention for π_l: `B(w)` carries the pose and
   velocity of `w`, and the ant overshoots or turns back. The mean of `B(s)` over states within
   0.5 of a point (as upstream builds `z_r`) works much better (oracle 58 % → 87 %).
   Approach A uses single-state `z_w = B(w)` for landmarks: this may explain its replans.
2. The privileged oracle does not clearly beat the baseline on medium. 70–80 % of failures
   (baseline 31/44, oracle 27/33) are the ant flipping over: a π_l failure. The headroom for any
   high-level method on medium is small unless its intentions reduce flips.

## Next

- After the a-frozen merge: commit `analysis/parity.py`, add a smoke test with the real model,
  redo the numbers with ≥ 5 eval seeds.
- After the c-planner merge: calibration script (`−log ρ / −log γ` vs. geodesic, wormhole count).

## Blockers

- None. Real-model work waits for the a-frozen merge.

## Requests to the coordinator

- API note: `run_episodes(controller, tasks, eval_seeds, n_episodes)` takes `TaskSpec`s instead of
  `(env_name, task_ids)`, because `z_r` comes from the model. `make_tasks(env_name, task_ids,
  model.infer_task_latent)` builds them.
- New file outside the brief list: `tests/test_eval_controllers.py` (tests of my controllers).
- Agreed: c-planner owns `analysis/planner_routes.py`. Aliases `planner`, `planner_high` added.
- To a-frozen: please make `_upstream` public (for example `upstream_module`). The parity script
  needs upstream `utils.evaluation`.
