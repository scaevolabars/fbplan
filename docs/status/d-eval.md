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

## Numbers on medium (5 eval seeds × 5 tasks × 50 episodes = 1,250 episodes per method)

Run with a-frozen's committed `frozen.py` (de9ac8f) in a scratch copy, CSVs in `runs/medium5/`.
I move them to `results/` after a-frozen merges and the `z_r` question (a-frozen fact 1) is decided.

| method | task 1 | task 2 | task 3 | task 4 | task 5 | all [95 % CI] |
|---|---|---|---|---|---|---|
| baseline (provided π_h) | 83.2 | 87.6 | 74.4 | 66.8 | 85.2 | 79.4 [77.2, 81.6] |
| upstream `evaluate` | | | | | | 78.6 [76.2, 80.8] |
| non-hier (π_l with z_r) | 78.8 | 86.0 | 70.4 | 8.8 | 82.4 | 65.3 [63.2, 67.4] |
| oracle (privileged; region, r 0.5, lookahead 2) | 79.6 | 88.8 | 86.4 | 76.4 | 84.4 | 83.1 [81.0, 85.1] |

- Paired against the baseline: non-hier −14.2 pp [−17.3, −11.0], p = 1e-16;
  oracle +3.7 pp [+0.7, +6.6], p = 0.018 (McNemar).
- Parity: ours − upstream = +0.9 pp, 95 % CI [−2.2, +4.1] (unpaired, 2,500 episodes). Holds.
- Deterministic: 5 and 10 workers give identical records; replays reproduce every failure.
- Throughput: 800–1,100 steps/s per worker; 7,000–9,800 steps/s total with 10 workers on the
  12-core Mac. One method × 5 eval seeds on medium takes 70–100 s.
- Oracle ablation (eval seed 0, 250 episodes, lookahead 1/2/3): single-state waypoints
  48.4 / 58.4 / 65.6 %; region r 0.5: 79.2 / 86.8 / 82.0 %; r 1.0: 81.2–82.8 %; r 1.5: 80.4–86.4 %.

## Findings (important for c-planner and the report)

1. `z = B(w)` of one dataset state is a poor intention for π_l: `B(w)` carries the pose and
   velocity of `w`, and the ant overshoots or turns back. The mean of `B(s)` over states within
   0.5 of a point (as upstream builds `z_r`) works much better (oracle 58 % → 87 %).
   Approach A uses single-state `z_w = B(w)` for landmarks: this may explain its replans.
2. The privileged oracle beats the baseline by only 3.7 pp on medium. 70–80 % of failures
   (baseline 31/44, oracle 27/33) are the ant flipping over: a π_l failure. The headroom for any
   high-level method on medium is small unless its intentions reduce flips.

## FB distance calibration (medium, 400 val states, 159,600 ordered pairs)

`analysis/calibration.py` (uncommitted until c-planner's `option_model` merges; it uses
`OptionModel.pair_table`). Wormhole = trusted pair whose FB distance `−log ρ / −log γ` is shorter
than the geodesic walk at the ant's top speed (p99 step length 0.268; median 0.138).

| F heads | ρ_min | trusted pairs | of which ρ = 1 (clipped) | wormholes with ρ < 1 | share of trusted |
|---|---|---|---|---|---|
| min | 0.3 | 62,200 | 1,118 | 3,391 | 5.5 % |
| min | 0.5 | 28,962 | 1,118 | 3,359 | 11.6 % |
| min | 0.7 | 12,146 | 1,118 | 3,003 | 24.7 % |
| mean | 0.3 | 67,397 | 1,569 | 4,296 | 6.4 % |
| mean | 0.7 | 13,799 | 1,569 | 3,702 | 26.8 % |

- The median FB distance grows steadily with the geodesic (about 30 steps at 0, 220 at 20 xy
  units), but the spread is wide, and 10 % of pairs have ρ = 0.
- A higher ρ_min does not remove wormholes: they concentrate at high ρ. The min over F heads
  removes about 20 %. Wormholes reach 27–30 xy units (across the whole maze).
  This supports c-planner's xy cap on edge length.

## Next

- After the a-frozen merge: commit `analysis/parity.py` (it imports a-frozen's `_upstream`),
  add a smoke test with the real model, copy the result CSVs to `results/`.
- After the c-planner merge: commit `analysis/calibration.py`; evaluate the planner with 5 seeds.
- Large: same runs when needed (the data are on disk; about 2–3 min per method).

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
