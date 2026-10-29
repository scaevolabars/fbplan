# Status: d-eval

## Done

- `fbplan/evaluate.py`: episode loop, `make_tasks`, `run_episodes` (one process),
  `run_parallel` (spawn workers over (task, eval_seed) jobs), `write_csv`.
  Reset seed = `SeedSequence([eval_seed, task_id, episode])`. The loop seeds the env, the
  action-space RNG and the global NumPy RNG: OGBench draws the start noise from the global RNG,
  so an env seed alone does not fix the start state (checked on medium).
- `fbplan/stats.py`: stratified bootstrap CI (strata = env × task, optional cluster level for
  retrained checkpoints), paired comparison (stratified bootstrap CI + exact McNemar), per-task tables.
- Tests: fake point env for the loop logic, plus the real `antmaze-medium` env (no checkpoint):
  identical start states across runs, full-length failures.

## Next

- `controllers/baseline.py`, `controllers/oracle.py`, CLI, `analysis/`.

## Blockers

- None. Real-model work waits for a-frozen (`load_model`, `make_env`, `load_dataset`).

## Requests to the coordinator

- API note: `run_episodes(controller, tasks, eval_seeds, n_episodes)` takes `TaskSpec`s instead of
  `(env_name, task_ids)`, because `z_r` comes from the model. `make_tasks(env_name, task_ids,
  model.infer_task_latent)` builds them.
