# Status: e-learner

## Done

- `fbplan/train_high.py`: retrains `π_h` on the frozen `F`, `B`, `π_l`. It calls the upstream
  `FBpiSwitchAgent.create` and `update`, so the network, loss and optimizer are upstream code.
  The agent config comes from the checkpoint's `flags.json`. A new `π_h` init per seed, as upstream
  `load_agent_from_frozen`. Logs train and validation metrics every 5k steps to
  `runs/train_high/<env>/seed<k>/train.csv`. Saves `params.pkl` + `flags.json` in the layout of the
  provided checkpoints to `~/.cache/fbplan/baselines/<env>/seed<k>/` (override `FBPLAN_BASELINE_DIR`).
- `fbplan/learned/sampling.py`: batches sampled on the device inside `lax.scan`. Same distribution
  as upstream `HGCDataset.sample` for the keys the loss reads (KS test against upstream in
  `tests/test_learned_sampling.py`, all three goal branches).
- Tests: `tests/test_learned_sampling.py`, `tests/test_learned_train_high.py` (frozen modules stay
  bit-identical, `π_h` init depends only on the seed, checkpoint round trip, CSV log).
- Speed: full-size step on Apple M2 Pro CPU = 34 ms, so 500k steps = 4.8 h per seed. Needs a GPU.
- GPU: vast.ai, a shared team account (shared credit, other members' instances on it).
  `scripts/gpu/vast.sh` (run locally: offers, create, list, push, train, status, pull, ssh,
  destroy), `remote_setup.sh` (uv, CUDA JAX, OGBench data, tests), `train_baselines.sh`
  (5 seeds in parallel on one GPU, in tmux). Guards for the shared account:
  - our instances carry the label `fbplan-gpu` (`VAST_LABEL`); push, train, status, pull,
    ssh and destroy refuse any instance with another label (tested with a fake `vastai`);
  - `create` attaches `~/.ssh/id_rsa.pub` to the new instance only (`vastai attach ssh`), never
    to the account; `list` shows only our label; `destroy` keeps vastai's confirmation prompt.
  The instance gets the committed tree at HEAD plus a `REVISION` file; each checkpoint records it.
- Merged main (a-frozen, b-theory). With the real medium checkpoint: the checkpoint config is the
  upstream `π_h` recipe (batch 1024, `γ` 0.99, `high_alpha` 0.1, lr 3e-4, uniform in-trajectory
  goals), a retrained `π_h` reloads through `load_baseline_model(env, seed)` (for d-eval), and
  a file with changed `F`/`B`/`π_l` is refused. 117 tests pass. CLI smoke run on CPU (10 steps,
  medium train split): loads, trains, logs, saves.
- `docs/plans/approach-b.md`: draft plan for approach B, waiting for the user's approval
  (7 decisions listed in its §10).

## Next

- After approval: implement B1 (option targets in JAX, `V_H`, AWR `π_h`, CLI, tests).
- First GPU job: medium, seeds 0-4. Ready; waits for the user's go (cost question in the brief).
- Revise `docs/plans/approach-b.md` with the facts from the merge (A's approved `R̂ = 0`,
  `ρ ≤ γ`, region-mean latents, small headroom on medium).

## Blockers

- None for Phase 1. Approach B waits for the user's approval of the plan.

## Requests to the coordinator

For a-frozen (`fbplan/frozen.py`). Updated after reading a-frozen's work in progress:

1. Done without a change in a-frozen: `train_high` uses `load_model(env).agent`, and
   `fbplan.train_high.load_baseline_model(env, seed)` gives d-eval an `FBModel` with a retrained
   `π_h`. This needs `FrozenFBModel` and `load_numpy_pickle` to stay public.
2. Make `_upstream(module)` public (for example `upstream`), so my tests do not add
   `third_party/ssm` to `sys.path` themselves (marked TODO in two test files).
3. `load_dataset(...)["terminals"]` = 1 at the last state of each trajectory (non-compact OGBench
   layout), as upstream `GCDataset` expects. (Checked: the real medium split meets it.)
