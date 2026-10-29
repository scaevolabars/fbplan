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
- GPU: the user chose vast.ai. `scripts/gpu/vast.sh` (run locally:
  offers, create, push, train, status, pull, ssh, destroy), `remote_setup.sh` (uv, CUDA JAX,
  OGBench data, tests), `train_baselines.sh` (5 seeds in parallel on one GPU, in tmux).
  The instance gets the committed tree at HEAD plus a `REVISION` file; each checkpoint records it.
  OGBench data for medium + large = 534 MB, downloaded on the instance.

- `docs/plans/approach-b.md`: draft plan for approach B, waiting for the user's approval
  (7 decisions listed in its §10).

## Next

- After approval: implement B1 (option targets in JAX, `V_H`, AWR `π_h`, CLI, tests).
- First GPU job (medium, seeds 0-4) when a-frozen is on main. Ask the user before it starts.

## Blockers

- a-frozen is not on main yet. `train_high.main()` needs `load_model(env).agent` and
  `load_dataset` (both exist in a-frozen's uncommitted work).
- No checkpoints and no OGBench data on this machine yet.
- vast.ai: the `vastai` CLI is installed, but no API key is set and the SSH key is not checked.
  The user runs `vastai set api-key <key>` and adds `~/.ssh/id_rsa.pub` in the vast.ai console.

## Requests to the coordinator

For a-frozen (`fbplan/frozen.py`). Updated after reading a-frozen's work in progress:

1. Dropped: `load_agent`. `train_high` now uses `load_model(env).agent`.
2. To evaluate retrained `π_h`, I will add `load_baseline_model(env, seed)` in `train_high.py`:
   `FrozenFBModel(env, from_state_dict(load_model(env).agent, <my params.pkl>))`. This needs
   `FrozenFBModel` and `load_numpy_pickle` to stay public. No change in a-frozen needed.
3. Make `_upstream(module)` public (for example `upstream`), so my tests do not add
   `third_party/ssm` to `sys.path` themselves (marked TODO in two test files).
4. `load_dataset(...)["terminals"]` = 1 at the last state of each trajectory (non-compact OGBench
   layout), as upstream `GCDataset` expects.
