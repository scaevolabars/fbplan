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

## Next

- GPU setup script in `scripts/gpu/` (waiting for the user: how to reach the GPU machine).
- Draft `docs/plans/approach-b.md` for the user to approve.

## Blockers

- `fbplan.frozen` is still a stub. `train_high.main()` needs `load_agent` and `load_dataset`.
- No checkpoints and no OGBench data on this machine yet.

## Requests to the coordinator

For a-frozen (`fbplan/frozen.py`):

1. `load_agent(env_name, ckpt_dir: Path | None = None)`: the restored upstream `FBpiSwitchAgent`
   (`get_config()` updated by `flags.json['agent']`, params from `params.pkl`). `train_high` needs
   the agent object to call upstream `create` and `update`.
2. `load_model(env_name, ckpt_dir: Path | None = None)`: so that d-eval can evaluate a retrained
   `π_h` from `~/.cache/fbplan/baselines/<env>/seed<k>/` (same layout as a provided checkpoint).
3. A public helper to import upstream modules (for example `upstream_module("agents.fbpiswitch")`).
   Until then, my two test files add `third_party/ssm` to `sys.path` themselves (marked TODO).
4. `load_dataset(...)["terminals"]` = 1 at the last state of each trajectory (non-compact OGBench
   layout), as upstream `GCDataset` expects.
