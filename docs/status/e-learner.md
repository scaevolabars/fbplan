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

## Next

- Draft `docs/plans/approach-b.md` for the user to approve.
- First GPU job (medium, seeds 0-4) when a-frozen is on main. Ask the user before it starts.

## Blockers

- `fbplan.frozen` is still a stub. `train_high.main()` needs `load_agent` and `load_dataset`.
- No checkpoints and no OGBench data on this machine yet.
- vast.ai: the `vastai` CLI is installed, but no API key is set and the SSH key is not checked.
  The user runs `vastai set api-key <key>` and adds `~/.ssh/id_rsa.pub` in the vast.ai console.

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
