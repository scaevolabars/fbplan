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
- `docs/plans/approach-b.md`: approved by the user (2026-10-29) with one change: B-clean keeps
  single-state latents; region latents are the variant B-region. B2 waits for a B1 gain.
- GPU job 1 (user approved, 2026-10-29): medium seeds 0-4 on vast.ai instance 58234389
  (RTX 3090, US, $0.209/h, label `fbplan-gpu`), code at commit 9557a55. Started 14:10 MSK.
  Speed: 5 ms/step per run with 4 runs in parallel, so about 45 min per run. Seed 2 crashed at
  start (thread limit, below) and was restarted at 14:22 with `taskset -c 0-1`, same code; its
  first log is kept as `stdout.crash1.log`, and the launcher log will count it as failed.
  Problems found on the instance and fixed in the scripts:
  - SSH direct port: uploads of 20 MB or more broke the connection; the vast.ai proxy carried
    60 MB both ways. `vast.sh` now uses the proxy.
  - The OGBench maze env builds a MuJoCo renderer; the headless instance needs `MUJOCO_GL=osmesa`
    and `libosmesa6` (EGL needs NVIDIA graphics libraries that the CUDA image lacks).
  - On the RTX 3090, JAX multiplies float32 in TF32 by default. Two a-frozen tests fail by
    about 1.6e-3 (`test_forward_ignores_the_scale_of_z`,
    `test_baseline_policy_matches_upstream_sample_actions`) and pass with
    `JAX_DEFAULT_MATMUL_PRECISION=highest`. `remote_setup.sh` tests at that precision; training
    keeps the JAX default, as upstream `main.py` does.
  - XLA sizes thread pools to the host's 80 cores (about 500 threads per run); the container
    allows 2304. `train_baselines.sh` now pins each run to 2 cores (39 threads per run).
- Medium baseline done (15:05 MSK, 55 min of GPU; cost in the GPU log below). 5 retrained `π_h` in
  `~/.cache/fbplan/baselines/medium/seed{0..4}/`, loadable with `load_baseline_model`; curves in
  `runs/train_high/medium/seed*/train.csv`, plot `runs/train_high/medium/curves.png`.
  Checks: train actor loss 791 -> 780 in every seed, no NaN; validation MSE 0.73 -> 0.72 (train
  0.711), no divergence. Cosine of each seed's `π_h` output with the provided `π_h` on 2000
  validation states x the 5 eval tasks: 0.955 (p10 0.90-0.92); between seeds also 0.955. So the
  provided `π_h` behaves like one more seed of our recipe. Ready for d-eval.
- At `create`, the vast.ai API answered "SSH key already associated with instance": the key may
  already be registered for the user in the team account, not only on our instance.

- B1 implemented (plan approved 2026-10-29 with B-clean on single-state latents):
  - `fbplan/learned/option_targets.py`: per-sample `ρ`, `R`, `V_1` in JAX, same formulas and head
    convention as `fbplan/option_model.py`; tested equal to it on a noisy fake FB and to
    b-theory's exact option tables; the residual form reaches the graph value-iteration fixed point
    with chaining gains on a low-rank fake.
  - `value.py` (stage 1: `V_H = V_1 + softplus(gain)`, expectile 0.9, `ρ ≤ 0.9`, `R = 0` for goal
    latents, EMA target), `high_policy.py` (stage 2: upstream AWR with the switch advantage),
    `train_b.py` (CLI; variants clean, wide, region; `load_b_model` for d-eval), `loop.py`
    (logged loop shared with `train_high`).
  - AWR temperature: chosen so that the spread of `α·A` equals upstream's (`0.1 × std` of the
    upstream advantage on the same data), so that B-clean changes only the critic.
  - CPU smoke run of both stages on the real medium checkpoint; 271 tests pass.
- Measured on medium (validation split, 4096 in-trajectory pairs): `ρ` median 0.78, raw `ρ ≥ 1`
  on 9.7 %, `ρ > 0.9` on 33 %. Correction to the plan (§5.1 said the cap "changes few targets"):
  the cap binds on a third of the pairs, mostly short hops. Kept as approved; the ablation
  `ρ_max ∈ {0.9, 1.0}` tests it. Exact `R` is noisy (10–90 %: −8.8 to +29), which supports
  `R = 0` for goals. One-hop gain with `R = 0` is positive on 26 % of pairs.

## GPU time and cost

vast.ai, a shared team account, label `fbplan-gpu`. Times in MSK, 2026-10-29.

| Instance | GPU, place | $/h (with disk) | Start | End | Hours | Cost |
|---|---|---|---|---|---|---|
| 58234389 | RTX 3090, US | 0.209 | 13:16 | running | | |

What ran on 58234389:

| From | To | What |
|---|---|---|
| 13:16 | 14:10 | setup: 2 pushes broke on the direct SSH port, 1 setup stopped at the MuJoCo renderer (fixed) |
| 14:10 | 15:05 | medium baseline seeds 0-4 (GPU 100 %) |
| 15:05 | 15:20 | pull, curve checks, user question |
| 15:20 | 15:41 | push of B1 code, 271 tests on the instance (2 GPU-only failures, below) |
| 15:45 | | push of the self-loop fix, tests; then stage 1 |

## Next

- Ask the user about the next GPU jobs (large baselines, B1 medium) or destroy instance 58234389.
- B1 GPU runs on medium (ask the user first): `value --latents single` seeds 0-4, then
  `policy --variant clean` and `wide`, then the region pair.

## Blockers

- None for Phase 1. Approach B waits for the user's approval of the plan.

## Requests to the coordinator

For a-frozen (`fbplan/frozen.py`). Updated after reading a-frozen's work in progress:

1. Done without a change in a-frozen: `train_high` uses `load_model(env).agent`, and
   `fbplan.train_high.load_baseline_model(env, seed)` gives d-eval an `FBModel` with a retrained
   `π_h`. This needs `FrozenFBModel` and `load_numpy_pickle` to stay public.
2. Done on main: `upstream_module`. My tests use it; no `sys.path` changes left.
3. `load_dataset(...)["terminals"]` = 1 at the last state of each trajectory (non-compact OGBench
   layout), as upstream `GCDataset` expects. (Checked on medium: 1000 trajectories of 1000 steps, every end marked.)

For the coordinator: on the GPU (full float32 precision) c-planner's
`test_planner_with_region_latents_solves_every_task_of_the_exact_tabular_maze` solved 13 of 15
tasks instead of 15 (269 of 271 tests passed; the other failure was mine, fixed in a189a49).
Small float differences change planner choices; on CPU it passes.

For the coordinator: a-frozen's tests assume CPU float32. On an Ampere GPU two of them fail
(TF32, see Done). Options: set `jax_default_matmul_precision` to `highest` in those tests, or
document that the suite runs on CPU.
