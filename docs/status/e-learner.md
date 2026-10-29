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

- First review (g-release, E1-E14, X3, X9, X12) and a-frozen's 3 requests: done, see
  "Review findings" below. Refactor check: short CPU runs of every training command give
  bit-identical parameters before (f1ae57d) and after (2ec73f9); only the logged `adv`,
  `adv_std` and `saturated` of the policy stage change (E6).
- Trained-modules format (a-frozen requests 1-3): `high_actor.pkl` (2.69 MB) + `flags.json` with
  paths relative to `~/.cache/fbplan`; loaders call `frozen.download_trained` for a missing file.
  Exported: medium and large baselines seeds 0-4, approach B `value-single` seeds 0-4. For each
  medium seed, the loaded `pi_h` and frozen modules equal the old full state bit for bit.
- Large baseline seeds 0-4 trained (16:11-17:55 on 58234389, code a189a49), pulled, exported.
- `value-single` seeds 0-4 trained (16:20-18:10, code a189a49), pulled, exported. Check of V_H
  against the maze geodesic (300 x 300 validation pairs, all 5 seeds agree):
  - long range: median V_1 at 20-24 xy units 7.7, V_H 16.3; 91 % of these pairs gain > 1.
    V_H falls from 53 to 13 over the maze, V_1 from 53 to 3.7 (the expected shape);
  - ordering: the rank correlation with distance is not better. Pooled: V_1 -0.59, V_H
    -0.55; far half: both -0.35. Per goal (mean over 300 goals): V_1 -0.61, V_H -0.56 to -0.57,
    V_H better for 13-17 % of goals only. So the gain lifts far values but does not point the
    way better. Likely cause: the 0.9 expectile over noisy targets also collects noise (plan §2,
    "a max over w collects the noise"). The policy runs, then their evaluation, test whether it
    matters for control; candidate fixes after that: expectile 0.7, rho_cap 1.0 (planned
    ablations, now with their own folders).
  Commands (CPU, after pull and export):
  `uv run python -m fbplan.learned.diagnostics --env antmaze-medium-navigate-v0 --value value-single --seeds 0 1 2 3 4`;
  the per-goal numbers came from the same functions in a scratch script (not committed).

## Review findings, first pass (E1-E14, X3, X9, X12)

| Item | What I did |
|---|---|
| E1 M | `VAST_LABEL` is required, no default; comments name no account, person or tab |
| E2 M | `value.switch_target` is a pure function; a test iterates it with a hop cost and a valid mask and matches graph value iteration (`tests/test_learned_option_targets.py`) |
| E3 M | one formula `ValueState.value(v1, s, z)`; stage 2 and the diagnostic use it |
| E4 M | one folder function, one writer, one example batch, one `SAMPLER_STREAM` and `Agent` (`fbplan/learned/checkpoint.py`). Two copies wait for other owners: `forward` (a-frozen A2/X9) and batched B (c-planner X4); both are marked TODO |
| E5 M | raw network output `delta`, `gain = softplus(delta)`, `delta_network` |
| E6 S | logged `adv`, `adv_std`, `saturated` count valid edges only |
| E7 S | `value --name`, `policy --name/--value-name`; defaults keep the main folders; test |
| E8 S | `loop.scan_updates`, used by train_high and both B stages |
| E9 S | `Split` is a pytree, `Stage` one static config; `train_policy` split into helpers |
| E10 S | help texts; choices from the `Literal` types; `train_high --out` is now `--out-dir` (X11) |
| E11 S | docstrings on the listed functions and classes |
| E12 S | `jq` listed; `push`/`pull` use `FBPLAN_CKPT_DIR`, `FBPLAN_BASELINE_DIR`, `FBPLAN_APPROACH_B_DIR` |
| E13 S | formula test removed; `advantage_stds` and `calibrate_alpha` tested against direct computations |
| E14 N | `run_seeds.sh` is 100755 in git; comment reworded; tests pass separate arrays |
| X3 | B's cap is `rho_cap` (CLI `--rho-cap`); exported flags rename `rho_max` |
| X4 (mine) | `region_latents` in `learned/sampling.py` is now `trajectory_window_latents` |
| X9, X12 | shared code in `fbplan/learned/checkpoint.py`; `fbplan.learned` no longer imports `train_high` |

Temporary, until their users switch (requests below): `train_high.load_baseline_model`,
`train_high.baseline_dir` and `train_b.load_b_model` stay importable for `fbplan.evaluate` and
`tests/test_evaluate.py`.

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
| 15:45 | 16:11 | push of the self-loop fix (code a189a49), 271 of 272 tests pass (c-planner's GPU one fails) |
| 16:11 | 16:18 | stage 1 as 15 runs: large baselines + value-single + value-region. GPU out of memory: a B run holds 2.5 GB (large baseline 0.86 GB), 15 runs need 29.6 GB of 24. 5 value runs crashed; I stopped all 10 value runs, kept the large ones, moved the logs to `runs/oom-attempt-1611/` |
| 16:20 | 18:10 | large baselines (16:11-17:55) + value-single seeds 0-4 (10 runs, 17 GB). Queue `queue-b1` then runs, one group of 5 at a time: value-region, policy clean, wide, region (two B groups need 25.2 GB) |
| 18:10 | | value-region seeds 0-4 (queue-b1) |
| 18:58 | | ablations approved (coordinator for the user): queue `runs/queue-ablations.sh` (below) holds queue-b1, then runs value expectile 0.7, resumes queue-b1 (policy clean, wide, region), then value rho cap 1.0 |

The ablation queue. The instance runs code a189a49, which has no `--name`; each ablation writes
below its own `FBPLAN_APPROACH_B_DIR` root, so it cannot replace `value-single`. queue-b1 is a
tmux pane, and tmux resumes a stopped pane, so the queue stops the tmux server too (its training
processes write to files and keep running; checked). Started with
`setsid nohup bash runs/queue-ablations.sh > runs/queue-ablations.log 2>&1 < /dev/null &`:

```
Q=37429 T=24127   # queue-b1 shell, tmux server
kill -STOP $T; kill -STOP $Q
while pgrep -f "python -m fbplan[.]learned[.]train_b value"; do sleep 60; done
FBPLAN_APPROACH_B_DIR=~/.cache/fbplan/approach_b/ablations/expectile0.7 bash scripts/gpu/run_seeds.sh \
  runs/train_b/medium/value-single-expectile0.7 "0 1 2 3 4" fbplan.learned.train_b value \
  --env antmaze-medium-navigate-v0 --latents single --expectile 0.7
kill -CONT $Q; kill -CONT $T
while kill -0 $Q; do sleep 60; done
FBPLAN_APPROACH_B_DIR=~/.cache/fbplan/approach_b/ablations/rhocap1.0 bash scripts/gpu/run_seeds.sh \
  runs/train_b/medium/value-single-rhocap1.0 "0 1 2 3 4" fbplan.learned.train_b value \
  --env antmaze-medium-navigate-v0 --latents single --rho-max 1.0
```

Same runs with the current code (bit-identical, see the refactor check), per seed k:
`uv run python -m fbplan.learned.train_b value --env antmaze-medium-navigate-v0 --seed k --latents single --expectile 0.7 --name value-single-expectile0.7`
and `... --rho-cap 1.0 --name value-single-rhocap1.0`. After the pull, the folders move to
`approach_b/medium/value-single-expectile0.7/` and `.../value-single-rhocap1.0/` and are exported.

Launch details (not in a script, so recorded here): every B group runs as
`env XLA_PYTHON_CLIENT_ALLOCATOR=platform bash scripts/gpu/run_seeds.sh runs/train_b/medium/<name>
"0 1 2 3 4" fbplan.learned.train_b <stage> --env antmaze-medium-navigate-v0 <--latents|--variant> ...`.
The allocator setting changes memory handling only. It did not lower the 2.5 GB, which is live memory.

## Next

- Queue on 58234389 (approved jobs): value-region (running since 18:10), value expectile 0.7,
  policy clean, wide, region, value rho cap 1.0; one group of 5 seeds each, about 55 min per
  group, end about 00:30 MSK. Then pull, export, destroy the instance; record the cost.
- After each value ablation: the geodesic check against V_1. If expectile 0.7 orders states
  clearly better than V_1, propose a clean policy run on it and ask the user first.
- Evaluation of the B variants by `fbplan.evaluate --checkpoint` (d-eval).
- Switch to the shared helpers when they are public (requests 2 and 4).

## Blockers

- None.

## Requests to the coordinator

For d-eval:

1. The trained loaders moved to `fbplan.learned.checkpoint` (`load_baseline_model`,
   `load_b_model`, `trained_dir`). Please import them from there in `fbplan/evaluate.py`, and in
   `tests/test_evaluate.py` check `trained_dir("baselines", MEDIUM, 0) / "high_actor.pkl"`
   (trained folders hold `high_actor.pkl`; `params.pkl` exists only in old local folders). Then
   I remove the temporary names `train_high.load_baseline_model`, `train_high.baseline_dir`,
   `train_b.load_b_model`.

For a-frozen:

2. Make `_forward` and `_example_batch` public and expose `save_numpy_pickle`, `maze_name` and
   `cache_dir` (review A2, X8, X9). `fbplan/learned/checkpoint.py` and `value.py` hold one local
   copy of each until then (marked TODO).
3. Ready to publish now: `~/.cache/fbplan/baselines/{medium,large}/seed{0..4}/` with
   `high_actor.pkl` and `flags.json` (keys `baselines/<maze>/seed<k>`; `params.pkl` there is the
   old full state, not for upload). Approach B folders follow after the queue.

For c-planner:

4. A public batched-B helper (review X4); `train_b.latent_table` will use it.

For g-release:

5. A session fixture for a small upstream agent in `tests/conftest.py` (review X14). Three of my
   test files build one each.

For the coordinator: on the GPU (full float32 precision) c-planner's
`test_planner_with_region_latents_solves_every_task_of_the_exact_tabular_maze` solved 13 of 15
tasks instead of 15 (269 of 271 tests passed; the other failure was mine, fixed in a189a49).
Small float differences change planner choices; on CPU it passes.

For the coordinator: a-frozen's tests assume CPU float32. On an Ampere GPU two of them fail
(TF32, see Done). Options: set `jax_default_matmul_precision` to `highest` in those tests, or
document that the suite runs on CPU.
