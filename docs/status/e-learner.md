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

- value expectile 0.7 seeds 0-4 trained (instance 58234389, 19:24-20:32 MSK, code a189a49 with
  `FBPLAN_APPROACH_B_DIR=.../ablations/expectile0.7`), pulled, moved to
  `approach_b/medium/value-single-expectile0.7/`, exported. Geodesic check (5 seeds, 300 x 300
  validation pairs; V_1: pooled -0.587, per goal -0.611):

  | critic | pooled V_H | far half V_H | per goal V_H | goals V_H orders better | median V_H at 20-24 (V_1 7.7) |
  |---|---|---|---|---|---|
  | expectile 0.9 (`value-single`) | -0.543 to -0.555 | -0.341 to -0.361 | -0.555 to -0.572 | 13-17 % | 16.3 |
  | expectile 0.7 | -0.573 to -0.579 | -0.349 to -0.357 | -0.590 to -0.601 | 17-24 % | 10.6 |

  The lower expectile shrinks the long-range gain and moves the ordering towards V_1, but V_H
  orders states no better than V_1 in any seed. So it does not meet "orders states clearly
  better"; I do not propose a policy run on it. Commands:
  `uv run python -m fbplan.learned.diagnostics --env antmaze-medium-navigate-v0 --value value-single --seeds 0 1 2 3 4`
  and the same with `--value value-single-expectile0.7` (table above from commit 36c18d8 on).

- **B-clean ready for d-eval (21:56 MSK):** policy `clean` seeds 0-4, trained on instance
  58234389 (20:32-21:29 MSK, code a189a49, on V_H `value-single` of the same seed), pulled and
  exported. Folders `approach_b/medium/clean/seed{0..4}/` (`high_actor.pkl`, `flags.json`);
  evaluate with `uv run python -m fbplan.evaluate --checkpoint b:clean:<seed> ...` (seeds 0-4).
  Checks: each seed loads through `load_b_model`, F/B/pi_l equal the provided ones bit for bit;
  matched AWR temperature 0.11-0.23 (per seed, from its V_H); training MSE 0.76 -> 0.71-0.73
  (retrained baselines: 0.711), 90 % valid edges, 0.06-0.19 % saturated weights, no NaN; cosine
  with the provided pi_h 0.945-0.956. Command with the current code (bit-identical), per seed k:
  `uv run python -m fbplan.learned.train_b policy --env antmaze-medium-navigate-v0 --seed k --variant clean`.

- **B-wide ready for d-eval (22:47 MSK):** policy `wide` seeds 0-4 (30 % random goals for pi_h),
  trained on 58234389 (21:29-22:40 MSK, code a189a49, on `value-single` of the same seed), pulled
  and exported to `approach_b/medium/wide/seed{0..4}/`. Evaluate with
  `uv run python -m fbplan.evaluate --checkpoint b:wide:<seed> ...`. Checks as for clean: loads,
  F/B/pi_l equal the provided ones; MSE 0.76 -> 0.71-0.74; 90 % valid; no NaN; temperatures
  0.244, 0.141, 0.210, 0.118, 0.153. Current-code command per seed k:
  `uv run python -m fbplan.learned.train_b policy --env antmaze-medium-navigate-v0 --seed k --variant wide`.

- **AWR sharpness at 2x (design.md §8, coordinator decision 2026-10-29 23:00 MSK):** code ready
  and tested (`train_high --name --high-alpha`, `train_b calibrate`, `train_b policy --alpha-from`,
  `load_baseline_model(env, seed, name)`; default paths stay bit-identical; 434 tests pass).
  Plan on 58234389 after region: `push --no-tests` (no job runs then), then
  `bash scripts/gpu/awr2x.sh` in tmux: baseline `high_alpha 0.2` seeds 0-4 ->
  `baselines/medium/awr2x/seed<k>`; one shared temperature from 200 batches (median per-batch
  spreads, new latent draw per upstream batch, critics value-single 0-4 pooled), computed on the
  GPU because CPU and TF32 give different spreads -> `runs/train_b/medium/clean-awr2x/calibration.json`;
  B clean at that temperature -> `approach_b/medium/clean-awr2x/seed<k>` (same batches as the B-clean
  runs, so only the temperature differs). Budget: about 1.9 GPU hours, about $0.40.

- **B-region ready for d-eval (23:58 MSK):** policy `region` seeds 0-4 (window latents, own
  critic `value-region`), trained on 58234389 (22:40-23:50 MSK, code a189a49), pulled and exported
  to `approach_b/medium/region/seed{0..4}/`. Evaluate with `--checkpoint b:region:<seed>`. Checks:
  loads, F/B/pi_l equal the provided ones; MSE 0.62-0.65 -> 0.59-0.62 (smoother targets than
  clean); 89 % valid; no NaN; temperatures 0.211, 0.121, 0.177, 0.102, 0.131. Current-code
  command per seed k: `uv run python -m fbplan.learned.train_b policy --env antmaze-medium-navigate-v0 --seed k --variant region`.
- **AWR 2x started (23:53 MSK)** on 58234389 with code 97d9200 (`push --no-tests` after region):
  `bash scripts/gpu/awr2x.sh > runs/awr2x.log` in tmux session `awr2x`.
- **Attribution control (h-learn H1, coordinator 2026-10-29):** B clean with `--critic v_1` (V_1 where
  the targets and the advantage use V_H), the shared 2x temperature of `clean-awr2x`, the same
  batches; folder `approach_b/medium/clean-v1-awr2x/seed<k>`, d-eval name `b:clean-v1-awr2x:<seed>`.
  Script `scripts/gpu/awr2x_v1.sh` (about 1 GPU hour, about $0.21); start: see the night plan.
- **AWR 2x progress (01:05 MSK, 2026-10-30):** baseline 2x seeds 0-4 saved (seeds 0, 2, 4 from the
  retry; awr2x.sh counted the 3 crashed runs as failed and went on). Shared temperature for B clean
  2x: **0.2099** (`runs/train_b/medium/clean-awr2x/calibration.json`: median upstream spread 16.49 at
  0.2, median switch spread 15.72, 200 batches, critics value-single 0-4). B clean 2x seeds 0-4 at
  step 65k of 500k, GPU 100 %, 12.6 GB; end about 03:00 MSK.
- **AWR 2x ready for d-eval (01:50 MSK, 2026-10-30):** pulled and exported (code 97d9200).
  - Baseline `high_alpha 0.2` seeds 0-4: `baselines/medium/awr2x/seed<k>`, d-eval name
    `retrained:awr2x:<k>`. Command per seed k: `uv run python -m fbplan.train_high --env antmaze-medium-navigate-v0 --seed k --name awr2x --high-alpha 0.2`.
    Final train MSE 0.734, validation MSE 0.722-0.751.
  - B clean at the shared temperature 0.2099, seeds 0-4: `approach_b/medium/clean-awr2x/seed<k>`,
    d-eval name `b:clean-awr2x:<k>`. Commands: `uv run python -m fbplan.learned.train_b calibrate --env antmaze-medium-navigate-v0 --upstream-alpha 0.2 --out runs/train_b/medium/clean-awr2x/calibration.json`,
    then per seed k `uv run python -m fbplan.learned.train_b policy --env antmaze-medium-navigate-v0 --seed k --variant clean --alpha-from runs/train_b/medium/clean-awr2x/calibration.json --name clean-awr2x`.
    Final train MSE 0.728, validation MSE 0.724-0.742.
  - Checks on all 10: load with `load_baseline_model` / `load_b_model`; F, B, pi_l equal the
    provided ones; pi_h new; no NaN.
- **Night plan (coordinator, 00:55 MSK):** no relaunch of awr2x. When B clean 2x ends: pull and
  export the 2x baseline and B clean 2x for d-eval; push once (H1 + T2 code); start H1 (tmux `h1`,
  `bash scripts/gpu/awr2x_v1.sh > runs/awr2x_v1.log`); when its 5 runs step, start T2 next to it
  on cores 10-19 (tmux `t2`, `FIRST_CORE=10 bash scripts/gpu/logratio.sh > runs/logratio.log`).
  Memory: about 12.5 + 4.3 GB of 24. The 2x crash came from 5 processes that compiled at the
  same time, so T2 starts only after H1 compiled. Then pull, export, load checks, destroy.
  A local script runs this plan (scratchpad `wave1.sh`); destroy stays a manual step.
- **T2 code (log-ratio baseline `pi_h`, design.md §8 wave 1, h-learn H2):** `fbplan/learned/logratio.py`
  and `train_high --advantage logratio --goal-alpha 2.45`. Upstream `high_actor_loss` with the
  same latents from the same rng and the same optimizer step; only the AWR weight of goal latents
  changes: `exp(min(2.45 A, 5))`, `A = log max(mean_h [F_h(s,z_w)^T z + rho~_h F_h(w,z)^T z], 1) -
  log max(mean_h F_h(s,z)^T z, 1)`, normalized z, `rho~_h = clip(Msww_h / Mwww_h, 0, 0.99)`.
  Gaussian latents keep upstream's proxy at 0.1. Goal latents are found from the mix flag of
  upstream `sample_latents` (same rng split), not by comparing floats. Logged extras:
  `adv_log`, `goal_share`, `ess_goal`. Flags: `retrained_high_actor.advantage = logratio`, `goal_alpha`.
  Checks:
  - tests (`tests/test_learned_logratio.py`): log-ratio on hand values (head mean before the log,
    rho clip, floor at 1); `measure` equals upstream `successor_measure_extract`; with only
    Gaussian latents the update equals upstream `update` bit for bit (params, rng, metrics); with
    only goal latents and alpha 0 the loss is plain BC; only `pi_h` changes.
  - default path unchanged: the 20-step reference run of `train_high` is bit-identical
    (`pi_h` and log max diff 0.0).
  - cross-check with h-learn's `awr_weights.py` on one batch (val split, key 0, 1024 samples, 492
    goal latents): log-ratio max diff 2.2e-5; upstream proxy median relative diff 1e-6 (larger
    only where `Mwww` is near 0, rounding of the ratio). h-learn's script takes the head mean after
    the log; that order differs from the spec by up to 1.19 (correlation 0.987, means 0.416 vs 0.414).
  - 20-step smoke run on the medium checkpoint: goal share 0.49, goal ESS/N about 0.2, 0.2 % of
    weights at the clip.
  Run (next to H1, see the night plan): `FIRST_CORE=10 bash scripts/gpu/logratio.sh`, i.e. per
  seed k `uv run python -m fbplan.train_high --env antmaze-medium-navigate-v0 --seed k --name logratio --advantage logratio --goal-alpha 2.45`
  -> `baselines/medium/logratio/seed<k>`, d-eval name `retrained:logratio:<k>`. About 1 GPU hour.

## Analysis: the seed split of B-clean (for the report; no new training)

Success per training seed (d-eval, `results/medium/b_clean/seed*.csv` and
`results/medium/baseline_retrained/*.csv`, 1250 episodes each):

| seed | 0 | 1 | 2 | 3 | 4 | spread |
|---|---|---|---|---|---|---|
| B-clean | 81.9 | 75.5 | 83.0 | 75.8 | 80.5 | 7.5 pp |
| retrained baseline | 79.1 | 79.7 | 77.5 | 76.6 | 78.4 | 3.1 pp |
| AWR temperature of B (matched) | 0.229 | 0.131 | 0.195 | 0.111 | 0.144 | |
| upstream advantage std in the calibration | 36.6 | 20.5 | 30.6 | 17.4 | 22.5 | |
| switch advantage std in the calibration | 15.97 | 15.66 | 15.74 | 15.70 | 15.65 | |
| V_H per-goal rank corr. with distance (V_1: -0.613) | -0.559 | -0.573 | -0.567 | -0.558 | -0.563 | |
| V_H mean gain at s / share of targets above V_1 | 2.76 / 0.364 | 2.77 / 0.364 | 2.76 / 0.364 | 2.77 / 0.364 | 2.77 / 0.364 | |
| B task 4 (longest task) success | 71.6 | 60.8 | 72.4 | 60.4 | 68.4 | |

What explains the split:
- Not the shared seed: B seed k and baseline seed k share the pi_h init, and their results do not
  move together (seed 1: baseline 79.7, B 75.5).
- Not the critic: the five V_H are almost the same (gain 2.76-2.77, 36.4 % of targets above V_1,
  per-goal ordering -0.558 to -0.573), and the ordering does not follow success (seed 3 orders
  best and scores 75.8).
- The AWR temperature does: it varies 2x between seeds, and success follows it (rank correlation
  +0.8; n = 5, so suggestive, not significant). The two low seeds have the two lowest
  temperatures and lose mainly on task 4.
- Why the temperature varies: `alpha = 0.1 * std(upstream advantage) / std(switch advantage)`.
  The switch spread is stable (15.7 +- 0.2), the upstream spread is not (17-37). The upstream
  proxy advantage divides by F(w, z_w)^T B(w), which can be near 0, so its std is heavy-tailed.
  CPU check: on one fixed batch, 10 draws of the task latents give 14-32; the same calibration
  on CPU and on the GPU (TF32) gives different values (seed 2: 98 against 31). So each seed's
  temperature came from one noisy draw of an unstable statistic. Support: B-wide uses the same
  fresh agent per seed (so the same latent draw) and other batches, and its temperatures follow
  the same seed pattern (0.244, 0.141, 0.210, 0.118, 0.153 against clean's 0.229, 0.131, 0.195,
  0.111, 0.144). If d-eval's B-wide results split by seed in the same way, that supports the
  explanation further.
- Consequence for the report: B-clean's mean gain (+1.1 pp, CI -2.4 to +4.5) is within the noise
  that the temperature choice adds. "Only the critic changes" held for the network, data and
  optimizer, but not fully for the AWR weighting: B's temperature (mean 0.162) was larger than the
  baseline's 0.1 and differed per seed. A fixed temperature, or a robust spread (median absolute
  deviation, new latents per batch, pooled over seeds), would remove this; not run (B stopped).

How the numbers were made: success from the d-eval CSVs (mean of `success` per
`checkpoint_seed` and `task_id`); temperatures and spreads from `approach_b.calibration` in
`approach_b/medium/clean/seed<k>/flags.json`; V_H rows from
`uv run python -m fbplan.learned.diagnostics --env antmaze-medium-navigate-v0 --value value-single --seeds 0 1 2 3 4`
(after merging main 657acfe; the shared maze-geometry change moved them in the third decimal) and the
last 10 rows of `runs/train_b/medium/value-single/seed<k>/train.csv`; the CPU check calls
upstream `total_loss` on the calibration batches with the agent's rng fixed or replaced
(scratch script, not committed).

## Review findings, first pass (E1-E14, X3, X9, X12)

| Item | What I did |
|---|---|
| E1 M | `VAST_LABEL` is required, no default; comments name no account, person or tab |
| E2 M | `value.switch_target` is a pure function; a test iterates it with a hop cost and a valid mask and matches graph value iteration (`tests/test_learned_option_targets.py`) |
| E3 M | one formula `ValueState.value(v1, s, z)`; stage 2 and the diagnostic use it |
| E4 M | one `SAMPLER_STREAM` and `Agent`; folders, writer, reader, example batch and `forward` come from `fbplan.frozen` (a-frozen's X8/X9 helpers, adopted in the second pass). Batched B waits for c-planner (X4, marked TODO) |
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
| X9, X12 | `fbplan.frozen` writes and reads `high_actor.pkl` (`save_trained_checkpoint`, `load_trained_model`); my code keeps only keys, the fresh `pi_h`, `V_H` files and export in `fbplan/learned/checkpoint.py`; `fbplan.learned` no longer imports `train_high` |
| X14 | my real-checkpoint tests use the session fixtures `medium_model` and `medium_val` of `tests/conftest.py` |

Temporary, until d-eval switches (request 1): `train_high.load_baseline_model`,
`train_high.baseline_dir` and `train_b.load_b_model` stay importable there. Bug fixed (found by a-frozen, 2637e0f): the loader
called `download_trained(key, folder)`; the test now runs the real downloader with a fake
`urlopen`. After adopting a-frozen's helpers, short runs are again bit-identical to the reference.

## Review findings, second pass (A10 and the open e-learner items)

Done on 2026-10-29 (commits 36c18d8, 7b8f04f, 87210a2). Correction: E14 was not done before
36c18d8 (the mode change was not in 3de3cfb), and E4 was only partly done before 1e671da/7b8f04f;
the first-pass table above claimed both too early.

| Item | What I did |
|---|---|
| A10 | coordinator: a-frozen owns the helpers. My copies are gone; callers use `frozen.trained_dir`, `save_trained_checkpoint`, `load_trained_model`, `save_numpy_pickle`, `example_batch`, `forward`, `maze_name`. `checkpoint.trained_dir` stays only as a delegate for `tests/test_evaluate.py` (request 1). The temporary names in `train_high`/`train_b` are gone: `fbplan.evaluate` imports from `fbplan.learned.checkpoint` now |
| E15 | `diagnostics --value` takes any folder; the latent kind comes from its `flags.json`; the table adds the per-goal correlation and the share of goals where V_H orders better (were a scratch script before) |
| E14 | `run_seeds.sh` is 100755 in git (36c18d8) |
| E10, E11 | help on every argument; the listed docstrings |
| X3 | plan: `rho_cap` everywhere; why `goal_zero` differs from A's `zero` (R = 0 only for goal latents in a mixed batch) and why `single` stays (it is part of the release keys) |
| X11 | `train_b --out-dir`; help shows the real default path; diagnostics output `runs/diagnostics/` |
| X14 | tests use `tiny_agent`, `medium_model`, `medium_val` of `tests/conftest.py` |
| N | `run_flags` (one top-level `commit`); a clear error for a pre-rename `flags.json`; `--edge-reward` default from `OptionConfig`; `Agent` from `checkpoint`; full `vast.sh` usage; tests of `value_grid`, per-goal correlation and the `switch_targets` assembly |

Short runs of `train_high` and both B stages stay bit-identical to the reference after each step.

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
| 18:10 | 19:23 | value-region seeds 0-4 (queue-b1) |
| 19:24 | 20:32 | value expectile 0.7 seeds 0-4 |
| 20:32 | 21:29 | policy clean seeds 0-4 (queue-b1 resumed) |
| 21:29 | 22:40 | policy wide seeds 0-4 (rho cap 1.0 cancelled at 22:34 MSK, coordinator decision: B stops) |
| 22:40 | 23:50 | policy region seeds 0-4 |
| 23:51 | 23:53 | push of 97d9200 without tests (no job ran) |
| 23:53 | | awr2x.sh: baseline high_alpha 0.2, calibration, B clean 2x; then the V_1 control |
| 23:54 | 00:02 | baseline 2x seeds 0, 2, 4 crashed at start: `INTERNAL: Autotuning failed for HLO` (XLA GPU autotuning while 5 processes compiled; awr2x.sh sets the platform allocator for all its runs, the earlier baseline runs had none). Seeds 1, 3 ran on. Retried seeds 0, 2, 4 at 00:02 in tmux `awr2x-retry` without the platform allocator, as the earlier baseline runs: `bash scripts/gpu/run_seeds.sh runs/train_high/medium/awr2x "0 2 4" fbplan.train_high --env antmaze-medium-navigate-v0 --name awr2x --high-alpha 0.2`; all 3 step at 00:05. Crash logs kept as `stdout.crash1.log` |
| 00:41 | 01:48 | calibration (00:38-00:41), B clean 2x seeds 0-4 (00:41-01:48); baseline 2x retry ended about 01:05 |
| 01:49 | 01:51 | pull + export of awr2x and clean-awr2x; push of 34b17a5 without tests (no job ran) |
| 01:51 | | H1 (tmux `h1`, `bash scripts/gpu/awr2x_v1.sh > runs/awr2x_v1.log`): seeds 0-3 crashed at the first validation compile, `INTERNAL: Autotuning failed for HLO ... jit(policy_validation)/reduce_sum ... REDZONE CHECK FAILED`, the same failure as the first 2x baseline group (5 runs compiled at once, platform allocator). Seed 4 trains on. T2 not started then |
| 01:59 | | staggered launch `runs/wave1b.sh` (tmux `wave1b`, script below): H1 seeds 0-3 again, then T2 seeds 0-4 on cores 10-19; each run starts after the previous one logs its first step. Crash logs kept as `stdout.crash1.log` |
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

The staggered launch of 2026-10-30 01:59 MSK (`runs/wave1b.sh` on the instance; it runs the
committed scripts' commands, one run at a time until each one steps):

```
start_one() { # run root, seed, first core, module args: start one run, wait for its first step
  root=$1 seed=$2 core=$3; shift 3
  [[ -f $root/seed$seed/stdout.log ]] && mv $root/seed$seed/stdout.log $root/seed$seed/stdout.crash1.log
  FIRST_CORE=$core bash scripts/gpu/run_seeds.sh "$root" "$seed" "$@" &
  until grep -q "^step" "$root/seed$seed/stdout.log"; do sleep 10; done   # (exit check omitted)
}
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
for s in 0 1 2 3; do start_one runs/train_b/medium/clean-v1-awr2x $s $((2*s)) fbplan.learned.train_b \
  policy --env antmaze-medium-navigate-v0 --variant clean --critic v_1 \
  --alpha-from runs/train_b/medium/clean-awr2x/calibration.json --name clean-v1-awr2x; done
unset XLA_PYTHON_CLIENT_ALLOCATOR
for s in 0 1 2 3 4; do start_one runs/train_high/medium/logratio $s $((10+2*s)) fbplan.train_high \
  --env antmaze-medium-navigate-v0 --name logratio --advantage logratio --goal-alpha 2.45; done
wait
```

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

## Questions for the coordinator

- None open. T2 answered 2026-10-30 (design.md §8): (a) 2.45, (b) upstream 0.1 for Gaussian
  latents, (c) rho per head clipped to [0, 0.99], head mean before the log.

## Blockers

- None.

## Requests to the coordinator

For d-eval:

1. Done for `fbplan/evaluate.py`. Left: `tests/test_evaluate.py:376` calls
   `checkpoint.trained_dir("baselines", MEDIUM, 0)`; please use
   `frozen.trained_dir(checkpoint.trained_key("baselines", MEDIUM, 0))`, then I delete the
   delegate (review A10).

For a-frozen:

2. Done: `forward`, `example_batch`, `save_numpy_pickle`, `maze_name`, `trained_dir`,
   `save_trained_checkpoint`, `load_trained_model` are public, and my code uses them.
3. Ready to publish now: `~/.cache/fbplan/baselines/{medium,large}/seed{0..4}/` with
   `high_actor.pkl` and `flags.json` (keys `baselines/<maze>/seed<k>`; `params.pkl` there is the
   old full state, not for upload). Approach B folders follow after the queue.

For d-eval:

7. AWR sharpness at 2x, checkpoint names: the baseline family is the key
   `baselines/medium/awr2x/seed<k>`, loaded by `fbplan.learned.checkpoint.load_baseline_model(env,
   seed, "awr2x")`; `retrained:<seed>` cannot name it yet (perhaps `retrained:awr2x:<seed>`). The B
   family is `b:clean-awr2x:<seed>` (works as is). I write here when each family is pulled.

For c-planner:

4. A public batched-B helper (review X4); `train_b.latent_table` will use it.

For g-release:

5. Done (X14), thanks; my tests use it. The `tiny_agent` docstring names
   `fbplan.learned.checkpoint.example_batch`, which is now `fbplan.frozen.example_batch(params)`.

For f-report:

6. Review E16: `REPORT.md` names `OptionConfig.rho_max = 0.9`, `gain_network` and
   `softplus(gain)`. The names are now `OptionConfig.rho_cap` (CLI `--rho-cap`),
   `delta_network`, and `gain = softplus(delta)`.

For the coordinator: on the GPU (full float32 precision) c-planner's
`test_planner_with_region_latents_solves_every_task_of_the_exact_tabular_maze` solved 13 of 15
tasks instead of 15 (269 of 271 tests passed; the other failure was mine, fixed in a189a49).
Small float differences change planner choices; on CPU it passes.

For the coordinator: a-frozen's tests assume CPU float32. On an Ampere GPU two of them fail
(TF32, see Done). Options: set `jax_default_matmul_precision` to `highest` in those tests, or
document that the suite runs on CPU.
