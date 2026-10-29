# Status: a-frozen

Last update: 2026-10-29.

## Done

- `fbplan/frozen.py`: `load_model`, `make_env`, `load_dataset`, `download_checkpoint`,
  `dataset_path`, `load_numpy_pickle`, class `FrozenFBModel` (implements `FBModel`).
  `tests/test_frozen.py`: 26 tests, all pass on this machine. A test that needs a checkpoint
  or an OGBench file skips when the file is absent.
- Checkpoints: all 4 are in `~/.cache/fbplan/checkpoints/<maze>/` and load.
  Drive file ids and sha256 are pinned in `frozen.py`.
  `uv run python -m fbplan.frozen [env_name ...]` downloads them (idempotent, hash-checked).
  `load_model` also downloads a missing checkpoint.
- Pickle check: each `params.pkl` references only `numpy._core.multiarray._reconstruct`,
  `numpy.ndarray` and `numpy.dtype`. `load_numpy_pickle` refuses all other classes, so loading
  cannot run code from the file. All 4 have the same keys and shapes (4.58 M values,
  `pi_h` trained 500k steps). In `flags.json` only `env_name` differs, and giant has discount 0.995.
- Restore: upstream `FBpiSwitchAgent.create` with `get_config()` updated by `flags.json['agent']`,
  then `flax.serialization.from_state_dict` with the full state (params, opt_state, rng, step).
  The restored params are equal to the pickle arrays, bit for bit (test).
- OGBench data in `~/.ogbench/data`: medium train and val, val of large, giant, teleport (340 MB).
  `load_dataset` downloads one split when it is absent.
- Sanity check (step 6): done. The result is below the paper; see below.

## Sanity check

Upstream `utils.evaluation.evaluate` on the restored agent, 5 tasks x 50 episodes per eval seed,
temperature 0, episode length 1000. CPU only (Apple M-series, 12 cores, no GPU).
Wall time: 66-77 s per eval seed on medium when it runs alone (about 0.5 ms per env step).
Stack: the lock of this branch (MuJoCo 3.1.6, see "Stack decision").

- Path "upstream": upstream `sample_actions`, exactly as upstream `main.py` evaluates.
- Path "FBModel": the same `evaluate` with an adapter that calls `low_action(s, high_latent(s, z_r))`.
  This is the path that the d-eval baseline will use.
- Both paths see the same start states (paired by seed, see fact 2).

Medium, eval seeds 0-4 (1250 episodes per path), success %:

| path | mean | 95 % CI (bootstrap) | per eval seed | per task 1-5 |
|---|---|---|---|---|
| upstream | 78.2 | 76.0 - 80.3 | 79.2, 76.4, 77.2, 78.4, 79.6 | 82.4, 82.8, 76.0, 62.4, 87.2 |
| FBModel | 79.6 | 77.4 - 81.7 | 78.0, 81.2, 80.8, 73.6, 84.4 | 84.0, 85.2, 77.6, 63.6, 87.6 |

Paired difference FBModel - upstream: +1.4 points, 95 % CI -1.6 to +4.6 (not significant).
392 of 1250 episodes change their outcome: the dynamics are chaotic, but the rate does not change.

All envs, eval seed 0 (250 episodes), success %:

| env | upstream | FBModel | paper (5 training seeds) |
|---|---|---|---|
| medium | 79.2 | 78.0 | 87 ± 6 |
| large | 30.8 | 44.8 | 66 ± 9 |
| teleport | 25.2 | - | 40 ± 6 |
| giant | 0.8 | - | 1 ± 1 |

**Large: the normalized `z_r` helps.** FBModel beats upstream on large by 14 points here, and by
7 points (40.4 against 33.6) with MuJoCo 3.17. One eval seed each, paired. `pi_h` was trained on
task latents of norm exactly sqrt(d); the upstream `z_r` on large has norm 0.90-0.95 sqrt(d)
(fact 1). So the upstream eval quirk seems to hurt the upstream baseline on large.

**The provided checkpoints score below the paper**, by 8 points on medium and by 21-35 points on
large. The user accepted this (2026-10-29): our re-evaluated numbers are the baseline
(design.md §5), and the gap goes into the report. Causes that we ruled out:

1. Restore: the restored params are bit-equal to the pickle (test).
2. Our wrappers: the FBModel path is equal to the upstream path within noise on medium.
3. MuJoCo version: success is the same with 3.17.0 (medium 79.6, 5 seeds) and 3.1.6 (78.2).

Causes that are still possible: the checkpoint of training seed 0 is weaker than the paper's mean
over 5 seeds; the paper used an eval protocol that differs from upstream `main.py`; a semantic
change in the JAX stack (unlikely: `pi_h` and `pi_l` use only Dense, tanh and gelu layers).
e-learner's retrained `pi_h` seeds will show if the `pi_h` training seed explains the gap.

## Stack decision (design.md §8)

Keep JAX 0.10.2, flax 0.12.8, distrax 0.1.9, tfp-nightly 0.26.0.dev20261028, numpy 2.4:

- The restore is exact. The upstream path gives the numbers above.
- PyPI keeps all 3028 tfp-nightly builds since 2018 (none yanked). `uv.lock` pins the build and
  its hash, so the install is reproducible.
- The pickles hold numpy arrays only, so an older stack (JAX 0.4.x with tfp 0.25) gives no
  benefit. distrax 0.1.9 needs tfp-nightly, because tfp 0.25 does not support new JAX.
- `np.in1d = np.isin` is not necessary: only upstream `main.py` sets it, and we do not import it.
- Cost: pytest shows one DeprecationWarning from inside tfp-nightly.

MuJoCo: pinned to 3.1.6 in `pyproject.toml` (user approved, 2026-10-29). The lock had 3.17.0.
A one-step replay of dataset transitions has a qpos error of 8.4e-3 (median) with 3.17.0 and
3.8e-7 with 3.1.6, so only 3.1.6 has the physics that produced the data that `pi_l` learned from.
The pin also moves dm-control from 1.0.50 to 1.0.20 (dm-control pins an exact MuJoCo version).
Success does not change (see the sanity check). The test `test_eval_env_reproduces_the_dataset_dynamics`
fails if the physics drift again.

## Facts for other components

1. **`z_r` norm (c, d, e).** Upstream `infer_latent` averages twice (uniform softmax weights 1/n,
   then a mean), so the raw latent is about 1e-7. The `1e-8` in `normalize_z` is then not
   negligible. `|z_r|` per task (sqrt(128) = 11.31): medium 11.09, 11.02, 10.83, 10.46, 10.91;
   large 10.70, 10.68, 10.34, 10.18, 10.18; teleport 10.49, 10.96, 10.96, 10.96, 10.49;
   giant 9.18, 8.96, 10.33, 8.90, 10.91. `infer_task_latent` returns this upstream value.
   Upstream gives this `z_r` to `pi_h` as it is. Our `high_latent` normalizes it (the contract).
   On medium this changes actions by 0.002 on average (max 0.023) and success not significantly.
   On large the normalized `z_r` scores 14 points higher (one seed, see the sanity check).
   `F(s, z_r)^T z_r` scales with `|z_r|`; compare values across tasks with care.
2. **Paired resets (d-eval).** OGBench `add_noise` draws the start xy from the global `np.random`.
   `env.reset(seed=s)` fixes only the joint noise. `info["goal"]` also depends on the RNG of
   `env.action_space` (5 random settling steps). This recipe gives equal start and goal
   observations (checked):
   `np.random.seed(s); env.action_space.seed(s); env.reset(seed=s, options={"task_id": t})`.
   `make_env` calls `env.reset()` once without a seed, as upstream does.
3. Env observations are float64. The jitted model methods cast them to float32.
4. JAX compiles each jitted method once per batch shape (about 0.2 s). In a hot loop, keep the
   batch size fixed.
5. `FrozenFBModel.agent` is the upstream `FBpiSwitchAgent` (on device). e-learner can retrain
   `pi_h` on it; upstream `evaluate` can run on it.
6. `load_dataset` returns a new dict of writable numpy arrays on each call, with `qpos` and `qvel`.
   Medium train: 1M rows, 1.4 s.
7. `infer_task_latent` takes about 1 s (100k states through `B`) and is deterministic.
8. Some tasks share a goal: teleport tasks 2, 3, 4 and tasks 1, 5; large tasks 4 and 5.

## Next

- a-frozen is done. The coordinator can merge `a-frozen` into `main`.
- Download the train splits of large, giant, teleport (240 MB each) when c-planner or e-learner
  need them.

## Blockers

- None.

## Requests to the coordinator

- README setup: add `uv run python -m fbplan.frozen` after `uv sync` (downloads 73 MB of checkpoints).
- design.md §2 says "`z_r` ... then normalized". Precise: norm 0.79-0.98 sqrt(d) (fact 1).
  Decide if the baseline gives `pi_h` the raw `z_r` (exact upstream) or the normalized one
  (contract, current).
- design.md §5: add the reset recipe of fact 2 to the paired design.
- design.md §1/§5: the provided checkpoints score below the paper numbers (see the sanity check).
- Baseline definition: on large, the normalized `z_r` (FBModel path) scores 14 points above the
  exact upstream path (one seed). Decide which path is "the baseline"; maybe report both.
- Merge: this branch changes `pyproject.toml` and `uv.lock` (mujoco==3.1.6, dm-control 1.0.20).
- Optional, `pyproject.toml`: a `filterwarnings` entry for the tfp-nightly DeprecationWarning.
