# Status: a-frozen

Last update: 2026-10-29.

## Done

- `fbplan/frozen.py`: `load_model`, `make_env`, `load_dataset`, `download_checkpoint`,
  `dataset_path`, `load_numpy_pickle`, class `FrozenFBModel` (implements `FBModel`).
  `tests/test_frozen.py`: 25 tests, all pass on this machine. A test that needs a checkpoint
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
Wall time: 66 s per eval seed on medium when it runs alone (about 0.5 ms per env step).

- Path "upstream": upstream `sample_actions`, exactly as upstream `main.py` evaluates.
- Path "FBModel": the same `evaluate` with an adapter that calls `low_action(s, high_latent(s, z_r))`.
  This is the path that the d-eval baseline will use.
- Both paths see the same start states (paired by seed, see fact 2).

Medium, 5 eval seeds (1250 episodes per path), success %:

| path | mean | 95 % CI (bootstrap) | per eval seed | per task 1-5 |
|---|---|---|---|---|
| upstream | 79.6 | 77.4 - 81.8 | 80.4, 80.0, 78.4, 78.8, 80.4 | 74.8, 87.6, 81.2, 65.6, 88.8 |
| FBModel | 78.5 | 76.2 - 80.6 | 80.0, 80.0, 78.4, 77.6, 76.4 | 81.2, 83.2, 80.8, 66.4, 80.8 |

Paired difference FBModel - upstream: -1.1 points, 95 % CI -4.2 to +2.0 (not significant).
396 of 1250 episodes change their outcome: the dynamics are chaotic, but the rate does not change.

All envs, eval seed 0 (250 episodes), success %:

| env | upstream | FBModel | paper (5 training seeds) |
|---|---|---|---|
| medium | 80.4 | 80.0 | 87 ± 6 |
| large | 33.6 | 40.4 | 66 ± 9 |
| teleport | 24.4 | - | 40 ± 6 |
| giant | 0.4 | - | 1 ± 1 |

**Open problem: the provided checkpoints score below the paper**, by 7 points on medium and
by 26-33 points on large. Causes that we ruled out:

1. Restore: the restored params are bit-equal to the pickle (test).
2. Our wrappers: the FBModel path is equal to the upstream path within noise (table).
3. `z_r` normalization (fact 1): small effect. It may help on large (40.4 against 33.6, one seed).
4. MuJoCo version. The lock has MuJoCo 3.17.0. A one-step replay of dataset transitions has a
   qpos error of 8.4e-3 (median) with 3.17.0 and 3.8e-7 with 3.1.6, so 3.17 does not reproduce
   the dataset dynamics. But success under 3.1.6 is the same: medium 79.2, large 30.8 (seed 0).

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

MuJoCo: see the question to the user below (pin 3.1.6 or keep 3.17.0).

## Facts for other components

1. **`z_r` norm (c, d, e).** Upstream `infer_latent` averages twice (uniform softmax weights 1/n,
   then a mean), so the raw latent is about 1e-7. The `1e-8` in `normalize_z` is then not
   negligible. `|z_r|` per task (sqrt(128) = 11.31): medium 11.09, 11.02, 10.83, 10.46, 10.91;
   large 10.70, 10.68, 10.34, 10.18, 10.18; teleport 10.49, 10.96, 10.96, 10.96, 10.49;
   giant 9.18, 8.96, 10.33, 8.90, 10.91. `infer_task_latent` returns this upstream value.
   Upstream gives this `z_r` to `pi_h` as it is. Our `high_latent` normalizes it (the contract).
   On medium this changes actions by 0.002 on average (max 0.023) and success not significantly.
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

- Waiting for the user's decision on the sanity gap (stop point of the brief).
- Download the train splits of large, giant, teleport (240 MB each) when c-planner or e-learner
  need them.

## Blockers

- Sanity number below the paper (see "Open problem"). Asked the user how to continue.

## Requests to the coordinator

- README setup: add `uv run python -m fbplan.frozen` after `uv sync` (downloads 73 MB of checkpoints).
- design.md §2 says "`z_r` ... then normalized". Precise: norm 0.79-0.98 sqrt(d) (fact 1).
  Decide if the baseline gives `pi_h` the raw `z_r` (exact upstream) or the normalized one
  (contract, current).
- design.md §5: add the reset recipe of fact 2 to the paired design.
- design.md §1/§5: the provided checkpoints score below the paper numbers (see "Open problem").
- Optional, `pyproject.toml`: a `filterwarnings` entry for the tfp-nightly DeprecationWarning.
