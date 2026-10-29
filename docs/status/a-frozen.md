# Status: a-frozen

Last update: 2026-10-29.

## Done

- `fbplan/frozen.py`: `load_model`, `make_env`, `load_dataset`, `download_checkpoint`,
  `dataset_path`, `load_numpy_pickle`, `upstream_module`, class `FrozenFBModel` (implements
  `FBModel`). `tests/test_frozen.py`: 28 tests, all pass on this machine. A test that needs a
  checkpoint or an OGBench file skips when the file is absent.
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
- Merged `main` (8b8a9d8, with b-theory) into `a-frozen`: fast-forward, 103 tests pass.
- Request of d-eval and e-learner: the upstream import helper is public now,
  `upstream_module(module)` (was `_upstream`). The old name has no alias. Its 4 callers are in
  `frozen.py` and are updated. 2 new tests: an upstream import, and the refusal of a module
  from outside `third_party/ssm`.
- Merged `main` (83dc7e1, all branches) into `a-frozen`: fast-forward, 251 tests pass.
- GPU precision (user approved, found by e-learner): on an Ampere GPU (RTX 3090) JAX uses TF32
  for float32 matmuls by default, and 2 tests failed by about 1.6e-3. These 2 tests now run
  inside `jax.default_matmul_precision("highest")`: `test_forward_ignores_the_scale_of_z` and
  `test_baseline_policy_matches_upstream_sample_actions`. The library code keeps the default
  precision. Checked on CPU: inside the context, `_forward`, `_low_action` and upstream
  `sample_actions` compile with `HIGHEST` (default: `DEFAULT`), and the jit cache compiles a new
  entry. CPU ignores the setting, so the GPU failure does not reproduce here.
- Merged `main` (386b4e5: f-report, g-release, rules "no remote, no push" and "at most 4 eval
  workers") into `a-frozen`: fast-forward.
- Item 7, code: `download_trained(key, target=None)` downloads one trained checkpoint from the
  GitHub releases of github.com/scaevolabars/fbplan, idempotent, sha256 pinned per file in
  `_TRAINED_FILES` (empty until the first upload). `key` is the folder below `~/.cache/fbplan`,
  for example `baselines/medium/seed0`; the asset name is the key with `-` for `/`, then `-` and
  the file name. `uv run python -m fbplan.frozen --trained` also gets every published one.
  `download_checkpoint` and `download_trained` share one fetch loop. Each download now writes its
  own `.part` file, so parallel eval workers that download one missing file do not mix their
  bytes (test). 8 new tests, 36 in `tests/test_frozen.py`; 284 tests pass.
- Item 7, format: the user chose to publish only the trained modules (2026-10-29). Nothing is
  uploaded; the upload waits for e-learner (see "Trained checkpoints").
- Item 9, clean-clone check: done (see "Clean-clone check").
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

## Trained checkpoints (item 7)

Present on this Mac only: `~/.cache/fbplan/baselines/medium/seed{0..4}/` (`params.pkl` 18.3 MB,
`flags.json`). `~/.cache/fbplan/approach_b/` does not exist yet.

**Content of a baseline `params.pkl`** (full upstream agent state, 18.3 MB):

| part | MB | trained? |
|---|---|---|
| `modules_high_actor` (`pi_h`) | 2.69 | yes |
| `opt_state` (Adam state of `pi_h`) | 5.37 | yes, needed only to resume training |
| `modules_forward_repr`, `modules_backward_repr`, `modules_actor` | 10.25 | no: equal to the provided checkpoint, bit for bit |

**Decision (user, 2026-10-29): publish only the trained modules.** The upload waits until
e-learner has the new format. Reasons: a reviewer downloads about 135 MB instead of about 900 MB
for about 50 checkpoints (baselines and B variants, 5 seeds); the release does not copy the
authors' F, B and `pi_l` again; the loader takes F, B and `pi_l` from `load_model(env)`, so they
cannot differ, and the equality check in `load_retrained_model` becomes unnecessary. Cost: no
optimizer state, so a published run cannot be resumed (no step needs that).

Proposed format for e-learner (e-learner owns the code; this is a proposal):
- `high_actor.pkl`: the `modules_high_actor` params only, as nested dicts of numpy arrays,
  `jax.tree_util.tree_map(np.asarray, flax.serialization.to_state_dict(params["modules_high_actor"]))`.
  `frozen.load_numpy_pickle` reads it. About 2.7 MB.
- `flags.json`: as now, with portable values (see the requests).
- Approach B: `high_actor.pkl` per variant; `value.pkl` per value run, unchanged (already modules
  only).
- Loader: `params = flax.core.unfreeze(load_model(env).agent.network.params)`, then
  `params["modules_high_actor"] = flax.serialization.from_state_dict(params["modules_high_actor"], saved)`,
  then `agent.replace(network=agent.network.replace(params=params))`.
- The 5 medium baselines: export once from the local `params.pkl`; no retraining.
- Checked on medium seed 0: export, then the loader above, gives params equal to
  `load_baseline_model(env, 0)`, bit for bit; the file is 2.69 MB.

Checks of the present files, done on a full-state staging (not uploaded). They apply to the new
files too:
- Pickles: every text value in the pickle stream is a key or a numpy class name
  (`numpy._core.multiarray._reconstruct`, `ndarray`, `dtype`); no path, no name.
- `flags.json`: two items. `retrained_high_actor.from` is `/root/.cache/fbplan/checkpoints/medium`
  (the GPU instance; no user name). `retrained_high_actor.commit` is the local commit `9557a55`; in
  the public repo it is `e5d66d7` (from `~/.cache/fbplan/publish/commit-map`). The release notes
  record both hashes. The release tag goes on the public training commit.
- Release notes: no match for the private-data patterns of `publish.sh` plus `/root/`, `/home/`,
  the user name, `vast`, `token`, `@`.

After e-learner's export: stage the files, repeat the checks, ask the user, upload, pin the
hashes in `_TRAINED_FILES`, download each asset into an empty folder to check it, commit.

## Clean-clone check (item 9)

2026-10-29, public `main` at `ebff808` (= local `386b4e5`). Clone with submodules into the
scratchpad; a new empty `HOME`, so no `~/.cache/fbplan` and no `~/.ogbench`. The uv package cache
and the uv Python installs of this Mac stay, so no package download over 1 GB.

| README step | result | time |
|---|---|---|
| `git clone --recurse-submodules ...` | OK, submodule at `22c938a` | 23 s |
| `uv sync` | OK, 76 packages (from the cache), Homebrew Python 3.11 | 3 s |
| `uv run python -m fbplan.frozen` | OK, 4 checkpoints, 72 MB from Google Drive | 87 s |
| `uv run pytest` | OK: 261 passed, **16 skipped** | 121 s |
| `fbplan.evaluate --method baseline`, 1 seed, 2 episodes, 2 workers | OK, 8 of 10 successes | 167 s (33 s episodes) |
| `devbox run setup` | **did not finish in 400 s**: Nix download of uv 0.13.5 from cache.nixos.org stalled | - |

Gaps for g-release (`README.md`, `devbox.json`):

1. The evaluation commands use `--workers 10`; design §7 now allows at most 4. The time "about
   1-2 min per method on medium with 10 workers" does not match this Mac: 228 env steps/s per
   worker and about 500 steps per episode give about 11 min for 1250 episodes with 4 workers
   (estimate; measured on 10 episodes).
2. `uv run pytest` on a new machine skips 16 tests: 15 need the medium val split (24 MB), 1 the
   medium train split (243 MB). OGBench files download only on first use; after the first
   evaluation the same run gives 276 passed, 1 skipped. Fix: a data step before the tests, or a
   sentence that these tests skip. (a-frozen can add a `--datasets` flag to `fbplan.frozen` if
   g-release wants one.)
3. No download step for the trained checkpoints. Without a GPU, a reviewer cannot evaluate the
   retrained baselines or approach B. Fix after the release: `uv run python -m fbplan.frozen
   --trained` in the setup.
4. `devbox.json` asks for `uv@latest` and the repo has no `devbox.lock`, so devbox installs a
   different uv on each day (0.13.5 today). Fix: pin the uv version and commit `devbox.lock`.
   The Nix stall can be a network problem of this Mac; the uv path works.
5. The evaluation commands write to the committed `results/medium/*.csv`. A reviewer who follows
   the README overwrites them. If that is the intent (compare with `git diff`), say so; else use
   `runs/`.

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
   `pi_h` on it; upstream `evaluate` can run on it. Import other upstream modules with
   `fbplan.frozen.upstream_module("utils.evaluation")`, not with your own sys.path change.
6. `load_dataset` returns a new dict of writable numpy arrays on each call, with `qpos` and `qvel`.
   Medium train: 1M rows, 1.4 s.
7. `infer_task_latent` takes about 1 s (100k states through `B`) and is deterministic.
8. Some tasks share a goal: teleport tasks 2, 3, 4 and tasks 1, 5; large tasks 4 and 5.
9. **GPU precision.** On an Ampere GPU the model methods use TF32 matmuls (JAX default), so
   results differ from CPU by about 1e-3. A test that compares two paths to 1e-4 or less must set
   `jax.default_matmul_precision("highest")` around both paths.

## Next

- Item 7: when e-learner has exported `high_actor.pkl` for medium seeds 0-4, stage and check
  the files, ask the user, upload, pin the hashes, check the download, commit. Then the same for
  the approach B checkpoints.
- Download the train splits of large, giant, teleport (240 MB each) when c-planner or e-learner
  need them.

## Blockers

- None.

## Requests to the coordinator

- For e-learner (item 7): `load_baseline_model` and `load_b_model` should call
  `frozen.download_trained(key, folder)` when a file is missing, with the key that matches the
  folder below `~/.cache/fbplan`:
  ```python
  ckpt_dir = baseline_dir(env_name, seed)
  if not (ckpt_dir / "high_actor.pkl").exists():
      frozen.download_trained(f"baselines/{env_name.split('-')[1]}/seed{seed}", ckpt_dir)
  ```
  (for B: `f"approach_b/{maze}/{variant}/seed{seed}"`). A key that is not published raises
  `ValueError` with the list of published keys.
- For e-learner (item 7): the user chose modules-only files. Please adopt a format (proposal
  under "Trained checkpoints"), and export `high_actor.pkl` for medium seeds 0-4. The upload
  waits for it.
- For e-learner: record portable values in `flags.json`. `retrained_high_actor.from` and
  `approach_b.value_dir` hold absolute paths of the training machine (`/root/...` on the GPU,
  the home folder with the user name on the Mac). A path relative to `~/.cache/fbplan`, for example
  `checkpoints/medium`, says the same and holds no user name.
- For g-release: the 5 gaps under "Clean-clone check".
- Open: the baseline definition. Does `pi_h` get the raw `z_r` (exact upstream) or the normalized
  one (contract, current)? On large, the normalized `z_r` scores 14 points higher (one seed).
  This waits for the user's decision.
- Optional, not done: a `filterwarnings` entry in `pyproject.toml` for the tfp-nightly
  DeprecationWarning.
- Info: on `main` (83dc7e1), `uv run ruff format --check .` reports 12 files of other components
  (4 in `analysis/`, `fbplan/controllers/oracle.py`, `fbplan/stats.py`, 5 test files,
  `docs/status/b-theory.md`). §7 requires
  only `ruff check`, which passes. The files of a-frozen are formatted.
- Applied by the coordinator (2026-10-29): README setup line, design.md §2 (`z_r` norm),
  §5 (reset recipe), §1/§5 (gap to the paper numbers), merge of `pyproject.toml` and `uv.lock`.
