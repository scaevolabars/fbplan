# Status: a-frozen

Last update: 2026-10-29.

## Done

- `fbplan/frozen.py`: `load_model`, `make_env`, `load_dataset`, `download_checkpoint`,
  `dataset_path`, `load_numpy_pickle`, `upstream_module`, class `FrozenFBModel` (implements
  `FBModel`); since the first review also `maze_name`, `cache_dir`, `trained_dir`,
  `download_trained`, `save_trained_checkpoint`, `load_trained_model`, `save_numpy_pickle`,
  `example_batch`, `forward`. `tests/test_frozen.py`: 51 tests. A test that needs a checkpoint
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
- Item 7, code: `download_trained(key)` downloads one trained checkpoint from the
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
- Merged `main` (f1ae57d: g-release's first review) into `a-frozen`: fast-forward.
- First review: A1-A9, K1-K4, X8, X9 and clean-clone gap 2 done (see "Review findings, first
  pass"). 357 tests pass.
- Merged `main` (`dded908`, public `7ae64a5`: e-learner's trained-modules format and exports)
  into `a-frozen`: fast-forward.
- e-learner's request 2 (public `forward`, `example_batch`, checkpoint writer, `save_numpy_pickle`,
  `maze_name`, `cache_dir`): already done in `a1a65b6` (first review), no new code. e-learner uses
  no private name of `frozen.py`. Mapping for e-learner: see the requests.
- Release `baselines-v1` of the 10 retrained baselines (medium and large, seeds 0-4): prepared and
  checked, not uploaded (see "Trained checkpoints").

## Review findings, first pass

From `docs/status/g-release.md` (review of 386b4e5). None rejected.

| Id | What changed |
|---|---|
| A1 M | The `z_r` norm is explained once, in `FrozenFBModel.infer_task_latent`, with the range of all 4 mazes (0.79-0.98 `sqrt(d)`) and a pointer to design.md §2 and §8. The class docstring and the 2 tests point there. The wrong "about 1e-3" is gone. |
| A2 M | `example_batch(params)` and `forward(agent, obs, z)` are public; `forward` is not jitted (the model method jits it). |
| A3 M | `download_trained` and `--trained` exist (item 7). The upload waits for e-learner's export (see "Trained checkpoints"). |
| A4 S | argparse CLI: `--help` works, an unknown env gives a usage error (exit 2, no traceback). argparse `choices` fails with `nargs="*"` and no argument on Python 3.11, so a `type=` check does it. |
| A5 S | The `observations` fixture resets with the design.md §5 recipe. |
| A6 S | The rho test is now 2 tests: row outputs of `forward` and `backward` do not depend on the batch order (permutation), and `F(w, B(w))^T B(w) > 0`. |
| A7 S | The `z_r` test asserts `0.75 < norm / sqrt(d) < 0.99`. |
| A8 S | `FrozenFBModel.__init__` puts the agent on the device (test). |
| A9 N | All 4 done: a hash error names the URL, not the deleted `.part` file; the env test checks that 2 resets put the goal at the cell centre (checked: with goal noise on, it fails) instead of reading `_add_noise_to_goal`; the param test compares the set of param paths instead of the count 36; the `load_numpy_pickle` docstring says that it refuses other classes; a test checks that a broken download leaves no file. |
| K1 M | `normalize_z` multiplies by `math.sqrt(d)`, so float32 numpy input stays float32 (test for numpy and JAX). Effect on results: only for numpy callers (`landmarks.py:87` with `space="backward"`, `learned/diagnostics.py`); no committed CSV uses `space="backward"`. |
| K2 M | `temperature` and `seed` are removed from `FBModel.low_action` and `high_latent` (no production caller used them), and from `FrozenFBModel`, with `_sampling_key` and 3 tests. The jitted policies take `dist.mode()` (the mean): outputs are bit-equal to the old `sample` at temperature 0 (checked on 256 validation states). `fbplan/tabular/fake_fb.py` keeps its own optional parameters, which still satisfy the protocol; b-theory can remove them. |
| K3 S | `TaskSpec.z_r`: "task embedding from `FBModel.infer_task_latent` (design.md §2)", no "raw". |
| K4 N | `TaskSpec` is `eq=False` (identity equality and hash; test); `step_info` docstring without "Optional"; module docstring in `tests/test_contracts.py`. |
| X8 S | `maze_name(env_name)` and `cache_dir(kind)` with kind `checkpoints`, `baselines` or `approach_b`. Each kind keeps its env variable (`FBPLAN_CKPT_DIR`, `FBPLAN_BASELINE_DIR`, `FBPLAN_APPROACH_B_DIR`), so `reproduce.sh --smoke` and e-learner's tests keep working. `trained_dir(key)` maps a key such as `baselines/medium/seed0` to its folder. Adoption by the others: see the requests. |
| X9 S | One writer and reader for the modules-only format: `save_trained_checkpoint(agent, flags, out_dir)` writes `high_actor.pkl` (pi_h only) and `flags.json`; `load_trained_model(env_name, key)` downloads a missing published checkpoint, checks `flags["env_name"]`, and puts the saved pi_h into the provided model. `save_numpy_pickle(tree, path)` writes numpy arrays and refuses object arrays. |
| gap 2 | `uv run python -m fbplan.frozen --datasets ENV ...` downloads the OGBench train and val splits of the envs (with the 4 provided checkpoints, as without the option). |

a-frozen commits no CSV to `results/`, so it has no command for `scripts/reproduce.sh`.

Refactor check (coordinator rule), on this Mac after all changes:
- Model outputs on 256 medium validation states (`forward`, `backward`, `low_action`,
  `high_latent`, `low_action(high_latent(z_r))`, `z_r`): bit-equal to the code before the review.
- `uv run python -m fbplan.evaluate --method baseline --env antmaze-medium-navigate-v0 --seeds 0
  --episodes 50 --workers 4 --out runs/refactor-check/baseline_seed0.csv`: the 250 records equal
  the eval-seed-0 rows of `results/medium/baseline.csv` in every column except `wall_time_s`
  (success 76.4 % in both).

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

**Large, one eval seed:** FBModel scored 14 points above upstream here. With 5 eval seeds
(d-eval, design.md §8) the difference is gone: large 38.0 % (normalized `z_r`) against 38.1 %
(raw), medium raw - normalized = +2.6 points, p = 0.11. The one-seed gap was noise.

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

**Decision (user, 2026-10-29): publish only the trained modules.** A trained folder holds
`high_actor.pkl` (the `modules_high_actor` params, numpy arrays, 2.7 MB instead of the 18.3 MB of
the full state) and `flags.json`; F, B and `pi_l` come from the provided checkpoint. Reasons: a
reviewer downloads about 135 MB instead of about 900 MB for about 50 checkpoints; the release
does not copy the authors' F, B and `pi_l` again; a trained folder cannot change them. Cost: no
optimizer state, so a published run cannot be resumed (no step needs that).

Writer and reader: `frozen.save_trained_checkpoint(agent, flags, out_dir)` and
`frozen.load_trained_model(env_name, key)` (X9). e-learner's `fbplan/learned/checkpoint.py`
(`save_high_actor`, `load_high_actor_model`) writes and reads the same format. Checked on medium
and large seed 0: `save_trained_checkpoint` rewrites e-learner's exported `high_actor.pkl` and
`flags.json` byte for byte, and both loaders give equal params.

**Release `baselines-v1`: prepared, not uploaded (the user decides).** Everything is in
`~/.cache/fbplan/release/baselines-v1/` (outside the repo):
- `assets/`: 20 files, 26 MB: `baselines-{medium,large}-seed{0..4}-{high_actor.pkl,flags.json}`,
  copies of e-learner's exports in `~/.cache/fbplan/baselines/` (key `baselines/<maze>/seed<k>`).
- `sha256.txt`; `notes.md` (release notes with the sha256 table); `pin.txt` (the `_TRAINED_FILES`
  entries to commit after the upload); `upload.sh` (checks the hashes, then
  `gh release create baselines-v1 --repo scaevolabars/fbplan --target 7ae64a5... --notes-file
  notes.md assets/*`). Not run.
- Tag target: public `main` `7ae64a5` (= local `dded908`), which has the loader for this format.
  Training commits, from `~/.cache/fbplan/publish/commit-map`: medium `9557a55` = public `e5d66d7`,
  large `a189a49` = public `6f6439b`. The notes give both hashes of each.

Checks:
- Pickles: every text value in the 10 pickle streams is a key (`Dense_0..2`, `actor_net`,
  `mean_net`, `kernel`, `bias`) or a numpy class name (`numpy._core.multiarray._reconstruct`,
  `ndarray`, `dtype`, `f4`). No path, no name.
- `flags.json`: every string value is a config value, `checkpoints/medium`, `checkpoints/large`
  (paths relative to `~/.cache/fbplan`), or one of the 2 local training commits. No absolute path.
- `notes.md`, `upload.sh`, `flags.json`: no match for the private-data patterns of `publish.sh`
  plus `/root/`, `/home/`, `/tmp`, the user name, `vast`, `token`, `@`.
- Download path without an upload: with the 20 hashes pinned and the staged files served in
  place of GitHub, `python -m fbplan.frozen --trained` fetched all 20 assets into an empty
  `FBPLAN_BASELINE_DIR`, and `load_trained_model` on each of the 10 keys gave params equal to
  e-learner's `load_baseline_model` on the local files.

After the upload: commit `pin.txt` into `_TRAINED_FILES`, download each asset from GitHub into
an empty folder, commit. This needs e-learner's fix of `download_trained(key, folder)` first
(see the requests).

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
   evaluation the same run gives 276 passed, 1 skipped. Fix: a data step before the tests.
   a-frozen added the tool: `python -m fbplan.frozen --datasets ENV ...` (see the requests).
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
   On medium this changes actions by 0.002 on average (max 0.023); success does not change
   significantly on medium or large (5 eval seeds, design.md §8).
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
5. `FrozenFBModel.agent` is the upstream `FBpiSwitchAgent`; the constructor puts it on the
   device. e-learner can retrain `pi_h` on it; upstream `evaluate` can run on it. Import other
   upstream modules with `fbplan.frozen.upstream_module("utils.evaluation")`, not with your own
   sys.path change.
6. `load_dataset` returns a new dict of writable numpy arrays on each call, with `qpos` and `qvel`.
   Medium train: 1M rows, 1.4 s.
7. `infer_task_latent` takes about 1 s (100k states through `B`) and is deterministic.
8. Some tasks share a goal: teleport tasks 2, 3, 4 and tasks 1, 5; large tasks 4 and 5.
9. **Shared helpers (X8, X9).** `maze_name(env_name)`; `cache_dir(kind)` and `trained_dir(key)`
   for the folders below `~/.cache/fbplan` (with their env variables); `save_numpy_pickle`;
   `save_trained_checkpoint` and `load_trained_model`; `example_batch(params)`; `forward(agent,
   obs, z)` (not jitted).
10. **Policies are deterministic (K2).** `low_action` and `high_latent` return the mean action
   and the mean intention; they have no temperature or seed.
11. **GPU precision.** On an Ampere GPU the model methods use TF32 matmuls (JAX default), so
   results differ from CPU by about 1e-3. A test that compares two paths to 1e-4 or less must set
   `jax.default_matmul_precision("highest")` around both paths.

## Next

- Item 7: after the user's decision, run `~/.cache/fbplan/release/baselines-v1/upload.sh`, pin
  `pin.txt`, check the real download, commit. Then the same for the approach B checkpoints.
- Download the train splits of large, giant, teleport (240 MB each) when c-planner or e-learner
  need them.

## Blockers

- None.

## Requests to the coordinator

- For e-learner, must fix before the release hashes are pinned:
  `fbplan/learned/checkpoint.py:79` (`load_trained_file`) calls
  `frozen.download_trained(key, folder)`. Since `a1a65b6`, `download_trained` takes only the key
  and writes to `frozen.trained_dir(key)`, which is the same folder with the same env variables.
  So a missing file gives a `TypeError` instead of a download. `tests/test_learned_checkpoint.py:118`
  replaces `download_trained` with a fake that has the old signature, so the test passes. Fix:
  `frozen.download_trained(key)`, or `frozen.load_trained_model(env, key)` for the whole loader.
- For e-learner (request 2 is done; X8, X9): the local copies in `fbplan/learned/checkpoint.py`
  and `value.py` can use the public names of `frozen.py`:
  - `maze_name` -> `frozen.maze_name` (also refuses an unknown env);
    `CACHE`, `_ROOT_VARS`, `trained_dir(kind, env, seed, name)` ->
    `frozen.trained_dir(trained_key(kind, env, seed, name))` (`frozen.cache_dir` for a root);
  - `save_numpy_pickle` -> `frozen.save_numpy_pickle` (also refuses object arrays);
  - `save_high_actor` -> `frozen.save_trained_checkpoint` (byte-equal output, checked);
    `with_high_actor` and `load_high_actor_model` -> `frozen.load_trained_model(env, key)`;
  - `example_batch(agent, obs_dim)` -> `frozen.example_batch(agent.network.params)` (reads
    `obs_dim` from the B weights);
  - `value.py:99` `forward` (TODO) -> `frozen.forward`;
  - `jax.device_put` before `frozen.FrozenFBModel(...)` (`checkpoint.py:106`): no longer needed,
    the constructor does it.
- For g-release: gap 2 has its tool now. `uv run python -m fbplan.frozen --datasets
  antmaze-medium-navigate-v0` downloads the 4 provided checkpoints and the medium train and val
  splits (270 MB); after it, `uv run pytest` skips no test of medium. Put it before `pytest` in
  the README and in `scripts/reproduce.sh`. Gap 3: `--trained` after the release. Gaps 1 and 4
  are fixed on `main` (4 workers; uv pinned in `devbox.json`). Optional (`pyproject.toml` is
  yours now): a `filterwarnings` entry for the tfp-nightly DeprecationWarning.
- For c-planner: `analysis/planner_routes.py:193` -> `frozen.maze_name(args.env)` (X8).
- For b-theory (K2): the protocol has no `temperature` or `seed` any more;
  `fbplan/tabular/fake_fb.py` may drop them (optional; extra optional parameters still satisfy
  the protocol).
- For the coordinator: K1-K4 are done in `fbplan/contracts.py` and `tests/test_contracts.py`.
  design.md §2 points to "per task values in `docs/status/a-frozen.md`", which will be removed
  before submission: move the `|z_r|` list of fact 1 into §2, or drop the pointer. design.md §7
  can name `fbplan.frozen.cache_dir` next to `FBPLAN_CKPT_DIR`.
- Applied by the coordinator (2026-10-29): README setup line, design.md §2 (`z_r` norm),
  §5 (reset recipe), §1/§5 (gap to the paper numbers), merge of `pyproject.toml` and `uv.lock`.
