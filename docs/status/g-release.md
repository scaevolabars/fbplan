# Status: g-release

Last update: 2026-10-29.

## Done

- `scripts/reproduce.sh`: one entry point with the stages checkpoints, train, eval, tables,
  figures; `--smoke` and `--workers N` (default 4, design.md §7). It prints every command that it
  runs and the time of each stage. Bash 3.2 compatible (macOS default shell); shellcheck clean.
  - `RUNS` holds one line per committed CSV in `results/medium/` (25 runs) plus the parity run.
    A line ends with raw `fbplan.evaluate` arguments (`--opt ...`, later `--checkpoint ...`).
    Checked against the config hashes in the files: the 13 oracle CSVs, and the 9 planner CSVs
    (`--opt landmarks.seed=k`, region runs also `landmarks.latent=region`) once the field
    `switch_steps`, added after these runs, is left out of the hash (X18).
  - Tables (`results/<maze>/tables.md`): the main table as in d-eval's command
    (`{baseline,baseline_raw,nonhier,oracle}.csv` and `planner*/*.csv`, `--label-root`,
    `--cluster config_hash`), the oracle sweep, failure classes of the 4 reference methods, and
    `analysis/report_figures.py` (f-report's request: time-to-goal figure and the numbers of
    report §8.4). `upstream.csv` stays out of `tables.py` (D2).
  - Figures: success figure (same CSVs as the main table), calibration figures and
    `results/calibration/` tables, `analysis/heldout_goals.py`, route plots, the tabular study
    and `fbplan.tabular.time_limit`.
  - Checkpoints stage: `python -m fbplan.frozen --trained` (frozen and published trained ones).
  - In `--smoke`, each evaluation prints how many episodes equal the committed episode with the
    same (task, eval seed, episode). With 1 episode per task, all 35 smoke episodes of the 7
    result groups equalled the committed ones (success and steps): the current code reproduces
    the committed CSVs, the planner included. It only prints, because another CPU can change
    the chaotic dynamics.
  - `--smoke` writes only to `runs/smoke/` (trained models via `FBPLAN_BASELINE_DIR`,
    `FBPLAN_APPROACH_B_DIR`). It skips the parity run (50 episodes per task are fixed, request 5)
    and `heldout_goals.py` (it always writes to `results/heldout_goals/`, see the CLI convention).
- Clean clone (commit f058ae6, cloned without `--recurse-submodules`, empty `FBPLAN_CKPT_DIR`):
  `--smoke` passed in 5.5 min with the datasets already in `~/.ogbench/data` (checkpoints 38 s,
  train 81 s, eval 53 s, tables 41 s, figures 98 s). The script initialized the submodule, and
  the clone stayed clean.
- `devbox.json`: uv pinned to 0.11.8 (the version that wrote `uv.lock`), `--locked` in every
  script, scripts `reproduce` and `smoke`. No `devbox.lock`: `devbox install` printed
  "Ensuring packages are installed" and then nothing for 15 min (the Nix download stalls on this
  Mac, as in a-frozen's check). So the README names uv as the primary setup path and devbox as
  a way to get the pinned uv.
- `.github/workflows/ci.yml`: `uv sync --locked`, `ruff check`, `pytest -ra` on ubuntu-24.04 with
  `MUJOCO_GL=osmesa`; packages only, no data. With an empty `HOME`, `pytest` downloads nothing.
  Linux check (user approved): the same steps in a `python:3.11-slim-bookworm` container
  (arm64, uv 0.11.8, `libosmesa6`, the 16 CUDA packages left out) on commit 8ce6821: ruff
  clean, pytest 247 passed, 30 skipped.
  GitHub Actions on the public repo (x86_64, CUDA wheels installed, about 2 min): the workflow
  works (sync, OSMesa, ruff), and pytest gives 335 passed, 40 skipped, 1 failed on every push
  since the merge. The failure is c-planner's
  `test_planner_with_region_latents_solves_every_task_of_the_exact_tabular_maze` (13 of 15
  tasks, float rounding on another CPU; e-learner saw the same on a GPU). c-planner's "CI fix"
  run was in progress at 15:50 UTC.
- `pyproject.toml` (X15): removed `h5py` (no import in our code, upstream or the packages we use;
  only dm_control's mocap loader imports it, which antmaze never loads; 343 tests pass without
  it); `addopts = "-ra"`; `pythonpath = [".", "tests"]` (`planner_fakes`, and tests of
  `analysis/`); ruff `per-file-ignores` for `E402` in the plot scripts. The 44
  `# noqa: E402` comments in 8 files of other owners are now redundant
  (`ruff check --extend-select RUF100` lists them); the owners can delete them.
- `tests/conftest.py` (X14): fixtures `require_files` (`.checkpoint(env)`, `.dataset(env,
  split)`: skip when the file is absent; works in parametrized tests), `medium_model` and
  `medium_val` (session scope: the real model loads once per session, not once per module).
  Checked with and without data. Owners replace their copies (`test_frozen.py:26-34`,
  `test_evaluate.py:298-301, 343-349`, `test_learned_train_high.py:135-142`,
  `test_learned_b.py:201-208`). Not in conftest, by choice: `MEDIUM` (a literal env id reads
  better than an import), and `same()` / the tiny upstream config (both only in e-learner's two
  test files; a `tests/learned_fakes.py` like `planner_fakes.py` fits them).
- README, gaps from a-frozen's clean-clone check: evaluation with `--workers 4` and its time
  (about 11 min per 5-seed run on medium, about 4 h for the eval stage, estimate); the eval stage
  writes the committed CSVs (`git diff` shows changes; `--out runs/...` for single runs); tests
  never download data and list their skips; `fbplan.frozen --trained` in the setup; methods
  table with `--checkpoint retrained:<seed>` and `b:<variant>:<seed>`.
- Merged `main` (f1ae57d, then dded908).
- `tests/conftest.py`, e-learner's request: session fixture `tiny_agent` (`TinyAgent`: the
  upstream agent with 16x16 networks and latent_dim 4, seed 123, obs dim 5, action dim 2; a
  read-only `config`; `latent_dim`). Its parameters equal those of the agent that
  `test_learned_train_high.py:33` builds now, and it works with `fbplan.learned.checkpoint`
  (`example_batch(tiny.agent, tiny.obs_dim)`, `with_fresh_high_actor(tiny.agent, seed,
  tiny.obs_dim)`). Variants come from `tiny.agent.replace(...)`, for example the copied F head of
  `test_learned_b.py:56`. `test_learned_checkpoint.py:34` uses seed 7 and no `batch_size`;
  e-learner checks that its tests still pass on seed 123.
- `RUNS`: the 5 retrained baselines (`--method baseline --checkpoint retrained:k`, 5 eval seeds).
  Checked: a rerun of seed 3 (eval seed 0, 2 episodes) gives 10 of 10 committed episodes. They
  are not in the main table: all 5 share the config hash `c232a50c`, so `--cluster config_hash`
  would make them one cluster (request 5). In `--smoke` this group evaluates the model that the
  smoke run trained, so its episodes differ from the committed ones.
- CI on the public repo is green from 7ae64a5 (= local dded908): ruff clean, pytest 345 passed,
  40 skipped, 74 s.
- Merged `main` 3e36461 (c-planner and d-eval fixes). `failures.py` now handles a CSV without
  failures (D9), so `reproduce.sh` no longer guards it; `--smoke` runs the parity check again.
- Merged `main` 3f169ae (review fixes of a-frozen, b-theory, e-learner). `reproduce.sh`: the
  checkpoints stage also gets the medium datasets (`fbplan.frozen --datasets`, a-frozen's new
  flag); `run_study --out-dir` (b-theory renamed the flag). README: the setup line gets the
  medium data, so the medium tests run instead of skipping; "none is published yet" for
  `--trained` (finding A12).
- Second review pass of a-frozen, b-theory and e-learner: section below.

## Next

- Second review pass of c-planner and d-eval when their runs end.
- Platform (coordinator's decision): canonical CSVs in `results/<env>/` from Linux x86_64 on CPU;
  the Mac CSVs moved to `results/macos/` (d-eval, 0b8e05e). Done: `JAX_PLATFORMS=cpu` for every
  stage but GPU training; the README platform paragraph (chaotic MuJoCo, other platforms give
  other episodes and the same success rates within noise); the smoke comparison uses
  `results/macos/` on a Mac and the canonical file elsewhere, and says so when no committed CSV
  exists yet. `RUNS` keeps its paths (d-eval keeps the file names). To do: add the large runs
  when their CSVs land; d-eval's Mac-vs-Linux comparison must confirm "within noise" before
  submission.
- Release `baselines-v1` (retrained pi_h, medium and large, 5 seeds each) exists. Checked with
  a clone of a-frozen's branch bfe4953 (not on main yet) and empty cache folders: `python -m
  fbplan.frozen --trained` downloads 20 files (32 MB) in 38 s, hash-checked, and an evaluation
  of the downloaded `retrained:3` (eval seed 0, 2 episodes) gives 10 of 10 episodes of
  `results/macos/medium/baseline_retrained/seed3.csv`: the release holds the models behind the
  results. When a-frozen's pins reach main: README "none is published yet" becomes "the
  retrained baselines are published; approach B follows".

## Blockers

- None. The retrained baselines and approach B wait for their CSVs and the release of the
  trained checkpoints.

## Requests to the coordinator

1. X18 (new, must-fix, c-planner and d-eval): a config field added after a run changes the
   `config_hash` of every committed CSV of that controller. c-planner added
   `PlannerConfig.switch_steps` (default `None`, the old behaviour) after the 9 planner runs.
   Consequences: a rerun of the same command writes a different `config_hash`, so "records
   identical" fails on that column; `analysis/failures.py:94` refuses to replay any committed
   planner CSV ("config hash differs"); and `tables.py` without `--label-root` would split old
   and new runs of one config. Proposed fix (d-eval owns `config_hash`): hash only the fields
   that differ from the dataclass defaults, so a new field with an old-behaviour default keeps
   every hash. This changes the hash of every committed CSV once, so rewrite the column after a
   check that a rerun of eval seed 0 gives identical records. Alternative: c-planner reruns.
2. Done (d-eval): `analysis/parity.py --episodes`. `--smoke` runs the parity check again
   (1 repetition, 2 episodes per task, 15 s).
3. F1 (new, should-fix, f-report): `analysis/report_figures.py` raises
   `KeyError: ['planner_region'] not in index` when one planner variant has no CSV in
   `--results`, so it cannot run on a subset. `reproduce.sh --smoke` now runs one CSV of each
   result group, so the smoke run passes. Fix: drop absent variants with a printed note, or
   raise a clear error that names the missing folder.
   F2 (new, should-fix, f-report): `analysis/report_figures.py:72` raises `ValueError: cannot
   convert float NaN to integer` for a method with no success (`int(median)` of an empty set).
   The smoke run hit it with 1 episode per task: planner landmark set 0 fails the 5 episode-0
   start states (as in the committed CSV). `--smoke` now runs 2 episodes per task, where every
   group has successes. Fix: report the median as missing when there is no success.
4. d-eval: the command for the table of the retrained baselines (the training seeds as a
   cluster level; `--cluster checkpoint_seed`?). I add it to the tables stage.
5. Decided (A10, coordinator, 2026-10-29): option (a). a-frozen owns the trained-checkpoint
   helpers; e-learner imports `fbplan.frozen.*` and deletes its copies. I check it in the next
   review pass of e-learner (and that a rerun of one retrained eval seed gives identical records).

## CLI convention (X11, proposal; the owners adopt it)

1. Inputs are positional paths: CSV files, or one folder.
2. Outputs: `--out FILE` for one file, `--out-dir DIR` for one folder. A script that writes
   figures and tables to different folders uses `--fig-dir` and `--table-dir`. A training CLI
   uses `--out-dir` for the checkpoint and `--run-dir` for the logs.
3. Every file that a script writes has a flag, so that `reproduce.sh --smoke` can send all
   output to `runs/smoke/`.
4. Defaults: a script that writes committed results defaults to their committed path in
   `results/` or `docs/figures/`. Other output goes to `runs/<script name>/`, never to a folder
   named after a branch. A quick or smoke option never defaults to a committed folder.
5. `--env` for one env (`choices=frozen.ENV_NAMES`), `--envs` for several. `--seeds` for eval
   seeds, `--seed` for one training or sampling seed; the help says which seed it is.
6. `--workers` defaults to 4 (design.md §7) in every script that starts processes.
7. Every argument has `help=`. The parser uses `description=` from the first docstring line and
   `formatter_class=argparse.ArgumentDefaultsHelpFormatter`, so `--help` shows the defaults. The
   usage line in the module docstring lists the real arguments.
8. `--method` takes an alias of `fbplan.controllers.ALIASES`; `Controller.name` equals the
   alias (X6), so the CSV `method` column is what the user typed.

Changes that this needs, by owner:

| Owner | Script | Change |
|---|---|---|
| d-eval | `fbplan/evaluate.py` | `--method` help from `sorted(ALIASES)`; `--env` choices; `--workers` default 4; help for `--tasks`, `--workers` |
| d-eval | `analysis/parity.py` | `--workers` 5 → 4; `--episodes` (request 2) |
| d-eval | `analysis/failures.py` | `--workers` 10 → 4 |
| d-eval | `analysis/calibration.py` | `--out-dir` → `--fig-dir` (it also has `--table-dir`) |
| d-eval | `analysis/heldout_goals.py` | add `--table-dir` (default `results/heldout_goals`) next to `--fig-dir` |
| d-eval | `analysis/tables.py`, `plots.py` | help for `--reference` etc.; defaults shown |
| c-planner | `analysis/planner_routes.py` | default `--out-dir runs/planner_routes`; help that `--seed` is the reset seed |
| c-planner | `fbplan/controllers/planner.py` | `name` = alias (`planner`, `planner_high`) |
| b-theory | `fbplan/tabular/run_study.py` | `--out` → `--out-dir`; `--workers` default 4; `--quick` defaults to `runs/run_study/` |
| b-theory | `fbplan/tabular/study_plots.py` | `--dir` → positional folder |
| b-theory | `fbplan/tabular/time_limit.py` | `--out` → `--out-dir` |
| e-learner | `fbplan/train_high.py` | `--out` → `--out-dir`; help with the real default path |
| e-learner | `fbplan/learned/train_b.py` | help text; `--name` / `--value-name` (E7) |
| f-report | `analysis/report_figures.py` | `--results` → positional folder |

When an owner renames a flag, it tells me, and I change `scripts/reproduce.sh` in the same
merge.

## Process files before submission (brief step 4)

Decision of the user (2026-10-29): remove `docs/briefs/` and `docs/status/` from the submitted
tree, in the final submission step only (until then they are the coordination channel). Before
that, each owner moves a fact that is still needed into REPORT.md or design.md. The coordinator
does the removal. Reasons:

1. They describe the multi-session workflow (tabs, coordinator, approvals). REPORT.md must not
   (f-report rule 6), and a reviewer reads them as noise next to the code.
2. They hold account details: the vast.ai team account name, an instance label with the
   user's surname, GPU costs (`docs/status/e-learner.md`).
3. They hold numbers that no committed CSV backs (probes, dev-set runs, early loaders). A
   reviewer can find a number that contradicts REPORT.md.
4. The facts that matter already live in `docs/design.md`, `docs/theory.md`,
   `docs/plans/approach-b.md` and REPORT.md. Before the removal, each owner moves a fact that is
   still needed (for example a-frozen's stack decision and sanity check) into REPORT.md or
   design.md.

Notes:
- Removal from the tree does not remove the files from git history. If the published history
  keeps the old commits, the files stay visible there.
- `docs/design.md` §6-§8 also describe the workflow (branches, tabs, merges). A short
  "repository layout and code rules" section could replace them (coordinator's file).
- Alternative: move the two folders to `docs/dev/`. This keeps the trail, but needs a scrub of
  account details and of numbers without a CSV.

## Review findings, second pass: a-frozen, b-theory, e-learner (2026-10-29, merged main 3f169ae)

Three read-only reviewers checked every M and S finding of these owners against the code at
e77587e (main 275dfe1 + 3f169ae), and read all code changed since 386b4e5. I checked the items
marked (checked). Verdicts: fixed / rejected (reason OK or weak) / partly / open. ruff passes on
all three areas; owner tests: a-frozen 52 passed, b-theory 115 passed, e-learner 45 passed.

### a-frozen: all M and S findings fixed, except A3 (waits for the release)

- Fixed: K1, K2 (in a-frozen's files), K3, K4, A1, A2 (a-frozen side), A4-A9; A6 is now two
  tests that can fail; A8 has a test that fails without `device_put`.
- A3 partly: `download_trained`, `--trained` exist; `_TRAINED_FILES` is empty until the release.
- X8, X9: the public helpers exist (`maze_name`, `cache_dir`, `trained_dir`,
  `save_numpy_pickle`, `example_batch`, `forward`); e-learner has not adopted them (see e-learner).
- X14 pending: `test_frozen.py:24-42` keeps its own skip helpers; `require_files` can replace
  them. Keeping its module `model` fixture is fine (checkpoint-only tests need no val split).
- K2 leftovers in other owners' files: `tabular/fake_fb.py:116-130, 242-244` (`temperature`,
  `seed`, `_require_greedy`) and its test `test_tabular_fake_fb.py:174-176`; test fakes with the
  old signature in `planner_fakes.py:75,82`, `test_eval_controllers.py:47,50`,
  `test_oracle_teleport.py:110,113`, `test_planner.py:66`. Owners: b-theory, c-planner, d-eval.

New findings:
- A10 M (checked; owners a-frozen and e-learner, coordinator decides). The trained-checkpoint
  format now has two writers and two readers: `frozen.save_trained_checkpoint`,
  `frozen.load_trained_model` (test callers only) and `learned/checkpoint.py`
  `save_high_actor`, `load_high_actor_model`, `with_high_actor` (used by `train_high`,
  `train_b`, `evaluate`). Also defined in both modules: `save_numpy_pickle`, `example_batch`,
  `trained_dir` (different signatures), `maze_name` (the e-learner copy does not reject unknown
  envs). X8/X9 meant to remove copies; now there are more. Pick one side before the release
  pins hashes: e-learner switches to the `frozen` names (a-frozen's status lists the mapping),
  or a-frozen deletes its unused pair.
- A11 S (d-eval). `evaluate.py:215-237` (`goal_latent`, `_val_dataset`) repeats the
  `infer_task_latent` pipeline (relabel, first n rows, `infer_latent`) with a different guard,
  and loads the val split a second time. Fix: a-frozen offers `infer_goal_latent(task_info)`,
  `infer_task_latent` calls it, d-eval deletes its copy.
- A12 S (a-frozen, g-release). `python -m fbplan.frozen --trained` prints nothing while nothing
  is published. Fix: print "no trained checkpoint is published yet". g-release: the README now
  says "none is published yet" (done).
- N: `test_frozen.py:428-435, 495` compare two compute paths at 1e-5 / rtol 1e-6 without
  `jax.default_matmul_precision("highest")` (fails on TF32 GPUs, a-frozen's own fact 11);
  `frozen.py:428` gives a wrong reason for not jitting `forward`; the cache kinds are written 3
  times (`frozen.py:104-109, 157`); `_known_env` repeats `maze_name`'s check; the CLI
  description hard-codes `~/.cache/fbplan/checkpoints`; `load_trained_model` hash-checks only on
  download; `contracts.py:4` still says "tell the user" (coordinator); the status file counts
  51 tests, pytest collects 47.

### b-theory: all M and S findings fixed in the code; the status file has no review response

- Fixed: B1-B14, and the b-theory parts of X2 (the study uses `OptionModel`, `trusted_edges` on
  raw rho, `value_iteration`, `q_values`, pinned by `test_tabular_study.py:37`), X3
  (`head_agg`, exact/paper/zero, `StudyPlannerConfig`), X7 (`test_tabular_maze.py:92-111`
  compares all 4 maps, tasks and teleporters with OGBench), X13 (`rho_bound_wormholes`; theory.md
  states both definitions). B15 partly.
- Numbers: the study code is unchanged since its CSVs were regenerated (89c4c74); the §9.1,
  §9.3, §9.4 and §9.5 numbers of theory.md match the committed CSVs.
- X11 partly: `run_study` adopted (`--out-dir`, 4 workers, `--quick` to `runs/run_study`);
  `study_plots --dir` and `time_limit --out` not adopted, no reason recorded.

New findings:
- B16 S (checked). `fbplan/tabular/time_limit.py:226-234`: `main` writes `time_limit.csv`
  before any folder exists (only `plot` creates it), so `--out <new folder>` fails with
  `OSError`; `reproduce.sh` works only because `run_study` creates the folder first. `--out` has
  no help. Fix: `mkdir(parents=True, exist_ok=True)` first, help text, `--out-dir` (X11).
- B17 S (checked). `docs/theory.md:175, 237` and `fbplan/controllers/oracle_teleport.py:16`
  refer to a §11 that does not exist (theory.md ends at §10). The time-limit results (walk
  thresholds 0.9977-0.9993, "the gamble fails with p >= 1/2") are written up nowhere. Fix:
  write §11 or point to existing sections.
- B18 S. `study.py:135-139` (`make_plan`) keeps its own argmax and strict `best_q > v_direct`;
  the planner now uses `graph.best_options` (1e-9 tie tolerance, ties by largest V_H, added by
  c-planner after the study refactor). theory.md:166 says the study "tests the same code that
  runs on AntMaze"; the commit timeout rule also differs. Fix: call `best_options`; reword.
- B19 S. K2 leftover: `fake_fb.py:115-130, 242-244` keeps `temperature`, `seed`,
  `_require_greedy` and `import jax`; `test_tabular_fake_fb.py:174-176` tests the dead guard.
- B20 S. `oracle_teleport.py:65-122` and `time_limit.py:43-75` both write the no-teleporter
  shortest walk and the risk-neutral cell MDP; a test pins only the discounted pair
  (`test_tabular_time_limit.py:55`); `oracle_teleport.py:35, 59-62` copies the oracle's
  neighbour helpers. Fix: build the oracle on `GridMaze(name, 1)` with `walk_policy` and
  `discounted_policy`, or pin `reach_plan` to `walk_policy` on every cell.
- B21 S. `docs/status/b-theory.md` has no review response and no X11 decision, and is stale
  (`high_advantage`, "1 min on 12 cores", "75 pass" against 115 now; no teleport oracle or
  time limit). B6: `run_study.py:3-5` says "14 min with 4 workers, 10 CPU min", which reads oddly.
- N: `oracle_teleport` has no alias in `controllers/__init__.py` and is not in the privileged
  set of `tables.py:35` / `plots.py:50` (coordinator, d-eval); new test copies of the teleport
  and medium maps and of `SAFE_WALK_CELLS` (`test_oracle_teleport.py:17-31, 182-197`,
  `test_tabular_time_limit.py:19, 59-61`) where `GridMaze` has them; `discounted_values` loops
  without a limit; `study.py:32` repeats `PlannerKind`; `matplotlib.use` with `noqa` remains
  (the per-file ignore makes the `noqa` redundant); B10's gap test asserts `> 0.3` while
  theory.md:70 marks 0.56/0.36 with ✓.

### e-learner: M findings fixed except E4 (copies of a-frozen's helpers remain)

- Fixed: E1 (`VAST_LABEL` required, neutral comments; the surname and account name are in no
  tracked file outside `docs/status/`, but they remain in git history, e.g. 386b4e5 — the
  publish step must handle history), E2 (the test iterates the production `switch_target`
  with hop costs and a random valid mask against `graph_value_iteration`), E3, E5, E6, E7, E8,
  E9, E12, E13; X4 (own part), X12.
- E4 partly, X8 open, X9 open: see A10. `value.py:99-104` copies `frozen.forward`, and its
  TODO says "when it exists" (it exists, `frozen.py:425`); e-learner's request 2 is out of date.
- E10, E11 partly: no help on `train_high.py:99 --env`, `train_b.py:311 --env, 318 --latents,
  321 --edge-reward, 324 --variant`; no docstring on `OptionTerms` (named in E11),
  `ValueState.create`, `Split.load/obs_dim/batch`, `save_flags`, `diagnostics.plot`,
  `parse_args`.
- E14 (checked): `scripts/gpu/run_seeds.sh` is still mode 100644 although commit 3de3cfb and the
  status table say it is executable. Fix: `git update-index --chmod=+x`.
- X3 partly: `rho_cap` done; R modes `goal_zero|exact|paper` against A's `exact|paper|zero` and
  latents `"single"` against `"state"` stay, with no reason recorded (`goal_zero` has a real
  difference, R = 0 for goal latents only: write it down). `approach-b.md:31,133,190` still say
  rho_max.
- X11 partly: `train_high --out-dir`, `--run-dir` done; help shows `<cache>/...` not the real
  path; `train_b` writes its model with no flag; `diagnostics` CLI: `--env` without choices, 4
  arguments without help, default `runs/train_b/diagnostics`.
- X14: `require_files`, `medium_model` not used; 3 own skip blocks, `MEDIUM` and `same()` 3
  times each; `tiny_agent` adoption pending (3 own tiny agents remain).

New findings:
- E15 S. `diagnostics.py:103, 110`: `--value` accepts only `value-single` and `value-region`
  and reads the latent kind from the folder name, so the planned check of the ablation folders
  (`value-single-expectile0.7`, ...) cannot run. Fix: any folder; latent kind from its
  `flags.json`.
- E16 S (checked; f-report's file). `REPORT.md:605, 613` name `OptionConfig.rho_max = 0.9`,
  `gain_network` and "softplus(gain)", which no longer exist (`rho_cap`, `delta_network`,
  softplus(delta)). e-learner sends the renames to f-report.
- N: `diagnostics.py` has no test (index arithmetic of `value_grid`), `main()` takes no `argv`,
  bin width written two ways; `switch_targets` assembly is untested (swapping s and w passes);
  two places build "flags + seed + steps + commit" with the commit in different keys
  (`train_high.py:131-137`, `train_b.py:297-300`); `vast.sh:180` prints a cut usage; an
  unexported value folder fails with a bare `TypeError` on `rho_max` (`value.py:60`);
  `--edge-reward` default is a literal; `high_policy.py:20` imports `Agent` from `value`;
  `evaluate.py:487` names the removed `load_retrained_model` (d-eval); `tests/test_evaluate.py:384`
  skips unless `params.pkl` exists, which new-format folders lack (d-eval); the status file
  claims E14 and E4 done.

## Review findings, first pass (2026-10-29, commit 386b4e5)

Six read-only reviewers read every line of `fbplan/`, `analysis/`, `tests/` and `scripts/gpu/`;
I checked the must-fix items that are marked (checked) myself. `ruff check` passes everywhere;
`ruff format --check` would change 13 files (4 in `analysis/`, `controllers/oracle.py`,
`stats.py`, 5 test files, and others), which design.md §7 does not require.

M = must-fix, S = should-fix, N = nit. Each item: file:line, problem, proposed fix.

### Cross-component (X)

- X1 M (checked). `fbplan/evaluate.py:241-243`, `controllers/baseline.py:23-40`. The eval CLI
  always loads the provided checkpoint. `load_baseline_model` (`train_high.py:151`) and
  `load_b_model` (`train_b.py:261`) have no caller. Also, `BaselineController` has no `config`,
  so every retrained seed and every B variant would get the same `config_hash` (`c232a50c`):
  `--cluster config_hash` would make one cluster and drop the training variance that design.md §5
  requires. Fix (d-eval, with e-learner's loaders): `--checkpoint provided|baseline/<seed>|b/<variant>/<seed>`
  in `fbplan.evaluate`, used by `build_controller` and by `analysis/failures.py`; add the
  checkpoint to the hash payload only when it is not `provided`, so committed hashes stay.
- X2 S. The tabular study writes the option model and value iteration again, with other math:
  `tabular/study.py:70-91` (head "mean" averages F before the ratio; `OptionModel` averages the
  per-head ratios; trust uses the clipped rho, so rho ≥ 1 becomes trusted), `tabular/options.py:116`
  against `graph.py:75` (additive hop cost against a factor). No test pins them together. Fix
  (b-theory): build the study on `OptionModel`, `trusted_edges`, `value_iteration`; rerun it.
- X3 S. One name, several meanings: `rho_max` (planner trust bound 1.0, B value cap 0.9); R modes
  `exact|paper|zero` (A), `goal_zero|exact|paper` (B), `full|proxy|zero` (study); `head_agg` /
  `aggregate`; `hop_discount` (factor) / `hop_cost` (added); latents `"state"` / `"single"`; two
  classes named `PlannerConfig`. Fix: glossary in design.md §3, then rename (B cap `rho_cap`,
  study class `StudyPlannerConfig`, one R-mode vocabulary).
- X4 S. `region_latents`: same function in `landmarks.py:96` and `controllers/oracle.py:181`; a
  different construction with the same name in `learned/sampling.py:160` (trajectory window).
  Batched B is written 4 times (`landmarks.py:115`, `oracle.py:189`, `train_b.py:80`,
  `option_model.py:155`). Fix: c-planner keeps one public `region_latents` and one batched-B
  helper; d-eval imports them; e-learner renames to `trajectory_window_latents`.
- X5 S. `analysis/planner_routes.py:52-97` copies `parse_options` (weaker: accepts an item
  without `=`), the reset and the episode loop of `evaluate.py`, and seeds resets with
  `SeedSequence([seed, task_id])`, so the plotted episode is in no eval CSV. Fix (c-planner,
  d-eval): d-eval makes the reset and `RecordingController` (from `failures.py`) public; the route
  script uses `parse_options`, `make_tasks`, `run_episode`, `reset_seed`.
- X6 S. The planner's `Controller.name` (`graph_planner`, `graph_plus_high`, `planner.py:101,
  324`) differs from its CLI alias (`planner`, `planner_high`). The CSV `method` column then
  differs from `--method`, and `tables.py --reference planner` finds nothing. Fix: c-planner sets
  the name to the alias; d-eval tests `name == alias` for every alias.
- X7 S. Maze geometry: `MazeGeometry` sits in a controller module (`oracle.py:51-170`) but
  `analysis/calibration.py` and `failures.py` use it; `planner_routes.py:115` reads OGBench
  private attributes to draw walls that `calibration.py:97` also draws; `tests/test_eval_controllers.py:21`
  holds a third copy of the medium map. Fix: d-eval moves it to `fbplan/maze.py` with one
  `draw_walls`; b-theory adds one test that compares all 4 `GridMaze` maps and tasks with OGBench.
- X8 S. Cache folders and maze names: `~/.cache/fbplan/<kind>` with an env var is written 3 times
  (`frozen.py:109`, `train_high.py:47`, `train_b.py:69`); `env_name.split("-")[1]` 6 times;
  `vast.sh:94,104,133` ignores the env vars. Fix: a-frozen exposes `maze_name` and `cache_dir`;
  others import them.
- X9 S. Checkpoint I/O: two pickle writers (`train_high.py:121`, `train_b.py:141`), the example
  batch built 3 ways, `learned/value.py:77` copies `frozen._forward`. The docstring at
  `train_high.py:124` says `fbplan.frozen` can load the folder; `load_model` cannot. Fix:
  a-frozen exposes `save_numpy_pickle`, `example_batch`, `forward`; e-learner uses them.
- X10 S. `analysis/calibration.py:51-62, 154-159` writes the planner's trust rule and the xy cap
  again (defaults 0.3 and 6.0 copied). Fix: call `graph.trusted_edges`, take defaults from
  `PlannerConfig()`.
- X11 S. CLIs: `--out` means a CSV, a checkpoint folder or an output folder; output folders are
  `--out-dir`, `--dir` or `--run-dir`; default outputs are `runs/`, `runs/c-planner` (a branch
  name), `results/figures` or `docs/figures/theory`; `--workers` defaults are 1, 5, 10 and
  `os.cpu_count()` (design.md §7 allows 4). Fix: `--out` = file, `--out-dir` = folder,
  workers default 4, run outputs to `runs/<script>/`.
- X12 N. Import cycle: `train_high.py` imports `fbplan.learned`, `learned/train_b.py` imports
  `train_high`; `SAMPLER_STREAM` and `Agent = Any` defined twice. Fix (e-learner): move shared
  helpers to `fbplan/learned/checkpoint.py`.
- X13 N. Two definitions of "wormhole", both functions named `wormhole_mask`
  (`tabular/study.py:149`: rho > 2γ^d; `calibration.py:51`: faster than the ant can walk).
  theory.md and design.md quote rates of each. Fix: distinct names, both definitions in §3.
- X14 N. Test helpers copied: the skip block for absent files (`test_evaluate.py:254`,
  `test_learned_b.py:202`, `test_learned_train_high.py:136`), `MEDIUM` in 4 files, `same()` twice
  with different contracts, the tiny upstream config twice. No `conftest.py`, so the real model
  loads once per module. Fix: a `tests/conftest.py` with session fixtures and skip helpers
  (needs an owner).
- X15 N. `pyproject.toml`: see request 2.
- X16 N. The "privileged" label is hard-coded twice (`tables.py:24`, `plots.py:47`), on the
  display label, so `oracle@<hash>` and sweep rows are not marked. Fix (d-eval): one constant,
  applied to the raw `method` column.
- X17 N. design.md §6 lists no owner for the tests of c-planner, d-eval, e-learner, or for
  `pyproject.toml`; it lists `controllers/oracle_teleport.py` and `results/{giant,teleport}/`,
  which do not exist yet.

### coordinator: `fbplan/contracts.py`, `tests/test_contracts.py`

- K1 M (checked). `contracts.py:24`: `* np.sqrt(latent_dim)` turns float32 numpy input into
  float64 (NumPy 2 promotion); the JAX path stays float32. Callers (`landmarks.py:87`,
  `learned/sampling.py:155`) get float64. Fix: `math.sqrt(latent_dim)`; a dtype test for both.
- K2 M. `contracts.py:57-66`: no production caller passes `temperature` or `seed` to
  `low_action` / `high_latent` (eval temperature is 0, design.md §5). The parameters keep
  `_sampling_key`, a guard in the tabular fake and 3 tests alive. `seed` is a PRNG key here and
  an int in `Controller.reset`. Fix: remove both, or rename `seed` to `key` and name the user.
- K3 S. `contracts.py:33` calls `z_r` the "raw task embedding", but "raw" means "not normalized"
  in `FBModel.backward`, and `infer_task_latent` returns an upstream-normalized vector. Fix:
  reword.
- K4 N. `TaskSpec` (`contracts.py:27`) is `frozen=True` but `==` raises and `hash` fails on its
  array fields. Fix: `eq=False`. `contracts.py:91`: "Optional" on a required Protocol method.
  `tests/test_contracts.py` has no module docstring.

### a-frozen: `fbplan/frozen.py`, `tests/test_frozen.py`

- A1 M. `frozen.py:231-232, 271-272`, `test_frozen.py:209-210`: the `z_r` norm is given as
  "0.92-0.98 sqrt(d)" (medium only; design.md §2 says 0.79-0.98) and the action change as "about
  1e-3" (measured: mean 0.002, max 0.023, and 14 points of success on large). The explanation is
  written 4 times. Fix: one place, correct range, point to design.md §2.
- A2 M. `frozen.py:215` `_example_batch` and `:302` `_forward` are private, so e-learner copied
  them (X9). Fix: make them public, un-jitted.
- A3 M. design.md §6 gives a-frozen "checkpoint download (frozen and trained)". Only the frozen
  download exists. `reproduce.sh` needs the trained one for its default path (brief step 1).
  Needs a hosting decision for the trained checkpoints.
- A4 S. `frozen.py:370-373`: the CLI reads `sys.argv`; `python -m fbplan.frozen --help` prints
  a traceback. Fix: argparse with `choices=ENV_NAMES`.
- A5 S. `test_frozen.py:41-49`: the `observations` fixture resets with `seed=task_id` only, so
  the start noise depends on the global RNG and on test order (design.md §5 recipe not applied).
- A6 S. `test_frozen.py:159-171`: `rho(w,w) = X/X` is 1 by construction; the name claims more.
  Fix: test batch-order independence directly and rename.
- A7 S. `test_frozen.py:259`: `norm(z_r) <= sqrt(d)` also passes for the 1e-7 raw vector. Fix:
  assert a band.
- A8 S. `frozen.py:211, 237`: callers must `device_put` the agent (`train_high.py:148` repeats
  it). Fix: do it in `FrozenFBModel.__init__`.
- A9 N. `frozen.py:140-155`: a hash mismatch names the deleted `.part` file. `test_frozen.py:273`
  reads back a private attribute; `:125` hard-codes 36 leaves. `load_numpy_pickle` docstring does
  not say that it refuses other classes. The "no partial file" guarantee has no test.

### b-theory: `fbplan/tabular/`, `tests/test_tabular_*.py`, `docs/theory.md`

- B1 M. Duplicate option model and planner in the study (X2), and `study_plots.py:192` rewrites
  `study.wormhole_mask` in log space.
- B2 M (checked). Dead parameters: `options.py:157` `replanning_policy(subgoals)` is never read;
  `options.py:120` `graph_value_iteration(hop_cost)` is never passed (the study adds it to R).
- B3 M. Dead branch: `fake_fb.py:63, 226` `high_advantage="full"` is set by no job and no test.
- B4 M. `maze.py:161`: the docstring says the fine cell "contains its centre (lower-left of the 4
  for even k)"; the code takes offset (k//2, k//2), the +x/+y cell, which does not contain the
  centre for even k.
- B5 M. `tests/test_tabular_theory.py:57`: `_graph_inputs` docstring describes return values that
  it does not return; no type hints.
- B6 M. `run_study.py:3` says "about 10 min on 8 cores"; theory.md and the status say 1 min on 12.
- B7 S. `run_study.py:196-198`: `--quick` overwrites the committed CSVs in `docs/figures/theory/`
  by default; `--workers` defaults to all cores (design.md §7: 4); no help text.
- B8 S. `options.py:134`: `iter(int, 1)` loops without a limit; with unclipped rho ≥ 1 the values
  diverge and the loop never ends. Fix: `max_iters` and raise.
- B9 S. Unknown `aggregate`, `low_policy`, `high_advantage` values fall back silently
  (`study.py:79`, `fake_fb.py:187`). Fix: `__post_init__` checks, `Literal` types.
- B10 S. `docs/theory.md:37-38, 69-70, 105` cites checks that no test makes (ρ ≤ γ^d, the proxy
  gap "in the same test", the counts 7/26 and 20/45). Fix: add the assertions or reword.
- B11 S. `study.py:70` requires `z_r` for the task-free ρ̂ (callers pass a dummy latent);
  `reward_rows` builds the same plan 3 times.
- B12 S. `study_plots.py:22, 28-35`: planner names, `FAR = 11` and the guarded config copy
  constants of `run_study.py`.
- B13 S. Names: the direct value is `v_direct`, `v_final` or `v_d`; V_H is `v_graph` or
  `v_land`; R is `r_before`, `r_hat` or `r_eff`. theory.md uses V_d and V_H.
- B14 S. Tests that can pass for the wrong reason: `test_tabular_fake_fb.py:131` (ties in the
  optimal policy), `test_tabular_study.py:96` (restates the formula), `test_tabular_env.py:17`
  (compares the env with the table it reads).
- B15 N. Missing docstrings on `plot_*` and `TabularFB.task_spec`; small duplicate helpers in
  `fake_fb.py` and `options.py`; `matplotlib.use` at import (6 `noqa`); the study timeout rule
  "+2" is not documented.

### c-planner: `option_model.py`, `landmarks.py`, `graph.py`, `controllers/planner.py`, `analysis/planner_routes.py`, tests

- C1 M. `option_model.py:6, 62, 67`, `graph.py:4`, `planner.py:322`: docstrings write
  `rho = F(s,B(w))ᵀB(w) / …` and "raw B(w)"; with region latents the code uses the region mean.
  Fix: write `z_w` everywhere and say once what region latents measure.
- C2 M. Duplicates in `planner_routes.py` (X5) and `landmarks.py` (X4).
- C3 M (checked). Dead code: `TaskPlan.reward`, `TaskPlan.direct` (`planner.py:93-94`);
  `step_info()["rho_waypoint"]` (`planner.py:177`, read by nothing; it is the only reason that
  `_choose` returns a tuple); `GraphSolution.iterations` (`graph.py:28`); `Landmarks.dataset_index`
  (`landmarks.py:56`, read only by its own test); public `landmarks.region_latents` used only by
  tests.
- C4 M. `tests/test_planner.py:347` `TaskPolicy` copies `NonHierController`.
- C5 M. `tests/test_planner.py:213-214`: name and comment describe the old clip to 1 ("rho at
  one"); distinct states are now clipped to γ and the raw value is 1.5.
- C6 S (checked). `planner.py:254`: the `while` loop in `_advance` has no bound; a cycle of tied Q
  values loops forever inside `act()`. `extract_route` guards the same case. Fix: bound it by the
  landmark count.
- C7 S. `_solve` accepts a value iteration that did not converge, without a signal (`graph.py:28`,
  `planner.py:192`).
- C8 S. `planner.py:235-279`: `_choose` and `_stop_values` repeat one Q pipeline. `pair_table`
  returns an all-zero reward in "zero" mode and `_solve` branches on the mode to get `None`.
- C9 S. `planner.py:77`, `landmarks.py:40`: `top_k` and `space="backward"` appear in no design
  decision and no result, but cost code and tests. Fix: an ablation that uses them, or remove.
- C10 S. `planner_routes.py:175-179`: no help text; `--seed` is ambiguous (reset or landmark
  seed); default output `runs/c-planner`.
- C11 S. `tests/test_planner.py:145, 225, 276`: `fb.cells[DIRECT]` with DIRECT = -1 gives the
  goal cell, so a wrong fallback to the task policy passes. `:129`: `rho_reached` has no effect
  under the default `reached_xy`.
- C12 N. Duplicate test constants (`all_cells`, two names for one cell); tests of default values
  only; missing docstrings (`Landmarks`, `Targets.take`, `PlannerConfig` fields); landmark tests
  live in `test_graph.py`.

### d-eval: `evaluate.py`, `stats.py`, `controllers/{__init__,baseline,oracle}.py`, `analysis/`, tests

- D1 M. No eval path for trained checkpoints, and one config hash for all of them (X1).
- D2 M (checked). `analysis/tables.py` pairs any two CSVs with the same episode keys.
  `results/medium/upstream.csv` has the same keys as `baseline.csv` but unseeded start states, so
  the README command `tables.py results/medium/*.csv` prints a paired test that is not valid.
  Fix: mark upstream rows as unpaired (show them in the success table, skip them in the paired
  table, print the unpaired difference from `stats.unpaired_comparison`).
- D3 M (checked). `analysis/plots.py:33`: `zip(..., strict=False)` drops every method after the
  6th, then `colors[method]` raises `KeyError` (for example with 5 planner landmark sets).
  The docstring promises fixed colors, but extra colors depend on the other methods. Fix: cycle
  the palette.
- D4 M. `analysis/plots.py:76-77`: without `--label-root`, all configs of one controller merge
  into one dot (the 12 sweep CSVs become one "oracle"); `tables.py` splits them. Fix: one shared
  load-and-label helper.
- D5 M. `analysis/parity.py:33-34`: the docstring says reruns give the same episodes; upstream
  `evaluate` resets without a seed, so they do not. Line 37 repeats `make_tasks`.
- D6 M (checked). `controllers/oracle.py:5, 11-12`: stale numbers (48/58/66 and 79/87/82; the
  committed sweep gives 44.8/56.4/65.2 and 80.8/86.4/78.0) and "estimates the best that a
  high-level method can get" (design.md §5: not an upper bound).
- D7 M. Duplicates: `oracle.py:181` `region_latents` (X4); `failures.py:73` repeats
  `build_controller`; `calibration.py:61, 156` the trust rule (X10); `calibration.py:101` the
  wall drawing (X7); `evaluate.py:82` reset and `:227` options copied by c-planner (X5).
- D8 M. `tests/test_evaluate.py:220-224`: docstring "replace with frozen.make_env after a-frozen"
  is stale; the helper is a second way to build the env (also `test_eval_controllers.py:109`).
- D9 S (checked). `analysis/failures.py:102-107`: a CSV with no failure gives
  `ProcessPoolExecutor(0)` and a `ValueError` (`reproduce.sh --smoke` guards against it for now);
  a CSV with 2 configs fails with an unclear unpack error; default `--workers 10` (parity: 5).
- D10 S. `evaluate.py:248-254`: `--method` help lists 3 of 6 aliases; `--env` has no `choices`;
  default `--seeds 0` while the protocol uses 5. `evaluate.py:83-84`: the comment says the
  settling actions randomize the start; they change only the goal observation.
- D11 S. `analysis/calibration.py:119-171`: `main` is 53 lines and builds a Markdown table by hand.
- D12 S. `controllers/baseline.py:52`: `RawBaselineController` needs `FrozenFBModel.agent` but
  types `FBModel` with a `type: ignore` (no type checker runs). `oracle.py:164` `MazeGeometry.path`
  is used only by tests.
- D13 S. `tests/test_stats.py:70` computes the expected McNemar value with the implementation's
  formula. Fix: the hand value 93/128.
- D14 S. `analysis/tables.py:104`: when the reference label is absent, the paired section
  disappears without a message.
- D15 S. No test covers `analysis/`; findings D3, D4, X16 went unnoticed. Fix:
  `tests/test_analysis.py` for the pure functions.
- D16 N. `evaluate.py:106` parameter `config_hash` shadows the function; `make_oracle(**options)`
  untyped; 3 baseline classes repeat init/reset/step_info; `test_evaluate.py:272` checks 3
  observations, not 4.

### e-learner: `train_high.py`, `fbplan/learned/`, `scripts/gpu/`, tests

- E1 M (checked). `scripts/gpu/vast.sh:26` (also 2, 18-20, 25): the default instance label
  holds the user's surname; comments describe a shared team account and an internal tab name. The repository will be public. Fix: neutral default (`fbplan`) or require
  `VAST_LABEL`; neutral comments. (`docs/status/e-learner.md` also names the account; see the
  process-file proposal.)
- E2 M (checked). `tests/test_learned_option_targets.py:148-171`: the test iterates
  `max(v1, max_w y)`, which is the graph value iteration itself, and never calls
  `value.switch_targets`. No test checks the production target
  `y = R + ρ·(V_1(w) + gain(w)) − hop_cost`. Fix: a pure `switch_target(...)` in `value.py`,
  iterated in the test with a hop cost and the valid mask.
- E3 M. `learned/value.py:72-74` `ValueState.value` is called only by a test; `high_policy.py:30`
  computes V_H again.
- E4 M. Duplicates: `value.py:77` forward, `train_b.py:252` example batch, `SAMPLER_STREAM`,
  `baseline_dir`/`approach_b_dir`, `save_value`/`save_checkpoint`, `latent_table` batched B
  (X8, X9, X12, X4).
- E5 M. `learned/value.py:1, 8, 44, 67-70`: "gain" names the raw network output and its softplus.
  Fix: raw output `delta`, `gain = softplus(delta)`.
- E6 S. `high_policy.py:53-56`: logged `adv`, `adv_std`, `saturated` include invalid edges, so
  `adv_std` in train.csv is not the calibrated quantity (`train_b.py:227` masks).
- E7 S. `train_b.py:332`: the value folder is always `value-<latents>/seed<k>`, so an ablation
  (`--edge-reward exact`, `--rho-max 1.0`, `--expectile 0.7`) overwrites the main V_H, and the
  policy stage loads whichever ran last. Fix: `--name` / `--value-name`.
- E8 S. `train_b.py:191-206, 106-123` and `train_high.py:76-95`: three copies of the jitted scan
  chunk and validation pattern. Fix: one `scan_updates` in `loop.py`.
- E9 S. `train_b.py:337-393`: `train_policy` is 57 lines with 8 jobs; jitted functions take 8-10
  positional arguments. Fix: a `Split` pytree and one static config dataclass.
- E10 S. CLIs (`train_b.py:269-290`, `train_high.py:168-176`): no help text; `--out` help names a
  function; choices repeat the `Literal` types.
- E11 S. Missing docstrings: `value_chunk`, `value_validation`, `config_from_dict`,
  `policy_chunk`, `policy_validation`, `base_flags`, `train_value`, `train_policy`, `ValueState`,
  `OptionTerms` and others.
- E12 S. `vast.sh:4-5`: `jq` is used but not listed; the cache paths ignore the env vars.
- E13 S. `tests/test_learned_b.py:191`: the alpha test restates the formula; `calibrate_alpha`
  and `advantage_stds` have no test.
- E14 N. `scripts/gpu/run_seeds.sh` is not executable in git (100644); comment "needs its own
  test" in `option_targets.py:76`; tests pass one array as two different arguments.
