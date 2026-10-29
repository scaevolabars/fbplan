# Status: g-release

Last update: 2026-10-29.

## Done

- `scripts/reproduce.sh`: one entry point with the stages checkpoints, train, eval, tables,
  figures; `--smoke` and `--workers N` (default 4, design.md §7). It prints every command that it
  runs. Bash 3.2 compatible (macOS default shell).
  - The run table (`RUNS`) reproduces every committed CSV in `results/medium/`. Checked: the
    options of the 13 oracle CSVs give the config hashes in the files.
  - Training writes to `~/.cache/fbplan` as before. `--smoke` sends trained models to
    `runs/smoke/models/` (`FBPLAN_BASELINE_DIR`, `FBPLAN_APPROACH_B_DIR`) and all outputs to
    `runs/smoke/`, so it never touches `results/`, `docs/figures/` or the real models.
  - Tables go to `results/<maze>/tables.md`. `upstream.csv` stays out of `tables.py`: its
    episodes have the same keys as ours but other start states, so `tables.py` would pair them
    (finding D2).
  - `--smoke` passed in this worktree (all 5 stages, 30 output files; an older version with
  the parity run). It took 17 min while six
    review agents used the CPU. Step times on an idle CPU: training 102 s (3 runs), one eval run
    about 30 s (model load, task latents, JIT), route plots 58 s, quick study 36 s. The parity run
    (`analysis/parity.py`) took 310 s even with 4 workers, because it always runs 50 episodes per
    task, so `--smoke` skips it (request 5).
- `devbox.json`: uv pinned to 0.11.8 (the version that wrote `uv.lock`), `--locked` in every
  script, new scripts `reproduce` and `smoke`.
- `.github/workflows/ci.yml`: `uv sync --locked`, `ruff check`, `pytest -ra` on ubuntu-24.04 with
  `MUJOCO_GL=osmesa`. It downloads packages only, no data. Not run yet: no remote, and Docker is
  not running on this machine. Checked locally: with an empty `HOME` (no checkpoints, no
  datasets), `pytest` gives 247 passed, 30 skipped, and downloads nothing.
- Code review, first pass: findings below.

- README: the Reproduce section points at `scripts/reproduce.sh`; the long list of analysis
  commands moved into the script; the layout table lists `fbplan/controllers/` and `scripts/`.

- Clean clone (commit f058ae6, cloned without `--recurse-submodules`, empty
  `FBPLAN_CKPT_DIR`): `scripts/reproduce.sh --smoke` passed in 5.5 min on the 12-core Mac with
  the datasets already in `~/.ogbench/data`. Stages: checkpoints 38 s (venv and download),
  train 81 s, eval 53 s, tables 41 s, figures 98 s. The script initialized the submodule, and
  the clone stayed clean (outputs only in the ignored `runs/`).

## Next

- Add rows to `RUNS` when new result CSVs are committed (see requests).
- Process files (brief step 4): proposal to the user.

## Blockers

- The full pipeline cannot reproduce the retrained baseline or approach B yet: the eval CLI has
  no way to load a trained checkpoint (D1), and trained checkpoints have no download (A3).

## Requests to the coordinator

1. For `scripts/reproduce.sh` to stay complete, every component that commits a CSV to `results/`
   writes the exact command in its status file. I add it to `RUNS`. Pending result sets that I
   know of: planner and planner_high (5 eval seeds, landmark seeds), planner with region latents,
   baseline_raw, retrained baselines, approach B, large, giant and teleport.
2. `pyproject.toml` has no owner in design.md §6. Proposed changes (X15): `addopts = "-ra"`
   (skips show on a clean clone), remove the unused `h5py`, ruff `per-file-ignores` for `E402`
   in the matplotlib scripts (20 `noqa` comments), and `pythonpath = ["tests"]` for
   `planner_fakes`. I can apply them if you give me the file.
3. design.md: a glossary in §3 for the names that differ across A, B and the tabular code (X3),
   both wormhole definitions (X13), and the gaps in the §6 table (X17).
4. Approve or reject each finding below, then send them to the owners.
5. d-eval: an `--episodes` option for `analysis/parity.py` (default 50), so that `--smoke` can
   run the parity check with 1 episode per task.

## Proposal: process files before submission (brief step 4, waits for the user)

Recommendation: remove `docs/briefs/` and `docs/status/` from the submitted tree.

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
