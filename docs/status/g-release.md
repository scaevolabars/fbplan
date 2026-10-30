# Status: g-release

Last update: 2026-10-30. Branch merged with `main` at 8b4795f.

## Finishing phase (coordinator, 2026-10-30)

Items: (1) final review pass over the whole repository, (2) README and `reproduce.sh` with the
recorded route (H1), the 10-seed protocol, wave 3 and the results pointer, (3) a clean-clone
check after the final merges, (4) the removal of the process files, prepared, not committed.

- (1) Done: section "Review findings, final pass" below (5 read-only reviewers, one per owner;
  I checked the items marked "checked" and the ones in my own files).
- (2) Done, see "Done" below. `reproduce.sh --smoke` passed in 8 min (train 111 s, eval
  336 s, tables 7 s, figures 63 s; datasets cached); `results/` unchanged. Medium reference runs, approach A
  (landmark set 0) and the oracle sweep: 10 of 10 episodes equal `results/macos/`; the approach
  A hashes differ (X18, decided). Trained-model runs differ as expected (smoke models).
  `heldout_selection.csv` and `platform_check.csv` rebuild byte for byte;
  `heldout_selection_awr2x.csv` differs (stale, D19).
- (3) Waits for the final merges.
- (4) Prepared: section "Process files: removal" below, with the facts that must move first
  and their target files.

## Done

- Merged `main` 8b4795f. ruff passes; pytest 501 passed.
- `scripts/reproduce.sh` covers every result of REPORT.md now:
  - `RUNS`: one line per committed evaluation CSV, with `{a..b}` for seed and landmark-set
    lists. It expands to 228 lines, exactly the 228 episode CSVs in `results/` (without
    `macos/`, `failures/`, `upstream.csv`), checked with the script's own expansion. For every
    line, the eval seeds, episodes per task, held-out task ids, checkpoint and env equal the
    CSV columns (228 of 228, checked). The config hash equals the CSV for all 46 lines whose
    config builds without data (oracle sweep, recorded route, trap-aware, teleport oracle;
    checked); the ensembles' hashes reproduce (d-eval reviewer); the approach A lines keep the
    hash of 84a8127 (X18, decided).
  - Commands come from the status files of c-planner, d-eval, b-theory and e-learner. Columns:
    CSV below `results/`, maze, eval seeds, episodes, method, more `fbplan.evaluate` arguments.
    `ROUTE`, `TRAP`, `TELEPORT_ORACLE` hold the `module:attribute` paths that have no alias.
  - Row order: the canonical Linux runs on the eval tasks ran one process per eval seed and
    were joined in seed order; a single `--seeds 0..9` run writes rows in task order (same
    records). The script does the same as the canonical runs: one process per eval seed, then
    a join, and a seed that an earlier line with the same command ran is reused. So
    `results/medium_10seeds/` costs only its seeds 5-9, and every CSV gets the committed row
    order. Held-out runs are one process (as committed).
  - `eval` also replays the 16 failure-class CSVs (output paths equal the committed ones,
    checked) and the parity run.
  - `train`: every family that the results use, 5 seeds each, with e-learner's commands:
    retrained `π_h`, `awr2x`, `logratio`; critics `value-single`, `value-region`,
    `value-single-expectile0.7`; policies `clean`, `wide`, `region`; `calibrate`, then
    `clean-awr2x`. `--smoke` trains seed 0 of each for 10 steps.
  - `tables`: every table of REPORT §11 and the main tables per maze into
    `runs/reproduce/tables.md`; it rewrites `heldout_selection.csv`,
    `heldout_selection_awr2x.csv` and `platform_check.csv` (in `--smoke` from the committed
    CSVs to `runs/smoke/`, with a byte comparison). Checked: the 25 table commands ran on
    the committed CSVs in 4 min without an error; the recorded-route rows equal d-eval's
    (medium 83.8 %, +5.3 / +3.3 / +2.2 pp; large 41.2 %, +3.5 pp).
  - `figures`: the 4 success figures to `docs/figures/report/` (explicit file lists; the
    REPORT §11 glob fails on medium, F12), time to goal, calibration (medium, large; giant and
    teleport to `docs/figures/theory/`), held-out goals, tabular study, time limit, and the
    critic check of B when the critics exist. Fixed: `study_plots` takes a positional folder,
    `time_limit --out-dir`, `report_figures.py` a positional folder (B24, F13).
  - `--smoke`: medium only, the first run of each folder below `results/medium/` (18 runs,
    the recorded route and the held-out path included), ensembles of one member, eval seed 0,
    2 episodes. It prints, per run, the episodes equal to the committed run and whether the
    config hashes are equal.
  - Bash 3.2 compatible (an empty array under `set -u` is guarded); shellcheck clean.
- README: results pointer to REPORT.md at the top; methods table with the recorded route, the
  ensembles, trap-aware control, the teleport oracle and all checkpoint families; the map from
  alias to the CSV `method` name (X6); protocol (eval seeds 0-4, 0-9, 10-19; held-out goals
  for every choice; the two-row rule); a table of the `results/` folders; platform with the
  hardware and the evidence of `results/platform_check.csv`; "Code versions" (X18 decision);
  time of the eval stage (about 244k episodes, 9 h on the 128-core machine); the smoke check;
  the H1 command as the single-run example; training; `vast.sh` does not destroy by itself
  (E19); the published checkpoints are `baselines-v1` only (A13, E17).
- `tests/conftest.py`: the `TinyAgent` docstring named a removed helper; now
  `fbplan.frozen.example_batch(tiny.agent.network.params)` (E27, checked).

Earlier work (first and second review passes, CI, devbox, X15, X14, the platform move, the
release check of `baselines-v1`) is in git history and in the sections below.

## Next

- (3) Clean-clone check of the README after the final merges: clone, `uv sync`,
  `fbplan.frozen --trained --datasets`, `pytest`, `reproduce.sh --smoke`.
- When aliases for the recorded route, trap-aware and teleport oracle land (C20, request 2):
  replace `ROUTE`, `TRAP`, `TELEPORT_ORACLE` in `RUNS` and the README methods table.
- When more trained checkpoints are published (request 1): update the README setup text.
- (4) The removal commit, when the coordinator says so.

## Blockers

- None.

## Requests to the coordinator

1. Release (E17, A3; public upload, so the user decides): 30 trained checkpoints that 90
   committed CSVs use are not published: `baselines/medium/{awr2x,logratio}` and
   `approach_b/medium/{clean,wide,region,clean-awr2x}`, seeds 0-4 (`high_actor.pkl`,
   `flags.json`). The critic check of REPORT §8.5 also needs
   `approach_b/medium/{value-single,value-single-expectile0.7}`, seeds 0-4. All 40 folders are
   in the local cache; their `flags.json` hold no personal path. Do not upload the old
   `params.pkl` of clean, wide, region. Without a release, a reviewer without a GPU cannot
   rerun these rows. Recommendation: a release `trained-v2` with these 40 folders, pinned by
   a-frozen.
2. Aliases (d-eval's file `fbplan/controllers/__init__.py`, C20, X11 rule 8):
   `recorded_route`, `trap_aware`, `oracle_teleport`; and `oracle_teleport` in `PRIVILEGED`
   (D23). The hash uses `controller.name` and the config only, so no committed hash changes.
3. X18 closed by the coordinator's decision (a): README "Code versions".
4. Git history: removing the process files from the tree does not remove them from history.
   History holds the instance label with the user's surname and the vast.ai team account name
   (in `docs/status/`, and in `scripts/gpu/vast.sh` before E1). If the published history is
   rewritten, the commit ids that the README and REPORT cite (84a8127, 0de6e0a, the training
   commits in each `flags.json`) dangle.
5. design.md §8 fixes a medium replication on eval seeds 10-19; no CSV exists (F4). Run it, or
   the report states it as not run.

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

## Process files: removal (prepared, not committed)

The commit removes `docs/briefs/` (8 files), `docs/status/` (11) and, if the coordinator
decides so, `docs/hypotheses/` (4). My files (README, `reproduce.sh`, CI, `pyproject.toml`,
`devbox.json`, `tests/conftest.py`) refer to none of them. The commands, when the coordinator
says so:

```
git rm -r -q docs/briefs docs/status            # + docs/hypotheses if decided
git grep -n -E 'docs/(status|briefs|hypotheses)' -- . ':!third_party'   # must print nothing
uv run --locked ruff check . && uv run --locked pytest
git commit -m "Remove the process files before submission"
```

**Dead references today** (the owners fix them before the commit):
- REPORT.md:1135, 1156, 1206, 1223, 1282, 1286, 1325, 1326, 1352, 1409, 1496, 1560 (f-report).
  1560 points to `docs/status/c-planner.md` for the H1 commands: point to `scripts/reproduce.sh`.
- docs/design.md:35, 163, 164, 177, 218, 221, 250, 258, and §6-§7 (tabs, coordinator,
  status files) (coordinator).
- docs/theory.md:320, 613 (status), 527, 576, 592, 604 (h-exec) (b-theory).
- fbplan/contracts.py:4 ("write the request in your docs/status ... tell the user")
  (coordinator).
- fbplan/controllers/baseline.py:11 (h-learn H3) (d-eval); scripts/gpu/awr2x_v1.sh:2 (h-learn)
  (e-learner).
- Names of process files or tabs in code: "h-papers H1" (graph.py:13, 133; planner.py:16, 439,
  671; test_planner.py:577; test_graph.py:160), "d-eval", "b-theory" (landmarks.py:10,
  planner.py:92, planner_fakes.py:9, test_option_model.py:132, test_planner.py:530, 700)
  (c-planner, C19); "h-learn H1/H2" (logratio.py:1, 25; logratio.sh:2; value.py:47;
  test_learned_b.py:178), "review X4" (train_b.py:82), tab names in approach-b.md (e-learner);
  "h-exec H1, M11" (trap_aware.py:1, 10) (b-theory); "plan B decision 8"
  (heldout_goals.py:1, heldout_selection.py:1) (d-eval).

**Facts that must move first**, by owner (source → target):
- a-frozen: `|z_r|` per task (a-frozen.md:278-282) → design.md §2 in place of the pointer at
  :35. `baselines-v1`: contents, training commits (medium 9557a55, large a189a49), the
  modules-only decision and the upload checks (a-frozen.md:170-220) → a comment at
  `frozen.py` `_TRAINED_FILES` or the README. Stack details (tfp-nightly pinned in uv.lock and
  kept on PyPI, distrax needs it, dm-control 1.0.20 with the MuJoCo pin) → design.md
  "Environment". design.md §2:45-46 ("large 31-45 %, 1 seed") → cite the 5-seed Linux rows.
- b-theory: the giant, teleport and T3 commands → done in `reproduce.sh`. The T3 tie rule, the
  control definition, the `F_goal` deviation from h-exec (top 64 pool states, 2.4-3.3 xy), the
  data checks (jumps ≥ 2.27, walks ≤ 0.37, 10-14 false alarms, AUC 0.991) and h-exec's M11
  number → theory.md §11.4. The gate-check and raster scripts (now in /private/tmp) →
  `analysis/` with commands (B23).
- c-planner: the H1 tie order (point < window < high, then smaller K), the falsification rule,
  the medium tie (window K150 = high K60 = 533/600), the open choices of the controller (goal =
  argmax of min_h F(w,z_r)ᵀz_r; walk choice; 100k pool for the first hop), its limits (C13) and
  the "H1 against the oracle on large" analysis → REPORT §8.7. `switch_steps` tested, not a
  default (held-out −2.0 / +5.3 pp) → design.md §4.A. Code commit per CSV group and the
  record-identity checks after c187bb7 and 06ea0a8 → REPORT appendix. Ablation option strings
  → done in `reproduce.sh`.
- d-eval: all evaluation commands and the join rule → done in `reproduce.sh`. Code commit of
  each result group (84a8127: Linux 5-seed runs, A, ablations, baseline seeds 5-9; 0be1249: T1;
  6c603a3: T4, 2×, T2, T3; 8a202e6: act-ahead; 5d7e642: H1 large; 2626146: wave 3; e6b99d1: B
  clean export) and the seed-0 identity check before each group → REPORT §11. Hardware → done
  in README. The held-out goal rule (≤ 50 % flipped states per goal ball; 6 of 10 large and 3
  of 10 giant goals changed) → design.md §5. The act-ahead pairing check (1,250 of 1,250
  records equal) → REPORT §8.4. Evaluation compute (about 9 h, about $8.3) → REPORT
  (optional; no account name, label or instance id).
- e-learner: training commands per family → done in `reproduce.sh train`; commit per family
  (baselines 9557a55 medium, a189a49 large; critics and clean, wide, region a189a49; awr2x and
  clean-awr2x 97d9200; logratio 34b17a5) → README or REPORT §11. GPU compute (one RTX 3090,
  13.9 h, about $2.90; 55 min per 5 seeds; 2.5 GB per B run) → REPORT (no account name or
  label). 2× α = 0.2099 (median spreads 16.49 / 15.72 over 200 batches, on the GPU; CPU gives
  another value) → REPORT §8.5 in place of the link at :1156. ρ measurements (median 0.78,
  raw ρ ≥ 1 on 9.7 %, ρ > 0.9 on 33 %, exact R 10-90 % from −8.8 to +29) → approach-b.md §5.1.
  Retrained `π_h` cosine 0.955 with the provided one → REPORT baseline section. T2 cross-check
  (max diff 2.2e-5) → REPORT T2 paragraph or a test (E20). Cancelled runs (ρ_cap 1.0, the
  policy on expectile 0.7, h-learn H1) → REPORT and approach-b.md.
- f-report: the figure commands with explicit file lists and how the tables are made → REPORT
  §11 (or a pointer to `reproduce.sh`). Items that will not land (route plots, large retrained
  row, wall-crossing count) → REPORT §10 as not done.
- coordinator: design.md §6-§7 describe the workflow; replace them with a short "repository
  layout and code rules" section, and the §8 citations of `docs/hypotheses/` with the facts.

**`docs/hypotheses/`.** REPORT §8.6-§8.7 and design.md §8 cite these files as the record of
hypotheses and readings fixed before any result, and some REPORT numbers have no other source
(F10). 7 lines address the coordinator (h-exec.md:77, 230, §5; h-learn.md:368;
h-papers.md:125-127, §7). Recommendation: keep the folder and delete those lines and sections;
removing it needs the facts above moved and weakens the "fixed in advance" trail.

## Review findings, final pass (2026-10-30, main 8b4795f)

Five read-only reviewers, one per owner, checked the open findings of the earlier passes and all
code changed since (a-frozen, b-theory, e-learner since e77587e; c-planner, d-eval since
386b4e5). Each ran ruff and its owner's tests: all pass (a-frozen 58, b-theory 147, c-planner
130, d-eval 106, e-learner 60). "checked" = reproduced by running code. M = must fix,
S = should fix, N = optional.

### Cross-component
- X18 closed (coordinator, option (a)): the approach A CSVs keep the hashes of 84a8127; README
  "Code versions" says that a rerun at a later commit repeats the records with another hash.
  53 of 54 approach A CSVs differ at HEAD (checked); all H1, oracle, ensemble, trap-aware and
  teleport-oracle hashes reproduce (checked).
- X4 open, now actionable (d-eval): `controllers/oracle.py:37-55` copies
  `landmarks.region_latents` and `option_model.batched_backward`; `oracle_teleport.py:32`
  imports the copy.
- X6 accepted: CSV names differ from aliases; a rename changes 54 CSVs. README and the REPORT
  appendix map them.
- Platform (d-eval reviewer): pooled over the 36 runs, Linux − Mac = −0.76 pp [−1.4, −0.1]
  (−0.66 without planner and upstream). Per run "within noise" holds; a shift below 1 pp is
  not excluded. It is small against the H1 gains but the size of the +1.0 pp oracle margin;
  f-report may add it to REPORT §6.

### a-frozen
- Fixed: X14 and most N items. A12 rejected, reason OK. A3 partly: only `baselines-v1` is
  pinned (20 of 20 hashes equal GitHub's digests, checked); see request 1. A10 partly (the
  e-learner side, E18). A11 open (d-eval side).
- A13 S (`frozen.py:668`): the `--trained` help says it gets "retrained baselines, approach B".
  Fix: name `baselines-v1` and the families that are not published. (README fixed.)
- A14 N (`frozen.py:655-657`): the CLI description shows `cache_dir('checkpoints')`; no
  `ArgumentDefaultsHelpFormatter`. A15 N (`test_frozen.py:29-32`): the `model` fixture only
  returns `medium_model`. A16 N (`frozen.py:636-641`): OGBench files are not hash-checked.

### b-theory
- Fixed: B15-B19, B21, X11, the N items. B20 partly: the walk and cell-MDP code stays
  duplicated (`oracle_teleport.py:65-122`, `time_limit.py:43-75`); the recorded reason (the
  Linux queue) expired. trap_aware.py matches design.md §8 T3 and theory.md §11.4: data only,
  no map, teleporters or `goal_xy`; the final config was selected on held-out goals only
  (same hash 25077c9a, checked). theory.md §9.4, §11.1-§11.4 numbers equal the CSVs (checked).
- B22 M (`theory.md:593-594`, checked): "other failures rise most on tasks 3 and 4" is wrong:
  tasks 1 and 2 rise most (29.8 → 46.4, 22.2 → 40.8 %) and lose most success (−10.8, −13.4 pp),
  while :586-587 says the gate is closed at their start. Fix: give all 5 tasks; reword the
  mechanism (the gate opens later in those episodes).
- B23 S: numbers from scripts outside the repo (/private/tmp): §11.4 gate check (58-77 steps,
  trap occupancy 0.15 → 0.11), §11.1-§11.2 raster bins, §11.2 giant replay (`cells_left`).
  Fix: commit the scripts to `analysis/` with commands, or mark the numbers.
- B24 (g-release): fixed in `reproduce.sh`.
- B25 N `trap_aware.py:20` "Lemma B1" vs theory.md:24 "Lemma 1". B26 N `trap_latent`
  (`trap_aware.py:93-95`) only for tests. B27 N `TrapAwareConfig` defaults (1.0, 2.5) are not
  the selected config (0.5, 1.5); `time_limit` exists only for tests. B28 N theory.md:202
  "14 min with 4 workers ... 10 CPU minutes"; `study_plots --help` prints "(default: csv_dir)
  (default: None)"; the giant bound is 58 steps per cell, not 59 (theory.md:402, 520).

### c-planner
- Fixed: C1, C3-C11, X10. C2 partly: X4 (d-eval side); X5: `planner_routes.py:67-87`
  `PathRecorder` repeats the public `evaluate.RecordingController`. C12 partly. X7 open for
  `planner_routes.py:122-133` (`maze.draw_walls` exists).
- H1 claims hold (checked): the controller reads the train split (observations, terminals, xy
  = obs[:2]), FB and `z_r`; no `goal_xy`, map or val split. Selection used task ids 101-110
  only, fixed before any result. Deterministic. `fbplan/` unchanged from 5d7e642 to HEAD.
- C13 S (`planner.py:509, 575`, checked): edge and first-hop weights are straight xy lengths,
  not walked lengths. On large 1.8 % of edges cross a wall, 21 % of shortest paths use one, and
  the route is 1.26× the geodesic at p90 (max 3.4×); medium 1.3 %, 6.7 %, 1.05×. REPORT:1350
  gives only the median. No test pins the weight (a mutant passes). Fix: state the limit in
  REPORT §8.7; add a test.
- C14 S (`planner.py:602-608`): the off-data fallback is untested (removing it passes).
- C15 S (`planner.py:267-280` vs `611-621`): `_route_point` repeats `point_along`.
- C16 S: the region mean is written twice (`landmarks.py:115-120`, `123-137`); "within radius,
  else nearest" twice (`landmarks.py:117-120`, `planner.py:604-608`); `from_dataset` draws the
  same pool 3 times; `RegionField.from_data(...)` repeated (`planner.py:203-205, 532-534`).
- C17 S stale docstring numbers: `landmarks.py:9-12` "58 % to 87 %" (committed: 57.2 → 80.4);
  `planner.py:66-78` probe numbers whose only source is docs/status: cite the committed
  ablations.
- C18 S: test fakes keep `temperature`, `seed` (`planner_fakes.py:80, 87`,
  `test_planner.py:73, 453, 457`).
- C19 S: process names in code (list in "Dead references").
- C20 S: alias `recorded_route` (d-eval's file, request 2).
- N: `planner.py:16` "plans without FB values" (the goal landmark is an FB value); fields that
  do not affect H1 enter its hash; the `__init__` docstring says xy [P, 2] for observations;
  `planner_routes.py` repeats `maze_name`, `--env` without choices, no H1 route plot.

### d-eval
- Fixed: D2-D5, D8, D10-D16, X1, X5 (own part), X7, X16, X17. D7 partly (X4). D9 partly:
  `--workers` defaults to 10 (`failures.py:126`, `parity.py:63`). X11 partly: `evaluate.py:518`
  `--workers` default 1 and no help for `--env`, `--tasks`, `--workers`; `calibration.py:159`
  `--out-dir`; `heldout_goals.py:72` always writes to `results/heldout_goals/`;
  `heldout_selection.py`, `platform_check.py` without help. A11 open: `evaluate.py:234-255`
  duplicates `frozen.infer_goal_latent`. The 43 ten-seed CSVs repeat their 5-seed CSVs
  record for record (checked).
- D17 M (`controllers/oracle.py:11-14`, checked): the docstring gives the Mac sweep
  (44.8/56.4/65.2, 80.8/86.4/78.0) and cites `results/medium/oracle_sweep/`, which holds the
  Linux runs (50.0/57.2/63.2, 78.4/80.4/82.4; lookahead 3 is best there). Fix: quote the
  Linux numbers; say that the defaults came from the Mac sweep.
- D18 S (`heldout_selection.py:44-46, 66`, checked): selection on float means: equal counts
  can give different floats (526/600 as ...667 and ...666), so a tie would go to rounding, not
  to the lower seed. No current selection is affected. Fix: select on counts; test with tied
  counts.
- D19 S (checked): `results/medium/heldout_selection_awr2x.csv` is stale; a rerun adds a
  `retrained_logratio` row (−3.4 pp, p 0.038). Fix: regenerate (the `tables` stage does).
- D20 S (`baseline.py:173-186`): `make_ensemble`, `RawEnsembleBaselineController`,
  `_raw_intention` have no test that runs without data; CI never runs T1/T4 code.
- D21 S (`evaluate.py:113-122`): the variant branch repeats `load_baseline_model`; `:535`
  names the removed `load_retrained_model`.
- D22 S (`platform_check.py:47-53`): includes `upstream` (unseeded resets) and planner rows
  whose code differs between platforms. Fix: skip unpaired runs; add a "code differs" column.
- D23 S (`controllers/__init__.py:19`): `PRIVILEGED` lacks `oracle_teleport`.
- N: D24 row order (documented in `reproduce.sh`); D25 `baseline.py:11` and "plan B decision 8"
  links; D26 test leftovers (`test_eval_controllers.py:40, 43, 109` fakes with `temperature`;
  `test_evaluate.py:352-358` shadows the conftest `medium_model`; `:391, 423` use the
  `trained_dir` shim); D27 missing docstrings and return types; D28 `.agent` with
  `type: ignore` (`baseline.py:157`, `evaluate.py:249`); D29 `evaluate.py` has 560 lines.

### e-learner
- Fixed: E10, E11, E14, E15, E16, X3, X11, X14. logratio.py matches design.md §8 T2 (2.45 for
  goal latents only, ρ per head clipped to [0, 0.99], head mean before the log), and the 2× α
  equals 0.2 × 16.49 / 15.72 (checked).
- E17 M: release coverage, request 1.
- E18 M (`learned/checkpoint.py:30-64`, A10): copies of `trained_key`, `save_flags`,
  `load_trained_file` and `Kind` remain, used by `train_high`, `train_b`, `value`,
  `diagnostics`, `evaluate`; the status says "my copies are gone". Fix: import from
  `fbplan.frozen` and delete them.
- E19 M (`vast.sh:162-165`): no step destroys the instance when the jobs end; the README said
  it does (README fixed). Fix: `vast.sh finish <id>`: wait for the jobs, pull, check the
  expected files, destroy.
- E20 S (`logratio.py:62-73`, checked): no test pins the wiring of `advantages()`; a mutant
  with F(s,z) for F(w,z) passes all 8 tests. Fix: compute A for goal samples from
  `frozen.forward` in a test.
- E21 S (`test_learned_b.py:290-300`, checked): the shared-α test does not pin the median or
  the new latent draw per batch; two asserts cannot fail.
- E22 S: approach-b.md is stale at :130-131 (α), :134-136 (`params.pkl`,
  `load_baseline_model`), :144 (the cap binds on 33 %), :213 (a missing `controller.py`; runs
  that never happened), :210 (process text).
- E23 S (`awr2x.sh:10-11`, `awr2x_v1.sh:10`): `XLA_PYTHON_CLIENT_ALLOCATOR=platform` with a
  reason that the status contradicts; `awr2x.sh:15, 20` hard-code `~/.cache/fbplan`.
- E24 S: the cancelled h-learn H1 control (`awr2x_v1.sh`, `--critic`, `ValueConfig.critic`,
  `test_learned_b.py:177-209`) has no model or CSV: delete it or say why it stays.
- E25 S (`train_b.py:77-90`): stale TODO; `option_model.batched_backward` exists.
- N: E26 CLI help and docstrings (`train_high.py:121, 139`; `train_b.py:313`; `main()`
  without docstrings); E27 repeated test helpers (`same()` 3 times, `MEDIUM`, `OBS_DIM`);
  E28 `MAX_LOG_WEIGHT` twice, `advantage_stds`, migration code.

### f-report
- Fixed: F1, F2 (checked; the time-to-goal PNG is byte-identical). 18 groups of REPORT numbers
  recomputed from the CSVs: all equal (headline, platform check, 10-seed medium, H1 large,
  2×, T1-T4, T3, large held-out grid, failure classes, act-ahead).
- F3 M (REPORT.md:3-7, 1422-1431, 1461, 1472-1474, 1495): wave 3 is merged, the report says
  "pending". Values (checked): large seeds 10-19 +1.8 [−0.7, +4.4] p 0.16 vs `baseline`, +3.7
  vs raw; seeds 0-19 +2.7 [+0.9, +4.5]; medium held-out window K150 88.8 % (tie with high K60
  broken by the order); medium seeds 0-9 83.8 %, +5.3 vs `baseline`, +3.3 [+1.2, +5.4] p 0.003
  vs raw, +2.2 p 0.037 vs oracle.
- F4 M (REPORT.md:1422-1431): the medium replication fixed in design.md §8 has no CSV; state
  it as not run (request 5).
- F5 M (REPORT.md:15-57): §1 contradicts the body: "no high-level choice has room on medium
  and large" (H1 passes on both); 5-seed values where §8 uses 10 seeds; "10 B tests" vs 12;
  T2 "pending"; no bullet for the recorded route.
- F6 S: the gap to the paper also for teleport (40 ± 6 vs 27.6 / 25.1 %) and giant.
- F7 S (:12): the platform line lacks "CPU"; :369-370 puts the recorded route in the wrong
  folder.
- F8 S (:1331-1354): one plain sentence on privileged information; "`π_h` executes" holds on
  large only (medium selected `window`).
- F9 S (:1135): names a workflow tab.
- F10 S (:1221-1237, 1338-1339, 1350-1352): numbers whose only source will be deleted or is
  not committed.
- F11 S (:637, :953, :1049): pending markers.
- F12 S (:1522-1524, checked): the §11 success-figure command fails on medium (the glob
  matches `heldout_selection*.csv`). The explicit list rebuilds the figure byte for byte;
  `reproduce.sh` uses it.
- F13 (g-release): fixed. F14 S (d-eval): no test of `report_figures.py`.
- N: F15 `report_figures.py` computes the gap twice, `int(median)` truncates, two envs give an
  unclear error; F16 large seeds 0-19 raw − baseline −2.0 [−3.7, −0.3] for REPORT §6;
  F17 design.md §6-§8 (coordinator).

### coordinator
- `contracts.py:4`, design.md:35 and §6-§8: see "Dead references". Requests 1, 2, 4, 5.

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
