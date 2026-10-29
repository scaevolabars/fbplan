# Status: f-report

Last update: 2026-10-29, 20:00. Branch merged with `main` at 275dfe1.

## Done

- `REPORT.md` step 1 of the brief: §2 setting, §3 theory summary with the tabular study, §4
  approach A, §5 approach B, §6 protocol; §8.1–§8.3.
- After the merge of 18725e8 (d-eval's medium planner results and `baseline_raw`; baseline
  decision in design.md §8):
  - §6: `baseline` (normalized `z_r`) is the main row, `baseline_raw` a reference row; the rule
    "a gain counts only if it also holds paired against `baseline_raw`".
  - §7.1: the medium table with `baseline_raw` and the 3 planner variants (3 landmark sets pooled),
    and the paired differences per landmark set against both `baseline` and `baseline_raw`.
  - §8.1: `baseline_raw` replaces the parity-based `z_r` statement (+2.6 pp, p = 0.11, paired).
  - §8.2: oracle against `baseline_raw` (+3.5 pp [+0.5, +6.5], p = 0.02).
  - §8.4 filled: why A is below the baseline on medium (little headroom, per-task pattern, later
    arrival at the goal, where failures end, landmark-set sensitivity, `planner_high`), and the
    hypothesis "region latents: good for acting, bad for planning".
  - §9 and §10: the items that the medium results allow; B, large and the ablations stay pending.
- `analysis/report_figures.py` (new): success within a step budget minus the baseline, per planner
  variant, and the tables that §8.4 quotes. Output
  `docs/figures/report/time_to_goal_antmaze-medium-navigate-v0.png`. Colors are those of
  `analysis/plots.py`; the 3 planner colors pass the dataviz validator over all pairs (the
  baseline is the zero line, because blue against the planner's purple fails it).
- After the merge of f1ae57d (g-release review items X3, X13):
  - Appendix "notation and names in the code": the trust bound `rho_max = 1.0` of A against the
    value cap `rho_max = 0.9` of B; the `R` modes in A (`exact`, `paper`, `zero`), B (`exact`,
    `paper`, `goal_zero`) and the tabular code (`full`, `proxy`, `zero`); the direct value
    (`direct`, `V_1`, `v_direct`, `v_d`, `v_final`); the graph `V_H` (`GraphSolution.values`,
    `v_graph`) against the learned `V_H`; `Δ_θ` against `gain`; the two wormhole definitions;
    and a method table (`--method` alias, CSV `method` such as `graph_planner`, results path).
  - §8.3 from `results/calibration/*.csv`: wormholes per `ρ_min` (a higher `ρ_min` keeps 88 % of
    them on medium), what the xy cap leaves (1,965 of 3,385 on medium, 1,420 of 3,992 on large),
    head min against head mean. §4 guard table: the xy cap removes 42 % (medium) and 64 % (large).
- After the merge of 275dfe1 (retrained baselines, failure classes, b-theory's rerun study):
  - §3.1 and §4: tabular numbers from the rerun `docs/figures/theory/study_*.csv` (single switch
    0.24, chained `R̂ = 0` 0.37, commitment 0.37 → 0.50); "cost per hop" is now the hop discount
    (0.37 → 0.11–0.13).
  - §7.1: the retrained baseline row (78.3 % [76.9, 79.8], 5 seeds pooled) and its paired rows
    against `baseline` (mean +0.5 pp) and `baseline_raw` (mean −2.2 pp).
  - §8.1: the `π_h` training seed is ruled out as the cause of the gap to the paper (76.9–80.6 %).
  - §8.2: failure classes from `results/medium/failures/`: 71 % flips with a high level; the oracle
    flips less (142 against 198 episodes); non-hier fails task 4 by getting lost, not by flipping.
  - §9, §10 and the appendix follow.
  - §8.4 correction: the test "plan with single-state latents, act with region latents" was run
    as a 100-episode probe by c-planner (no gain, not committed); §10 lowers that next step.
- Private defense sheet for the reviewer call (coordinator request): written outside the
  repository, not committed.
- Every number in the report names a committed file. Recomputed with `analysis/tables.py`
  (`--reference baseline` and `--reference baseline_raw`, `--cluster config_hash`) and
  `analysis/report_figures.py`. The paired `upstream` row of `tables.py` is not used.

## Results the report waits for

| # | Result | Report section | Owner | Expected place |
|---|---|---|---|---|
| 2 | A guard and execution ablations on medium: `R̂` exact and paper against zero, raw `ρ̂ ≥ 1` trusted, no xy cap, `two_way`, `reached_xy` 1.5 and the `ρ` rule, no look-ahead, `replan_every_step`; the `ρ̂` statistics of region latents | §4 (evidence column), §8.4, §9 | c-planner (assigned) | `results/medium/planner_ablations/` |
| 3 | route plots of A on medium and large, committed | §8.4 | c-planner (`analysis/planner_routes.py` writes to `runs/`) | `docs/figures/report/` or c-planner's results |
| 4 | failure classes of the 3 planner variants (per landmark set), later B | §8.4, §10 | d-eval (assigned) | `results/medium/failures/` |
| 5 | whether the wormholes that the xy cap leaves cross a wall (a column of the calibration CSV) | §8.3, §4 | d-eval | `results/calibration/` |
| 6 | B1 on medium: `V_H`, then B-clean, B-wide, B-region, 5 eval seeds | §7.1, §8.5, §9 | e-learner (GPU training), d-eval (evaluation) | `results/medium/` |
| 7 | B offline diagnostics as a committed file: `V_H` and `V_1` against maze distance, `ρ` cap and clip rates (numbers so far only in e-learner's status) | §8.5 | e-learner | `results/medium/` |
| 8 | large: baseline, `baseline_raw`, oracle, planner variants, retrained baselines, B | §7.2, §8.1, §10 | d-eval (running); e-learner for large training | `results/large/` |
| 9 | giant and teleport | §7.3 | b-theory | `results/{giant,teleport}/` |
| 10 | Linux reruns, canonical from now on (coordinator decision): every evaluation CSV that the report quotes. Mac CSVs move to a separate folder | all numbers in §6–§10, figures | d-eval | `results/` (Mac copies: d-eval's choice of folder) |

Dropped: the per-target wormhole diagnostic (coordinator: no new run; §8.3 keeps it as a
hypothesis). Done: A on medium (`results/medium/planner*/`), the baseline definition, the
calibration tables (`results/calibration/`), the retrained medium baselines
(`results/medium/baseline_retrained/`), the failure classes of baseline, `baseline_raw`, oracle
and non-hier (`results/medium/failures/`). Seen, for §7.3 later: b-theory's time-limit study
(`docs/figures/theory/time_limit*`).

Every needed result has an owner. The medium results of A are negative, as the brief expected;
nothing changes the story of the report.

## Plan for the Linux reruns

- When the Linux CSVs land: rerun `analysis/tables.py` (both references, both cluster levels) and
  `analysis/report_figures.py`, and replace every number and figure in the report that they change.
- One sentence on the Mac-vs-Linux agreement, from `analysis/tables.py` with `--label-root results`
  and the Linux CSV as `--reference`: the paired difference with CI, and the share of episodes
  with the same outcome (from the discordant counts). Checked on a copy: `tables.py` pairs the same
  method across two folders by episode, so no new code is needed.
- Report inputs from evaluation runs: `baseline`, `baseline_raw`, `nonhier`, `oracle`, `upstream`
  (parity), `oracle_sweep/`, `planner*/`, `baseline_retrained/`, `failures/` (they replay
  episodes, so they must be rerun on the same platform as their CSVs). Inputs without episodes:
  `results/calibration/`, `docs/figures/theory/`. Any input that is not rerun on Linux keeps its
  Mac number, and the report says so.

## Next

- Fill §7.2, §7.3, §8.5, the rest of §9 and §10 when the results above arrive; write §1 last.
- Merge `main` when new results are merged.

## Blockers

- None.

## Requests to the coordinator

- g-release: please add `uv run python analysis/report_figures.py` (time-to-goal figure and
  tables of report §8.4) to the "Reproduce" section of `README.md` and to
  `scripts/reproduce.sh`. Not on main at f1ae57d.
- d-eval: the calibration CSV counts the wormholes within the xy cap (1,965 on medium) but not
  whether they cross a wall. Your status says they cross none (short pairs, FB overconfident at
  short range). Please add that count as a column (item 5), so §8.3 can state it.
- d-eval: please include the 3 planner variants (per landmark set) in the failure-class CSV
  (item 4). §8.4 needs their flip share to separate `π_l` failures from route failures.
- d-eval (Linux reruns): the report quotes these evaluation inputs. Please rerun all of them on the
  Linux instance, or tell me which stay Mac-only, so the report can mark them: `baseline`,
  `baseline_raw`, `nonhier`, `oracle`, `upstream` (parity), `oracle_sweep/`, `planner*/`,
  `baseline_retrained/`, and `failures/` for each of these (a replay must run on the platform of
  its CSV). Please also tell me the folder of the Mac copies.
