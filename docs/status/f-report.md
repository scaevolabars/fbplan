# Status: f-report

Last update: 2026-10-29. Branch merged with `main` at 18725e8.

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
- Every number in the report names a committed file. Recomputed with `analysis/tables.py`
  (`--reference baseline` and `--reference baseline_raw`, `--cluster config_hash`) and
  `analysis/report_figures.py`. The paired `upstream` row of `tables.py` is not used.

## Results the report waits for

| # | Result | Report section | Owner | Expected place |
|---|---|---|---|---|
| 1 | retrained baseline `π_h` seeds 0–4 on medium, 5 eval seeds each | §7.1, §8.1, §10 | d-eval (after large) | `results/medium/` |
| 2 | A guard and execution ablations on medium: `R̂` exact and paper against zero, raw `ρ̂ ≥ 1` trusted, no xy cap, `two_way`, `reached_xy` 1.5 and the `ρ` rule, no look-ahead, `replan_every_step`; the `ρ̂` statistics of region latents | §4 (evidence column), §8.4, §9 | c-planner (assigned) | `results/medium/planner_ablations/` |
| 3 | route plots of A on medium and large, committed | §8.4 | c-planner (`analysis/planner_routes.py` writes to `runs/`) | `docs/figures/report/` or c-planner's results |
| 4 | failure classes as a CSV: baseline, `baseline_raw`, oracle, non-hier, and the 3 planner variants; later B | §8.2, §8.4, §10 | d-eval (assigned) | `results/medium/` |
| 5 | calibration table as a CSV: trusted pairs and wormholes per `ρ_min`, min and mean heads, medium and large | §8.3 | d-eval (assigned) | `results/` |
| 6 | B1 on medium: `V_H`, then B-clean, B-wide, B-region, 5 eval seeds | §7.1, §8.5, §9 | e-learner (GPU training), d-eval (evaluation) | `results/medium/` |
| 7 | B offline diagnostics: `V_H` and `V_1` against maze distance; `ρ` cap and clip rates | §8.5 | e-learner | `results/medium/` |
| 8 | large: baseline, `baseline_raw`, oracle, planner variants, retrained baselines, B | §7.2, §8.1, §10 | d-eval (running); e-learner for large training | `results/large/` |
| 9 | giant and teleport | §7.3 | b-theory | `results/{giant,teleport}/` |

Dropped: the per-target wormhole diagnostic (coordinator: no new run; §8.3 keeps it as a
hypothesis). Done: A on medium (`results/medium/planner*/`), the baseline definition.

Every needed result has an owner. The medium results of A are negative, as the brief expected;
nothing changes the story of the report.

## Next

- Fill §7.2, §7.3, §8.5, the rest of §9 and §10 when the results above arrive; write §1 last.
- Merge `main` when new results are merged.

## Blockers

- None.

## Requests to the coordinator

- g-release: please add `uv run python analysis/report_figures.py` (time-to-goal figure and
  tables of report §8.4) to the "Reproduce" section of `README.md`.
- d-eval: please include the 3 planner variants (per landmark set) in the failure-class CSV
  (item 4). §8.4 needs their flip share to separate `π_l` failures from route failures.
