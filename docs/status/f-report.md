# Status: f-report

Last update: 2026-10-29.

## Done

- `REPORT.md` step 1 of the brief: §2 setting, §3 theory summary with the tabular study, §4
  approach A, §5 approach B, §6 protocol. §8.1 (gap to the paper), §8.2 (headroom on medium:
  non-hier, oracle, oracle sweep), §8.3 (FB calibration and wormholes). §7.1 has the final
  reference rows (baseline, upstream, non-hier, oracle) and empty rows for the pending methods.
- Every number in the report names a committed file. Checked by recomputation:
  - `analysis/tables.py` on `results/medium/{baseline,nonhier,oracle,upstream}.csv` and
    `results/medium/oracle_sweep/*.csv`;
  - parity with `fbplan.stats.unpaired_comparison`: −2.2 pp [−5.3, +1.0];
  - the tabular tables against `docs/figures/theory/study_{noise,guards,reward,wormholes}.csv`
    (all equal to `docs/theory.md`);
  - wormhole counts from the titles of `results/figures/wormholes_*.png`.
- Numbers that exist only in status files are not in the report. Their slots say *(pending)*.

## Results the report waits for

| # | Result | Report section | Owner | Expected place |
|---|---|---|---|---|
| 1 | retrained baseline `π_h` seeds 0–4 on medium, 5 eval seeds each | §7.1, §8.1 | d-eval (checkpoints from e-learner exist) | `results/medium/` |
| 2 | approach A on medium, 5 eval seeds: `planner`, `planner` with region latents, `planner_high`, several landmark seeds | §7.1, §8.4 | d-eval (runs), c-planner (configs) | `results/medium/planner*/` |
| 3 | A ablations on medium that back each guard: `R̂` exact and paper against zero, raw `ρ̂ ≥ 1` trusted, no xy cap, `two_way`, `reached_xy` 1.5 and the `ρ` rule, no look-ahead, `replan_every_step` | §4 (evidence column), §9 | c-planner | `results/medium/planner_ablations/` |
| 4 | route plots of A on medium and large, committed | §8.4 | c-planner (`analysis/planner_routes.py` writes to `runs/`). f-report can render them into `docs/figures/report/` with that script when A's config is final | `docs/figures/report/` |
| 5 | failure classes as a CSV: baseline, oracle, non-hier; later A and B | §8.2, §8.4 | d-eval (`analysis/failures.py` prints only) | `results/medium/` |
| 6 | calibration table as a CSV: trusted pairs and wormholes per `ρ_min`, min and mean heads, medium and large | §8.3 | d-eval (`analysis/calibration.py` prints only) | `results/figures/` or `results/<maze>/` |
| 7 | B1 on medium: `V_H`, then B-clean, B-wide, B-region, 5 eval seeds | §7.1, §8.5 | e-learner (GPU training), d-eval (evaluation) | `results/medium/` |
| 8 | B offline diagnostics: `V_H` and `V_1` against maze distance; `ρ` cap and clip rates | §8.5 | e-learner | `results/medium/` |
| 9 | large: baseline, paired raw-`z_r` row, retrained baselines, A, B | §7.2, §8.1 | d-eval; e-learner for the large `π_h` and B training | `results/large/` |
| 10 | giant and teleport | §7.3 | b-theory | `results/{giant,teleport}/` |
| 11 | optional: wormhole count per target against `F(w,z_w)ᵀB(w)` (tests the hypothesis in §8.3) | §8.3, §10 | d-eval (`analysis/calibration.py`) | `results/figures/` |
| 12 | decision: normalized or raw `z_r` for the main baseline row (design.md §8) | §6 | user | design.md §8 |

Every needed result has an owner. Nothing changes the story of the report yet.

## Next

- Fill §7, §8.4, §8.5, §9 and §10 when the results above arrive; write §1 last.
- Merge `main` into this branch when new results are merged.

## Blockers

- None.

## Requests to the coordinator

- d-eval: `analysis/tables.py` prints a paired row for `upstream` (+2.2 pp, p = 0.19). Upstream
  resets are unseeded, so that pairing is not valid. The report uses `unpaired_comparison`
  instead. Please let `tables.py` skip or mark unpaired methods in the paired table.
- d-eval: please make `analysis/failures.py` and `analysis/calibration.py` write their tables as
  CSV and commit them (items 5 and 6). The report cannot quote the printed numbers.
- c-planner: please commit the ablation CSVs that back the approved guards (item 3). The report's
  evidence column for approach A waits for them.
- Scratch check (not in the report): on the 400 medium states of `calibration.py`, wormholes
  concentrate more on targets than on sources (top 10 targets hold 19 % of them, top 10 sources
  8 %), and targets with a low `F(w,z_w)ᵀB(w)` collect more (Spearman −0.25). This supports the
  §8.3 hypothesis; item 11 would make it a committed result.
