# Status: f-report

Last update: 2026-10-29, 23:04. Branch merged with `main` at 28f6aba.

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
- After the merge of 3e36461 (Mac results moved to `results/macos/`; Linux reruns pending):
  - The report points to `results/macos/` until the Linux CSVs land, and says so in a platform
    note under the status line.
  - §8.4 point 5: A's failure classes per landmark set (`results/macos/medium/failures/`): A flips
    as often as the baseline (159–210 against 198) but has 2–4 times as many slow failures
    (136–312 against 72).
  - §7.3: the limits of giant and teleport before any run, from b-theory's committed files
    (Proposition 4, `time_limit*.csv`, giant calibration); §9 and §10 follow.
  - Appendix (g-release E16): `OptionConfig.rho_cap`, `delta_network`, softplus(delta).
  - `analysis/report_figures.py` (g-release F1, F2, CLI convention): the results folder is
    positional (default `results/medium`); a method without CSVs is left out with a note on
    stderr, a missing baseline stops with a clear error; a method without successes or failures
    shows "–". `--help` shows the defaults. The old `--results` flag still works (hidden) until
    `scripts/reproduce.sh` switches. Checked: on `results/macos/medium` the figure is identical
    byte for byte and the tables are unchanged; a folder without `planner_region`, a planner
    without successes, and a folder without the baseline all behave as described.
- After the merge of 529f7ff (canonical Linux results for medium, large, teleport; design.md §8
  on the two baseline rows): the report runs on the Linux numbers.
  - Status line and §1 Summary (draft, B pending) with the central finding: against the stronger
    baseline row the privileged oracle gains +0.5 pp [−2.6, +3.5] on medium and +1.9 pp
    [−1.8, +5.6] on large, so with this frozen `π_l` no high-level choice has measurable room;
    giant about 2 % (preliminary); teleport bounded by the time limit and the trap.
  - §6: the two baseline rows per maze (raw stronger on medium, normalized on teleport, no
    significant difference on large) and the rule "the stronger row is the bar"; the platform check
    (`analysis/platform_check.py`: Linux − Mac −4.6 to +2.0 pp over 23 runs, CI contains 0 for 22;
    55–73 % of episodes with the same outcome); parity marked as a Mac run.
  - §7.1 medium, §7.2 large (new), §7.3 teleport measured (new); pooled paired CIs from the new
    `tables.py` output; success figures per task made with `analysis/plots.py --out-dir
    docs/figures/report`.
  - §8.1 (large gap), §8.2 rewritten around the oracle test, §8.3 from the updated calibration
    CSVs (trusted pairs exclude raw `ρ̂ ≥ 1`; medium 3,405 wormholes at `ρ_min` 0.3), §8.4 on Linux
    numbers (time-to-goal figure regenerated), §9, §10, §11, appendix paths.
  - Marked as Mac-only: the parity run, the oracle sweep and the failure classes on medium. Giant:
    preliminary numbers from `docs/theory.md` §11.3, no CSV yet.
- After the merge of 7c42d54 (B-clean on Linux; Linux reruns of parity, oracle sweep and medium
  failure classes; `results/platform_check.csv`): B is written as a negative result (coordinator
  decision: B stops at the gate).
  - §7.1: B-clean row (79.3 % [76.4, 82.2]) and paired rows; +1.1 pp [−2.4, +4.5] over the
    retrained baselines, −1.5 pp [−5.1, +2.1] against `baseline_raw`; the seed split (seeds 0, 2, 4
    beat seeds 1, 3 by +4.6 to +7.4 pp, all p ≤ 0.005; retrained seeds: largest gap +3.0 pp,
    p = 0.07).
  - §8.5 rewritten, "Why approach B gives no gain": the critic check (`V_H` lifts far values,
    16.4 against 7.8 at 20–24 xy units, but orders states no better than `V_1`, at expectile 0.9
    or 0.7), the seed split, and the headroom bound (oracle +0.5 pp against `baseline_raw`).
  - Critic check: I ran e-learner's committed `fbplan.learned.diagnostics` for both critics (CPU,
    no rollouts) and committed its output and the seed-0 plots to `docs/figures/report/`
    (`b_critic_check_medium.txt`, `b_critic_medium_expectile{0.9,0.7}_seed0.png`), so §8.5 names
    a committed file. Same numbers as e-learner's status, except 16.4 / 7.8 against 16.3 / 7.7.
  - §5 gate paragraph (B-argmax, B2 and the expectile 0.7 policy did not run); §1, §9, §10, §11,
    appendix.
  - Linux reruns: §6 parity (upstream − baseline +0.9 pp [−2.3, +4.1], − `baseline_raw`
    −2.9 [−6.0, +0.2]) and platform check from `results/platform_check.csv` (36 runs, 35 CIs
    contain 0); §8.1 upstream 77.9 %; §8.2 oracle sweep and failure classes; §8.4 point 5.
  - Correction: on Linux the oracle does not flip less than `baseline_raw` (176 against 179; Mac:
    142 against the baseline's 198). §8.2 now says so and drops the "shorter route, fewer flips"
    hypothesis.
  - §8.4 point 7 softened: c-planner's act-ahead probe (Mac, not committed) leaves the share of
    the route open; the Linux run is in d-eval's queue.
- After the merge of 28f6aba (held-out seed selection, e-learner's seed-split analysis, the
  AWR-sharpness experiment in design.md §8, A ablations, act-ahead, giant, teleport extras):
  - B's conclusion is open (coordinator): §8.5 is now "what the critic and the seeds show
    *(conclusion pending)*", with the three readings of the 2× experiment as fixed in design.md
    §8. §1, §5 (gate), §9 and §10 say "no gain so far; conclusion pending".
  - §8.5 finding 2: the seeds split by their AWR temperature (0.111–0.229, from one calibration
    batch per seed; rank correlation with success +0.8, n = 5), not by their critic. Temperatures
    and spreads checked against `approach_b/medium/clean/seed<k>/flags.json`. §5 now says that
    the temperature calibration was noisy and that "only the critic changes" did not fully hold.
  - §7.1: the held-out seed selection table (`results/medium/heldout_selection.csv`, rerun with
    `analysis/heldout_selection.py`: identical); B-clean pooled on the held-out goals −0.7 pp
    [−3.8, +2.4] against the retrained baselines.
  - §4: the guard evidence column and an ablation table (medium: 9 changes; large: 3), paired with
    `planner/l0`. No change beats the defaults; `R̂ = 0`, look-ahead and reach radius matter; the
    xy cap matters on large only; head min, commitment, route following, `two_way`: no effect.
  - §8.4 point 8 (new): act-ahead 77.0 %, +3.2 pp [−0.3, +6.6] over `planner_high` l0, level with
    `baseline`, below `baseline_raw` (−3.8) and the oracle (−4.2, p = 0.01). The region
    hypothesis paragraph uses it.
  - §7.3: teleport non-hier and the teleport-aware oracle (5.6 % with `π_l`, 21.4 % with `π_h`,
    both below the baseline; gains only on task 2); giant measured (every method below 3 %;
    oracle +2.0 pp [+1.1, +3.0], significant; no `baseline_raw` run). New figure
    `success_antmaze-giant-navigate-v0.png`; teleport figure regenerated.
  - Central finding reworded in §1, §8.2 and §9 for giant: the oracle's gain there is significant
    but small; "no measurable room" now says "on medium and large", and giant and teleport are
    "bounded by the time limit and `π_l`".
- Private defense sheet for the reviewer call (coordinator request): written outside the
  repository, not committed; updated for B, the seed selection, the ablations and giant.
- Every number in the report names a committed file. Recomputed with `analysis/tables.py`
  (`--reference baseline` and `--reference baseline_raw`, `--cluster config_hash`) and
  `analysis/report_figures.py`. The paired `upstream` row of `tables.py` is not used.

## Results the report waits for

| # | Result | Report section | Owner | Expected place |
|---|---|---|---|---|
| 1 | AWR sharpness at 2× (design.md §8): baseline and B-clean, 5 seeds each; decides B's conclusion | §1, §5, §8.5, §9, §10 | e-learner (training), d-eval (evaluation) | `results/medium/` |
| 2 | B-wide and B-region on medium, pooled and held-out selection | §1, §7.1, §9 | d-eval | `results/medium/` |
| 3 | route plots of A on medium and large, committed | §8.4 point 8 | c-planner | `docs/figures/report/` or c-planner's results |
| 4 | teleport: failure classes with the trap class (non-hier and the teleport-aware oracle done) | §7.3 | d-eval | `results/teleport/` |
| 5 | giant: oracle failure classes (Linux); optional `baseline_raw` | §7.3 | d-eval | `results/giant/` |
| 6 | large: retrained baselines; failure classes of the A variants | §7.2 | d-eval | `results/large/` |
| 7 | whether the wormholes that the xy cap leaves cross a wall (a column of the calibration CSV) | §8.3, §4 | d-eval | `results/calibration/` |

Every needed result has an owner. The central finding (no measurable room for a high level with
this `π_l`) was set by the coordinator from the Linux results; the report states it in §1 and §8.2,
with the giant wording adjusted to the Linux giant runs (oracle +2.0 pp, significant, at 2.5 %).

## Next

- When the 2× experiment lands: B's conclusion in §8.5, then §1, §5, §9, §10, by the reading
  fixed in design.md §8. When B-wide and B-region land: §7.1 rows and the held-out table. Then the
  other items above.
- Merge `main` when new results are merged.

## Blockers

- None.

## Requests to the coordinator

- d-eval: the calibration CSV counts the wormholes within the xy cap (1,965 on medium) but not
  whether they cross a wall. Your status says they cross none (short pairs, FB overconfident at
  short range). Please add that count as a column (item 7), so §8.3 can state it.
- d-eval: teleport needs failure classes with the trap class (item 4).
- d-eval: in `analysis/plots.py`, the two teleport-aware oracles get red and green (a pair that
  colour-blind readers confuse) and no "(privileged)" label; please add `oracle_teleport` to the
  privileged set and give it fixed colours (it is the method name of both CSVs).
- d-eval (optional): giant has no `baseline_raw` run, so the two-row rule of §6 applies there with
  one row only; the report says so.
- e-learner, two notes on the seed-split analysis: (1) B-wide's calibration uses the same upstream
  advantage spread per seed as B-clean (equal to 7 digits in the flags), so its temperatures are
  not independent support for the explanation; (2) the rank correlations are more negative =
  better, so seed 1 has the best-ordering `V_H` and seed 3 the worst (your status says seed 3
  orders best). The conclusion "the order does not follow success" holds either way.
- g-release: `analysis/report_figures.py` takes the results folder as a positional argument now
  (CLI convention). Please switch `scripts/reproduce.sh` to
  `py analysis/report_figures.py "$dir" --out-dir "$report_dir"`; then I remove the hidden
  `--results` flag.
- d-eval: please add a test of `analysis/report_figures.py` to `tests/test_analysis.py` (I own no
  test file): a folder without one planner variant runs and leaves it out; a method with no
  success shows "–"; a folder without `baseline.csv` stops with an error.
- b-theory: `docs/theory.md` §11.2 says the 17-cell giant walk fits at ≤ 59 ant steps per cell;
  17 × 59 = 1003 > 1000, so the bound is 58. The report uses 58.
