# Status: f-report

Last update: 2026-10-30, 05:25. Branch merged with `main` at 8b4795f.

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
- After the merge of 4c2ee44 (final protocol with 10 eval seeds, T1 ensembles, B-wide and
  B-region, the hypothesis files, wave 1 in design.md §8):
  - §7.1 split into the final protocol (10 eval seeds, `results/medium_10seeds/`: both baseline
    rows, retrained, B-clean, B-wide, B-region; paired against both rows and the retrained
    baselines) and the rows with 5 eval seeds (non-hier, oracle, A; paired on seeds 0–4). Every
    number recomputed with `analysis/tables.py`; equal to d-eval's table.
  - `baseline_raw` − `baseline`: +2.0 pp [−0.2, +4.2], p = 0.07 with 10 eval seeds; +0.3 pp
    [−2.6, +3.4] on eval seeds 5–9 alone (my check: the rows with `eval_seed >= 5`, same
    `tables.py`). §6 shows both the 10-seed and the 5-seed rows.
  - No pooled B family gains (+0.5 to +1.5 pp over retrained, −0.6 to −1.6 against raw). The B-clean
    seed split holds (+4.2 to +6.2 pp, p ≤ 3e-4); with 10 eval seeds the retrained seeds also
    differ (seed 1 against 3: +3.6 pp, p = 0.002). Temperatures of B-wide and B-region read from
    their flags: same upstream spread per seed as B-clean, so the same pattern (rank correlation
    with success +0.9 each, not independent evidence).
  - Held-out selection table with B-wide and B-region and 5- and 10-seed columns. **With 10 eval
    seeds the selected B-clean seed 2 is +2.2 pp [+0.1, +4.3], p = 0.047, against `baseline_raw`
    and +4.2 pp against `baseline`**, so it passes the letter of the rule. The report states it
    with three limits: 8 B tests against raw (Holm needs p < 0.006); +2.2 pp [−0.7, +5.3],
    p = 0.16 on eval seeds 5–9 alone; seed 2 is the second-sharpest temperature, so 2× and H1
    decide the cause. Coordinator's answer (option A): shown as a pass at the edge with the
    limits and a multiplicity statement (Holm-adjusted p 0.37 over the 8 B tests).
  - §8.6 (new): the hypothesis tests with the readings of design.md §8 (T1–T4, 2×, H1), a short
    summary of the three hypothesis files, and T1 filled: no ensemble passes `baseline_raw`
    (best +0.5 pp, B-region); B-region ensemble − retrained ensemble +3.0 pp [+0.9, +5.3]
    (chaining condition holds), with four limits (the CI covers episodes only, pooled +1.5 pp
    [−0.8, +3.8]; no ensemble differs from its own seeds; B-region also changes the latent type;
    two B ensembles ran).
  - §7.3: teleport trap shares (from `docs/theory.md` §11.3, Linux) replace the pending bullet;
    giant oracle failures from `results/giant/failures/oracle.csv` (77.6 % still moving, 21.7 %
    flips).
  - §1, §5 (gate), §8.1, §8.2 (oracle: 5 eval seeds), §8.5 (10-seed numbers, B-wide and B-region
    temperatures, H1 control), §9, §10 (rows with 5 eval seeds; region-target baseline as a next
    step), §11, appendix.
  - Not done: a 10-seed figure. `analysis/plots.py` cycles red and green over 9 series, so the
    figure could not tell the methods apart; the tables carry the numbers.
- After the merge of 0433cf3 (2× experiment, medium oracle on 10 eval seeds, T3, T4, held-out
  ensembles; coordinator: close B as negative, H1 cancelled):
  - §8.5 is now "why it gives no gain", with finding 4: at matched 2× sharpness B-clean 80.2 %
    against the baseline at 2× 80.9 % (−0.8 pp [−4.3, +2.5]), both at `baseline_raw` (+0.1, −0.6);
    reading: `π_h` tuning effect, no gain of chaining. The selected seed's pass does not carry
    over: at 0.210 the family is −0.6 pp against raw; selected B-2× seed 3 +0.2 pp; selected
    B-clean seed 2 level with the selected baseline 2× (+1.2 pp, p 0.43). Conclusion written.
  - New in finding 4 (not in the coordinator's message): with one temperature, B-clean seed 3
    recovers (75.8 → 81.0 %), but seed 1 stays low in both families (B 75.0 %, baseline 77.2 %;
    5.4–7.8 and 4.1–5.4 pp below each other seed) and fails task 4 (36.4 %, 48.4 %). So finding 2
    now says "partly by the temperature". At 2× both families gain on tasks 1, 2, 5 and lose on
    task 4.
  - Multiplicity: T4 adds 2 B ensembles, so 10 B tests on the final protocol; Holm-adjusted p of
    the selected seed 10 × 0.0466 = 0.47 (0.37 over the 8 before T4, kept in a bracket).
  - §8.6: T3 (fails, −9.7 pp; selection on teleport held-out goals), T4 (no gain; B-region margin
    over the retrained ensemble +0.9 pp, p 0.44, on the raw path), held-out ensembles (B-region
    ahead, +2.2 and +4.3 pp, the second p 0.034), 2× and H1 rows filled. B-region is ahead of the
    retrained ensemble in all four comparisons, significant in two; stated as an open point that
    does not isolate the critic.
  - Oracle on 10 eval seeds: +1.0 pp [−1.2, +3.2] against `baseline_raw` (+3.1 against
    `baseline`); §1, §7.1, §8.2, §8.4, §8.5, §9 use it.
  - Teleport baselines on 10 eval seeds in §6 (normalized +2.5 pp, p 0.011); T3 bullet in §7.3.
  - §1, status line, §5 gate, §9, §10, §11, appendix (2× families, T4, T3 rows).
- After the merge of 2626146 (T2; the 2× families on 10 eval seeds; gate G0 on large; the
  recorded-route controller of h-papers H1 passes on large; wave 3 in design.md §8; coordinator:
  add it as the method that passes, keep §1 for after wave 3):
  - §8.7 (new), "Recorded-route tracking: the one method that passes (large)": the method with the
    coordinator's three framings (the route comes from recorded trajectories, not from FB values;
    FB gives only the goal landmark and the executor; `π_h` executes a sequence of nearby targets),
    the protocol (G0, held-out check, final run), the held-out grid (selected `high` K 150, 58.8 %;
    5 of 6 configs 3.5–5.5 pp above `baseline`), the final run (41.2 %, +3.5 pp [+0.9, +6.0],
    p 0.009 against `baseline`, +5.6 pp against `baseline_raw`), per task (gains on 1, 2, 4;
    −9.8 pp on task 3), both eval-seed halves (+3.2, +3.8 pp), and the multiplicity: about 20
    candidate tests on 10 eval seeds, Holm-adjusted p ≤ 20 × 0.0087 = 0.17. Wave 3 (R) and (M)
    with their readings and *(pending)* results.
  - Against the oracle the recorded route is +2.2 pp [−0.4, +4.9], p 0.10 (not significant); the
    coordinator's "+2.3 pp" is the difference of the rounded rates. The report says "ahead, not
    significantly".
  - §7.2: the final protocol on large (10 eval seeds: baseline 37.7, raw 35.5, oracle 38.9,
    recorded route 41.2); the 5-seed rows stay below. §6 (large rows), §8.1, §8.2 (oracle 10 seeds
    +1.2 pp; title now "no measurable room for the privileged oracle"; the oracle is not a bound:
    the recorded route is ahead of it), §8.4 point 8 (act-ahead is the same executor on A's route).
  - §8.5 finding 4 on 10 eval seeds: baseline 2× 80.6 %, B 2× 80.1 % (−0.6 pp [−3.8, +2.3]); the
    baseline rises +2.2 pp [+0.0, +4.3], B +0.2 pp over B-clean. Selected B-2× seed 3: +0.6 pp
    against raw, +0.5 pp against the selected baseline 2× (seed 4). New: seed 2 is level with the
    best baseline-2× seed (+0.6 pp, p 0.57) but +2.2 pp (p 0.043) above the selected one, seed 4
    (the second weakest on the eval tasks). Seed 1 stays low (B −6.1 to −7.8 pp, baseline −2.6 to
    −4.1 pp). The "H1" control is now "the `V_H = V_1` control", so H1 means only h-papers H1.
  - Multiplicity of B: 12 tests on the final protocol (with B 2× pooled and selected), Holm 0.56,
    P(at least one p ≤ 0.05) 0.46 (§7.1, §8.5, §9).
  - §8.6: T2 fails (76.1 %, −4.4 pp [−7.2, −1.9] against raw, −2.3 pp against retrained; selected
    seed 0 77.5 %, −3.0 pp against raw); header without "pending".
  - §9 (T2, recorded route, 2×, room rows), §10 (the pass is not yet replicated; next steps: wave 3,
    then the FB arm h-papers H2), §11 (large and held-out commands; 2× on 10 seeds), appendix (T2,
    recorded route).
  - §1 is unchanged on purpose (coordinator); the status line says what it does not yet include.
- After the merge of 8b4795f (wave 3: large replication, H1 on medium; c-planner's note on the
  oracle; design.md §8 fixes the medium replication; coordinator: finishing phase, write the H1
  section, §9, §10, then §1 after the medium replication):
  - §8.7 rewritten as the H1 section: what it is (recorded-walk edges, xy weights, route re-planned
    every step; region latent of the route ahead as the intention; `π_l` or `π_h` acts; selected
    `window` K 150 on medium, so `π_h` is not used there; `high` K 150 on large), why it is a
    sequence of intentions, a table of what FB contributes (goal landmark by `F`, intentions by
    `B`, the executor) and what it does not (edges, weights, route; chain value, `ρ̂`, `V_H`).
  - Results, all seed sets: medium 0–9 83.8 %, +3.3 pp [+1.2, +5.4] (p 0.0027) against
    `baseline_raw`, +5.3 against `baseline`, +2.2 [+0.2, +4.3] (p 0.037) against the oracle; no
    task below `baseline`. Large 0–9 +3.5 (passes), 10–19 +1.8 (p 0.16, does not confirm), 0–19
    +2.7 (p 0.004). Medium 10–19 pending. Held-out grids of both mazes in one table: the
    configurations differ little (medium all six within 2.5 pp).
  - New from my checks: on medium the gain shrinks on the new eval seeds (against raw: 0–4
    +4.0 pp, p 0.010; 5–9 +2.6 pp, p 0.11). Not faster: median steps of successes 371 against 364
    (medium). Large 20 eval seeds: `baseline` − `baseline_raw` +2.0 pp [+0.3, +3.7], p 0.023 (§6).
  - c-planner's oracle note in §8.7: on large tasks 2, 4, 5 (same cells) H1 − oracle −1.5 pp; on
    tasks 1, 3 (another equally short path) +7.9 pp; `π_h` adds +1.8 pp (n.s.) on held-out goals.
    The paired split is recomputed from the committed CSVs with a task filter; the cell check
    itself is c-planner's scratch script (named as such). "Matches the oracle" on large; above it
    on medium.
  - Multiplicity: about 21 candidate tests on 10 eval seeds; medium pass Holm-adjusted p ≤
    21 × 0.0027 = 0.057 (P of a p this small by chance 0.055); large ≤ 0.17.
  - §7.1 (H1 row and paragraph), §7.2 (replication table), §6 (large 20 seeds), §8.2 (room for route
    following, not for the oracle), §8.4, §9 (H1 row: worked on medium, not confirmed on large),
    §10 (limits; next steps: medium replication, FB arm H2, forced-oracle check, distil H3), §11,
    appendix, status line. §1 untouched until the medium replication lands.
  - Not renumbered: H1 stays in §8.7, because the hypothesis files cite REPORT section numbers
    about 60 times; §1 will point to §8.7.
- Private defense sheet for the reviewer call (coordinator request): written outside the
  repository, not committed; updated for B, the seed selection, the ablations, giant, and the
  10-seed protocol and T1.
- Every number in the report names a committed file. Recomputed with `analysis/tables.py`
  (`--reference baseline` and `--reference baseline_raw`, `--cluster config_hash`) and
  `analysis/report_figures.py`. The paired `upstream` row of `tables.py` is not used.

## Results the report waits for

| # | Result | Report section | Owner | Expected place |
|---|---|---|---|---|
| 1 | medium replication: the recorded route (`window`, K 150) and both baseline rows on medium eval seeds 10–19 | §8.7, §7.1, §9, §10, then §1 | d-eval | `results/medium_seeds10-19/` (expected) |
| 2 | optional: c-planner's route-cell check committed (`analysis/planner_routes.py`), so §8.7 cites a committed file for "same cells" | §8.7 | c-planner | `analysis/` |
| 3 | route plots of A on medium and large, committed | §8.4 point 8 | c-planner | `docs/figures/report/` or c-planner's results |
| 4 | non-hier and approach A on eval seeds 5–9 (optional) | §7.1 | d-eval | `results/medium_10seeds/` |
| 5 | giant: optional `baseline_raw` | §7.3 | d-eval | `results/giant/` |
| 6 | large: retrained baselines; failure classes of the A variants | §7.2 | d-eval | `results/large/` |
| 7 | whether the wormholes that the xy cap leaves cross a wall (a column of the calibration CSV) | §8.3, §4 | d-eval | `results/calibration/` |

Every needed result has an owner. The central finding (no measurable room for a high level with
this `π_l`) now holds for the privileged oracle only: the recorded route passes the bar on medium
and matches or beats the oracle (§8.7). §1 still states the old wording; it is rewritten when the
medium replication lands (coordinator).

## Next

- When the medium replication lands: its row in §8.7 by the reading of design.md §8, medium
  eval seeds 10–19 and 0–19 in §7.1 and §8.7, §9, §10; then §1, written last.
- Then the other items above.
- Merge `main` when new results are merged.

## Blockers

- None.

## Requests to the coordinator

- d-eval: the calibration CSV counts the wormholes within the xy cap (1,965 on medium) but not
  whether they cross a wall. Your status says they cross none (short pairs, FB overconfident at
  short range). Please add that count as a column (item 7), so §8.3 can state it.
- d-eval (optional, item 4): the medium oracle on eval seeds 5–9 (1,250 episodes), so the
  headroom bound of §8.2 pairs with the 10-seed baseline rows. With 5 eval seeds the normalized
  baseline sat 1.4 pp below its 10-seed value, so the oracle's +4.2 pp against it may shrink.
- Answered (coordinator, 2026-10-30): option A plus a multiplicity statement. The report shows the
  selected B-clean seed as a pass at the edge, with the three limits, and adds: 8 B tests on the
  final protocol; P(at least one p ≤ 0.05 | no effect, 8 independent tests) = 0.34; Holm-adjusted
  p of the smallest (McNemar 0.0466) = 0.37, no pass. B's conclusion stays open until 2× and H1.
- d-eval: the oracle request (eval seeds 5–9) was routed by the coordinator.
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
