# Status: b-theory

Last update: 2026-10-30 (after d-eval's T3 runs). Branch merged with `main` 416d90c. Current
work: none open. T3 of wave 1 (design.md §8), the trap-aware controller on teleport, is done and
fails: −9.7 pp [−11.6, −7.7] against the baseline on eval seeds 0–9 (result below, theory.md
§11.4). Item 15 (bonus mazes) done: all final numbers are d-eval's Linux runs, theory.md §11 uses
them. Review: first pass (B1–B15, X2, X13, X7 my part, K2) and second pass (B16–B21) done.

## T3: trap-aware controller (teleport, wave 1)

- Code: `fbplan/controllers/trap_aware.py`, factory `make_trap_aware` (commit 7123688). Tests:
  `tests/test_tabular_trap_aware.py`, 24 tests on the exact tabular teleport maze.
- Method (h-exec H1): `z_t = normalize(π_h(s, z_r) − weight · g(s, t) · ẑ_trap)`.
  - Trap from the train data only: the states after a trajectory's last jump (xy step > 2) are
    trap states if they all stay within `trap_radius` of their mean xy. `ẑ_trap` = normalized
    centred mean of B over a seeded pool of 200k train states.
  - Gate (h-exec's rule): open while `log ρ̂_walk / log γ ≤ 1000 − t`, with
    `ρ̂_walk = min_h F_h(s, z_c)ᵀẑ_r / F_goal` and `z_c = normalize(ẑ_r − weight · ẑ_trap)`. One
    change: h-exec takes F_goal over the goal-ball states, which the controller cannot know
    without `goal_xy`. I take the 64 pool states with the highest `B(s)ᵀẑ_r` (median distance to
    the goal 2.4–3.3 xy; F_goal within 20 % of the goal ball's on 5 custom pairs).
  - With weight 0, or while the gate is closed, the actions equal the baseline's (tested).
- Tabular tests (exact): the rule marks exactly the trap states of the trajectories that end in
  the trap. The gate counts the steps of the walk that avoids the entries (41, 28, 24, 46, 47
  fine steps on tasks 1–5). Where the walk fits, the controller walks (success 1.0, trap 0) and
  the baseline jumps (trap 0.42–0.72). Where it does not fit, the gate hands back to the
  baseline (ungated: success 0).
- Real data (Mac, analysis only, no evaluation episode): jumps are ≥ 2.27 xy and walking steps
  ≤ 0.37. Stay radius 2.5–4.0 catches all 247 train trajectories that end in the trap cell (the
  map checks the rule, the rule does not use it), with 10–14 false alarms; 16.4 % of the states.
  `B(s)ᵀẑ_trap` separates the trap cell with AUC 0.991. Build 1.8 s per worker, 4 ms per step.
- Two findings that limit T3:
  - The radius is not a knob: `ẑ_trap` for radius 1.5, 2.5 and 4.0 have cosines ≥ 0.997.
  - The FB gate cannot see the walk early in an episode. The decision falls when the walk needs
    600–1,000 steps, where `ρ = 0.99^steps` ≤ 0.0025, far below FB's noise floor on teleport
    (17.8 % of pairs have ρ̂ ≤ 0; theory.md §11.1). On h-exec's 5 custom pairs (walks of 12, 20,
    19, 10 and 12 cells) the walk estimates at the start are 337, 256, none, 203 and none steps.
    h-exec's ungated penalty gained on pairs 1 and 5 and lost on 2 and 4. This gate stays open
    on 2 and 4 and closed on 5. Two other estimates did not do better: c-planner's landmark
    graph with R = 0 (4–11 steps per cell, wrong order) and Dijkstra over train trajectories
    (6–16 steps per cell; π_l needs about 40–65).

### T3 held-out selection (rule fixed on 2026-10-30, before any held-out result)

- Grid, gated (candidates): weight ∈ {0.5, 1, 2} × trap_radius ∈ {1.5, 2.5}, gate `time`.
  Control, not a candidate (h-exec M11): gate `none`, trap_radius 2.5, weight ∈ {0.5, 1, 2}.
  References: baseline, baseline_raw.
- Episodes: the teleport held-out goals (`--heldout`), eval seeds 0 1 2, 20 episodes per goal
  and eval seed (600 per row), as for medium.
- Rule: the candidate with the highest held-out success; ties go to the smaller weight, then to
  radius 2.5. Nothing else is tuned. Final run: that candidate on the 5 eval tasks, eval seeds
  0–9, 50 episodes, paired against baseline and baseline_raw on the same seeds (design.md §8).

Commands for d-eval (Linux; a code tree at main cc11eb3 or later: it has
`fbplan/controllers/trap_aware.py` and `results/heldout_goals/antmaze-teleport-navigate-v0.csv`).
Step 1, held-out grid (11 runs, 600 episodes each):

    E="--env antmaze-teleport-navigate-v0 --heldout --seeds 0 1 2 --episodes 20 --workers N"
    T=fbplan.controllers.trap_aware:make_trap_aware
    D=results/teleport/heldout
    uv run python -m fbplan.evaluate $E --method $T --opt weight=0.5 trap_radius=1.5 --out $D/trap_aware/w0.5_r1.5.csv
    uv run python -m fbplan.evaluate $E --method $T --opt weight=0.5 trap_radius=2.5 --out $D/trap_aware/w0.5_r2.5.csv
    uv run python -m fbplan.evaluate $E --method $T --opt weight=1.0 trap_radius=1.5 --out $D/trap_aware/w1.0_r1.5.csv
    uv run python -m fbplan.evaluate $E --method $T --opt weight=1.0 trap_radius=2.5 --out $D/trap_aware/w1.0_r2.5.csv
    uv run python -m fbplan.evaluate $E --method $T --opt weight=2.0 trap_radius=1.5 --out $D/trap_aware/w2.0_r1.5.csv
    uv run python -m fbplan.evaluate $E --method $T --opt weight=2.0 trap_radius=2.5 --out $D/trap_aware/w2.0_r2.5.csv
    uv run python -m fbplan.evaluate $E --method $T --opt weight=0.5 gate=none --out $D/trap_aware/w0.5_ungated.csv
    uv run python -m fbplan.evaluate $E --method $T --opt weight=1.0 gate=none --out $D/trap_aware/w1.0_ungated.csv
    uv run python -m fbplan.evaluate $E --method $T --opt weight=2.0 gate=none --out $D/trap_aware/w2.0_ungated.csv
    uv run python -m fbplan.evaluate $E --method baseline --out $D/baseline.csv
    uv run python -m fbplan.evaluate $E --method baseline_raw --out $D/baseline_raw.csv

Step 2, selection (mechanical, the rule above): among the 6 gated CSVs (`w*_r*.csv`), take the
highest success over all 600 episodes; ties go to the smaller weight, then to trap_radius 2.5.
The ungated rows and the baselines are not candidates.

Step 3, final run with the selected W and R:

    uv run python -m fbplan.evaluate --env antmaze-teleport-navigate-v0 --tasks 1 2 3 4 5 --seeds 0 1 2 3 4 5 6 7 8 9 --episodes 50 --workers N --method fbplan.controllers.trap_aware:make_trap_aware --opt weight=W trap_radius=R --out results/teleport_10seeds/trap_aware.csv

It pairs with baseline and baseline_raw on eval seeds 0–9 (d-eval adds seeds 5–9 to both rows).
d-eval ran it with W = 0.5, R = 1.5 and wrote it next to the 10-seed baselines in
`results/teleport_10seeds/` (the path above is the one used).

Check on the 10 teleport held-out goals (Mac, no episodes; controller reset and the gate at the
start state, weight 1):

- Only 3 of the 10 goals (103, 107, 110) have a teleport entry on the plain route; 4 of the 5
  eval tasks have one. So the held-out grid mostly measures the cost of the penalty and will
  likely favour a small weight. The rule stays as fixed.
- The gate's walk estimates at the start do not follow the safe walks (10–23 cells): goal 110
  (23 cells) gets 61 steps, goal 109 (10 cells) 772, goals 104, 107 and 108 no estimate
  (ρ̂ ≤ 0). The data-only goal states lie 1.5–3.9 xy from the goal (103: 6.8).

### T3 result (d-eval, Linux; analysis on the Mac, no episodes)

- Held-out grid (`results/teleport/heldout/`, 600 episodes per row): baseline 48.2 %,
  baseline_raw 46.5 %; gated λ 0.5 46.5 / 44.7 % (radius 1.5 / 2.5), λ 1 40.2 / 39.8 %, λ 2
  27.2 / 29.7 %; ungated λ 0.5 / 1 / 2: 44.2 / 37.8 / 29.8 %. The rule selects λ = 0.5, radius
  1.5 (d-eval applied it; I checked it). No row beats the baseline (best −1.7 pp [−6.2, +2.8]).
- Final run (`results/teleport_10seeds/trap_aware.csv`, 2,500 episodes, paired): 17.9 %
  [16.4, 19.4] against 27.6 % (baseline, −9.7 pp [−11.6, −7.7], p 3e-21) and 25.1 %
  (baseline_raw, −7.2 pp [−9.2, −5.2], p 8e-13). Trap 54.8 → 51.8 %, other failures 17.6 →
  30.2 %, median steps of successes 502 → 574. T3 fails the rule.
- Mechanism (theory.md §11.4):
  - The penalty moves 3 points from the trap to the time limit, and it costs 7.1 pp on the 4
    held-out goals where the baseline rarely ends in the trap (nothing to avoid there).
  - The time gate has no measurable effect: gated − ungated = +2.3 pp (p 0.33) at λ 0.5.
  - At the start states of eval tasks 3–5 the gate's walk estimate is 58–77 steps (the jump
    route; safe walks 14–27 cells), at every λ from 0 to 4 in `z_c`; on tasks 1 and 2 it has
    no estimate (ρ̂ ≤ 0). F's trap occupancy under `z_c` falls only from 0.15 to 0.11. So F
    does not model the trap-averse route, and the gate stays open on the long walks.
  - The held-out goals are easier (48 % against 28 %), so they under-predicted the cost
    (−1.7 against −9.7 pp).
- Gate check at the eval starts: `make_trap_aware(model, weight=λ, trap_radius=1.5)`, `reset`
  on each eval task, `walk_steps` at the env reset state for reset seeds
  `reset_seed(0, task, s)`, s = 0–4; λ set to 0, 0.5, 1, 2, 4. Scripts (not committed):
  `/private/tmp/claude-501/-Users-ssbars-Documents-work-tbank-robots--worktrees-b-theory/9eb5871f-f0ef-430e-94a3-2c926e648b0f/scratchpad/t3/eval_gate.py`
  and `gate_weight.py` in the same folder; `split.py` there gives the trap / other split. I can
  commit them if you want them in the repo.
- Reading: three ways to walk instead of jump now lose to the baseline on teleport (planner,
  teleport-aware oracle, T3). The frozen π_l is too slow for the safe walks, and FB cannot tell
  where a walk fits. I do not propose a T3 variant; a faster π_l (h-exec H2) comes first.

## Done

- Brief items (earlier): `fbplan/tabular/` (exact mazes, fake FB, option theory, study),
  `docs/theory.md` §1–10, tests `tests/test_tabular_*.py`.
- Item 15, giant and teleport (design.md §6):
  - Train splits of giant and teleport downloaded (240 MB each).
  - `fbplan/controllers/oracle_teleport.py`: teleport-aware oracle (privileged). Plans on the cell
    MDP with the jumps: `objective=reach` (default) walks the shortest route that enters no
    teleporter; `objective=discounted` maximises `E[γ_cell^H]` as FB does. Entry cells use the
    trigger disk (1.5) as waypoint region. `follow=pi_h` gives the waypoint to π_h.
    Tests: `tests/test_oracle_teleport.py` (new file, not in my brief list; please assign it).
  - `fbplan/tabular/time_limit.py`: P(H ≤ T) vs. `E[γ^H]` on the cell mazes, the walk threshold
    per task; outputs `docs/figures/theory/time_limit*.{csv,png}`.
  - `docs/theory.md` §11: teleport (risk-neutral value vs. success metric) and giant (time budget,
    ρ at long range).
  - Evaluations on the Mac stopped by the coordinator's decision (Linux is the platform for final
    numbers). d-eval ran the commands below on Linux; theory.md §11.3 uses only those CSVs. The
    old Mac CSVs stay in `runs/mac/{giant,teleport}/` (not committed).
  - FB calibration on giant (again, with d-eval's corner fix) and on teleport (new), figures and
    tables in `docs/figures/theory/`. Found a second geodesic leak (Requests).

## Review (g-release, first pass)

- X2 / B1: the study now gets ρ̂, R̂ and V_d from c-planner's `OptionModel`, and trust, value
  iteration and Q-values from `fbplan/graph.py`. One vocabulary: `head_agg`, `edge_reward`
  exact | paper | zero, `hop_discount` (a factor, not an added cost); class `StudyPlannerConfig`.
  `options.py` keeps only exact quantities from the true model (docstring says so);
  `graph_value_iteration` is now `graph.value_iteration` with every edge trusted (e-learner's
  test calls it). New test pins `OptionModel` on the exact fake to the exact option tables, and
  the "paper" R̂ to the value of following π_w forever. The study was rerun (numbers in
  theory.md §9 changed; see there).
- B2: removed `replanning_policy(subgoals)` and `graph_value_iteration(hop_cost, n_iters, init)`.
- B3: removed `FakeFBConfig.high_advantage` (only the upstream proxy remains).
- B4: `centre_state` docstring fixed (+x, +y state of the 4 for even k); test added.
- B5: `_graph_inputs` docstring and types.
- B6: `run_study` docstring (time measured with 4 workers).
- B7: `--quick` writes to `runs/run_study/` by default; `--out-dir`; `--workers` defaults to 4;
  help texts.
- B8: value iteration has `max_iters` and raises when it does not converge.
- B9: `FakeFBConfig` and `StudyPlannerConfig` reject unknown values; tests.
- B10: new tests for every ✓ that theory.md cites: `ρ ≤ γ^d` (d counts teleport jumps),
  `ρ = γ^d` for a shortest-path option, the proxy gap (0.56 medium, 0.36 teleport), the
  contraction and the self-loop counterexample, the 7/26 and 20/45 re-planning counts.
- B11: the task-free ρ̂ table needs no z_r (`LandmarkModel.table`); `reward_rows` builds each
  plan once (`make_plan`, then `plan_actions` and `plan_route` from the plan).
- B12: `study_plots` imports `FAR`, the planner names and the output folder from `run_study`.
- B13: names `v_direct` (V_d), `v_graph` (V_H), `r_before` (R) in all my files.
- B14: the fake-FB test compares values, not actions (ties); the env test checks geometry, not
  the transition table; the study test that restated R̂ is replaced by the X2 pin test.
- B15 (N): docstrings for `plot_*` and `task_spec`; the "+2" of the commit timeout documented.
  Not done: `matplotlib.use` at import (the Agg backend must be set before pyplot loads).
- X13: `wormhole_mask` → `rho_bound_wormholes` (ρ̂ > 2γ^d); its docstring names d-eval's other
  definition.
- X7 (my part): `tests/test_tabular_maze.py` compares all 4 maps, tasks and teleporters with
  OGBench.
- Speed: the fake's latent grouping uses a checked random projection instead of `np.unique` on
  rows: 3× faster, byte-identical study output (quick run compared).
- K2: `temperature` and `seed` dropped from `fake_fb.py` (and the test fakes) after a-frozen's
  protocol change.

## Review (g-release, second pass)

- B16: `time_limit` creates its output folder first; `--out` → `--out-dir` with help; test.
- B17: theory.md §11 exists (commit fc62526, after the review's snapshot) and holds the
  time-limit results (walk thresholds 0.9977–0.9993, success ≤ 1/2).
- B18: `make_plan` calls `graph.best_options` (the planner's tie rules). theory.md §8 now says
  that the study shares the plan code, not the execution (reach radius, look-ahead, commit
  timeout). Study rerun: only exact-model ties change (see theory.md §9.4).
- B19: done in 8f58343 (after the review's snapshot).
- B20: new pin tests in `tests/test_tabular_time_limit.py`: on every cell and teleport task the
  oracle's `reach_plan` has the same hitting-time distribution as `walk_policy` and never
  jumps; `discounted_plan` reaches the tabular optimum (γ 0.67 and 0.99). I kept the oracle's
  own code instead of rebuilding it on `GridMaze`: the Linux queue runs the oracle at commit
  84a8127, and the pins show the plans are equivalent. The neighbour helper stays: it takes the
  map as an argument (the oracle needs it on the map with the entries walled), and
  `MazeGeometry._free_neighbours` is private and bound to its own map.
- B21: this status has review responses for both passes; run_study's time is "about 14 min with
  4 workers"; test count below.
- N: test maps now come from `GridMaze` (no copies; `SAFE_WALK_CELLS` stays once, in the
  time-limit tests, as the hand count); `discounted_values` has `max_iters` and raises (test);
  `PlannerKind` is the single source of the kinds (`get_args`); redundant `noqa: E402` removed
  (the per-file ignore covers them); the proxy-gap test asserts 0.561 / 0.355, the values in
  theory.md. Not mine: the `oracle_teleport` alias and the privileged set in `tables.py` /
  `plots.py` (d-eval, coordinator).
- X11 adopted: `run_study --out-dir` (done before), `study_plots [csv_dir] [--out-dir DIR]`
  (positional input folder), `time_limit --out-dir`; all three use
  `ArgumentDefaultsHelpFormatter` (request to g-release below).
- Tests: all pass (`uv run pytest`), ruff clean.

## Results

Linux (d-eval, main 28f6aba), 5 eval seeds × 5 tasks × 50 episodes, planner rows pool landmark
sets 0–2; success % [95 % CI], paired against the baseline on the same episodes. Trap: failed
episode whose final goal distance is within 2 xy units of the trap centre's.

| teleport | all | vs. baseline (pp) | trap |
|---|---|---|---|
| baseline | 27.7 [25.3, 30.2] | – | 54.7 % |
| baseline_raw | 24.2 [21.8, 26.6] | −3.5 [−6.2, −1.0] | 57.4 % |
| nonhier | 19.0 [16.9, 21.2] | −8.6 [−11.2, −6.2] | 65.4 % |
| planner | 18.6 [16.5, 21.0] | −9.1 [−12.2, −6.0] | 64.2 % |
| planner_high | 24.1 [21.5, 26.4] | −3.6 [−6.7, −0.5] | 53.9 % |
| teleport-aware oracle, π_l | 5.6 [4.3, 6.9] | −22.1 [−24.7, −19.5] | 19.9 % |
| teleport-aware oracle, π_h | 21.4 [19.4, 23.3] | −6.3 [−9.2, −3.4] | 18.2 % |

| giant | all | vs. baseline (pp) |
|---|---|---|
| baseline | 0.5 [0.2, 0.9] | – |
| nonhier | 0.5 [0.2, 0.9] | +0.0 [−0.6, +0.6] |
| planner | 0.4 [0.2, 0.7] | −0.1 [−0.6, +0.4] |
| planner_high | 0.8 [0.5, 1.1] | +0.3 [−0.2, +0.9] |
| oracle (privileged) | 2.5 [1.7, 3.4] | +2.0 [+1.1, +3.0] |

- Teleport: no method beats the baseline. 54–65 % of the episodes of the learned agents end in
  the trap. An exact risk-neutral agent (max `E[γ^H]`, γ = 0.99) jumps on every task and
  succeeds with probability ≤ 1/2 (theory.md §11.1). The FB checkpoint has learned the jump:
  for pairs 60–75 xy units apart on foot, `ρ̂` is 0.15–0.22, against 0.004–0.008 for the walk.
- Teleport-aware oracle: the risk-free route cuts the trap share to 18–20 %, but π_l and π_h
  still walk into entry (4,6) next to the start of tasks 3 and 5 (37–39 % trapped there), and
  the long safe walks time out. On task 2 (no entry on the route) it gains +16.8 pp
  [+8.0, +25.6] with π_h.
- Giant: no planner gain. Even privileged waypoints give 2.5 %: 77.6 % of the oracle's failures
  are still moving at the time limit, 21.7 % are flips (Linux, `results/giant/failures/`). π_l
  covers 5–15.5 cells in 1,000 steps (Mac replay, eval seed 0), and the walks are 17–30 cells. FB
  distance grows with the walk at 1.2–1.7 times the walk time, but from 5 cells on, the upper
  quartile is at the cap (`ρ̂` < 0.05).

## Commands for d-eval (Linux evaluation instance)

Every run: `--seeds 0 1 2 3 4 --episodes 50`. `M=fbplan.controllers.oracle_teleport:make_teleport_oracle`.

    uv run python -m fbplan.evaluate --env antmaze-teleport-navigate-v0 --seeds 0 1 2 3 4 --episodes 50 --workers N --method <method> [--opt ...] --out results/teleport/<csv>
    uv run python -m fbplan.evaluate --env antmaze-giant-navigate-v0    --seeds 0 1 2 3 4 --episodes 50 --workers N --method <method> [--opt ...] --out results/giant/<csv>

In the `RUNS` format of `scripts/reproduce.sh` (maze, CSV, eval seeds, method, options):

    teleport baseline.csv            5 baseline
    teleport nonhier.csv             5 nonhier
    teleport planner/l0.csv          5 planner landmarks.seed=0
    teleport planner/l1.csv          5 planner landmarks.seed=1
    teleport planner/l2.csv          5 planner landmarks.seed=2
    teleport planner_high/l0.csv     5 planner_high landmarks.seed=0
    teleport planner_high/l1.csv     5 planner_high landmarks.seed=1
    teleport planner_high/l2.csv     5 planner_high landmarks.seed=2
    teleport oracle_teleport.csv     5 fbplan.controllers.oracle_teleport:make_teleport_oracle
    teleport oracle_teleport_pi_h.csv 5 fbplan.controllers.oracle_teleport:make_teleport_oracle follow=pi_h
    giant    baseline.csv            5 baseline
    giant    nonhier.csv             5 nonhier
    giant    oracle.csv              5 oracle
    giant    planner/l0.csv          5 planner landmarks.seed=0
    giant    planner/l1.csv          5 planner landmarks.seed=1
    giant    planner/l2.csv          5 planner landmarks.seed=2
    giant    planner_high/l0.csv     5 planner_high landmarks.seed=0
    giant    planner_high/l1.csv     5 planner_high landmarks.seed=1
    giant    planner_high/l2.csv     5 planner_high landmarks.seed=2

Analysis for theory.md §11.2–11.3 (done on the Mac; tables only):

    uv run python analysis/tables.py results/teleport/*.csv results/teleport/planner*/*.csv --label-root results/teleport --cluster config_hash
    uv run python analysis/tables.py results/giant/*.csv results/giant/planner*/*.csv --label-root results/giant --cluster config_hash
    uv run python analysis/failures.py results/giant/oracle.csv --method oracle

Docs figures and tables (Mac, light):

    uv run python -m fbplan.tabular.time_limit        # docs/figures/theory/time_limit*.{csv,png}
    uv run python analysis/calibration.py --env antmaze-giant-navigate-v0 --out-dir docs/figures/theory --table-dir docs/figures/theory
    uv run python analysis/calibration.py --env antmaze-teleport-navigate-v0 --out-dir docs/figures/theory --table-dir docs/figures/theory
    uv run python -m fbplan.tabular.run_study --workers 4 && uv run python -m fbplan.tabular.study_plots

## Next

- None for T3 (done).
- When d-eval fixes the slit leak in `MazeGeometry.geodesic_distances` (Requests): run the two
  calibration commands again, and replace the raster-check numbers in theory.md §11.1–11.2 with
  the bins of the fixed figure.

## Blockers

- None.

## Requests to the coordinator

- Please assign `tests/test_oracle_teleport.py` and `fbplan/controllers/trap_aware.py` to
  b-theory in design.md §6 (line 157 lists neither). The T3 tests are in
  `tests/test_tabular_trap_aware.py`, which my `tests/test_tabular*.py` already covers.
- T3, A/B question: closed. A ran as registered. B was to follow only if a held-out row beat
  the baseline; none did, so B is not needed.
- g-release (X11 flag renames): `scripts/reproduce.sh:272-273` still use the old flags and fail
  with the current CLIs. They become `py -m fbplan.tabular.study_plots "$theory_dir"` and
  `py -m fbplan.tabular.time_limit --out-dir "$theory_dir"`. The giant and teleport runs are in
  "Commands for d-eval" above.
- d-eval: `MazeGeometry.geodesic_distances` still lets a point pass through walls. Two wall cells
  that share an edge leave a slit of 2e-6 (`_WALL_SHRINK`), and a segment that lies exactly on
  that grid line passes between them. The corner graph has many such segments (corner to corner
  along a grid line). Example: giant task 5, (32, 16) → (28, 8), measures 21.2 xy units via the
  corners (34, 18) → (34, 6), down x = 34 between walls (4, 9) and (4, 10); the walk is 17 cells.
  Check against an octile Dijkstra on a 0.5-unit raster (at most 8 % too long): pairs of the
  400-state calibration sample more than 15 % too short: giant 31 %, large 13 %, teleport 16 %,
  medium 0.2 %. Task geodesics, MazeGeometry vs. raster: giant 76.4/75.1/69.0/51.1/21.2 vs.
  91.7/81.8/83.8/78.2/51.9; large task 2 21.2 vs. 55.7. It affects the large and giant
  calibration figures and wormhole counts (`results/calibration/`), not medium. A possible fix:
  block a segment that runs along a grid line where the cells on both sides are walls (or test
  the midpoint of each grid-cell crossing against the unshrunk walls). My check script:
  `/private/tmp/claude-501/-Users-ssbars-Documents-work-tbank-robots--worktrees-b-theory/9eb5871f-f0ef-430e-94a3-2c926e648b0f/scratchpad/geo_check.py`
  (I can commit it if you want it in the repo).
