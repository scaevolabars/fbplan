# Status: b-theory

Last update: 2026-10-29. Branch merged with `main` 1903bf2. Item 15 (bonus mazes): code, theory
and preliminary Mac runs done; final runs handed to d-eval (Linux), commands below. First-pass
review items B1–B15, X2, X13, X7 (my part) and K2 done.

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
    numbers). Finished Mac CSVs are in `runs/mac/{giant,teleport}/` (not committed); theory.md
    §11.3 quotes them as preliminary.

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

## Results (preliminary, Mac, MuJoCo 3.1.6, planner code of f1ae57d)

5 eval seeds × 5 tasks × 50 episodes; success % [95 % CI]; paired against the baseline.

| maze | baseline | nonhier | planner (landmarks 0) | oracle |
|---|---|---|---|---|
| teleport | 28.8 [26.4, 31.2] | 19.0 [16.9, 21.2], −9.8 pp | 18.2 [16.2, 20.4], −10.6 pp | – |
| giant | 0.6 [0.2, 1.0] | 0.7 [0.3, 1.2], +0.2 pp | – | 2.2 [1.5, 3.0], +1.7 pp (privileged) |

- Teleport: 55 % (baseline), 65 % (nonhier), 66 % (planner) of all episodes end in the trap exit.
  An exact risk-neutral agent (max `E[γ^H]`, γ = 0.99) takes the teleporter on every task and
  succeeds with probability ≤ 1/2; it walks only for γ ≥ 0.9977–0.9993 (theory.md §11.1).
  π_l itself walks into the entries (the data route through them: 91 up vs. 2 down at (5,6)).
- Giant: even privileged waypoints give 2.2 %. π_l covers 5–15.5 cells in 1,000 steps (walks
  are 17–30 cells; 19 % of oracle failures are flips). The executor, not the route, is the
  bottleneck. FB distance stops tracking the geodesic beyond about 45 xy units.
- Teleport oracle on 30 dev episodes (eval seed 100): 2/30 with π_l, 6/30 with `follow=pi_h`,
  baseline 8/30.

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

After the runs (analysis, for theory.md §11.2–11.3):

    uv run python analysis/tables.py results/teleport/*.csv results/teleport/planner*/*.csv --label-root results/teleport --cluster config_hash
    uv run python analysis/tables.py results/giant/*.csv results/giant/planner*/*.csv --label-root results/giant --cluster config_hash
    uv run python analysis/failures.py results/giant/oracle.csv --method oracle

Docs figures and tables (Mac, light):

    uv run python -m fbplan.tabular.time_limit        # docs/figures/theory/time_limit*.{csv,png}
    uv run python analysis/calibration.py --env antmaze-giant-navigate-v0 --out-dir docs/figures/theory --table-dir docs/figures/theory
    uv run python -m fbplan.tabular.run_study --workers 4 && uv run python -m fbplan.tabular.study_plots

## Next

- When d-eval commits the Linux CSVs: replace the preliminary numbers in theory.md §11.3 (and
  the trap shares: final distance within 2 xy units of the trap centre's distance to the goal).

## Blockers

- None.

## Requests to the coordinator

- Please assign `tests/test_oracle_teleport.py` to b-theory (design.md §6).
- d-eval: `MazeGeometry.geodesic_distances` lets a point pass between two walls that touch only
  at a corner (giant 4 places, medium 2, teleport 2). The ant cannot pass there, so some
  calibration pairs look closer than they are (giant task 5: 21 xy units, walk 17 cells).
- d-eval: `analysis/calibration.py` failed on teleport (`geodesic.max()` is inf: the trap exit
  cannot be reached on foot). `analysis/failures.py` could report a "trapped" class
  (`cells_left = −1`); I used `replay_failures` from a scratch script.
- d-eval: please run the commands in "Commands for d-eval" on the Linux instance and commit the
  CSVs to `results/{giant,teleport}/`.
