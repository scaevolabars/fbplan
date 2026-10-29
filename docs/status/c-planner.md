# Status: c-planner

Last update: 2026-10-29.

## Done

- `fbplan/graph.py` (pure numpy): trusted edges (`rho_min`, top-k, self-loops removed),
  SMDP value iteration with a cost per hop, Q-values from any source, route extraction.
- `fbplan/option_model.py`: batched `rho(s,w)`, `R(s,w)`, direct value `F(s,z_r)ᵀz_r`;
  head aggregation min / mean; edge reward modes exact / paper / zero; clip counts.
- `fbplan/landmarks.py`: farthest-point sampling on xy or on normalized `B(s)`, full observations.
- `fbplan/controllers/planner.py`: `GraphPlannerController`, `GraphPlusHighController`,
  `PlannerConfig` (frozen dataclass). Edge table cached per env, plan cached per task.
  `route(obs)` gives the planned landmark sequence for route plots.
- CLI factories for d-eval's `fbplan.evaluate`: `fbplan.controllers.planner:make_planner` and
  `:make_planner_high`. Options are `key=value`; a dotted key sets a nested field, for example
  `--opt rho_min=0.4 landmarks.count=500 options.head_agg=mean`. Unknown keys raise an error.
- Tests: 52 pass (`uv run pytest`), ruff clean. They run on an exact tabular fake,
  `tests/planner_fakes.py` (`rho = gamma^distance`; knobs for one-head wormholes and a horizon
  beyond which FB and `pi_z` fail). Closed loop on the fake: the planner reaches a goal 10 steps
  away with 4-step edges via 2 waypoints, where the task policy alone stands still.
  Mutation checks: removing the reached-exclusion, the timeout or the reached check each fails a test.

## Implementation choices (no change to the math in design.md)

1. Batching of the N×N table lives in `OptionModel.pair_table` (it needs FB). `graph.py` stays pure numpy.
   Pairs go through F one batch at a time, padded to a power of two (few jit shapes, bounded memory).
2. Heads: `rho_h`, `R_h`, `direct_h` per head, then min or mean. `rho` is clipped to [0, 1] after aggregation.
   A target with `F_h(w,z_w)ᵀB(w) <= 0` gets `rho_h = 0` (unreachable).
3. Cost per hop: a factor `hop_discount` on `rho` in the Bellman backup. It does not depend on the
   scale of `z_r`, and in shortest-path form it adds `-log(hop_discount)` to the edge cost `-log rho`.
4. `edge_reward = "zero"` (R = 0) is a third mode next to exact and paper: the shortest-path case of §3.
5. At a replan, landmarks with `rho(s,w) >= rho_reached` are excluded. Without this, the landmark under
   the agent has Q = V_H, wins ties, and the agent stalls.
6. No trusted edge from `s`: the task policy `pi_l(z_r)` acts. This is what the graph Bellman equation gives.
7. The task policy option is also a commitment: replan after `commit_timeout` steps.

## Findings on the real model (early probe)

Probe with a-frozen's uncommitted loader (loaded by path, read-only), medium, landmarks from the
validation split. Scripts are throw-away (not committed). Redo the numbers after a-frozen merges.

- Cost on CPU: N = 100 / 200 / 400 landmarks: edge table 0.3 / 0.5 / 1.5 s, one replan 3 / 4 / 10 ms.
  Landmark count is not a cost problem in this range.
- Before clipping: 8–10 % of pairs have rho < 0 and about 1 % have rho > 1 (N = 200: 3950 / 401 of 40000).
- Calibration (FB steps = log rho / log gamma, N = 200): xy distance [0,2) → median 26 steps;
  [8,16) → 154; [16,∞) → 230, p90 2749.
- Default config (exact R, min heads, rho_min 0.3) gives bad routes: 13–14 hops, jumps of 18 xy units,
  route end up to 15.6 from the goal. Causes:
  1. Exact R is noise that the max collects: on trusted edges median +2.1, p99 +68 (max direct value 96).
  2. Pairs with rho > 1 are clipped to 1 and become free hops across the maze. rho_min and top-k cannot remove them.
  3. Edges with 0.7 <= rho < 1 still jump 16–19 xy units (wormholes).
- Guards tried (R = 0, N = 200, rho_min 0.5; mean xy distance route end → goal over 5 tasks; longest hop):
  none: 11.5 / 12.2 — drop rho >= 1: 3.1 / 17.4 — drop rho >= 1 + xy cap 6: 2.3 / 5.8 —
  drop rho >= 1 + two-way min(rho(u,w), rho(w,u)): 5.4 / 15.5; with rho_min 0.7 the graph disconnects.
- Closed loop, 5 tasks × 2 episodes (too small to rank): baseline pi_h 6/10, pi_l(z_r) 8/10,
  planner with drop rho >= 1 + xy cap 6 = 6/10, same + pi_h 5/10. The planner replans 12–41 times
  per episode and often runs to 1000 steps: rho_reached = 0.9 and the 100-step timeout need tuning.

## Next

1. After a-frozen merges: time the edge table and the per-step cost on medium (CPU).
   Choose the landmark count and `rho_min` (trade-off: ask the user).
2. Route plots on the medium maze (xy) for the 5 tasks; edge count and clip statistics.
3. Smoke run: 5 tasks × 10 episodes, planner vs. baseline, with a minimal loop.
4. Switch the tests to `fbplan/tabular/fake_fb.py` when b-theory merges it.

## Blockers

- Waiting for the user: which wormhole guards to adopt (drop rho >= 1 instead of clipping it to 1;
  xy cap or two-way consistency; R = 0 as the default for navigate tasks). This changes the clipping
  rule in my brief and the guard list in design.md §3.
- Final numbers wait for a-frozen to merge (`load_model`, `load_dataset`, `make_env`).

## Requests to the coordinator

- `fbplan/controllers/__init__.py` belongs to d-eval. Until it exists, `fbplan.controllers.planner`
  imports as a namespace package. That works; no action needed unless d-eval's `__init__` re-exports.
- d-eval: please add aliases `planner` -> `fbplan.controllers.planner:make_planner` and
  `planner_high` -> `fbplan.controllers.planner:make_planner_high` to `ALIASES`.
- Landmark tests are in `tests/test_graph.py`, because my brief lists no `tests/test_landmarks.py`.
- I need a file for the route-plot / edge-stats / smoke-run script. Proposal: `analysis/planner_routes.py`,
  owned by c-planner (d-eval owns the rest of `analysis/`).
