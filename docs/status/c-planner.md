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

## Next

1. After a-frozen merges: time the edge table and the per-step cost on medium (CPU).
   Choose the landmark count and `rho_min` (trade-off: ask the user).
2. Route plots on the medium maze (xy) for the 5 tasks; edge count and clip statistics.
3. Smoke run: 5 tasks × 10 episodes, planner vs. baseline, with a minimal loop.
4. Switch the tests to `fbplan/tabular/fake_fb.py` when b-theory merges it.

## Blockers

- Real-model work waits for a-frozen (`load_model`, `load_dataset`, `make_env`).

## Requests to the coordinator

- `fbplan/controllers/__init__.py` belongs to d-eval. Until it exists, `fbplan.controllers.planner`
  imports as a namespace package. That works; no action needed unless d-eval's `__init__` re-exports.
- d-eval: please add aliases `planner` -> `fbplan.controllers.planner:make_planner` and
  `planner_high` -> `fbplan.controllers.planner:make_planner_high` to `ALIASES`.
- Landmark tests are in `tests/test_graph.py`, because my brief lists no `tests/test_landmarks.py`.
- I need a file for the route-plot / edge-stats / smoke-run script. Proposal: `analysis/planner_routes.py`,
  owned by c-planner (d-eval owns the rest of `analysis/`).
