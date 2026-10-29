# Status: c-planner

Last update: 2026-10-29.

## Summary

Approach A is implemented, tested and runs on the real medium model. With the approved guards,
the planned routes are clean (they follow the corridors and end at the goal). Waypoint following,
not the plan, was the bottleneck: an earlier switch (reached_xy 3) plus a look-ahead along the
route lifted the planner from 56/100 to 79/100 on held-out medium episodes. The baseline pi_h
gets 84/100 on the same start states, so the planner does not beat it on medium yet.
All real-model numbers come from a probe with a-frozen's uncommitted loader; redo them after
a-frozen merges.

## Done

- `fbplan/graph.py` (pure numpy): trusted edges (`rho_min <= rho < rho_max`, top-k, self-loops
  removed), SMDP value iteration with a cost per hop, Q-values from any source, route extraction.
- `fbplan/option_model.py`: batched `rho(s,w)`, `R(s,w)`, direct value `F(s,z_r)ᵀz_r`;
  head aggregation min / mean; edge reward modes exact / paper / zero; clip counts.
- `fbplan/landmarks.py`: farthest-point sampling on xy or on normalized `B(s)`, full observations.
- `fbplan/controllers/planner.py`: `GraphPlannerController`, `GraphPlusHighController`,
  `PlannerConfig` (frozen dataclass). Edge table cached per env, plan cached per task.
  `route(obs)` gives the planned landmark sequence for route plots.
- CLI factories for d-eval's `fbplan.evaluate`: `fbplan.controllers.planner:make_planner` and
  `:make_planner_high`. Options are `key=value`; a dotted key sets a nested field, for example
  `--opt rho_min=0.4 landmarks.count=500 options.head_agg=mean`. Unknown keys raise an error.
- Decisions approved by the user on 2026-10-29, after the medium probe (numbers below):
  1. An edge with rho >= 1 is untrusted (`rho_max = 1`; rho is still clipped and counted).
  2. Edges longer than `max_edge_xy = 6` xy units are untrusted (default guard against wormholes).
     `two_way = True` uses `min(rho(u,w), rho(w,u))`: an FB-only option for the report.
  3. `edge_reward = "zero"` is the default; exact and paper stay as ablations.
  4. A waypoint is reached within `reached_xy` xy units; `reached_xy = None` gives the brief's
     rule `rho(s,w*) >= rho_reached`.
  5. Look-ahead along the planned route (user: "follow your recommendation"): `lookahead = True`
     moves on to the next route stop when Q(s, next) >= Q(s, current) and the edge s -> next is
     trusted.
  The same rules apply to the landmark graph and to the first hop from `s`.
  Defaults changed by c-planner after a held-out check (user approved 1.5 before; please confirm):
  `reached_xy = 3.0` and `lookahead = True` (79/100 against 56/100 for 1.5 without look-ahead).
- Tests: 67 pass (`uv run pytest`), ruff clean. They run on an exact tabular fake,
  `tests/planner_fakes.py` (`rho = gamma^distance`; knobs for one-head wormholes and a horizon
  beyond which FB and `pi_z` fail). Closed loop on the fake: the planner reaches a goal 10 steps
  away with 4-step edges via 2 waypoints, where the task policy alone stands still.
  Mutation checks: each guard, the timeout, both reached rules, the reached exclusion and its
  order before top-k are each caught by a test.

## Implementation choices (no change to the math in design.md)

1. Batching of the N×N table lives in `OptionModel.pair_table` (it needs FB); `graph.py` stays pure
   numpy. Pairs go through F one batch at a time, padded to a power of two (few jit shapes, bounded memory).
2. Heads: `rho_h`, `R_h`, `direct_h` per head, then min or mean. rho is clipped to [0, 1] after
   aggregation. A target with `F_h(w,z_w)ᵀB(w) <= 0` gets `rho_h = 0` (unreachable).
3. Cost per hop: a factor `hop_discount` on rho in the Bellman backup. It does not depend on the
   scale of `z_r`; in shortest-path form it adds `-log(hop_discount)` to the edge cost `-log rho`.
4. At a replan, landmarks that count as reached are excluded before top-k. Without this, a landmark
   next to the agent has Q ≈ V_H, wins ties, and the agent stalls.
5. No trusted edge from `s`: the task policy `pi_l(z_r)` acts, as the graph Bellman equation gives.
6. The task policy option is also a commitment: replan after `commit_timeout` steps.

## Findings on the real model (probe)

a-frozen's uncommitted loader (loaded by path, read-only), medium, 200 landmarks from the
validation split. Probe scripts are throw-away and not committed.

- Cost on CPU: N = 100 / 200 / 400 landmarks: edge table 0.3 / 0.5 / 1.5 s, one replan 3 / 4 / 10 ms.
  Landmark count is not a cost problem in this range.
- Edges: rho clipped low / high in 3950 / 401 of 40000 pairs. With the defaults: 5466 trusted edges
  (27.3 per landmark); 35 of them have a maze geodesic length above 1.5 × xy length.
- Calibration (FB steps = log rho / log gamma): geodesic distance [0,2) → median 26 steps;
  [4,8) → 79; [8,16) → 146; [16,∞) → 223, p90 2749.
- Design defaults (exact R, rho clipped to 1 and trusted) give bad routes: 13–14 hops, jumps of
  18 xy units. Causes: exact R is noise that the max collects (per trusted edge median +2.1, p99 +68;
  largest direct value 96); clipped rho = 1 edges are free hops across the maze; edges with
  0.7 <= rho < 1 still jump 16–19 xy units.
- Guards (R = 0, rho_min 0.5; mean xy distance route end → goal over 5 tasks / longest hop):
  none 11.5 / 12.2; drop rho >= 1: 3.1 / 17.4; + xy cap 6: 2.3 / 5.8; + two-way instead: 5.4 / 15.5
  (with rho_min 0.7 the two-way graph disconnects).
- Reached rule: within 0.5 xy units of a waypoint FB rho peaks at 0.88–0.90, so rho >= 0.9 rarely
  fires there (16 of 90 waypoints timed out); elsewhere it fired a median 2.3 units away.
- Closed loop, paired start states (global NumPy RNG seeded, as d-eval found), 5 tasks × 6 episodes:

  | Method | Success | Median steps |
  |---|---|---|
  | baseline pi_h | 22/30 | 482 |
  | oracle 2 cells ahead + pi_l (privileged, d-eval code) | 17/30 | 738 |
  | oracle 1 cell ahead + pi_h (privileged, probe variant) | 20/30 | 581 |
  | planner + pi_l, rho rule 0.9 | 10/30 | 1000 |
  | planner + pi_l, xy rule 1.5 (probe subclass) | 23/30 | 584 |
  | planner + pi_l, committed defaults | 20/30 | 714 |
  | planner + pi_h, committed defaults | 21/30 | 671 |

  The committed controller also uses the xy rule to exclude landmarks at a replan; the probe
  subclass excluded only landmarks with rho >= 0.9.
- Smoke run (brief item), committed defaults, paired start states, 5 tasks × 10 episodes:

  | Method | Success | Per task 1–5 | Median steps |
  |---|---|---|---|
  | baseline pi_h | 34/50 | .9 .8 .7 .5 .5 | 492 |
  | graph_planner (pi_l) | 31/50 | .4 .5 .8 .5 .9 | 770 |
  | graph_plus_high (pi_h) | 33/50 | .4 .6 .8 .6 .9 | 692 |

  Level with the baseline within noise; better on task 5, worse on tasks 1–2, slower overall.
- Look-ahead and reached radius, on the 50 smoke-run episodes (a dev set; selection among
  12 configurations on it is biased upward):

  | Planner config | pi_l | pi_h |
  |---|---|---|
  | reached_xy 1.5 | 31/50 | 33/50 |
  | reached_xy 1.5 + look-ahead | 29/50 | 30/50 |
  | reached_xy 2.0 | 31/50 | 36/50 |
  | reached_xy 3.0 | 37/50 | 35/50 |
  | reached_xy 4.0 | 32/50 | 25/50 |
  | reached_xy 3.0 + look-ahead | 40/50 | – |

  Held-out check, 5 tasks × 20 new episodes (paired, seeds 1000·task + 10..29):

  | Method | Success | Per task 1–5 | Median steps |
  |---|---|---|---|
  | baseline pi_h | 84/100 | .85 .90 .80 .70 .95 | 442 |
  | planner pi_l, reached_xy 1.5 (old default) | 56/100 | .35 .60 .85 .50 .50 | 841 |
  | planner pi_l, reached_xy 3.0 + look-ahead (new default) | 79/100 | .80 .75 .75 .75 .90 | 500 |
  | planner pi_l, reached_xy 3.0 | 62/100 | .55 .70 .85 .40 .60 | 744 |
  | planner pi_h, reached_xy 2.0 | 64/100 | .35 .90 .85 .60 .50 | 792 |

  The baseline's 84 % matches the paper (87 %); the 68 % in the smoke run came from a hard set
  of start states. Neither the radius nor the look-ahead alone helps much; together they do.
- Route plots: `runs/c-planner/routes_medium.png` (not committed). Planned routes follow the
  corridors, cross no wall and end at the goal. Executed paths wobble around waypoints and
  sometimes loop back: execution, not the plan, costs the time.

## Next

1. Long horizons: medium is short for pi_h (84 %). The planner's case is large and giant (paper
   baseline 66 % and 1 %). Run the same probe on large with landmarks from its validation split.
2. After a-frozen merges: redo the numbers with landmarks from the train split, and hand the
   factories to d-eval for a full evaluation (the probe samples are too small to rank methods).
3. Switch the tests to `fbplan/tabular/fake_fb.py` when b-theory merges it.

## Blockers

- Final numbers wait for a-frozen to merge (`load_model`, `load_dataset`, `make_env`).

## Requests to the coordinator

- design.md §3 lists the guards `rho >= rho_min`, min over heads, cost per hop. Please add the
  approved decisions 1–4 above (rho >= 1 untrusted, xy cap, two-way option, R = 0 default, xy reached rule).
- d-eval: please add aliases `planner` -> `fbplan.controllers.planner:make_planner` and
  `planner_high` -> `fbplan.controllers.planner:make_planner_high` to `ALIASES`.
- d-eval / report: in the probe the privileged oracle (waypoint = dataset state nearest to a cell
  centre, `pi_l(B(w))`) was not an upper bound (17/30 against 22/30 for the baseline). The report
  should not call it "the best any high-level method can get" without that caveat.
- I need a file for the route-plot / edge-stats / smoke-run script. Proposal: `analysis/planner_routes.py`,
  owned by c-planner (d-eval owns the rest of `analysis/`).
- `fbplan/controllers/__init__.py` belongs to d-eval. Until it exists, `fbplan.controllers.planner`
  imports as a namespace package. No action needed.
- Landmark tests are in `tests/test_graph.py`, because my brief lists no `tests/test_landmarks.py`.
