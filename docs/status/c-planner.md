# Status: c-planner

Last update: 2026-10-29.

## Summary

Update after the merge of main (a-frozen, b-theory): main merged into c-planner, 173 tests pass.
rho is now clipped to [0, gamma] for distinct states (design.md §3); trust uses the raw estimate,
so decision 1 still holds. New: a switch to re-plan every step (design.md §4.A),
`analysis/planner_routes.py`, tests on b-theory's tabular fake. With the merged loader (MuJoCo
3.1.6), 5 tasks × 6 paired medium episodes: baseline 24/30, planner (defaults) 17–22/30
depending on the landmark set. Region-mean landmark latents (user-approved, now the default)
did not help on 100 held-out medium episodes: single-state 68/100, region 59/100, baseline 92/100.
Open, for the user: which latent default to keep.

Approach A is implemented, tested and runs on the real medium model. With the approved guards,
the planned routes are clean (they follow the corridors and end at the goal). Waypoint following,
not the plan, was the bottleneck: an earlier switch (reached_xy 3) plus a look-ahead along the
route lifted the planner from 56/100 to 79/100 on held-out medium episodes. The baseline pi_h
gets 84/100 on the same start states, so the planner does not beat it on medium yet.
On large (probe, 5 tasks × 10) the planner is worse than the baseline (9–15/50 against 22/50):
plans are mostly clean, but the agent leaves the planned route and replans about 40 times.
Numbers below the section "Findings on the real model (probe)" come from a-frozen's
uncommitted loader; the section "After the merge" uses the merged one.

## Done

- `fbplan/graph.py` (pure numpy): trusted edges (`rho_min <= rho < rho_max`, top-k, self-loops
  removed), SMDP value iteration with a cost per hop, Q-values from any source, route extraction.
- `fbplan/option_model.py`: batched `rho(s,w)`, `R(s,w)`, direct value `F(s,z_r)ᵀz_r`;
  head aggregation min / mean; edge reward modes exact / paper / zero; clip counts.
- `fbplan/landmarks.py`: farthest-point sampling on xy or on normalized `B(s)`, full observations.
- `fbplan/controllers/planner.py`: `GraphPlannerController`, `GraphPlusHighController`,
  `PlannerConfig` (frozen dataclass). Edge table cached per env, plan cached per task.
  `route(obs)` gives the planned landmark sequence for route plots.
  `replan_every_step = True` rescores all landmarks at every step (design.md §4.A).
- `analysis/planner_routes.py`: route plots, edge statistics and one episode per task, as PNG and
  JSON in `runs/c-planner/` (`uv run python analysis/planner_routes.py --env <env> [--opt ...]`).
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
  Defaults changed by c-planner after a held-out check (user approved 1.5 before; not yet
  confirmed; design.md §4.A still says 1.5): `reached_xy = 3.0` and `lookahead = True`
  (79/100 against 56/100 for 1.5 without look-ahead; after the merge 22/30 against 13/30).
- Tests: 179 pass in the merged tree (`uv run pytest`), ruff clean. My tests use two fakes:
  b-theory's `fbplan/tabular/fake_fb.py` for ground truth on the medium maze (rho equals the
  exact hitting discount; the planner solves all tasks with the exact model and beats the task
  policy under rho noise 0.02, 27 against 7 of 45 episodes), and `tests/planner_fakes.py` for
  controller tests that place an error on one pair or head (one-head wormholes; a horizon
  beyond which FB and `pi_z` fail). Mutation checks: each guard, the timeout, both reached rules,
  the reached exclusion and its order before top-k, the raw/clipped split and the look-ahead
  trust check are each caught by a test.

## Implementation choices (no change to the math in design.md)

1. Batching of the N×N table lives in `OptionModel.pair_table` (it needs FB); `graph.py` stays pure
   numpy. Pairs go through F one batch at a time, padded to a power of two (few jit shapes, bounded memory).
2. Heads: `rho_h`, `R_h`, `direct_h` per head, then min or mean. After aggregation rho is clipped
   to [0, gamma] for distinct states and to [0, 1] for a state and itself (same observation),
   as design.md §3 says (coordinator request, 2026-10-29). `OptionTable.rho_raw` keeps the
   estimate before clipping: trust rules use it (so raw rho >= 1 stays untrusted), values use
   the clipped rho. A target with `F_h(w,z_w)ᵀB(w) <= 0` gets `rho_h = 0` (unreachable).
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
- Large (probe, 400 landmarks from the validation split, defaults, 5 tasks × 10 paired episodes):

  | Method | Success | Per task 1–5 | Median steps | Replans / advances |
  |---|---|---|---|---|
  | baseline pi_h | 22/50 | .3 .5 .9 .3 .2 | 1000 | – |
  | planner pi_l | 9/50 | .2 .3 .2 .1 .1 | 1000 | 40 / 24 |
  | planner pi_h | 15/50 | .2 .4 .6 .2 .1 | 1000 | 36 / 19 |

  FB is noisier on large: rho < 0 in 22 % of pairs (medium 10 %). Trusted edges that cross a wall
  (geodesic > 1.5 × xy + 0.5): large 17 of 11407, medium 51 of 5451. Planned routes from the 5
  start states: 16–20 hops, one wall-crossing hop in total (task 2: xy 5.9, geodesic 19.3).
  Route plot `runs/c-planner/routes_large.png`: in tasks 4–5 the agent takes another way than the
  plan, so replans change the route on the way (two ways around a loop can have close values).
- Route plots: `runs/c-planner/routes_medium.png` (not committed). Planned routes follow the
  corridors, cross no wall and end at the goal. Executed paths wobble around waypoints and
  sometimes loop back: execution, not the plan, costs the time.

## After the merge (merged loader, MuJoCo 3.1.6, rho clipped to [0, gamma])

Paired medium episodes (design.md §5 reset), 5 tasks × 6, seeds 1000·task + 0..5:

| Method | Success | Per task 1–5 | Median steps |
|---|---|---|---|
| baseline pi_h | 24/30 | 1.0 .67 .83 .67 .83 | 456 |
| planner defaults, 300 landmarks, train split | 17/30 | .67 .50 .67 .50 .50 | 730 |
| planner defaults, 200 landmarks, train split | 18/30 | .17 .83 .67 .83 .50 | 706 |
| planner defaults, 300 landmarks, val split | 20/30 | .67 .83 1.0 .33 .50 | 693 |
| planner defaults, 200 landmarks, val split | 22/30 | .50 .83 .67 .67 1.0 | 570 |
| planner reached_xy 1.5, no look-ahead, 200 val | 13/30 | .17 .83 .33 .50 .33 | 1000 |

- The landmark set moves the planner by a few episodes in 30 (within noise); the full
  evaluation should average over landmark seeds (`landmarks.seed`).
- `analysis/planner_routes.py` on medium (300 train landmarks, seed 0): 0/5. In task 2 the agent
  circles 1–2 units from the goal: near the goal, the max over landmark values can beat the
  direct value, so the planner keeps choosing landmarks next to the goal.
- d-eval (5 eval seeds): baseline 79.4 % [77.2, 81.6]; oracle with region-mean latents 83.1 %.
  70–80 % of medium failures are the ant flipping over, so the headroom on medium is small.

## Region-mean landmark latents (user-approved 2026-10-29, now the default)

`LandmarkConfig.latent = "region"`: z_w = mean of raw B over the pool states within
`region_xy = 0.5` of the landmark, used for rho, V_H and the action. `latent = "state"` keeps B(w).
Held-out medium episodes, merged loader, design.md §5 reset, seeds 1000·task + 30..49 (not used
for tuning), 300 landmarks from the train split:

| Method | Success | Per task 1–5 | Median steps | Replans |
|---|---|---|---|---|
| baseline pi_h | 92/100 | .95 1.0 .85 .85 .95 | 406 | – |
| planner pi_l, single-state latents | 68/100 | .55 .70 .80 .60 .75 | 656 | 21 |
| planner pi_l, region latents (default now) | 59/100 | .40 .70 .65 .65 .55 | 790 | 42 |
| planner pi_l, plan single-state, act region (probe) | 59/100 | .65 .55 .60 .45 .70 | 810 | 22 |
| planner pi_h, region latents | 62/100 | .65 .65 .75 .60 .45 | 772 | 36 |

- The 9-point gap is about 2 standard errors: suggestive, not conclusive.
- With region latents rho is higher and flatter (pairs within 6 xy: median raw rho 0.73 against
  0.58), and 8.4 % of them reach raw rho >= 1 and become untrusted (single-state: 2.2 %).
  "Reach the region" looks easy from farther away, so rho separates near and far landmarks less.
- d-eval's oracle gained from region latents with waypoints 2 cells (8 xy units) ahead on the
  true shortest path; my waypoints are at most 6 xy units away and switch at 3.
- The planner is far below the baseline on this set (68 against 92). The landmark set also moves
  it by several points (see "After the merge").

## Next

1. Waiting for the user: keep region latents as the default, or go back to single-state.
2. Large: find why the agent leaves the route (route switching between near-equal ways around a
   loop?). Candidate fix: hysteresis.
3. Near the goal: the agent circles landmarks next to the goal (see above).

## Blockers

- None.

## Requests to the coordinator

- design.md §4.A says `reached_xy = 1.5`. The code default is 3.0 with look-ahead (held-out
  evidence above); it waits for the user's confirmation.
- c-planner can merge into main: d-eval needs it for its 5-seed planner evaluation and
  `analysis/calibration.py`.
- d-eval / report: in the probe the privileged oracle (waypoint = dataset state nearest to a cell
  centre, `pi_l(B(w))`) was not an upper bound (17/30 against 22/30 for the baseline). The report
  should not call it "the best any high-level method can get" without that caveat.
- Landmark tests are in `tests/test_graph.py`, because my brief lists no `tests/test_landmarks.py`.
