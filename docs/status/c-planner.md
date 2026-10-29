# Status: c-planner

Last update: 2026-10-29.

## Summary

Act-ahead (2026-10-29, time-boxed, done): planner_high acting on the region latent 8 xy ahead
along its own route closes about a third of its gap to the privileged oracle on held-out medium
episodes (+4.4 of 12.8 points, p 0.30), level with the baseline (0.0) and baseline_raw (−1.6):
no gain. Option `act_ahead_xy` stays, off by default. Details in "Act-ahead" below.

Latest (2026-10-29, evening): the approach A ablations moved to the Linux cloud (user decision);
the exact commands for d-eval are in "Approach A ablations" below, with the Mac numbers as a
cross-check (no single change beats the defaults; the look-ahead and the reached radius matter
most). Review findings C1–C12, X4, X5 and X10 are fixed and merged (c187bb7);
the refactor keeps the records identical (defaults and region latents, eval seed 0).

Time-boxed fix (2026-10-29, see "Time-boxed fixes" below): on dev traces, near the goal the
full replan overrode the plan's hand-over to the task policy in 74 % of the cases, and on large
about half of the reach replans were a stall cycle. New option `switch_steps` (route following
at a reach) removes both, but on held-out episodes success does not change beyond noise
(medium −2.0 pp, p = 0.69; large +5.3 pp, p = 0.4). Defaults unchanged; recommendation: keep it
as an ablation, not a default.

Update after the merge of main (a-frozen, b-theory): main merged into c-planner, 173 tests pass.
rho is now clipped to [0, gamma] for distinct states (design.md §3); trust uses the raw estimate,
so decision 1 still holds. New: a switch to re-plan every step (design.md §4.A),
`analysis/planner_routes.py`, tests on b-theory's tabular fake. With the merged loader (MuJoCo
3.1.6), 5 tasks × 6 paired medium episodes: baseline 24/30, planner (defaults) 17–22/30
depending on the landmark set. Region-mean landmark latents did not help on 100 held-out medium episodes (single-state 68/100,
region 59/100, baseline 92/100). The user chose (2026-10-29): single-state is the default,
region stays an option, and d-eval evaluates both with 5 eval seeds.

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
  Defaults `reached_xy = 3.0` and `lookahead = True`, confirmed by the user on 2026-10-29
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

## Region-mean landmark latents (option; default is single-state)

`LandmarkConfig.latent = "region"`: z_w = mean of raw B over the pool states within
`region_xy = 0.5` of the landmark, used for rho, V_H and the action. The default
`latent = "state"` uses B(w). History: the user approved region latents as the default, then,
after the table below, chose single-state as the default and a 5-seed evaluation of both.
Held-out medium episodes, merged loader, design.md §5 reset, seeds 1000·task + 30..49 (not used
for tuning), 300 landmarks from the train split:

| Method | Success | Per task 1–5 | Median steps | Replans |
|---|---|---|---|---|
| baseline pi_h | 92/100 | .95 1.0 .85 .85 .95 | 406 | – |
| planner pi_l, single-state latents | 68/100 | .55 .70 .80 .60 .75 | 656 | 21 |
| planner pi_l, region latents | 59/100 | .40 .70 .65 .65 .55 | 790 | 42 |
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

## Act-ahead: an oracle-like executor on the planner's route (2026-10-29, time-boxed)

Question (coordinator): how much of approach A's execution gap closes when A acts like the
privileged oracle, i.e. with the region-mean latent (radius 0.5) of a point 1–2 cells ahead
instead of the single-state latent of its waypoint? Under the rule (gain against baseline_raw)
no gain is possible on medium: on Linux baseline_raw 80.8 %, oracle 81.3 %.

New option `act_ahead_xy` (commit de2c3b4, off by default): planning stays on the single-state
landmarks; pi_l (or pi_h as task, planner_high) gets the region latent
(`landmarks.RegionField`, the landmark data pool, radius `act_region_xy` 0.5) of the point
`act_ahead_xy` xy units ahead along the planned route (agent → waypoint → best_next stops), and
z_r past the last landmark. No privileged data. Tests: the latent of the point 2 and 4 xy ahead,
z_r past the end, pi_h as task, the field from the landmark pool, all 15 tabular tasks solved;
4 mutants caught.

Dev (eval seed 100, 5 tasks × 20 episodes, medium, 4 niced workers on the Mac):

| run | success /100 |
|---|---|
| baseline | 80 |
| baseline_raw | 75 |
| planner, defaults | 76 |
| planner_high, defaults | 74 |
| planner, act-ahead 4 / 8 | 72 / 74 |
| planner_high, act-ahead 4 / 8 | 68 / **77** |
| same, route extended to the landmark of largest V_H (probe, reverted) | planner 69 / 69, planner_high 73 / 72 |
| oracle (d-eval, privileged: true path, val regions, 2 cells ahead) | 89 |
| my executor on the true path (privileged probe): 4 / 8 xy ahead, 2 cells snapped | 81 / 84 / 83 |

- The executor works: on the true path it gets 81–84 (oracle 89; the difference is within
  noise or comes from the val-split regions). On the planner's own route the same executor gets
  68–77, no better than acting on the waypoint (74–76). So the remaining gap is the route
  (straight segments between landmarks up to 6 xy apart, FB edge values), not the latent type.
- The route point ahead is rarely off the data (1 % of steps, median gap 0.02 xy); z_r is used
  in 39 % of steps (the route's last landmark can be several xy from the goal). FB's best goal
  estimate is the landmark of largest V_H (0.4–1.3 xy from the goal on medium and large;
  B(s)ᵀz_r is 2–44 xy off), but extending the route to it did not help (probe above).
- Chosen on dev for the single held-out check: planner_high, act-ahead 8.

Held-out check, once (eval seed 300, 5 tasks × 50 episodes = 250 paired episodes per run, Mac;
CSVs in `runs/c-planner/ahead/`, not committed). Difference in points [95 % CI], McNemar p:

| run | success | − planner_high | − baseline | − baseline_raw |
|---|---|---|---|---|
| baseline | 78.0 % | | – | |
| baseline_raw | 79.6 % | | | – |
| planner, defaults | 72.4 % | | | |
| planner_high, defaults | 73.6 % | – | −4.4 [−11.2, +2.4], p 0.27 | |
| **planner_high, act-ahead 8** | **78.0 %** | +4.4 [−2.8, +11.6], p 0.30 | 0.0 [−7.2, +7.2], p 1.0 | −1.6 [−8.4, +5.2], p 0.74 |
| my executor on the true path, 8 xy ahead (privileged probe) | 80.0 % | +6.4 [−0.8, +13.2], p 0.11 | +2.0, p 0.66 | +0.4, p 1.0 |
| oracle (privileged) | 86.4 % | +12.8 [+5.6, +19.6], p 7e-4 | +8.4, p 0.02 | +6.8, p 0.05 |

Answer: acting like the oracle closes about a third of A's execution gap, within noise. The gap
oracle − planner_high is 12.8 points; act-ahead 8 recovers 4.4 of them (34 %; CI −2.8 to +11.6,
p 0.30), which brings planner_high level with the baseline (0.0) and baseline_raw (−1.6), with no
gain. Point estimates of the rest: the route (my executor on the true path instead of the
planner's route) +2.0, and the oracle's own executor (cell centres snapped 2 cells ahead,
regions from the validation split) +6.4. None of these steps alone is significant.

Command for d-eval's Linux 5-seed run (optional; pairs: `results/medium/planner_high/l0.csv`,
`baseline.csv`, `baseline_raw.csv`):

    uv run python -m fbplan.evaluate --method planner_high --env antmaze-medium-navigate-v0 \
        --seeds 0 1 2 3 4 --episodes 50 --workers 4 --opt landmarks.seed=0 act_ahead_xy=8.0 \
        --out results/medium/planner_ablations/high_act_ahead8/l0.csv

## Teleport: do routes avoid the teleporter entries? (2026-10-29, light Mac analysis)

`analysis/planner_routes.py --env antmaze-teleport-navigate-v0 --method planner|planner_high`,
episode 0 of eval seed 0 per task, defaults (plots in `runs/planner_routes/teleport/`, not
committed). Entries (4, 6) and (5, 1) trigger within 1.5 xy of their centres and send the agent
to one of 3 exits; exit (1, 7) is a cell walled in on all four sides (the trap).

| method | successes | episodes that teleport | end in the trap | routes within 0.3 xy of an entry |
|---|---|---|---|---|
| planner | 0/5 | 5/5 (9 jumps) | 3 | tasks 2, 5 |
| planner_high | 1/5 | 5/5 (8 jumps) | 2 | tasks 2, 5 |

- 15 of 300 landmarks lie within 3 xy of an entry; 161 of 5,849 trusted edges cross a trigger
  zone (the xy cap blocks the teleport jump itself, not edges that pass through an entry).
- The main path into the trap is the task policy: in tasks 3 and 4 the plan hands over to
  pi_l(z_r) after 0–1 hops, and pi_l(z_r) walks into the nearest entry (a 1/3 gamble on landing
  next to the goal of tasks 2–4; a discounted value can prefer it to a long walk, the success
  rate does not).
- The offline data shows the teleporters without the maze map: 752 one-step jumps longer than
  2 xy (largest normal step 0.37 xy; p99 0.27). Jump sources form two clusters at (19.9, 11.5)
  and (0.1, 16.7), within 1.5 xy of the entry centres; jump targets form the three exits.
- An FB-only guard does not work here: the task policy's discounted measure on the jump-source
  zones (F(s, z_r)ᵀb_E normalised at the zone) at the 5 start states is −0.49 to 1.18, and 0.017
  for the entry that task 3 walked into (4.8 xy away).
- Every task has a route that avoids both entries (checked on the map, for diagnosis only).

Decision (coordinator, 2026-10-29): no guard (option C); the report shows this failure mode with
the analysis above. Reason: b-theory's docs/theory.md §11 bounds what walking can reach on
teleport with this pi_l (about 24–34 % for tasks 1–4, because of the time limit), close to the
baseline's 27.7 %, so a guard cannot give a clear gain. The proposal, not implemented: a
data-derived jump guard, switch
`avoid_jumps` (off by default; it does nothing when the data has no jumps):
1. Entry zones: dataset states whose next state is more than 2 xy away (above any ant step).
2. Landmarks within r_zone of an entry zone are not used, and an edge whose xy segment passes
   within r_zone of an entry zone is not trusted (like the xy cap). r_zone about 2.5–3 xy
   (trigger 1.5 plus the ant's overshoot), chosen on dev seeds, not on eval seeds.
3. The task policy becomes an edge to an estimated goal centre (the landmark with the largest
   task value F(w, z_r)ᵀz_r, no privileged goal): the hand-over to pi_l(z_r) needs the same
   trust rule (xy cap and no entry zone on the segment); otherwise the agent keeps following
   landmarks.
Tests on b-theory's tabular teleport maze (exact M); evaluation on Linux by d-eval (teleport,
5 eval seeds, paired with the baseline and the planner defaults). About 2–3 h of work.

## Approach A ablations: commands for the Linux rerun (d-eval runs them)

The user decided (2026-10-29) that all final numbers come from the Linux cloud, so d-eval reruns
the whole ablation table there. One change against the defaults per run, landmark set 0,
5 eval seeds × 5 tasks × 50 episodes = 1,250 episodes, paired against the defaults and the
baseline on the same episodes. `--workers` does not change the records. Run from the repository
root:

    E="uv run python -m fbplan.evaluate --method planner --seeds 0 1 2 3 4 --episodes 50 --workers 4"
    M=antmaze-medium-navigate-v0; L=antmaze-large-navigate-v0
    # defaults (the pairs; d-eval's main planner runs, landmark set 0)
    $E --env $M --opt landmarks.seed=0 --out results/medium/planner/l0.csv
    $E --env $L --opt landmarks.seed=0 --out results/large/planner/l0.csv
    # medium: 9 ablations
    $E --env $M --opt landmarks.seed=0 max_edge_xy=None --out results/medium/planner_ablations/no_xy_cap/l0.csv
    $E --env $M --opt landmarks.seed=0 reached_xy=1.5 lookahead=False --out results/medium/planner_ablations/reached_1.5_no_lookahead/l0.csv
    $E --env $M --opt landmarks.seed=0 lookahead=False --out results/medium/planner_ablations/no_lookahead/l0.csv
    $E --env $M --opt landmarks.seed=0 options.edge_reward=exact --out results/medium/planner_ablations/edge_reward_exact/l0.csv
    $E --env $M --opt landmarks.seed=0 options.edge_reward=paper --out results/medium/planner_ablations/edge_reward_paper/l0.csv
    $E --env $M --opt landmarks.seed=0 options.head_agg=mean --out results/medium/planner_ablations/head_agg_mean/l0.csv
    $E --env $M --opt landmarks.seed=0 two_way=True --out results/medium/planner_ablations/two_way/l0.csv
    $E --env $M --opt landmarks.seed=0 replan_every_step=True --out results/medium/planner_ablations/replan_every_step/l0.csv
    $E --env $M --opt landmarks.seed=0 switch_steps=1e999 --out results/medium/planner_ablations/route_following/l0.csv
    # large: the 3 guard-table runs
    $E --env $L --opt landmarks.seed=0 max_edge_xy=None --out results/large/planner_ablations/no_xy_cap/l0.csv
    $E --env $L --opt landmarks.seed=0 reached_xy=1.5 lookahead=False --out results/large/planner_ablations/reached_1.5_no_lookahead/l0.csv
    $E --env $L --opt landmarks.seed=0 lookahead=False --out results/large/planner_ablations/no_lookahead/l0.csv

Notes: `no_lookahead` keeps reached_xy 3.0 (guard table (c)); `reached_1.5_no_lookahead` is the
old default (b); `1e999` is inf for `ast.literal_eval`. Time on the Mac (4 niced workers, load
40–95): 10–20 min per medium run, 18–40 min per large run; `edge_reward=exact` and
`replan_every_step` are the slowest (a full row of F per replan, or per step).

### Mac cross-check (2026-10-29, MuJoCo 3.1.6, not the final numbers)

Same commands on the 12-core Mac; CSVs kept in `runs/mac_ablations/` (git-ignored). Success in
%, paired difference in points with the McNemar p-value. Medium pairs with d-eval's committed
`results/medium/{planner/l0,baseline}.csv`; large with d-eval's uncommitted large runs
(`planner_l0.csv`, `baseline.csv` in its worktree). `replan_every_step` and `route_following`
did not finish before the stop.

| run | medium | − defaults | large | − defaults |
|---|---|---|---|---|
| baseline pi_h | 77.8 | +6.5 (p 2e-4) | 38.0 | +5.6 (p 0.003) |
| planner, defaults | 71.4 | – | 32.4 | – |
| no xy cap | 69.8 | −1.6 (p 0.38) | 27.0 | −5.4 (p 0.003) |
| reached 3.0, no look-ahead | 67.6 | −3.8 (p 0.03) | 22.9 | −9.5 (p 5e-8) |
| reached 1.5, no look-ahead (old default) | 50.1 | −21.3 (p 4e-27) | 24.6 | −7.8 (p 5e-6) |
| head_agg mean | 69.4 | −1.9 (p 0.28) | – | – |
| two_way | 69.0 | −2.3 (p 0.21) | – | – |
| edge_reward paper | 62.1 | −9.3 (p 5e-8) | – | – |
| edge_reward exact | 42.3 | −29.0 (p 4e-54) | – | – |

- No single change beats the defaults; each guard and choice costs or is neutral.
- The look-ahead and the reached radius matter most: without them the planner loses 4–21 points
  on medium and 8–10 on large.
- Exact R collapses task 4 on medium (1.2 %): the max over routes collects R noise.
- The xy cap and the min over heads do not change medium success much (−1.6, −1.9, within
  noise), but the cap matters on large (−5.4 without it).
- The earlier ablation runs used the code before the tie fix and before the merge of main; eval
  seed 0 of `results/medium/planner/l0.csv` gave identical records with both later codes
  (250 of 250, twice), so all Mac runs are consistent.

## Review findings, first pass (g-release, 2026-10-29): response

All M and S findings of my section are fixed; C12 (N) partly. Commit c187bb7, merged into
c-planner. Record identity after the refactor (Mac, eval seed 0, 10 episodes per task,
4 workers, nice): `results/medium/planner/l0.csv` 50 of 50 identical,
`results/medium/planner_region/l0.csv` (region latents, which use the new `batched_backward`)
50 of 50 identical. `planner_routes.py` reproduces eval-CSV rows: episode 1 of eval seed 0 gives
the same outcomes and step counts for all 5 tasks.

| finding | change |
|---|---|
| C1 M | Docstrings write z_w (`option_model`, `graph`, `GraphPlusHighController`); the option model docstring says once what a region latent measures. |
| C2 M | See X4 and X5. |
| C3 M | Removed `TaskPlan.reward`, `TaskPlan.direct`, `step_info()["rho_waypoint"]` (`_choose` now returns the waypoint and its Q, which `_on_reach` uses), `GraphSolution.iterations`, `Landmarks.dataset_index`. `landmarks.region_latents` stays public: X4 asks d-eval to import it. |
| C4 M | Tests use `NonHierController` instead of a copy. |
| C5 M | `test_a_raw_rho_at_or_above_one_is_not_a_trusted_edge`, comment fixed (raw 1.5, valued at gamma). |
| C6 S | `_advance` runs at most N steps (a route visits each landmark once); test with a tied loop. |
| C7 S | `_solve` raises `RuntimeError` when value iteration has not converged after `vi_max_iters` sweeps; test. With rho < 1 the max-product optimum is reached in at most N sweeps. |
| C8 S | One Q function (`_q`) for choices and route stops. `pair_table` returns no reward table in zero mode; `_solve` asks for R only when `OptionModel.has_reward` (the N × N pass per task is the reason for the branch). |
| C9 S | Removed `top_k` and `space="backward"` (no decision or result used them). |
| C10 S | `planner_routes.py`: help texts, `--eval-seed` and `--episode` (the reset of that eval-CSV row), default output `runs/planner_routes/`. |
| C11 S | `cell_of()` fails on the task policy instead of indexing the last cell. This exposed one test: the first-hop test now asserts the task policy, the correct choice there. The timeout test no longer sets `rho_reached`. |
| C12 N | U-maze constants and `all_obs()` live in `planner_fakes`; landmark tests moved to `tests/test_landmarks.py`; docstrings for `Landmarks`, `Targets.take`, the PlannerConfig fields; two tests of default values removed. Not done: other duplicate test constants. |
| X4 S | `fbplan.option_model.batched_backward(fb, obs, batch_size=4096)`: raw B in the model's dtype, batches padded to a power of two (padding does not change B: checked on the medium model). Landmarks and the option model use it. For e-learner: `train_b.latent_table` can call it with `batch_size=65536`. For d-eval: `oracle.region_latents` can be `fbplan.landmarks.region_latents`. |
| X5 S | `planner_routes.py` uses `parse_options`, `make_tasks`, `task_latent`, `run_episode` and `config_hash` from `fbplan.evaluate`, and builds the planner with `make_planner` (as the eval CLI). It keeps a small recorder (`PathRecorder`) until d-eval makes `RecordingController` public. |
| X10 S | `fbplan.controllers.planner.trusted_edges(rho_raw, sources, targets, config=PlannerConfig(), exclude_diagonal=False)`: raw-rho bounds and the xy cap. The controller uses it; d-eval's calibration can call it. The rho bounds stay in `fbplan.graph.trusted_edges(rho, rho_min, exclude_diagonal, rho_max)`, which b-theory's tabular study imports. |

## CI fix: ties between options (2026-10-29)

CI (ubuntu-24.04, x86_64) failed `test_planner_with_region_latents_solves_every_task_of_the_exact_tabular_maze`
with 13 of 15; it passed on macOS arm64. Cause: exact ties. With region latents, rho to a region
counts the hit one step before w, so in the exact tabular maze a landmark behind the agent can
have exactly the same Q as one ahead (task 2 at (13, 13): Q = 2.1523… for both). `argmax` then
followed the last bits, which depend on the CPU and BLAS build; on x86 the agent turned back and
forth between two cells for 200 steps. Reproduced in Docker (linux/amd64: 11/15) and on the Mac
by perturbing F by 1e-12 (13/15 for 4 of 5 seeds).

Fix (`graph.best_options`, used by value iteration and the controller): Q values within 1e-9
relative are ties; the task policy wins a tie with the landmarks (as before, now with a
tolerance), and among tied landmarks the one with the largest V_H wins (the most progress),
then the lowest index. The look-ahead moves on at a tie. The region test now also runs with F
perturbed by 1e-12 (5 cases); all pass on the Mac and on linux/amd64 (Docker). On the real
model the rule changes nothing: eval seed 0 of `results/medium/planner/l0.csv` rerun with the
fix gives identical records (250 of 250).

## Time-boxed fixes: near-goal circling and route switching (2026-10-29)

Approved by the user through the coordinator: about 3 hours, tuning only on my own dev
episodes, defaults unchanged unless a held-out check supports a change. Episodes use d-eval's
reset seeds `SeedSequence([eval_seed, task, episode])`: dev = eval seed 100, held-out = eval seed
200. d-eval's eval set is eval seeds 0–4, so the sets are disjoint. 300 landmarks, train split.

### Diagnosis (dev, defaults; traces with every replan and waypoint)

| | medium (100 episodes) | large (50 episodes) |
|---|---|---|
| success (baseline pi_h on the same episodes) | 76 | 16 (16) |
| failures that are flips | 13 of 24 | 11 of 34 |
| reaches where the plan says "task policy next" / full replan picks a landmark instead | 110 / 81 | 30 / 21 |
| reach replans per episode / within 2 steps of the previous one | 18.0 / 4.5 | 37.9 / 17.6 |
| task-policy segments that start < 8 xy from the goal / time out after 100 steps | 133 / 55 | 28 / 10 |

Three mechanisms:

1. Near the goal (problem 1): when the agent reaches the last landmark of the route, the plan
   says "task policy". The full replan from s compares one task value F(s, z_r)ᵀz_r with the
   max over many landmark values, and the max wins in 74 % of these reaches (medium). The
   agent then heads for a landmark next to the goal and passes the goal.
2. A stall cycle (most of problem 2): at a reach, the plan's next stop is often more than
   `max_edge_xy` = 6 xy from s (it is at most 6 from the landmark, and s is up to 3 from it),
   so it is not trusted from s. The full replan picks a landmark behind the agent; the
   look-ahead then advances to the reached landmark; the next step replans again. On large
   17.6 of the ~38 reach replans per episode are this cycle.
3. The task policy pi_l(z_r) near the goal drifts away: in the 56 medium segments of 100 or
   more steps without success, the median distance to the goal grew from 4.5 to 7.0 xy.

### Fix: route following at a reach (`switch_steps`)

`PlannerConfig.switch_steps = k`: at a reach of w*, take the plan's next stop `best_next[w*]`
(a landmark or the task policy) instead of a full replan, unless the full replan finds a choice
worth more than `gamma^-k` times its Q (it saves more than k FB steps). While that stop is not
trusted from s, the agent keeps heading for w* (until the timeout). `None` (default) keeps the
old full replan; `k = inf` (CLI: `--opt switch_steps=1e999`) always follows the plan between
timeouts. Tests on the U maze: the old rule stalls between two cells, and circles near the goal
when FB underestimates the task value at one state; route following solves both, and a
switch away from the plan needs a gain of more than k steps.

Variants on the dev episodes (paired with the defaults on the same episodes; "+a/−b" = episodes
that only the variant / only the defaults solve):

| Variant | medium | large |
|---|---|---|
| defaults (full replan at a reach) | 76/100 | 16/50 |
| baseline pi_h | – | 16/50 |
| route following, k = inf | 66/100 (+15/−25, p 0.15) | **25/50** (+16/−7, p 0.09) |
| route following, k = 10 | 63/100 (+14/−27, p 0.06) | 19/50 (+13/−10, p 0.68) |
| route following, but full replan when the next stop is not trusted (probe) | 73/100 (+12/−15, p 0.7) | 14/50 (+11/−13, p 0.84) |
| task policy through pi_h (probe) | 68/100 (+6/−14, p 0.12) | 15/50 (+4/−5, p 1.0) |
| route following k = inf + task policy through pi_h (probe) | 68/100 | 24/50 |

- The large gain comes from keeping the reached waypoint until the plan's next stop is trusted:
  without that hold (third row) the gain is gone.
- On medium the same hold costs: the agent spends a mean 60 steps per episode within 3 xy of
  its current waypoint (defaults: 5). `pi_l(B(w))` does not settle on `w` (it hovers 1–3 xy
  away, d-eval saw the same overshoot), so the next stop stays untrusted, and flips rise
  (20 against 13).
- Sending the task policy through pi_h did not help; I removed that option.

### Held-out check (eval seed 200, not used for any choice)

Medium 5 tasks × 50 episodes, large 5 tasks × 30 episodes, d-eval's evaluation CLI, 4 workers
(CSVs in `runs/c-planner/fix/heldout/`, not committed):

| Method | medium | large |
|---|---|---|
| baseline pi_h | 81.2 % [76.4, 86.0] | 35.3 % [28.7, 42.0] |
| planner, defaults | 72.4 % [66.8, 78.0] | 32.7 % [25.3, 40.0] |
| planner, route following (`switch_steps=1e999`) | 70.4 % [64.8, 75.6] | 38.0 % [30.7, 45.3] |
| route following − defaults, paired | −2.0 pp [−9.6, +5.6], p = 0.69 | +5.3 pp [−5.3, +16.0], p = 0.4 |

Recommendation: do not make route following the default. It removes the two mechanisms it
targets (no landmark overrides of the plan's hand-over, no stall cycles), but success does not
change beyond noise on either maze. The +18-point dev gain on large shrank to +5 points on the
held-out episodes, as expected after choosing among 5 variants on 50 dev episodes. On medium,
most failures are flips (13 of 24 on dev) and pi_l drifting near the goal, which route
following does not touch. The option stays, off by default, as an ablation.

## Next

1. d-eval reruns the ablation table on Linux (commands above). When the CSVs are on main, I
   compare them with the Mac cross-check and write the ablation table for the report.
2. d-eval: `oracle.region_latents` can import `fbplan.landmarks.region_latents`, and
   `analysis/failures.RecordingController` can become public so that `planner_routes.py` imports
   it (X4, X5). e-learner: `train_b.latent_table` can import `batched_backward` (X4).

## Blockers

- None.


## Requests to the coordinator

- Merge c-planner: review fixes (c187bb7), ablation commands and Mac cross-check. API changes
  for other components: `fbplan.graph.trusted_edges(rho, rho_min, exclude_diagonal, rho_max)`
  lost `top_k` (b-theory's study and d-eval's calibration call it compatibly); new public
  `fbplan.controllers.planner.trusted_edges` (full rule with the xy cap) and
  `fbplan.option_model.batched_backward`; removed `LandmarkConfig.space`, `top_k`,
  `Landmarks.dataset_index`, `GraphSolution.iterations`, `step_info()["rho_waypoint"]`.
- design.md §4.A could note that route following (`switch_steps`) was tested and is not a
  default (held-out: medium −2.0 pp, large +5.3 pp, both within noise).
- d-eval / report: in the probe the privileged oracle (waypoint = dataset state nearest to a cell
  centre, `pi_l(B(w))`) was not an upper bound (17/30 against 22/30 for the baseline). The report
  should not call it "the best any high-level method can get" without that caveat.
