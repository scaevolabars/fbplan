# Hypotheses from the papers of the assignment (h-papers)

Session h-papers, 2026-10-30. Brief: `docs/briefs/hypotheses.md`. Lens: the papers and repos that
the assignment lists, read in full, and the mechanisms in them that could give a gain over the
stronger baseline under our constraints:

| Source | Read | Main mechanism for us |
|---|---|---|
| Barreto et al. 2016, Successor Features for Transfer in RL (arXiv 1606.05312) | main text, appendix | GPI over a library of policies |
| Touati and Ollivier 2021, Learning One Representation to Optimize All Rewards (2103.07945) | main text, appendix | `z_R = E[r·B]`, value of any `π_z` for any reward, "sup over z" |
| Touati, Rapin and Ollivier 2022, Does Zero-Shot RL Exist? (2209.14935) and `facebookresearch/controllable_agent` | main text, appendix, agent code | inference rules for `z`, head pessimism |
| Tirinzoni et al. 2025, Zero-Shot Whole-Body Humanoid Control via Behavioral Foundation Models, "Meta Motivo" (**2504.11054**) and `facebookresearch/metamotivo` | main text, appendix, model and agent code | tracking a sequence of target states with a window of `B` |
| Stojanovic and Proutiere 2026, FB π-Switch (2605.13207) | main text, appendix | future work: chains of subgoals; the low level limits the gain |

The start message gives 2410.20092 for Meta Motivo. That id is the OGBench paper (Park et al.).
Meta Motivo is 2504.11054; this file uses that paper.

New numbers come from scratch scripts outside the repository (§1, §6). They ran on this Mac,
4 workers, about 4 minutes of wall time in total, on the 10 medium held-out goals and on the
offline data only. Nothing ran on the eval tasks.

## Summary

1. **Meta Motivo's window does not add a lever to the executor.** On the true route, the mean of
   the waypoint latents of the next 1–3 cells scores +1.0 pp [−2.2, +4.3] above the single
   waypoint of the privileged oracle (600 paired held-out episodes, M1). Longer windows score
   lower. So the window applied to approach A's routes can recover at most about 1 pp over A's
   act-ahead, which ended level with `baseline` and below `baseline_raw`.
2. **The rest of A's gap is the route (c-planner), and the offline data give near-true routes
   without the map.** Keep only landmark edges that a recorded trajectory travels (within 0.5 xy of
   both ends, at most 60–150 steps apart), weight them by xy length, and the routes on medium are
   0–1 % longer than the true shortest path at the median, 3–6 % at the 90th percentile; no edge
   crosses a wall (M2). This is the Meta Motivo idea in its plain form: track recorded states,
   not states that FB values promise. FB `ρ̂` edges have 5.6 % wormholes on medium (REPORT §8.3).
3. **GPI (Barreto) is weak for goal reaching here.** Its guarantee is against the best *committed*
   policy of the library, not against the switching `π_h`; the committed value of a policy toward
   another goal is about 0 far from the goal; and the max collects value noise (§3, H4).
4. **The FB π-Switch paper agrees with the central finding.** Its Table 2 swaps low-level
   policies between FB + `π_h` and FB π-Switch and concludes that "the limited gains are
   primarily due to the quality of the low-level policy rather than the hierarchical
   decomposition itself". Its future work names chains of subgoals, with "error accumulation
   across chained subgoals" as the key challenge (§9 of the paper).
5. **The headroom bound may be lower than the truth under the final protocol.** The +0.5 pp of
   the oracle against `baseline_raw` comes from eval seeds 0–4, where `baseline_raw` was +3.8 pp
   above `baseline`; on eval seeds 5–9 that gap is +0.3 pp (REPORT §7.1). On the held-out goals
   the oracle is +4.0 pp [+0.3, +7.7] above `baseline` (M1), and `baseline_raw` is level with
   `baseline` there (h-repr A5: −0.7 pp). A Linux run of the oracle on eval seeds 5–9 (G0) decides
   whether any route method can pass the rule on medium. It comes before H1–H3.

## 1. Measurements (this session)

**M1. A Meta Motivo tracking window on the true route (privileged diagnostic).** Meta Motivo
tracks a motion with `z_t = proj(mean_{k=1..L} B(s*_{t+k}))`, a forward window that skips the
current state, `L = 8`, with the policy trained on the same 8-state segment latents
(`metamotivo/fb/model.py:154-159`; paper App. C.5.3; 60 % of its training latents are such
segments, Table 9). The paper has no ablation of `L` or of a window against a single state. Our
form: the privileged oracle (`fbplan/controllers/oracle.py`: true shortest cell path, region-mean
latent of radius 0.5, `π_l` follows), but with `z_t = mean_k n(region(cell_{t+k}))` over the
cells `k = first..last` ahead (`n` = unit norm; `π_l` normalizes the mean). Past the goal the path
repeats the goal cell, so the window shrinks onto the goal, as in Meta Motivo. Medium held-out
goals 101–110, eval seeds 0 1 2, 20 episodes per goal: 600 paired episodes per row (Mac).

| Row | success | vs `baseline` (pp) | vs oracle (pp) | median steps of successes |
|---|---|---|---|---|
| `baseline` | 85.5 [82.7, 88.2] | – | −4.0 [−7.7, −0.3], p 0.043 | 265 |
| oracle, 2 cells ahead | 89.5 [87.2, 91.8] | +4.0 [+0.3, +7.7], p 0.043 | – | 257 |
| window, cells 1–3 | 90.5 [88.2, 92.7] | +5.0 [+1.5, +8.7], p 0.009 | +1.0 [−2.2, +4.3], p 0.62 | 246 |
| window, cells 2–4 | 88.7 [86.2, 91.0] | +3.2 [−0.5, +7.0], p 0.12 | −0.8 [−4.3, +2.7], p 0.71 | 248 |
| window, cells 1–5 | 87.3 [84.7, 89.8] | +1.8 [−2.0, +5.7], p 0.40 | −2.2 [−5.7, +1.3], p 0.27 | 250 |

- My threshold, set in my notes before the run but not committed: the window is a lever if it
  gains ≥ +3 pp over the single waypoint. It does not: +1.0 pp at best, and a longer window loses.
  The best window is a little faster (246 against 257 steps). So it is an executor option, not a
  hypothesis of its own.
- A mean of `B` over spread route points is the latent of the reward "occupy any of these
  regions" (FB linearity). With discounting the nearest points dominate, so a long window adds
  little direction and can pull the latent off the path at corners. This agrees with the oracle
  sweep, where region radius and look-ahead change little (REPORT §8.2).
- The oracle is +4.0 pp above `baseline` on these goals, the same size as on the eval tasks
  (+4.2 pp, REPORT §8.2). See Summary 5 and G0.

**M2. Routes from data-certified edges (offline, no map).** Landmarks: farthest-point sampling on
xy over 100k train states. An edge `u → w` is certified when one train trajectory passes within
`r` xy of `u` and later within `r` of `w`, at most `K` steps later. Two weights: the smallest
recorded travel time (steps between the two closest passes), or the xy length of the edge. Routes:
shortest paths over the certified edges. Score (privileged, for diagnosis only): the route
stretch = maze-geodesic length of the route through its landmarks divided by the geodesic from its
first to its last landmark, on 3,000 sampled pairs more than 8 xy units apart
(`fbplan.maze.MazeGeometry.geodesic_distances`). Reference: the best route through the same
landmarks (edges = pairs within 10 xy of geodesic distance, weight = geodesic).

| Maze | landmarks, `r`, `K` | edges | reachable pairs | stretch, travel-time weights: median / 90 % | stretch, xy weights: median / 90 % / 99 % | reference: median / 90 % |
|---|---|---|---|---|---|---|
| medium | 150, 0.5, 150 | 12,411 | 0.973 | 1.168 / 1.291 | 1.000 / 1.059 / 1.216 | 1.001 / 1.004 |
| medium | 300, 0.5, 100 | 33,940 | – | 1.290 / 1.461 | 1.006 / 1.059 / 1.140 | 1.000 / 1.001 |
| medium | 300, 0.5, 60 | 19,011 | – | 1.299 / 1.472 | 1.010 / 1.032 / 1.092 | 1.000 / 1.001 |
| large | 300, 0.5, 100 | 16,603 | 0.980 | 1.219 / 1.478 | 1.022 / 1.223 / 2.337 | 1.002 / 1.180 |
| large | 300, 0.5, 60 | 9,315 | 0.980 | 1.219 / 1.478 | 1.019 / 1.208 / 2.307 | 1.002 / 1.180 |

- **Certified edges with xy weights give near-true routes on medium.** No certified edge is
  faster than 0.5 xy per step (the data's median step is 0.138 xy), so no edge crosses a wall.
  The data certify that an edge can be walked; the xy length orders the walkable edges.
- **Recorded travel times are a poor weight.** The fastest recorded traversal favours edges that
  some fast segment happened to cover; routes are 17–30 % longer. With `r = 1.0` both weights are
  worse, because each hop may cut up to 2 xy units (not tabulated).
- **On large the tail comes from the landmark density, not from the data.** The reference itself
  has 90 % stretch 1.18 with 300 landmarks; more landmarks would be the fix.
- Not measured: FB `ρ̂` weights on the certified edges (H2), and A's current graph on the same score.

## 2. Gate G0 (a measurement, not a hypothesis): the oracle under the final protocol

The privileged oracle has eval seeds 0–4 only. Under the final protocol (10 eval seeds) the two
baseline rows differ by +2.0 pp, and by +0.3 pp on seeds 5–9 alone. Run the oracle on medium eval
seeds 5–9 on Linux (1,250 episodes, under 0.5 CPU h; command as `results/medium/oracle.csv` with
`--seeds 5 6 7 8 9`) and pair the 10-seed oracle with both baseline rows. Reading, fixed now:

| Oracle − `baseline_raw`, 10 eval seeds | Consequence |
|---|---|
| ≤ +1 pp | no route method can pass the rule on medium: drop H1–H3 on medium, keep large only |
| +1 to +3 pp, not significant | H1 has a chance only if it matches the oracle; run H1's held-out check first |
| significant | the true route has room under the final protocol; H1 is the first test |

The same holds for any other route or waypoint method (A, act-ahead).

## 3. Hypotheses, ranked by expected value

Expected value = P(gain by the rule) × size / cost. The probabilities are my judgement from §1
and the committed evidence.

| # | Claim (short) | Maze | P(gain by the rule) | Size if real | Cost | Kind |
|---|---|---|---|---|---|---|
| H1 | recorded-route tracking: data-certified landmark routes + the oracle's executor | medium, large | 0.10 | +2 to +4 pp | ~3 CPU h, a new controller | sequence of intentions, no FB in the route |
| H2 | FB chaining on certified edges (`ρ̂` weights): the FB arm of H1 | medium, large | 0.05 | +1 to +3 pp | +1 CPU h on H1's code | FB chaining |
| H3 | distil H1's route latent into `π_h` (amortized recorded-route oracle) | medium | 0.04 | +2 to +4 pp | ~1 GPU h + ~10 CPU h | `π_h` training |
| H4 | conservative GPI over the library of trained `π_h` seeds (lower bound + margin) | medium | 0.03 | +1 to +2 pp | ~2 CPU h | test-time selection |
| H5 | in-distribution task latents for `π_h`: train on ball-mean goal latents, as `z_r` is built | medium | 0.03 | +1 to +3 pp | ~1 GPU h + ~10 CPU h | `π_h` training |

All five probabilities are low. The reason is the same for all: with this `π_l`, the true route
gains +4 pp over `baseline` but only +0.5 pp over `baseline_raw` on eval seeds 0–4 (G0 checks the
second number), and most failures are flips, which no intention removes (REPORT §8.2; h-exec M1–M7).
H1 ranks first because its cheap test passed (M2), it needs no GPU, and it attacks the measured
cause of A's loss.

### H1. Recorded-route tracking: data-certified routes and the oracle's executor

1. **Claim.** A planner whose routes run over landmark edges certified by recorded trajectories
   (M2, xy weights) and whose executor gives `π_l` the region latent of the route point about
   8 xy ahead (the oracle's rule, or the window of M1) reaches the oracle's success without the map,
   and beats both baseline rows on medium or on large.
2. **Mechanism.** Approach A loses mostly time (REPORT §8.4), and c-planner split A's gap to the
   oracle: on the true route its own executor scores 81–84 % against 68–77 % on A's route (dev,
   `docs/status/c-planner.md`, "Act-ahead"), so "the remaining gap is the route". A's routes come
   from FB `ρ̂`, which has wormholes at every `ρ_min` (5.6 % of trusted pairs on medium, 10.7 % on
   large; the xy cap leaves 1,965 and 1,420; REPORT §8.3) and a wide spread at every distance. A
   certified edge is an observed walk, so it cannot be a wormhole, and xy weights over walkable
   edges give near-true routes (M2). This is Meta Motivo's tracking in its basic form: follow
   recorded states, which `π_l` was trained on, instead of states that a value promises. It can
   beat the stronger row only to the extent that the oracle does: on medium only if G0 shows room;
   on large the bar is `baseline` (37.4 %), and the oracle is +1.9 pp [−1.8, +5.6] there, with
   +6.8 to +12.0 pp on tasks 1, 2, 4 and 5 and −24 pp on task 3 (REPORT §7.2).
3. **Evidence.** For: M2; M1 (the oracle's executor is +4.0 pp over `baseline` on held-out goals);
   act-ahead, with A's own route, recovered 3.2 of the 7.4 points between `planner_high` and the oracle (REPORT §8.4, p 0.07; 34 % in c-planner's Mac check). Against:
   the oracle is only +0.5 pp over `baseline_raw` on medium eval seeds 0–4; c-planner's executor on
   the true path was still 6.4 points below the oracle in one 250-episode check (point estimate);
   on large every waypoint method loses 24–38 pp on task 3; straight segments between landmarks
   can cut corners; the first hop from the agent's state is not certified by M2.
4. **Cheap falsification test.** Done in part (M2: route quality). The rest, about 10 minutes on
   the 600 medium held-out episodes of M1: a scratch controller with (a) the first hop certified
   the same way (dataset states within 0.5 xy of `s`, the landmarks that their trajectories reach
   within `K` steps), (b) the route over certified xy-weighted edges, (c) the point 8 xy ahead
   along the recorded segments of the route, region latent from `landmarks.RegionField`, `z_r`
   past the last landmark. Falsified if it scores more than 2 pp below the oracle row of M1
   (89.5 %) or not above `baseline` (85.5 %) on the same episodes.
5. **Full test.** c-planner adds the controller (about 100 lines; certification is a pass over
   the train split, seconds). Selection on the medium and large held-out goals only, rule fixed
   before the runs: executor ∈ {`π_l` on the point 8 xy ahead, `π_l` on the window at 4/8/12 xy,
   `π_h` with the point as its task}, `K` ∈ {60, 150}; highest held-out success, ties to the
   simpler (point, smaller `K`). Then medium eval seeds 0–9 (2,500 episodes) and large eval seeds
   0–4 (1,250 episodes; seeds 5–9 for both rows if the CI is close), paired against `baseline`
   and `baseline_raw` (`analysis/tables.py --reference baseline baseline_raw`). About 3 CPU h on
   Linux, no GPU. The rule decides as written. The report must say that the route comes from the
   data, not from FB values; H2 is the FB arm.
6. **Constraint check.** `F`, `B`, `π_l`, `π_h` frozen. Offline data only: the train split's
   observations and trajectory order, and xy from the observation (as A's xy cap). The route is a
   lookup of recorded states, not a generative model. No map: the geodesic of M2 scores, it does
   not plan.
7. **Prior work.** SoRB (Eysenbach et al. 2019) and Sparse Graphical Memory (Emmons et al. 2020)
   plan over replay states with learned distances; here the edges are certified by the data
   instead. Meta Motivo's tracking inference (Tirinzoni et al. 2025). FB π-Switch's future work on
   chains of subgoals.

### H2. FB chaining on certified edges

1. **Claim.** H1 with FB `ρ̂` (min over heads, `R̂ = 0`) as the edge weight on the certified
   edges, instead of the xy length, beats both baseline rows, and beats H1.
2. **Mechanism.** Certification removes the wormholes, the error that A's max exploits (REPORT
   §8.3), but keeps FB's travel-time estimate, which knows the dynamics (for example that turning
   costs time) where xy length does not. This is the assignment's question in a cleaner form:
   does FB chaining help once its edges are guarded by data?
3. **Evidence.** For: A's three guards that matter are against FB errors (`R̂ = 0`, look-ahead,
   reach radius; REPORT §4); B's targets use only in-trajectory edges, the same guard, but B's
   regression over noisy targets did not order states better than `V_1` (REPORT §8.5); a graph
   does exact value iteration instead. Against: `ρ̂` has a wide spread at every distance
   (REPORT §8.3), and a shortest path picks the edges whose `ρ̂` is overestimated (tabular study,
   REPORT §3.1); xy weights already give near-true routes on medium (M2), so FB can only add noise
   there; large is where FB might add something.
4. **Cheap falsification test** (about 5 minutes, c-planner's edge table on landmark set 0).
   Score the routes of three graphs with the M2 stretch: A's current trusted edges, certified
   edges with `ρ̂` weights, certified edges with xy weights. Falsified if certified `ρ̂` routes are
   not shorter than A's routes, or if they are more than 5 % longer than certified xy routes at the
   90th percentile on medium and large.
5. **Full test.** One more arm in H1's held-out selection and, if selected, the same final runs.
   Attribution reading, fixed now: H2 − H1 paired on the eval tasks; chaining adds value only if it
   is positive and significant. About 1 CPU h on top of H1.
6. **Constraint check.** As H1.
7. **Prior work.** As H1; approach A (REPORT §4).

### H3. Distil the recorded-route latent into `π_h`

1. **Claim.** `π_h` trained offline to output, for a state `s` and a goal latent, the region latent
   of the point 8 xy ahead on the certified route from `s` to the goal (H1's executor latent),
   beats both baseline rows with the baseline's own test-time controller.
2. **Mechanism.** `π_h` re-selects an intention at every step, and its regression target is
   `B(w)` of an in-trajectory subgoal; its output is a blurred mean (cosine 0.85 to a region mean,
   h-repr A1). The oracle sweep shows that `π_l` executes region latents ahead best (REPORT
   §8.2). A route-derived target gives `π_h` the target that the oracle's executor uses, from data
   only, and it needs no graph at test time. This is plan B's stage B2 (`docs/plans/approach-b.md`
   §6) with a better teacher: certified routes instead of A, which lost.
3. **Evidence.** For: M1, M2; the B-region ensemble, whose targets are trajectory-window means,
   is the best `π_h` row (81.0 %, REPORT §8.6), though not significant against `baseline_raw`.
   Against: a regression averages the targets of a state over goals (validation MSE 0.72 per
   dimension, `docs/status/e-learner.md`), so `π_h` may blur the route latent again; B-region gives
   no pooled gain over B-clean (79.9 % both); H1 must pass its held-out check first.
4. **Cheap falsification test.** Only after H1's held-out check passes. Offline: for 2,000
   validation states and the held-out goals, the cosine between the provided `π_h`'s intention and
   the route latent. If it is above 0.95, `π_h` already outputs it: stop.
5. **Full test.** e-learner: a regression target from the route (landmark routes precomputed once;
   the target per sample is a table lookup), plain behaviour cloning or AWR with the upstream
   advantage, 5 seeds, about 1 GPU h. Held-out selection between the two losses, then 10 eval
   seeds paired against both rows and the retrained baselines. About 10 CPU h.
6. **Constraint check.** Only `π_h` is retrained; targets come from the offline data; no
   generative model.
7. **Prior work.** Distilling graph search into a goal-conditioned policy (PIG, Kim et al. 2023);
   HIQL's high-level policy (Park et al. 2023).

### H4. Conservative GPI over the library of trained `π_h` seeds

1. **Claim.** At each step, choose among the intentions of the provided `π_h` and the 15 trained
   seeds (retrained, B-clean, B-region) the one with the highest pessimistic value
   `min_h F_h(s,z_k)ᵀz_r`, but leave the default (provided `π_h`) only if the gain exceeds a margin
   `κ` chosen on held-out goals. This beats both rows.
2. **Mechanism.** GPI: acting with the best policy of a library, by its value under the new reward,
   is at least as good as every policy of the library (Barreto et al. 2016, Thm 1). The theorem
   also holds with state values only, when the selection is repeated at every step (the policy
   switching result of Chang, Givan and Chong 2004). The seeds differ per task (B-clean seeds
   split 76.5–83.0 %, mostly on task 4; REPORT §8.5), and T1 averaged them instead of choosing.
3. **Evidence.** Against, and stronger: the GPI loss bound is `2ε/(1−γ)` = 200ε at γ = 0.99, so it
   is empty unless the value error ε is far below the value gaps; the guarantee is against the
   *committed* policies `π_l(·|·,z_k)`, not against the switching `π_h`; Barreto's own successes
   have a shared goal or dense rewards, and his fix for an overestimated max is an online update
   that we cannot do (App. B.1.2); the seed outputs are close (cosine 0.955, e-learner), so the gaps
   are small; h-learn rejected the plain argmax for the same reasons.
4. **Cheap falsification test** (offline, about 5 minutes, held-out goals). For 2,000 validation
   states, the 16 candidate intentions, their values per head, and the decoded progress of each
   candidate (privileged geodesic, diagnosis only). Falsified if, for states more than 8 xy from
   the goal, the spread of the values over candidates is below the head disagreement, or if the
   selected candidate's decoded progress is not above the default's.
5. **Full test.** A controller (d-eval, about 40 lines; 16 `π_h` and 16 `F` calls per step).
   `κ` ∈ {0, 0.5σ, 1σ} of the head disagreement, selected on the held-out goals; then 10 eval seeds,
   paired against both rows and against the T1 ensembles. About 2 CPU h.
6. **Constraint check.** Nothing trained; frozen modules; test-time only.
7. **Prior work.** GPI (Barreto et al. 2017; the arXiv 2016 version read here); policy switching
   (Chang, Givan and Chong 2004); FB's "sup over z" (Touati and Ollivier 2021, Thm 4, point 3;
   "not necessary in our experiments").

### H5. In-distribution task latents for `π_h`

1. **Claim.** `π_h` retrained with goal latents built as `z_r` is built at test time (normalized
   mean of `B` over the dataset states within 0.5 xy of the goal state), in its input and in the
   AWR advantage, instead of `B(g)` of one state, beats both rows.
2. **Mechanism.** Two papers say that the test latent should come from the distribution the
   network was trained on: Touati and Ollivier (p. 4: "if prior information is available on the
   rewards r, the corresponding distribution of z_R may be used instead"), and Meta Motivo, whose
   tracking window has the length of the training segments (App. C.5.3) and whose authors explain
   the lag of a tracking baseline by a mismatch between its training goals and the test goals
   (Goal-GAIL, Table 1). Our `π_h` was trained on `B(g)`
   of single states and gets `z_r`, a mean over 485–1,897 ball states, whose cosine to any single
   ball state's latent is 0.57–0.89 (h-repr A1). The ball mean is also the FB latent of the true
   task reward (the 0.5-xy indicator; `controllable_agent` `fb_ddpg.py:216-218`), so the advantage
   would value subgoals for the reward that OGBench scores.
3. **Evidence.** For: the latent mismatch above. Against: `π_h` already acts better on `z_r` than
   on the in-distribution single-state latent (h-repr A1: worse decoded progress on 15 of 15
   goals), so the mismatch does not seem to hurt; region-mean subgoal targets (B-region) gave no
   pooled gain; it is close to h-repr H1, which changes the output targets instead.
4. **Cheap falsification test** (offline, about 3 minutes). h-learn's progress correlation
   (`docs/hypotheses/h-learn.md` §0 B) of the upstream advantage with ball-mean goal latents
   against single-state ones. Falsified if no distance bin beyond 8 xy gains ≥ 0.05.
5. **Full test.** e-learner: one option in `train_high` (ball means precomputed per train state
   with a KD-tree), 5 seeds, about 1 GPU h; held-out selection rule of design.md §8; 10 eval seeds
   paired against both rows and the retrained baselines. About 10 CPU h.
6. **Constraint check.** Only `π_h` is retrained; offline data; no generative model.
7. **Prior work.** As in the claim; HIQL's goal representation (Park et al. 2023).

## 4. What each paper says, in one block

- **Barreto et al. 2016.** GPI (Thm 1) and transfer (Thm 2: loss ≤ `2/(1−γ)·(φ_max·min_j‖w − w_j‖ + ε)`).
  The tasks in its experiments share the goal (four-room: the same goal G in all 250 tasks) or
  have dense rewards (reacher). It does not test a library of policies toward other sparse goals.
  It treats each library policy as committed, so it does not value "go to `w`, then switch",
  which the one-switch advantage of `π_h` does. → H4, rejected items.
- **Touati and Ollivier 2021.** `π_z` and `Q* = F(s,a,z_R)ᵀz_R` need an action-level `F` with
  `π_z = argmax_a`; our `F(s,z)` has no actions, so no FB optimality result covers `π_l(z_r)`.
  `Fᵀz_r` is the value of `π_z` for any `z` (the basis of `π_h`'s advantage). Approximation
  (Thm 9): the loss comes from *ranking* errors between neighbouring choices along the optimal
  path, not from constant offsets: a reason to compare only nearby candidates, as A's xy cap does.
  The maze horizons are short (30–50 steps). No sequencing, no planning. → H4, H5.
- **Touati, Rapin and Ollivier 2022 and `controllable_agent`.** No selection among `z`, no GPI,
  no test-time search: "zero-shot" means "no additional planning or learning". `z` is fixed per
  episode. Goals use one state `B(g)`; in their maze `B` takes only the xy goal, which is the
  analogue of our region means. Every value use takes the min over the 2 `F` heads. FB is not the
  best method on their maze goals (Laplacian and LRA-P match or beat it). → rejected items.
- **Meta Motivo (2504.11054) and `metamotivo`.** Tracking: forward window of 8 next states,
  mean then projection; trained on 60 % segment latents; no ablation of the window. Goal: one pose.
  Reward: softmax-weighted regression, which for an indicator reward is the ball mean. No
  test-time search. Its fix for policy failures is training the policy on the test latents
  (FB-CPR, online, 30M steps): an unfrozen `π_l`. Its own offline variant (FB-AW) is far below
  online FB-CPR (reward about 0.25 against 0.62, read from Fig. 4): offline policy learning is the bottleneck there too.
  → M1, H1, H3, H5.
- **FB π-Switch (2605.13207).** Table 2: the low-level policy limits the gain of the hierarchy.
  Future work: base-model quality, which task properties favour hierarchy, generative planners,
  and chains of subgoals with error accumulation. Its evaluation turns off goal noise at reset
  (App. C.5), which our loop also does (`fbplan/frozen.py:610`), so this is not a source of the
  gap to the paper's numbers. → H1–H3.

## 5. Rejected hypotheses

- **Meta Motivo's window over route points, as a separate lever on A's routes.** On the true route
  it adds +1.0 pp [−2.2, +4.3] over one waypoint, and longer windows lose (M1). It stays an
  executor option in H1.
- **Meta Motivo segment latents (8 consecutive states) as `π_h` targets.** Consecutive states keep
  the pose, which `π_l` executes worst (REPORT §8.2); B-region's trajectory windows gave no pooled
  gain over B-clean; close to h-repr H1.
- **FB-CPR (discriminator-regularized policy).** It retrains the policy online with expert data:
  new interaction and an unfrozen `π_l`.
- **Reward inference by weighted regression (Meta Motivo, temperature 10).** For an indicator
  reward it is the ball mean, which is `z_r`.
- **Goal latent of one pose, `z = B(g)`.** Worse than `z_r` on 13–15 of 15 goals (h-repr A1).
- **Plain GPI over landmark or goal policies.** The committed value of a policy toward another goal
  is about 0 far from the goal, so the max selects by noise; GPI does not value a switch.
- **"Sup over z near z_R" at test time (Touati and Ollivier, Thm 4).** The authors found it not
  necessary; the direction of `z_r` is stable and its training norm is the best (h-repr).
- **Action-level GPI with a learned one-step head `G(s,a,z) ≈ E[F(s',z)]`.** One-step value gaps are
  about `(1−γ)V`, below the `F` noise; the action carries no flip signal (h-exec M8: AUC 0.652 →
  0.660); it changes `π_l`'s actions.
- **The exact switching advantage instead of the proxy for `π_h`.** The paper finds the proxy's
  subgoals better (App. D.1, Fig. 8); the exact `R̂` costs A −29.3 pp (REPORT §4).
- **Geometric policy composition (Thakoor et al. 2022) and jumpy world models (Farebrother et
  al. 2026).** They need a generative γ-model or FB at several γ.
- **Min over the `F` heads in `π_h`'s advantage.** Upstream averages the heads' AWR weights
  (per head, then the mean of the weights over heads and samples: `third_party/ssm/agents/fbpiswitch.py:67-78`); B uses the min and gains nothing; a `π_h` tuning
  change in h-learn's lens.
- **ZSRL's Q-loss, orthonormality, mix ratio and dimension.** They retrain `F` and `B`.
- **Distribution-shift correction of `z_R` (Touati and Ollivier, Prop. 15).** The train and
  validation splits give the same `z_r` (h-repr A1).
- **Fine-tuning `Q = Fᵀz_R + q_θ` (Touati and Ollivier, p. 5).** Offline, this is approach B, which
  gave no gain.
- **A better low-level policy (π-Switch Table 2, Meta Motivo's FB-CPR).** Both papers point there;
  it is h-exec H2, out of scope for now.
- **Online adaptation of `z` (Sikchi et al. 2025; Meta Motivo's future work).** It needs
  interaction beyond evaluation.
- **An upright weight in the goal-ball mean.** The eval goal balls hold no flipped states (h-repr
  A1), and safety latents are covered (h-exec M7).

## 6. Reproduction of the measurements

Scratch scripts in this session's scratchpad, not in the repository; each step below is enough to
redo them.

- M1: `hp_window.py` subclasses `OracleWaypointController`; `act` walks `next_cell` from the
  agent's cell for `last` moves and averages the unit-norm region latents of the cells
  `first..last`. Runs: `PYTHONPATH=<scratch> uv run python -m fbplan.evaluate --env
  antmaze-medium-navigate-v0 --heldout --seeds 0 1 2 --episodes 20 --workers 4 --method
  <baseline | oracle | hp_window:make_window_oracle --opt first=1 last=3> --out <csv>`; tables by
  `analysis/tables.py <csvs> --label-root <dir> --reference baseline_heldout oracle2`.
- M2: `hp_datagraph.py <env> <r> <K> <landmarks>`. Train split; landmarks by
  `fbplan.landmarks.farthest_point_sampling` on xy of a 100k pool (`default_rng(0)`); each state
  is labelled with its nearest landmark if within `r`; per trajectory (`terminals` mark the ends),
  the visits are runs of one label, and the pass time of a visit is its step of closest approach;
  an edge keeps the smallest pass-to-pass gap ≤ `K`; `scipy.sparse.csgraph.shortest_path`; the
  stretch uses `MazeGeometry.geodesic_distances` between consecutive landmarks of the route.

## 7. Notes for the coordinator

- The Meta Motivo id in the start message (2410.20092) is OGBench; Meta Motivo is 2504.11054.
- G0 (§2) is cheap and bears on every route method and on the report's central finding: the
  oracle has no run on eval seeds 5–9.
- For the report: FB π-Switch's Table 2 is independent support for "the low level limits the
  hierarchy" (REPORT §8.2), from the authors themselves.
