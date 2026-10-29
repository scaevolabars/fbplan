# Hypotheses for a gain: representations and the task latent (h-repr)

Session h-repr, 2026-10-29. Brief: `docs/briefs/hypotheses.md`. Lens: how the task latent and
the intention latents are built from the frozen `B`, what the FB option model and γ imply, and
which FB-side changes could create headroom.

New numbers come from scratch scripts on this Mac (CPU, at most 4 workers), outside the
repository; the appendix describes each check so that it can be repeated. They are not committed
results. The rollouts use only the 10 held-out goals of medium; no eval task was run.

## 1. Summary

1. **The direction of `z_r` is not a lever.** It is stable: a bootstrap over the validation
   trajectories inside the goal ball gives cosine ≥ 0.985 (5 % quantile) for 14 of 15 medium
   goals. It is robust: a goal radius of 1–2 xy units, a Gaussian kernel, the mean of normalized
   `B`, the train split or slow states change `π_h`'s intention about as much as a 20 % change of
   the norm, or less, on 13 of 15 goals, and two offline proxies do not move (A1).
2. **The norm of `z_r` at `π_h`'s input is a monotone look-ahead knob**, but the training norm is
   already at its peak. `π_h` has no LayerNorm on its input. With the input at `c·√d`, its
   decoded subgoal on medium lies 2.5 xy units ahead at `c = 0.5`, 3.4 at `c = 1` and 6.2 at
   `c = 2` (same pattern on large and teleport, A2). `baseline` is `c = 1`; `baseline_raw` is
   `c = 0.93–0.98` per task, an accident of the `1e-8` in the double average. On 600 paired
   held-out episodes, success is 81.3 / 82.2 / **85.7** / 84.0 / 83.8 % at
   `c` = 0.7 / 0.85 / 1.0 / 1.2 / 1.4; `c = 0.7` is −4.3 pp [−8.5, −0.2], p = 0.048 (A5).
3. **The raw-versus-normalized gain of medium does not replicate on held-out goals.**
   `baseline_raw` against `baseline` on the same 600 held-out episodes: −0.7 pp [−4.5, +3.3],
   p = 0.81 (A5). On the eval tasks it is +3.8 pp [+0.7, +6.8] (REPORT.md §6). Large (−2.3 pp)
   and teleport (−3.5 pp) have the sign of the held-out curve. So "the task latent is a lever"
   holds for the norm offline, not for success; the +3.8 pp is most likely noise or specific to
   the eval tasks, and the medium bar may sit above the true success of the same policy (§4).
4. **The offline data contain the absorbing states that cause most failures.** 10 % of the
   medium train states are flipped ants, and 210 of the 1,000 train trajectories end flipped. A
   data-only rule finds the teleport trap without the map: 17.3 % of the train states, against
   18 % from the map (theory.md §11.1). FB is linear in the reward, so the frozen model gives the
   discounted occupancy of these sets under any intention (A3, A4).
5. **γ and the option model.** One γ gives one point of the Laplace transform of the hitting
   time. No transform of `ρ̂` at γ = 0.99 gives another γ or `P(H ≤ 1000)`, so "several γ" means a
   second `F` (H6). The look-ahead knob is the test-time axis that a change of horizon would move,
   and on medium it shows no room.

## 2. Ranking

Expected value = P(gain by the rule) × size / cost. CPU hours assume this Mac's rate: 600
episodes take about 75 s and 1,250 about 2.5 min with 4 workers.

| Rank | Hypothesis | Env | Cost | P(gain) | Size if it works |
|---|---|---|---|---|---|
| 1 | H1 region-mean intention targets for `π_h` | medium | ~1.5 GPU-h, ~1 CPU-h | 0.10 | +2 to +5 pp |
| 2 | H2 absorbing-state-aware choice among `π_h`'s intentions | medium (flips), teleport (trap) | ~3 CPU-h, a new controller | 0.06 | +3 to +10 pp |
| 3 | H3 look-ahead knob `c`, selected on held-out goals | large, teleport (medium falsified) | ~1 CPU-h per maze, a 5-line option | 0.04 | +2 to +4 pp |
| 4 | H4 look-ahead that depends on the distance to the goal | medium | ~2 CPU-h | 0.04 | +2 to +4 pp |
| 5 | H5 one AWR scale for all goals: normalized reward vector, bounded `ρ̂` | medium | ~1.5 GPU-h, ~1 CPU-h | 0.05 | +2 to +3 pp |
| 6 | H6 FB-side: a pose-light `B`, or `F` at a second γ | all | GPU-days; unfreezes `F`, `B`, `π_l` | < 0.05 here | unknown |

None of these is a strong candidate. The evidence of this lens agrees with REPORT.md §8.2: the
room left for a high level with this `π_l` is small, and the largest representation effect
measured in the project (single-state against region-mean waypoint latents for `π_l`) points to
the intention latents, not the task latent.

## 3. Hypotheses

### H1. Region-mean intention targets for `π_h`

1. **Claim.** `π_h` retrained with the upstream recipe, but with the regression target
   `n(mean of B over the dataset states within 0.5 xy units of w))` instead of `n(B(w))`, beats
   both baseline rows.
2. **Mechanism.** `π_l` executes region-mean latents much better than single-state ones: on the
   true shortest path, region-mean waypoints score 77–82 % and single-state ones 50–63 %
   (REPORT.md §8.2, oracle sweep). A single-state `B(w)` encodes the pose and velocities of `w`,
   which `π_l` then tries to match. `π_h` regresses onto single-state targets, and its output is
   neither kind: its best cosine to a single-state latent is 0.84, to a region mean 0.85 (A1).
   Region targets make `π_h` emit the latent type that `π_l` executes best. This acts on the
   execution of every intention, so it can beat the stronger row, unlike a better route (the
   oracle gains +0.5 pp over `baseline_raw` with region latents and the true route).
3. **Evidence.** For: the oracle sweep (+14 to +32 pp over single-state latents at the same
   look-ahead); act-ahead on `planner_high` (region latent for acting) +3.2 pp, p = 0.07
   (REPORT.md §8.4). Against: A with region latents is worse, but there the latent also enters
   `ρ̂` (REPORT.md §4, §8.4); `π_h`'s output is already a weighted mean over subgoals, so part of
   the smoothing exists; B-region (window of ±5 trajectory steps, with B's critic) is pending.
4. **Cheap falsification test.** Read B-region against B-clean when d-eval reports it
   (`results/medium/b_region/`, pooled, `--cluster checkpoint_seed`). If B-region is not above
   B-clean, drop H1: the latent type then does not help through `π_h`. No new run.
5. **Full test.** One change to the upstream recipe: the AWR target of `high_actor_loss` becomes
   the 0.5-xy region mean of `w` (precomputed per train state with a KD-tree); the advantage
   keeps `B(w)`. Variant: the region latent in the advantage too (consistent option, but `ρ̂`
   gets flatter, REPORT.md §4). 5 seeds, 500k steps, `high_alpha` as the retrained baselines;
   about 1.5 GPU-h for 5 parallel seeds (e-learner's rate for the retrained baselines). Evaluate
   5 × 1,250 episodes and the 600-episode held-out set per seed (about 1 CPU-h). Decide by the
   gain rule: pooled and held-out-selected rows, significant against `baseline` and
   `baseline_raw`; the controlled reference is the 5 retrained baselines.
6. **Constraint check.** Only `π_h` is retrained, as for the retrained baselines. `F`, `B`, `π_l`
   frozen. Targets come from the offline data. No generative model.
7. **Prior work.** HIQL predicts subgoals in a learned representation instead of raw states
   (Park et al. 2023); HILP's temporal-distance latents (Park et al. 2024).

### H2. Choose among `π_h`'s own intentions with an FB value that penalizes absorbing states

1. **Claim.** At each step, take `K` candidate intentions from `π_h` (its output at input norms
   `c ∈ {0.85, 1.0, 1.2}`) and act on the one with the highest FB value of the reward
   `1_goal − λ·1_abs`, where `abs` is the flipped ant (medium) or the data-detected trap
   (teleport). This beats both rows.
2. **Mechanism.** Most failures end in an absorbing state: 70–75 % of the medium failures are
   flips (REPORT.md §8.2), and 54.7 % of all teleport baseline episodes end in the trap
   (`docs/status/d-eval.md`, teleport table). `π_h` was trained on the goal value only, so it never
   trades goal progress against flip or trap risk. By FB linearity, `z_abs = E[1_abs·B]` gives
   the occupancy `F(s,z_w)ᵀz_abs` for any intention. The candidates come from `π_h`, so the
   choice stays inside `π_h`'s distribution, and a max over 3 candidates collects little FB noise
   (A's failure was a max over hundreds of landmarks, REPORT.md §3.1). The two terms must be in
   reward units (unnormalized `E[r·B]`): with normalized latents a trap step costs only about
   `P(goal ball)/P(trap)` = 0.3–0.8 % / 17 % ≈ 0.02–0.05 of a goal step (up to the norms of the
   two mean latents), far too weak against the jump, whose value is
   about 50 times that of the walk (theory.md §11.1).
3. **Evidence.** For: the FB flip occupancy `F(s_t, B(s_{t+25}))ᵀz_flip` predicts a flip within
   100 steps in the medium train data with AUC 0.64 (base rate 2.2 %; torso tilt at `t`: 0.54;
   A4); the trap is found from data alone (A3), and FB knows that it is hard to leave (2 % of the
   pairs out of the trap have raw `ρ̂ ≥ 0.3`, theory.md §11.1); the candidates lead to different
   outcomes (140 of 600 held-out episodes differ between `c = 1.2` and `c = 1.0`, A5). Against:
   fed as a penalty direction into `π_h`, the latent barely moves the intention (cosine 0.91–0.98
   at λ = 1) and lowers the predicted absorbing mass by at most 7 % (A3), so the candidates may
   not differ in risk; the true route does not lower flips (oracle 176, `baseline_raw` 179,
   REPORT.md §8.2), so flips may come from `π_l` whatever the intention; on teleport, avoiding
   the trap helps only if `π_l` can walk, and the teleport-aware oracle scores 5.6 % and 21.4 %
   (REPORT.md §7.3); most outcome differences between candidates may be chaos, not risk.
4. **Cheap falsification test** (offline, about 5 min). For 2,000 upright medium states (and
   teleport states within 3 xy units of an entry), compute the 3 candidates and their flip (trap)
   occupancy in reward units. If the spread of the occupancy over the candidates of one state is
   below 10 % of its spread over states, the choice has nothing to choose: reject. Also check
   that the argmax with λ = 2 picks a lower-risk candidate than `c = 1` in at least 20 % of the
   states near an entry.
5. **Full test.** A controller (about 60 lines, d-eval or a new owner) and the data-only
   absorbing rules of A3. λ ∈ {0.5, 1, 2, 4} in reward units on the medium held-out goals
   (4 × 600 episodes); teleport first needs 10 held-out goals (`analysis/heldout_goals.py`).
   Select λ* by held-out success, then 1,250 eval episodes per maze, paired against both rows.
   Report the failure classes per λ (`analysis/failures.py`). Each step costs 3 `π_h` and 3 `F`
   calls, so about 2× the baseline: about 1.5 CPU-h per maze.
6. **Constraint check.** Test time only; all modules frozen. The absorbing sets come from the
   offline observations by rules that use no map: the torso orientation (part of the observation)
   and the xy spread of the rest of a trajectory. No generative model.
7. **Prior work.** Generalized policy improvement over successor-feature values (Barreto et al.
   2017, 2020); reward composition by linearity of FB (Touati and Ollivier 2021); dead-end
   detection from offline data (Fatemi et al. 2019, 2021).

### H3. The look-ahead knob, selected on held-out goals of large and teleport

1. **Claim.** Feeding `π_h` the task latent at norm `c·√d`, with one `c` per maze selected on its
   held-out goals, beats both rows on large or teleport. On medium it is falsified (below).
2. **Mechanism.** In training, `π_h`'s goal input always had norm `√d`. A larger norm
   extrapolates along the goal direction: the intention turns toward `z_r` (cosine 0.30, 0.41,
   0.55 at `c` = 0.5, 1, 2 on the medium eval goals) and its decoded subgoal moves further ahead
   (A2). `c → ∞` tends to the direct policy `π_l(z_r)` (non-hier, −11.6 pp on medium) and `c → 0`
   drops the goal, so success has an interior maximum. A maze with longer walks or a time limit
   that binds (large: 72 % of the failures are slow, REPORT.md §7.2) may have its optimum above 1.
3. **Evidence.** For: the offline effect is monotone on all 15 goals of medium and large and on
   both goals of teleport (A2); larger `c` reaches goals faster on medium held-out goals (median
   steps of successes 240 at `c = 1.2` against 269 at 1.0, A5). Against: on medium the held-out
   curve peaks at `c = 1.0` (A5), so the knob gives no gain there; large (`baseline_raw` at
   `c ≈ 0.90–0.95`, −2.3 pp) and teleport (−3.5 pp) agree with the medium curve below 1 and say
   nothing about `c > 1`; a gain would be a tuned input of the given `π_h`, not chaining, and
   every method that uses `π_h` would then get the same `c`.
4. **Cheap falsification test.** Done on medium (A5): falsified there. For large, 3 × 600
   held-out episodes at `c` = 1.0, 1.2, 1.4 (about 4 min): if neither 1.2 nor 1.4 is above 1.0,
   drop H3. First repair the `z_r` of large held-out goals 102 and 103, which are 74–77 %
   flipped states (A1).
5. **Full test.** A `scale` option for `BaselineController` (d-eval, about 5 lines; the scratch
   controller of A5 is the reference). Held-out grid `c` ∈ {0.85, 1.0, 1.15, 1.3, 1.5} plus
   `baseline_raw`, design.md §8 protocol (eval seeds 0 1 2, 20 episodes per goal); rule fixed in
   advance: the highest held-out success, ties to the `c` nearer 1. Then 1,250 eval episodes at
   `c*`, paired against both rows. About 1 CPU-h per maze. Teleport needs held-out goals first.
6. **Constraint check.** No training, all modules frozen, `z_r` from offline data as upstream.
7. **Prior work.** HIQL's subgoal step `k` fixes the look-ahead at training time (Park et al.
   2023). The guidance scale of classifier-free guidance extrapolates along a conditioning
   direction in the same way (Ho and Salimans 2022). FB zero-shot RL normalizes `z` (Touati and
   Ollivier 2021); we know no work that uses its norm as a test-time knob.

### H4. A look-ahead that depends on the distance to the goal

1. **Claim.** `c(s) = c_far` when the direct value `V_d(s) = F(s,z_r)ᵀz_r` is below a threshold
   τ and `c_near` above it, with the three parameters selected on held-out goals, beats both rows.
2. **Mechanism.** Far from the goal, a short look-ahead keeps the intention on reliable short
   FB ranges; near the goal, a long one (up to the direct policy) finishes faster. The direct
   policy fails mainly far away (task 4: 8.0 %; tasks 1 and 2: at or above `π_h`, REPORT.md
   §7.1), and the time limit ends about a quarter of the failures on medium and 72 % on large
   (REPORT.md §8.2, §7.2). `V_d` orders far states by the maze distance with rank correlation 0.61–0.76 (A1),
   so it is a usable "near" signal. SoRB acts directly when the goal is within the reliable range.
3. **Evidence.** For: `c = 1.2` is the fastest held-out setting and loses only −1.7 pp
   [−5.5, +2.2] (A5). Against: the constant-`c` curve on medium is flat or falling away from 1, so
   a schedule must find its gain in the interaction alone; three parameters on 600-episode
   held-out sets (standard error about 1.4 pp per point) invite overfitting of the selection.
4. **Cheap falsification test.** A 2 × 2 grid of (`c_far`, `c_near`) ∈ {0.85, 1.0} × {1.2, 1.5}
   with τ at the median `V_d` of states 3 xy units from the goal, on the 600 medium held-out
   episodes (about 5 min). If no cell beats `c = 1` by +2 pp or more, drop H4.
5. **Full test.** As H3, with the selected triple, on medium; about 2 CPU-h.
6. **Constraint check.** As H3.
7. **Prior work.** SoRB (Eysenbach et al. 2019): plan only beyond the range where the distance
   is reliable.

### H5. One AWR scale for all goals in the upstream advantage

1. **Claim.** `π_h` retrained with the upstream advantage, but with the reward latent normalized
   and `ρ̂` clipped to `[0, γ]` (rows with raw `ρ̂ ≥ 1` dropped), beats both rows.
2. **Mechanism.** Upstream `high_actor_loss` uses the raw `B(g)` as the reward vector
   (`successor_measure_extract(obs, z_goals, …)` with unnormalized `z_goals`), so the advantage
   of a goal scales with `|B(g)|`. On medium validation states `|B|` has mean 4.55, std 1.50, and
   5–95 % range 2.97–6.77 (A1): the effective AWR temperature varies 2.3× between goals in one
   batch. `ρ̂` divides by `F(w,z_w)ᵀB(w)` and is heavy-tailed where it is near 0 (REPORT.md §8.5).
   This is where a "per-task scaling" acts in this system: in training, not in `z_r`.
3. **Evidence.** For: AWR sharpness matters at this level (the B-clean seeds split by their
   temperature, REPORT.md §8.5). Against: B-clean already normalizes all latents and caps `ρ̂` at
   0.9 (with its critic) and gains only +1.1 pp pooled (REPORT.md §7.1); the AWR-at-2× experiment
   (design.md §8) tests sharpness directly; this overlaps the learner lens.
4. **Cheap falsification test** (offline, about 2 min). On 50 upstream batches, the share of the
   AWR weight mass `exp(min(α·A, 5))` on the 10 % of goals with the largest `|B(g)|` and on rows
   with raw `ρ̂ ≥ 1`. If both shares are within 2 percentage points of their share of rows, the
   mechanism is too small: reject.
5. **Full test.** 5 seeds with the two changes, about 1.5 GPU-h; evaluation and the rule as for
   H1, with the retrained baselines as the controlled reference.
6. **Constraint check.** Only `π_h` is retrained. Offline data only.
7. **Prior work.** AWR (Peng et al. 2019), AWAC (Nair et al. 2020), IQL (Kostrikov et al. 2022).

### H6. FB-side headroom: a pose-light `B`, or `F` at a second γ

1. **Claim.** A retrained FB (and `π_l`) whose `B` weights the position over the joint pose, or a
   second `F` at γ' ≈ 0.998, would give any high level room that the frozen model does not give.
2. **Mechanism.** Two FB-side limits bind. Single-state latents carry the full pose, and `π_l`
   executes them poorly (oracle sweep, −14 to −32 pp against region means). One γ is one point of
   the Laplace transform of the hitting time, so no frozen value separates "slow" from
   "sometimes never": on teleport the walk wins only at γ ≥ 0.9977–0.9993 (theory.md §11.1). Two
   γ give two equations, which separate the trap probability from the walking time.
3. **Evidence.** For: the oracle sweep; the teleport analysis; on giant (γ = 0.995) the FB
   distance saturates beyond about 45 xy units (REPORT.md §7.3). Against: a larger γ degrades
   long-range `ρ̂` (giant); the comparison would be against a new FB model, not a new high level.
4. **Cheap falsification test.** None with the frozen model. A tabular check on b-theory's
   teleport maze: recover `P(trap)` from exact `ρ` at γ and γ'.
5. **Full test.** Out of scope: FB pretraining per maze (hours of GPU per seed, 5 seeds), a new
   `π_l`, and retrained baselines on it for fairness.
6. **Constraint check.** Unfreezes `F`, `B` and `π_l`; this is why it ranks last.
7. **Prior work.** Learning over multiple horizons (Fedus et al. 2019); HILP (Park et al. 2024).

## 4. A note on the bar on medium

`baseline` and `baseline_raw` are the same policy at two values of one knob (`c = 1` and
`c ≈ 0.95`). On the 600 held-out episodes they do not differ (−0.7 pp [−4.5, +3.3], A5), and
the knob's curve falls below `c = 1`. On the eval tasks the raw row is +3.8 pp [+0.7, +6.8]
(Linux) and +2.6 pp, p = 0.11 (Mac, `docs/status/d-eval.md`). If this is noise, the rule
"significant against both rows" sets the medium bar about 3 pp above the true success of the
policy, and a real gain of 2–3 pp cannot pass it. Two readings remain open: noise, or an effect
specific to the 5 eval tasks. A re-run of both rows on 5 more eval seeds (2 × 1,250 episodes,
about 0.3 CPU-h) would separate them without any selection on the eval tasks.

## 5. Rejected

| Idea | Reason |
|---|---|
| Goal radius 1–2 xy, Gaussian kernel, mean of normalized `B`, train split, slow states only | on 13 of 15 goals: direction cosine ≥ 0.95 to `z_r`, intention cosine ≥ 0.991, about as much change as a 20 % norm change; offline proxies flat (A1) |
| Fix the double average (drop the `1e-8`) | changes only the norm: this is `c = 1`, the `baseline` itself |
| Upright states only in the goal ball | no flipped state in any eval goal ball of medium, large or teleport, so the latents do not change; it repairs large held-out goals 102 and 103 (74–77 % flipped), a fix for the selection set (A1) |
| Single-state goal latent `B(g*)`, as in `π_h`'s training | cosine 0.57–0.89 to `z_r`; worse value order on 13 and worse decoded progress on all 15 goals (A1) |
| Ensemble or bootstrap of goal latents | `z_r` is already stable (A1): nothing to average out |
| Penalty direction fed into `π_h`, `n(z_r) − λ·n(z_abs)` | the intention barely moves and the predicted absorbing mass falls by ≤ 7 % (A3); only as a choice (H2) |
| A per-task norm | the only non-selected per-task rule is `baseline_raw`'s accident of the `1e-8` (smaller norm for goals with fewer ball states: 0.925 at 485 states, 0.980 at 1,897), and it does not help on held-out goals (A5) |
| A smaller look-ahead (`c < 1`) on medium | −4.3 pp at `c = 0.7` (p = 0.048), −3.5 pp at 0.85 on held-out goals (A5) |
| "Several γ" from the frozen `F` by `ρ̂^κ` | exact only for a fixed hitting time; for a risky jump it keeps the trap factor as `(2/3)^κ` and so values the jump above a true γ' would; it cannot give `P(H ≤ T)` |
| Isotonic calibration of `ρ̂` against the data offsets `γ^k` | the data offset only bounds `H` from above (`π_z` can be faster than the data), so the labels are biased; A and B use `ρ̂` only through ranks or under a cap |
| Region latents for planning (`ρ̂`) | measured: −12.1 pp (REPORT.md §7.1) |

## Appendix: the cheap checks

All on CPU with the frozen checkpoints; scripts outside the repository (session scratchpad).
`n(·)` scales a vector to norm `√d`.

**A1. Construction of `z_r` (medium; 5 eval goals and 10 held-out goals).** For each goal, the
val states within the goal radius 0.5 (as upstream), my `z_r` matches `infer_task_latent` with
cosine 1.000000. The ball holds 485–1,897 val states from 17–65 trajectories. Bootstrap over these
trajectories (200 draws): cosine to `z_r` ≥ 0.985 at the 5 % quantile, except goal 109 (0.965).
Candidates: the mean of normalized `B`; radius 1.0, 1.5, 2.0; a Gaussian kernel (σ = 1); the train
split (400k-state sample: 2,638–5,500 ball states from 185–688 trajectories), radius 0.5 and 1.0;
the slower half of the ball states; the single ball state nearest the goal. On 2,000 random val
states: the cosine of `π_h`'s intention to the baseline's, the mean absolute action change of
`π_l`, the rank correlation of `V(s) = min_h F_h(s,z)ᵀz` with minus the maze geodesic distance
(states more than 3 xy units away), and the decoded progress (the val state of a 30k pool whose
normalized `B` best matches the intention; its geodesic distance to the goal against the state's).

| Candidate | intention cosine | action change | value order | decoded progress (xy) |
|---|---|---|---|---|
| `z_r` (baseline) | 1 | 0 | 0.61–0.76 | 1.74–3.32 |
| radius, kernel, normalized mean, train split, slow states; 13 goals (not 4, 110) | ≥ 0.991 | ≤ 0.0060 | within ±0.06 | within ±0.42 |
| the same, goal 4 (larger radii) / goal 110 (radius 1.0–2.0, kernel) | ≥ 0.989 / ≥ 0.928 | ≤ 0.0072 / ≤ 0.018 | up to +0.03 / up to +0.03 | up to +0.39 / down to −0.91 |
| norm `c = 0.9` / `c = 0.8` (for scale) | 0.997–0.998 / 0.989–0.992 | 0.0025–0.0033 / 0.0050–0.0065 | unchanged | −0.15 to −0.50 / −0.29 to −0.89 |
| single ball state | 0.953–0.988 | 0.008–0.017 | lower on 13 of 15 | lower on 15 of 15 |

`|B|` over medium val states: mean 4.55, std 1.50, 5–95 % range 2.97–6.77 (`√d` = 11.31).
`π_h`'s intention (goal 1): best cosine 0.837 to a single-state latent of the pool, 0.850 to a
region mean (0.5 xy) of the pool. Flipped states in the goal balls (torso z-axis below the
horizon, `analysis/failures.py:upright`): none for the eval goals of medium, large and teleport
and for the medium held-out goals; large held-out goals 102 and 103 hold 460 of 626 and 860 of
1,124 (cosine of `z_r` to the upright-only mean 0.41 and 0.62), goal 105 173 of 1,236.

**A2. The norm at `π_h`'s input.** Input `c·n(z_r)`; same states and decoding as A1. Means over
goals (eval goals / all goals with held-out ones; teleport has 2 distinct eval goals and no
held-out set):

| `c` | medium: progress | medium: subgoal reach (xy) | medium: cos(intention, `z_r`) | large: progress | large: reach | teleport: progress | teleport: reach |
|---|---|---|---|---|---|---|---|
| 0.5 | +0.91 / +0.93 | 2.71 / 2.53 | 0.30 / 0.37 | +0.87 / +0.46 | 3.48 / 3.32 | +1.15 | 3.38 |
| 0.8 | +1.85 / +1.74 | 3.27 / 3.00 | 0.37 / 0.44 | +1.48 / +1.31 | 3.74 / 3.61 | +2.60 | 4.12 |
| 1.0 | +2.49 / +2.28 | 3.77 / 3.42 | 0.41 / 0.48 | +1.95 / +1.92 | 4.11 / 4.03 | +3.53 | 5.04 |
| 1.25 | +3.41 / +3.06 | 4.50 / 4.01 | 0.46 / 0.53 | +2.84 / +2.95 | 4.83 / 4.88 | +4.58 | 6.04 |
| 2.0 | +6.42 / +5.60 | 6.98 / 6.18 | 0.55 / 0.62 | +4.90 / +5.94 | 7.06 / 8.21 | +10.76 | 10.58 |

Per goal, the progress rises with `c` on every medium goal. Upstream norms `|z_r|/√d`: medium
0.925–0.980, large 0.899–0.945 (held-out down to 0.712), teleport 0.927–0.969. The decoded
progress is a proxy: at large `c` the decoded state jumps toward the goal region, as the
intention approaches the goal latent.

**A3. Absorbing states in the train split (1M states, 1,000 trajectories).** Flipped: medium
10.0 % of the states, 210 trajectories end flipped; teleport 11.2 %, 221. "Absorbed": the rest
of the trajectory, at least 200 steps, stays within 2 xy units of its mean. On teleport, 172,750
absorbed states lie in the cell `(1, 7)`, the trap (17.3 % of the states; theory.md §11.1: 18 %).
Latent penalty `z = n(n(z_r) − λ·n(z_abs))`, 2,000 upright states, λ ∈ {0, 0.25, 0.5, 1}: the
intention's cosine to λ = 0 at λ = 1 is 0.91–0.94 (teleport, trap or flip) and 0.95–0.98
(medium, flip); the FB absorbing occupancy `min_h F_h(s,z_w)ᵀn(z_abs)/√d` under the intention
falls by at most 7 % (teleport trap 2.245 → 2.090) and changes by −4 % to +1 % for medium
flips. λ here is in
normalized units, a weak penalty (H2, mechanism).

**A4. Does FB predict flips?** 20,000 upright medium train states `s_t` (`upright > 0.5`) whose
trajectory lasts 100 more steps; label: a flip within `(t, t+100]` (2.2 %). AUC of
`min_h F_h(s_t, B(s_{t+25}))ᵀ n(z_flip)` (the data's own subgoal as the intention): 0.641; with
`B(s_t)`: 0.618; torso tilt `−upright(s_t)`: 0.543; torso height: 0.458.

**A5. Rollouts on the medium held-out goals.** Scratch controller: the baseline with
`c·n(z_r)` at `π_h`'s input (`c = 1` is the baseline); `fbplan.evaluate --heldout --seeds 0 1 2
--episodes 20 --workers 4`, 600 paired episodes per row; differences paired against `c = 1`,
bootstrap stratified by goal, exact McNemar p.

| Row | success | against `c = 1` (pp) | discordant (+/−) | p | median steps of successes |
|---|---|---|---|---|---|
| `c = 0.7` | 81.3 % | −4.3 [−8.5, −0.2] | 67 / 93 | 0.048 | 268 |
| `c = 0.85` | 82.2 % | −3.5 [−7.7, +0.5] | 67 / 88 | 0.11 | 250 |
| `c = 1.0` | 85.7 % | – | – | – | 269 |
| `c = 1.2` | 84.0 % | −1.7 [−5.5, +2.2] | 65 / 75 | 0.45 | 240 |
| `c = 1.4` | 83.8 % | −1.8 [−5.3, +2.0] | 61 / 72 | 0.39 | 260 |
| `baseline_raw` | 85.0 % | −0.7 [−4.5, +3.3] | 72 / 76 | 0.81 | – |

A first pass on 100 of these episodes (eval seed 0, 10 per goal) gave `c = 0.7` +9 pp; the
600-episode run reverses it. The retrained baselines score 83.7–87.8 % on the same held-out
protocol on Linux (`results/medium/heldout_selection.csv`), so `c = 1` at 85.7 % is typical. The
rollouts used 9.0 min of wall time with 4 workers in 9 runs, within the brief's 10 minutes.
