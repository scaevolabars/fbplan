# Hypotheses: execution and the low-level policy (h-exec)

Lens: the frozen `π_l` caps success. Which intentions `z`, or which rules for when and how to
switch them, change what `π_l` does? Does success depend on properties of `z` that `π_h` does not
use? Is unfreezing `π_l` worth its cost? Brief: `docs/briefs/hypotheses.md`.

## Summary

- **On medium, no test-time choice of `z` that I tested lowers the flips.** Smoothing, holding and
  a safety reward in `z` leave them unchanged or raise them (§1, M4, M7). Flips are a per-step
  hazard of the frozen `π_l`, equal to the hazard of the train data (M1). Neither the action nor
  the position in the data predicts them well (M8). `z` steers where the ant goes, not how stably
  it walks.
- **On teleport, `z` can steer away from the trap** (a trap latent from the offline data halves
  the trap share), but the saved episodes then time out, so success does not move (M11). A time
  gate is the open part (H1, rank 1).
- **Unfreezing `π_l`** is worth testing for pace on large and giant (H2), not for flips (H5). It is
  fair only if every row, the baselines included, runs on the new `π_l`, and then its lift is not
  a high-level gain.
- **Power:** paired episodes are almost independent (M6), so a gain must be about +4 pp to pass
  the rule. All five hypotheses have a low probability of a gain (0.02–0.12).

## 1. New measurements (this session)

All runs below are cheap checks on macOS (arm64), 4 workers, at most 2 minutes each. Medium
runs use the 10 held-out goals (`task_id` 101–110), 40 episodes per goal, eval seed 50: 400
episodes, the same start states in every run. Nothing ran on the 5 eval tasks. Mac and Linux
agree on success rates, not on single episodes (REPORT §6), so these numbers are not comparable
episode by episode with `results/`. Scripts stay outside the repository (brief, Rules); §6 lists
what they compute, so that any owner can redo them.

**M1. Flips come from the data.** In the medium train split, 10.0 % of the states are flipped
(torso z-axis below the horizon), 210 of the 1,000 episodes flip, 193 end flipped, and none
recovers. The flip hazard of the data is 0.23 per 1,000 upright steps (large: 0.27). The
evaluation hazard, bounded from the CSVs and the failure classes (flip time unknown, so a range):

| medium, Linux | flips | hazard per 1,000 steps |
|---|---|---|
| train data (noisy expert) | 210 / 1,000 episodes | 0.23 |
| non-hier (`π_l(z_r)`, one fixed intention) | 155 | 0.20–0.24 |
| oracle | 176 | 0.26–0.36 |
| `baseline_raw` | 179 | 0.27–0.37 |
| `baseline` | 202 | 0.29–0.41 |

Sources: `results/medium/{baseline,baseline_raw,oracle,nonhier}.csv` and
`results/medium/failures/*.csv`. `π_l` acts at temperature 0, yet it flips about as often per step
as the noisy expert whose data it learned from.

**M2. FB represents flips.** With `ẑ_flip = mean((1[flipped] − p)·B(s))` over 200k train states
(the FB reward latent of the flip indicator), `B(s)ᵀẑ_flip` separates flipped states with AUC
0.985. In the baseline rollouts, `min_h F_h(s,z_t)ᵀẑ_flip` predicts a flip onset within 50 steps
with AUC 0.71. But the same value with `z_r` in place of `π_h`'s `z_t` predicts as well (0.71),
and their difference has AUC 0.47. So `F` knows where flips happen, but the two intentions do not
differ in predicted risk: the risk belongs to the state.

**M3. The warning is short.** From the last step with upright ≥ 0.7 to the flip onset: median 7
steps (10–90 %: 3–12). A reflex triggered by tilt has almost no time to act.

**M4. Intention jitter precedes flips, but does not cause them.** The 10-step mean of
`1 − cos(z_t, z_{t−1})` is 40 % higher in the 50 steps before a flip (AUC 0.63). Interventions
on the same 400 episodes:

| held-out, Mac | success | paired diff (pp) | flips | timeouts | jitter `1 − cos` | median steps of successes |
|---|---|---|---|---|---|---|
| baseline (normalized `z_r`) | 87.0 % | – | 32 | 20 | 0.0033 | 258 |
| EMA on `z`, β = 0.5 | 87.0 % | +0.0 [−4.5, +4.2] | 35 | 17 | 0.0009 | 284 |
| EMA on `z`, β = 0.8 | 88.2 % | +1.2 [−3.0, +5.5] | 34 | 13 | 0.0003 | 238 |
| hold `z` for 4 steps | 86.3 % | −0.8 [−5.2, +4.0] | 40 | 15 | 0.0025 | 262 |
| `baseline_raw` path (raw `z_r`) | 83.8 % | −3.2 [−8.0, +1.5] | 49 | 16 | 0.0033 | 282 |

EMA cuts the jitter by a factor of 3.5 to 12, and the flips do not move. So the jitter before a
flip is an effect of the unusual state, not its cause.

**M5. The raw `z_r` path is not better on held-out goals.** On the eval tasks (Linux),
`baseline_raw` beats `baseline` by +3.8 pp [+0.7, +6.8]. On the 10 held-out goals (Mac) it is
−3.2 pp [−8.0, +1.5], with 49 flips against 32. Both runs give the same jitter. This bears on
the bar, not on my lens: see the note to the coordinator in §5.

**M6. Paired episodes are almost independent.** On Linux medium, 30.5–33.2 % of the episodes
end differently between `baseline` and `baseline_raw`, the oracle, a retrained seed or a B seed.
Two independent coin flips at 78 % success differ in 34 % of the pairs. So the start state fixes
little of the outcome, and pairing removes little variance. At 1,250 episodes the paired
difference has a standard error of about 1.6 pp: the smallest significant gain is about +3.1 pp,
and 80 % power needs about +4.4 pp. Against `baseline_raw` (240 failures, 179 flips), +4.4 pp
means 55 fewer failures: a cut of a third of the flips, with no new timeouts.

**M7. Composing a safety reward into `z` does not lower flips.** FB is linear in the reward, so
`z + λ·ẑ_safe` with `ẑ_safe = −ẑ_flip` (both at norm `√d`) is the latent of "reach the
intention, and avoid flipped states". `π_l` was trained on Gaussian latents half of the time
(upstream `actor_latent_mix_prob = 0.5`), so such mixtures are inside its training range.
Same 400 held-out episodes:

| held-out, Mac | success | paired diff (pp) | flips | timeouts |
|---|---|---|---|---|
| baseline | 87.0 % | – | 32 | 20 |
| λ = −0.5 (toward flips, control) | 88.7 % | +1.8 [−2.5, +6.0] | 34 | 11 |
| λ = +0.5 | 80.2 % | −6.8 [−11.2, −2.0], p = 0.007 | 42 | 37 |
| λ = +1.0 | 74.8 % | −12.2 [−17.0, −7.0], p < 0.001 | 57 | 44 |

The safe direction loses more as λ grows, and flips rise with it. In the train data, `ẑ_flip` is
almost the negative of the speed latent (cosine −0.94: a flipped ant does not move), so `ẑ_safe`
is a generic "be upright and move" direction. Adding it dilutes the goal signal; it does not make
`π_l` walk more carefully.

**M8. In the data, the action carries no flip information beyond the state.** An MLP on the train
split (episode-level 70/30 split, upright states) predicts "flip onset within 30 steps" with test
AUC 0.652 from pose and velocity, 0.660 with the action added, and 0.652/0.654 with the xy
position as well. And the frozen `F` gives a fine-tune little to learn from: on train states 5–50
steps before a flip, the one-step flip advantage `γF(s',z)ᵀẑ_flip − F(s,z)ᵀẑ_flip` (z = `B` of the
state 50 steps ahead) has AUC 0.54, and its spread is 56 times smaller than that of the task
advantage. The task advantage itself does not flag these transitions (AUC 0.48).

**M9. Pace.** In the baseline rollouts `π_l` walks at 1.41 (mean xy speed while upright), the data
at 1.46; the data's 90th percentile is 2.21. Median steps to success are 363–390 for every
high-level row on medium (baseline, `baseline_raw`, oracle, 5 retrained, 5 B-clean seeds;
`results/medium/`): the high level does not change the pace. On large, success still rises by
8.7 pp over the last 200 steps (`results/large/baseline.csv`, 37.4 % at 1,000 steps, 28.7 % at
800), against 4.2 pp on medium; 72 % of large failures are timeouts (REPORT §7.2).

**M10. The teleport trap is visible in the offline data.** Of the train trajectories that jump
(an xy step above 2 units) and then stay in a 4.5-unit box until the end, 231 of 233 end in cell
(1,7), the trap. These states are 16 % of the train split. The latent
`ẑ_trap = mean((1[trap] − p)·B(s))` separates them with AUC 0.977, and it is almost orthogonal to
`ẑ_flip` (cosine −0.04). No maze map is needed for this definition.

**M11. A trap penalty in `z` halves the trap share on teleport, but success does not rise.**
`z_t = norm(π_h(s,z_r) − λ·ẑ_trap)`, with `ẑ_trap` at norm `√d`. Five start–goal pairs that are not
eval tasks (cells (1,2)→(7,8), (7,3)→(1,9), (4,8)→(7,2), (1,4)→(5,10), (3,2)→(5,8)), 40 episodes
each, eval seed 50, Mac; teleport has no held-out goal set yet:

| teleport, Mac | success | paired diff (pp) | trapped | flipped | median steps of successes | success per pair |
|---|---|---|---|---|---|---|
| λ = 0 (baseline) | 15.0 % | – | 52.5 % | 14.5 % | 371 | 0.12, 0.15, 0.00, 0.45, 0.03 |
| λ = 1 | 14.5 % | −0.5, p = 1.0 | 26.5 % | 15.0 % | 626 | 0.25, 0.03, 0.03, 0.30, 0.12 |
| λ = 2 | 5.5 % | −9.5, p < 0.01 | 13.0 % | 10.5 % | 753 | 0.10, 0.00, 0.00, 0.03, 0.15 |

The lever exists: `z` steers the ant away from the entries, and the trap share falls by half at
λ = 1. But the episodes that no longer jump walk, and the walks do not fit in 1,000 steps at
`π_l`'s pace (median steps to success 371 → 626). Two pairs gain (+12.5 and +10 pp), two lose (−12.5
and −15 pp). This is the trade-off of `docs/theory.md` §11.1 (`E[γ^H]` against `P(H ≤ T)`), now
measured with the agent's own route instead of the teleport-aware oracle's.

## 2. What the measurements say

1. **`z` steers where `π_l` goes, not how stably it walks.** Smoother intentions (M4), held
   intentions (M4) and a safety reward in `z` (M7) do not lower flips. The flip hazard at
   evaluation is the data's hazard (M1), the warning is about 7 steps (M3), and neither the
   action nor the position in the data predicts flips better than AUC 0.66 (M8). On medium, where
   75 % of `baseline_raw`'s failures are flips, no test-time choice of `z` that I can find has a
   lever on the main failure class.
2. **The only medium lever left in this lens is exposure time**, and the high level does not
   change the pace (M9). So, for medium, my lens confirms REPORT §8.2 with a mechanism: the flips
   are a per-step hazard of the frozen `π_l`, inherited from the data.
3. **Navigation-level risks are different.** The teleport trap is a "where" failure, and a `z` that
   steers away from it does act (M11). But without a faster walk the saved episodes time out. Pace
   on large and giant is a "how" property of `π_l`, so only a changed `π_l` can act on it (H2).
4. **Power.** Paired episodes are almost independent (M6). Any full test needs a true gain of
   about +4 pp against the stronger row to pass the rule with reasonable probability.

## 3. Hypotheses, ranked by expected value

Expected value = P(gain by the rule) × size / cost. The probabilities are my judgement from §1.

| # | Claim (short) | maze | P(gain by the rule) | size if real | cost | why this rank |
|---|---|---|---|---|---|---|
| H1 | trap penalty in `z`, applied only where the walk fits the time left | teleport | 0.12 | +4 to +8 pp | ~3 CPU h, no GPU | the lever is measured (M11); the gate is the open part |
| H2 | a faster `π_l`: unfrozen, progress-weighted AWR fine-tune; every row gets it | large, giant | 0.05 as a high-level gain; 0.3 as an absolute lift | +3 to +8 pp absolute | ~1 GPU h + ~4 CPU h | pace binds on large (M9); fair only as a new shared executor |
| H3 | goal anchoring near the goal: `z = norm((1−β)z_πh + β·z_r)` when `V_d(s)` is high | medium | 0.04 | +1 to +3 pp | ~1 CPU h | global β = 0.3 is level (−1.0 pp): fewer flips, more timeouts |
| H4 | pose-free intention: replace `π_h`'s `z` by the mean `B` of its nearest data states | medium | 0.04 | +1 to +3 pp | ~1 CPU h | region latents help waypoints, but `π_h`'s `z` is already a blurred mean |
| H5 | a flip-averse `π_l`: unfrozen, AWR with a separately learned flip critic | medium, large | 0.02 | +2 to +5 pp absolute | ~2 GPU h + ~4 CPU h | M8: the data shows no action-level flip signal |

None of these has a high probability. On medium in particular, every test-time change of `z` that
I tried leaves the flips where they are (M4, M7, H3), and the bar is `baseline_raw`. The ranking
puts H1 first because its failure class is the largest of all mazes (55 % of teleport episodes
end in the trap, `docs/theory.md` §11.3) and its lever is measured, and because it costs no GPU.

### H1. Trap-averse intentions with a time gate (teleport)

1. **Claim.** On teleport, `z_t = norm(π_h(s,z_r) − λ·g(s)·ẑ_trap)`, with the gate `g(s) = 1` only
   where a walk still fits in the steps left, beats both baseline rows on the 5 teleport eval tasks.
2. **Mechanism.** FB is linear in the reward, so `−λẑ_trap` adds "avoid the absorbing trap" to the
   intention's reward, and `F(s,z)ᵀẑ_trap` is the discounted trap occupancy of `π_z`. This moves
   the agent from the risk-neutral `E[γ^H]` toward `P(H ≤ T)`, which no baseline row does
   (`docs/theory.md` §11.1). Unlike the teleport-aware oracle, it keeps `π_h`'s own route and
   pace, and it is soft: `π_h` can still jump when the walk is hopeless. The gate is needed
   because M11 shows that an ungated penalty saves trap episodes only to lose them to the time
   limit. It beats the stronger row (`baseline`, 27.7 % on teleport) only if the gated version
   keeps the gains on the pairs where the walk fits (+10 and +12.5 pp in M11) and drops the losses.
3. **Evidence.** For: 54.7 % of baseline episodes end in the trap (`docs/theory.md` §11.3);
   the trap is identifiable from data alone and `B` separates it (M10); λ = 1 halves the trap
   share (M11). Against: ungated, success does not move (M11); the walks of 3 eval tasks are
   23–27 cells and fit only at ≤ 37–43 steps per cell, and `π_l` follows routes more slowly
   (`docs/theory.md` §11.3); `planner_high` and the oracle, which also walk, are below the
   baseline (REPORT §7.3).
4. **Cheap test** (≤ 10 min, 4 workers, on the 5 pairs of M11, not the eval tasks). The gate
   estimates the length of the trap-free walk from FB: `z_c = norm(z_r − λẑ_trap)`,
   `ρ̂_walk(s) = min_h F_h(s,z_c)ᵀz_r / F_goal`, where `F_goal` is the median of `F(s',z_c)ᵀz_r` over
   the train states that earn reward 1 in the relabelling of `z_r` (the goal ball, as `z_r` uses
   it), and `d̂(s) = log ρ̂_walk / log γ`. Then `g(s) = 1` while `d̂(s) ≤ 1000 − t`, else 0.
   λ ∈ {0.5, 1}. Falsified if no gated setting beats λ = 0 by ≥ 5 pp on the 200 episodes, or if
   the trap share does not fall.
5. **Full test.** d-eval first adds teleport held-out goals (`select_heldout_goals`, as for medium).
   Select λ ∈ {0.5, 1, 2} × gate {none, time} on them (3 eval seeds × 20 episodes per goal; teleport
   episodes run to about 900 steps, about 4 min per setting with 4 workers). Then one run on the 5
   eval tasks × 50 episodes × 5 eval seeds, paired with `results/teleport/baseline.csv` and
   `baseline_raw.csv` (`analysis/tables.py --reference baseline baseline_raw`). Cost: about 3 CPU
   hours on Linux, no GPU. The rule decides as written: significant against both rows.
6. **Constraints.** `F`, `B`, `π_l`, `π_h` frozen; `ẑ_trap` is a mean of `B` over train states;
   test-time only; no generative model. The gate uses only `F`, `B` and the step counter.
7. **Prior work.** FB reward composition (Touati and Ollivier 2021; Touati, Rapin and Ollivier
   2023); successor features with GPI for new reward mixtures (Barreto et al. 2017); constrained
   and risk-sensitive RL with an absorbing failure set. The time gate is a crude form of
   time-indexed or finite-horizon values, which FB at one γ cannot give exactly.

### H2. A faster `π_l` as a shared executor (large, giant)

1. **Claim.** An offline fine-tune of `π_l` that weights data actions by progress toward the
   intention walks faster, and with it the high level gets room on large.
2. **Mechanism.** On large 72 % of the failures are timeouts and success still rises 8.7 pp over
   the last 200 steps (M9); on giant the walks of tasks 1–4 fit only at ≤ 33–38 steps per cell (REPORT §7.3).
   `π_l` walks at the data's mean speed (M9), and the data holds faster actions (90th percentile
   +53 %). AWR can only reweight data actions, so a sharper weighting on the one-step progress
   `γF(s',z)ᵀz − F(s,z)ᵀz` moves `π_l` toward the faster ones that the data supports. M8 says the
   action barely changes the flip risk, so faster actions should not cost many flips.
   **Fairness.** A better `π_l` also lifts the baseline. So every row (`baseline`,
   `baseline_raw`, oracle, A, B) runs on the same `π_l'`, and the `π_l'` lift is reported as an
   executor change, not as a high-level gain. The high-level question on `π_l'` is then whether
   the oracle gains room over the baseline; only if it does is a high-level method worth a run.
   I recommend this reading. The other reading ("`π_h` + `π_l'` beats `π_h` + `π_l`") changes the
   frozen setting of the assignment, and the coordinator must decide on it.
3. **Evidence.** For: M9; REPORT §7.2–7.3; `docs/theory.md` §11.2 (on giant, 65–200 steps per cell
   for the oracle against 40 on medium). Against: `F` is the successor measure of the old `π_l`.
   A changed `π_l` makes `F`, `ρ̂`, B's critic and `π_h`'s training signal slightly off-policy; a
   KL or BC term keeps `π_l'` close, which also limits the speed-up. Faster walking may hit
   walls more.
4. **Cheap test** (offline, CPU, minutes). On train transitions, with `z = norm(B(s_{t+k}))`:
   (a) the mean xy speed of the actions in the top 10 % of AWR weights at the sharper temperature,
   against all actions; (b) the flip rate within 50 steps after those actions. Falsified if the
   top-weighted actions are not ≥ 15 % faster, or if they flip more often.
5. **Full test.** Fine-tune `π_l'` from the frozen weights, 5 seeds, about 200k steps each on one
   GPU (5 `π_h` seeds took 55 min on an RTX 3090 at $0.21/h; e-learner's log). Evaluate on large
   and giant: `baseline`, `baseline_raw` and the oracle, each with `π_l` and with `π_l'`, 1,250
   episodes per row (about 4 CPU hours on Linux for large and giant). Report the executor lift.
   The high-level gain rule applies only on `π_l'`: a method beats `baseline`/`π_l'` and
   `baseline_raw`/`π_l'`.
6. **Constraints.** Unfreezes `π_l`. Why: on large and giant the pace of `π_l`, not the choice of
   intentions, bounds success (M9, REPORT §8.2). Cost: about 1 GPU hour, re-evaluation of every
   row, and stale `F`. Offline data only; no generative model.
7. **Prior work.** AWR / AWAC / IQL policy extraction (Peng et al. 2019; Nair et al. 2020; Kostrikov
   et al. 2022); HIQL trains its low-level policy by AWR toward subgoals in the same way.

### H3. Goal anchoring near the goal (medium)

1. **Claim.** Blending the task latent into the intention, `z = norm((1−β)z_πh + β·z_r)`, only
   where the direct value `V_d(s) = F(s,z_r)ᵀz_r` is high, lowers the flips without new timeouts.
2. **Mechanism.** Non-hier, one fixed intention, has the lowest evaluation flip hazard (M1), and
   it scores above the baseline on tasks 1 and 2 (REPORT §7.1); it fails only on the long task 4,
   where `z_r` has no direction far from the goal. A gate on `V_d` keeps `π_h` where the route
   matters and uses the steadier anchored intention near the goal.
3. **Evidence.** For: M1; held-out β = 0.3 everywhere has 28 flips against 32 (hazard 0.19
   against 0.22 per 1,000 steps). Against: the same run has 28 timeouts against 20 and is level,
   −1.0 pp [−5.0, +3.2] (Mac, 400 episodes); M4 and M7 show that changes of `z` move flips little.
4. **Cheap test.** As in M4, 400 held-out episodes, β ∈ {0.3, 0.6} applied only when `V_d(s)`
   exceeds its median along successful baseline episodes. Falsified if the flips do not fall by
   ≥ 25 % (to ≤ 24) or the timeouts rise.
5. **Full test.** Select β and the gate on the medium held-out goals (Linux, 600 episodes per
   setting, 4 settings, about 10 CPU min each), then 1,250 eval episodes paired with
   `results/medium/baseline*.csv`. About 1 CPU hour. The rule as written.
6. **Constraints.** Test-time only; frozen modules; no data beyond `z_r`.
7. **Prior work.** Approach A's "task policy if `V_d(s)` is higher" rule (REPORT §4) is the hard
   (β = 1) form of this gate.

### H4. Pose-free intentions (medium)

1. **Claim.** Replacing `π_h`'s intention by the mean of `B` over its k nearest train states in
   latent space (`top-k of B(s)ᵀz_t`, k ≈ 256) makes `π_l` walk faster or more steadily.
2. **Mechanism.** With the true route, region-mean waypoint latents beat single-state ones by
   14–32 pp (REPORT §8.2): a mean over many poses keeps the position and drops the pose. If
   `π_h`'s mean output still carries pose, a neighbourhood mean removes it.
3. **Evidence.** For: the oracle sweep (`results/medium/oracle_sweep/`). Against: `π_h`'s mean
   output has norm 0.57 `√d` before normalization (my rollouts), so it is already an average over
   many targets; the oracle with region latents is not faster than `π_h` (median steps to success
   384 against 363–368, `results/medium/`).
4. **Cheap test.** As in M4, 400 held-out episodes, k ∈ {64, 256}, pool of 200k train states.
   Falsified if the median steps to success do not fall by ≥ 10 % and the success does not rise
   by ≥ 3 pp.
5. **Full test.** Select k on the medium held-out goals, then 1,250 eval episodes, paired; about
   1 CPU hour (one 200k × 128 matrix product per step).
6. **Constraints.** Test-time only; the pool holds real train states (a lookup, not a generative
   model), as the region latents of approach A do.
7. **Prior work.** Region latents of approach A and the oracle; latent nearest-neighbour
   projection onto the data manifold, as in SoRB's graph over replay states.

### H5. A flip-averse `π_l` (medium, large)

1. **Claim.** An AWR fine-tune of `π_l` with a penalty from a separately trained flip critic
   `Q_flip(s,a)` (TD on the flip indicator of the train data) lowers the flip hazard.
2. **Mechanism.** Flips are 75 % of `baseline_raw`'s failures on medium. M2 and M7 show that `z`
   cannot change the gait, so only `π_l` can. The frozen `F` gives almost no action-level flip
   signal (M8), so the critic must be learned from the flip label, which the data gives exactly.
3. **Evidence.** For: 210 flip episodes in the medium train data (M1). Against: M8, the action
   adds 0.008 AUC to a flip classifier, so the data shows no action that avoids flips; the
   evaluation hazard equals the data's (M1).
4. **Cheap test.** Done in part (M8). The rest: the same classifier with a 10-step action history
   instead of one action. Falsified if the test AUC gain stays below 0.03.
5. **Full test.** As H2 (fine-tune 5 seeds, every row on `π_l'`), with the flip penalty in the
   AWR weights; medium and large; about 2 GPU hours and 4 CPU hours.
6. **Constraints.** Unfreezes `π_l` and trains one new critic offline; same fairness rule and
   stale-`F` cost as H2.
7. **Prior work.** Safety critics and conservative safe offline RL (for example Srinivasan et al.
   2020; Xu et al. 2022).

## 4. Rejected hypotheses

- **Smoothing `z` (EMA) to stop flips.** EMA cuts the jitter by 3.5–12× and the flips stay at
  34–35 against 32 (M4).
- **Re-selecting `z` every k steps.** k = 4 gives 40 flips against 32 and −0.8 pp (M4).
- **A safety reward in `z` (`z + λẑ_safe`).** Success falls monotonically with λ, and flips rise
  (M7).
- **Risk-gated choice among candidate intentions with `F(s,z)ᵀẑ_flip`.** The candidates do not
  differ in predicted risk: the AUC of the difference is 0.47 (M2).
- **A tilt-triggered recovery intention.** The warning is about 7 steps (M3), and no train
  episode recovers from a flip (M1).
- **A speed reward in `z` on large.** Not run; M7 shows that a non-goal direction in `z` dilutes
  the goal signal and raises timeouts and flips.
- **An action low-pass filter on `π_l`'s output.** It changes `π_l`, not `z`; the baseline would
  get it too; M8 says the immediate action barely moves the flip risk.
- **Tuning the scale of `z_r` per maze or per goal.** That is selection on the eval tasks, and
  the order of the raw and normalized paths reverses on the held-out goals (M5).
- **Sampling `π_l` or `π_h` at temperature > 0.** The protocol fixes temperature 0, and noise
  adds variance, not a lever.
- **Retraining `F` and `B` with a larger γ, or `π_l` from scratch.** That is retraining FB, which
  the task forbids.

## 5. Notes for the coordinator

- **The `baseline_raw` bar on medium may be goal-specific.** On the held-out goals the raw path
  scores −3.2 pp [−8.0, +1.5] against the normalized one (M5, Mac, 400 episodes), the opposite
  sign of the eval tasks (+3.8 pp). Not significant, and Mac only, but a Linux run of
  `baseline_raw` on the held-out goals (600 episodes, a few CPU minutes) would show whether the
  bar reflects these 5 goals rather than the method. The rule stays as fixed.
- **Power.** Paired episodes are almost independent (M6): a gain must be about +4 pp to pass the
  rule with reasonable probability at 1,250 episodes.
- **H1 needs teleport held-out goals** (d-eval).
- **H2 and H5 need a decision on scope**: an unfrozen `π_l` is fair only as a new executor for
  every row, and then its lift is not a high-level gain.

## 6. Reproduction of the measurements

Scratch scripts outside the repository; each step below is enough to redo them.

- M1: upright = `1 − 2(q_x² + q_y²)` from `obs[3:7]` of the train split; a flip onset is the first
  step with upright < 0. Evaluation hazard: flips from `results/medium/failures/<method>.csv`;
  exposure between the total steps and the total steps minus 1,000 per flip.
- M2, M7: `ẑ_flip = mean((1[upright<0] − p)·B(s))` over 200k train states (`default_rng(0)`);
  rollouts call `π_h`, `π_l` and `F` through `agent.network.select(...)` exactly as
  `fbplan/frozen.py` does; flip onset = start of the final run of upright < 0 when the last 50
  steps are flipped (as `analysis/failures.py`).
- M4–M7, H3: held-out goals 101–110 of `results/heldout_goals/antmaze-medium-navigate-v0.csv`,
  `reset_env` and `reset_seed` from `fbplan/evaluate.py`, eval seed 50, episodes 0–39.
- M6: `results/medium/*.csv` joined on (`task_id`, `eval_seed`, `episode`).
- M8: an MLP (2 × 256, GELU, Adam 1e-3, 3,000 steps of 1,024) on upright train states, label
  "flip onset within 30 steps", episodes split 70/30 with `default_rng(0)`, negatives subsampled
  to 6 per positive and weighted 6:1.
- M10, M11: trap = states after the last jump (xy step > 2) of train trajectories that then stay
  in a 4.5-unit box; goal latents of the custom pairs from `FrozenFBModel.infer_goal_latent`.
