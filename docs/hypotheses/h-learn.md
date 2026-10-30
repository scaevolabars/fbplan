# Hypotheses: high-level learning (`π_h` and approach B)

Written on 2026-10-29. Lens: the training of `π_h`
(AWR temperature, advantage, subgoal and goal distribution, seeds, ensembles, distillation),
what HIQL, SoRB, QRL and CRL do differently, and the seed stability of approach B.
Notation: `docs/design.md` §3. Written at 23:45 MSK, before any result of the AWR-sharpness-at-2×
experiment (design.md §8).

## Summary

One offline check (below, §0) changes the picture of B and of the upstream `π_h`:

1. **Upstream AWR spends most of its weight on states that are already at the goal.** At
   `high_alpha = 0.1`, 56 % of the goal-latent AWR weight falls on samples with the state within
   0.5 xy units of the goal (inside OGBench's success radius, where a test episode has already
   ended), and 4 % on goals 12 or more xy units away (24 % of the samples). At 2×, 64 % of all
   weight sits on clipped samples.
2. **The long-range direction signal of the upstream advantage is the term `F(s,z_w)ᵀz`,
   and approach B removed it.** B sets `R = 0` for goal latents, as A does. Its switch advantage
   then anti-correlates with true progress at long range (rank correlation −0.16 and −0.20 for
   goals 16–40 xy units away), where the upstream proxy has +0.32 to +0.34.
3. **B's per-seed temperatures were not matched in effective sharpness.** Matching the std of
   `α·A` gave B weights with an effective sample size (ESS) of 2.7–5.2 % of the goal samples,
   against 16.8 % for the upstream `π_h`. The sharpest B seeds scored best.

So my ranking puts first two `π_h` training changes that keep the proxy's signal and fix the
weighting (H1, H2). They are `π_h` tuning gains, not gains of chaining, and the report must say so.

| Rank | Hypothesis | Kind | P(gain by the rule) | GPU h | CPU h |
|---|---|---|---|---|---|
| H1 | sharp AWR on the one-switch advantage without `V_H`: the attribution control for B at 2× and a gain candidate | tuning; attribution | 0.20 | 0.9 | 15 |
| H2 | log-ratio proxy advantage: scale-free AWR weights over goal distance | tuning | 0.20 | 2.8 | 35 |
| H3 | test-time ensemble of the 5 `π_h` seeds of a family | variance | 0.10 | 0 | 2 per family |
| H4 | ESS-matched AWR temperature for every family: B stable across seeds, sharpness matched for real | stability; fairness | 0.05 (gain), high (stability) | 0–0.9 | 0.2 (check); 15 if the run is needed |
| H5 | B critic in distance space (QRL-like shortest-path TD on FB `ρ`) | chaining | 0.08 | 1.8 | 25 |
| H6 | HIQL value for the long-range term of the `π_h` advantage | other critic | 0.08 | 2.3 | 25 |

The probabilities are my estimates. The bar on medium is `baseline_raw` (80.8 %). A pooled family
of 5 training seeds needs about +3.5 pp against it for a significant paired difference, so about
84 %. B's two best single seeds reached 81.9 % and 83.0 %. A hypothesis that changes only the
training of `π_h` must also be evaluated on large, where the oracle has more room
(+1.9 pp against `baseline`, +4.2 pp against `baseline_raw`; REPORT.md §8.2).

**Predictions for the 2× experiment** (fixed now, before its result):

- P1. The B-at-2× seeds do not split, because they share one temperature. My estimate of that
  temperature is about 0.2 × 16.8 / 15.5 ≈ 0.22 (medians of the per-batch std on the validation
  split, §0), close to B seed 0 (0.229). So B at 2× should score near the sharp B seeds, 80–83 %.
- P2. The baseline at 2× changes by at most 2 pp against the retrained baselines, because the
  extra sharpness goes mostly to clipped samples at the goal (§0, table A).
- P3. If B at 2× then beats the baseline at 2×, the two families are not at matched sharpness:
  goal-latent ESS about 3 % for B against 35 % for the baseline at 2× (§0). The pre-registered
  reading "chaining adds value" then needs the control H1 before the report states it.

## 0. The offline check (not committed)

**What it computes.** Medium validation split, provided checkpoint, 32 batches × 1024 samples
`(s, g, w)` drawn as upstream `HGCDataset.sample` (`fbplan.learned.sampling.sample_high_indices`),
50 % Gaussian task latents as upstream. Per sample: the upstream proxy advantage per F head,
exactly as `agents/fbpiswitch.py:high_actor_loss` (raw latents in the dot products); its three
terms; the clean one-switch advantage with B's guards (`option_terms` with the default
`OptionConfig`: `R = 0` for goals, min over heads, `ρ ≤ 0.9`, invalid edges dropped,
normalized `z`); B-clean's switch advantage with `V_H` of `value-single` seed 0
(`fbplan.learned.high_policy.switch_advantage`); log variants. For goal latents, the true progress
`p = d(s,g) − d(w,g)` with the exact geodesic of `fbplan.maze.MazeGeometry` (privileged; used only
to diagnose, never by a method). CPU, 1 process, 4 threads, about 1 minute per run. Scripts:
`awr_weights.py`, `awr_analyse.py`, `awr_terms.py`, `awr_more.py`, `awr_final.py`, outside the
repository.

**Limits.** The metrics show what the AWR regression target selects on the training
distribution, not closed-loop success. Comparisons pool different `(s, g)` within a distance bin;
the regression only compares samples with similar `(s, g)`. One split, one critic seed.

**A. Where the upstream AWR weight goes** (goal latents; weight share per bin of `d(s,g)` in xy units):

| `d(s,g)` | < 0.5 | 0.5–2 | 2–4 | 4–8 | 8–12 | 12–16 | 16–20 | 20–40 |
|---|---|---|---|---|---|---|---|---|
| share of samples | 16.8 % | 15.3 % | 9.8 % | 18.1 % | 16.5 % | 12.6 % | 6.5 % | 4.4 % |
| share of weight, α = 0.1 | 55.8 % | 17.4 % | 7.9 % | 9.8 % | 5.0 % | 2.4 % | 1.1 % | 0.5 % |
| share of weight, α = 0.2 | 42.3 % | 23.9 % | 9.7 % | 13.9 % | 6.8 % | 2.3 % | 0.8 % | 0.3 % |

- At α = 0.1, 1.3 % of the head-samples reach the clip `α·A ≥ 5` and carry 26.7 % of all
  weight; 96 % of them have a goal latent, and their mean `d(s,g)` is 0.8. At α = 0.2: 10.8 % of
  head-samples, 64.2 % of the weight.
- Why at the goal: if `s ≈ g`, the proxy `F(s,z_w)ᵀz + ρ·F(w,z)ᵀz − F(s,z)ᵀz` counts the reward at
  `s` in the first term and adds `ρ·V(w)`, so every subgoal gets a large positive advantage (the
  double count of REPORT.md §3).
- Why so many such samples: the goal is a uniform future state of the trajectory, and in 15.3 %
  (train) / 16.5 % (val) of draws it is within 0.5 xy units of `s`; 7.9 % of dataset steps move
  less than 0.005 xy units (for example flipped or stuck ants; not checked).
- OGBench ends an episode within 0.5 xy units of the goal (`terminate_at_goal`), so test time
  never queries `π_h` there.
- The 28.1 % of goal samples whose subgoal lies past the goal on the trajectory carry 35.4 % of
  the goal weight (mean weight 15.5 against 11.1).

**B. Which term carries the progress signal** (Spearman rank correlation of the advantage with
the true progress `p`, goal latents, per bin of `d(s,g)`; invariant to the temperature):

| Advantage | < 0.5 | 0.5–4 | 4–8 | 8–12 | 12–16 | 16–20 | 20–40 |
|---|---|---|---|---|---|---|---|
| upstream proxy | +0.32 | +0.62 | +0.57 | +0.50 | +0.43 | +0.34 | +0.32 |
| its term `F(s,z_w)ᵀz` alone | +0.16 | +0.15 | +0.33 | +0.45 | +0.41 | +0.35 | +0.29 |
| its terms `ρ·F(w,z)ᵀz − F(s,z)ᵀz` | +0.43 | +0.59 | +0.32 | +0.18 | +0.13 | +0.10 | +0.15 |
| one-switch, `R = 0`, B's guards (B with `V_H = V_1`) | +0.38 | +0.58 | +0.32 | +0.18 | +0.15 | +0.13 | +0.18 |
| **B-clean switch advantage** (`V_H` seed 0) | +0.35 | +0.57 | +0.29 | +0.09 | **−0.02** | **−0.16** | **−0.20** |
| B with the proxy term put back (`F(s,z_w)ᵀz + ρ·V_H(w) − V_H(s)`) | +0.29 | +0.62 | +0.57 | +0.46 | +0.33 | +0.17 | +0.11 |
| clean proxy: proxy with B's guards and normalized `z` | +0.30 | +0.63 | +0.60 | +0.54 | +0.47 | +0.39 | +0.38 |
| **log-ratio proxy** (H2) | +0.46 | +0.67 | +0.58 | +0.51 | +0.45 | +0.38 | +0.34 |

Samples per bin: 2754, 4123, 2973, 2716, 2075, 1071, 721.

- Beyond 8 xy units, the progress signal comes from `F(s,z_w)ᵀz`, the value for the goal of
  following the intention `z_w`. The one-switch part `ρ·V(w) − V(s)` is weak there (+0.10 to +0.18).
- `R = 0` removes that term. Approach A needed `R = 0` because a hard max over landmarks collects
  the noise of `R̂` (REPORT.md §4). AWR is an in-sample soft weighting, and here the term carries signal.
- `V_H` makes it worse at long range. Its gain grows with the distance to the goal (REPORT.md
  §8.5: `V_H` falls from 53 to 13 over the maze, `V_1` to 3.7), so `ρ·gain(w) − gain(s)` rewards a
  subgoal farther from the goal. This is a mechanism for the B-clean result.

**C. Effective sharpness** (ESS/N = `(Σw)² / (N·Σw²)` over goal-latent samples):

| Weights | ESS/N |
|---|---|
| upstream, α = 0.1 / 0.2 / 0.4 | 0.168 / 0.351 / 0.541 (the clip at 5 makes ESS rise with α past α ≈ 0.04, where it is 0.069) |
| B-clean `V_H` s0, α = 0.111 / 0.162 / 0.229 (the range of the trained seeds) | 0.052 / 0.033 / 0.027 |

At equal ESS, the gain in weighted-mean progress over the bin mean (xy units; the bin mean
progress is 3.4 at 12–16, 4.7 at 16–20 and 5.6 at 20–40):

| Weights | ESS/N | 12–16 | 16–20 | 20–40 | weight at `d < 0.5` | weight at `d ≥ 12` |
|---|---|---|---|---|---|---|
| upstream α = 0.1 | 0.168 | +1.46 | +1.58 | +1.82 | 56 % | 4 % |
| upstream α = 0.2 | 0.351 | +2.53 | +2.66 | +2.51 | 42 % | 3 % |
| B-clean `V_H`, α = 0.071 | 0.168 | +0.28 | +0.16 | −0.10 | 12 % | 31 % |
| one-switch `R = 0`, B's guards, α = 0.069 | 0.149 | +1.03 | +1.13 | +1.42 | 10 % | 35 % |
| log-ratio proxy, α = 2.45 | 0.177 | +2.71 | +3.46 | +3.74 | 17 % | 32 % |
| B-clean `V_H`, α = 0.175 | 0.030 | +1.48 | +1.61 | +2.83 | 6 % | 28 % |
| one-switch `R = 0`, B's guards, α = 0.19 | 0.043 | +3.04 | +3.07 | +3.67 | 5 % | 39 % |

At ESS 0.03 the 20–40 bin has about 22 effective samples, so its cells are noisy. The upstream
proxy and the clean proxy cannot reach an ESS below about 0.07 and 0.17: the clip at 5 flattens
their positive tail.

## H1. Sharp AWR on the one-switch advantage without `V_H` (B with `V_H = V_1`)

1. **Claim.** A `π_h` trained like B-clean but with `V_H` replaced by `V_1`, at B's 2×
   temperature, scores at least as high as B at 2×; so B's best seeds owe their score to sharp
   weighting, not to the multi-switch critic, and this `π_h` may pass `baseline_raw`.
2. **Mechanism.** The good B seeds (81.9 %, 83.0 %) had the sharpest weights (ESS 0.027–0.03).
   At that sharpness the one-switch advantage selects more long-range progress than B's
   (§0 C: +3.0 to +3.7 against +1.5 to +2.8 at 12–40 xy units), because `V_H`'s gain grows with
   distance and rewards moving away (§0 B). It also moves the weight off the at-goal samples
   (5 % at `d < 0.5`, against 56 % upstream). So it should beat the retrained baselines on the long
   task 4, where the sharp B seeds gained about 6 pp (71.6–72.4 % against 65.9 %).
3. **Evidence.** For: B seed split follows α (REPORT.md §8.5; success order 83.0, 81.9, 80.5,
   75.8, 75.5 equals the ESS order); B-wide repeats the pattern with the same per-seed α
   (`results/medium/b_wide/`, B wide 81.0, 77.0, 82.2, 72.3, 79.6 %); §0 B and C. Against: the
   clean one-switch has a weaker progress signal than the proxy at 8–20 xy units (+0.15 to +0.18
   against +0.34 to +0.50); the oracle gains only +0.5 pp against `baseline_raw` on medium
   (REPORT.md §8.2). A sharp weighting raises the variance of the fit (ESS 3–4 %).
4. **Cheap falsification test** (CPU, 2 minutes, after the 2× calibration exists). Read α from
   `runs/train_b/medium/clean-awr2x/calibration.json`. Rerun §0 C at that α for B's advantage and
   for the one-switch advantage. Falsified if the one-switch weights select less progress than B's
   at 8–40 xy units, or if their ESS differs from B's by more than 1.5×.
5. **Full test.** A "no critic" policy variant is added to the training code (`fbplan/learned/`;
   `y = R + ρ·V_1(w)`, baseline `V_1(s)`; everything else as `train_b policy --variant clean`), at
   the shared 2× temperature.
   5 seeds, one GPU batch (about 55 min on the RTX 3090, 0.9 GPU h, about $0.20). Evaluation:
   5 × 1250 episodes on medium (about 10 CPU h on the Linux queue) plus held-out goals
   (5 × 600 episodes, about 5 CPU h). Decision: (a) attribution: pooled paired difference against
   B at 2×; if it is ≥ 0 or not significant, the report does not credit a B gain to chaining;
   (b) gain: pooled and the held-out-selected seed, paired against `baseline_raw` and the
   normalized baseline; both significant → a `π_h` tuning gain. Then large (same cost).
6. **Constraint check.** F, B, `π_l` frozen; offline data; no generative model; only the AWR
   advantage of `π_h` changes. It needs no trained critic.
7. **Prior work.** AWR (Peng et al. 2019) and IQL/HIQL policy extraction with a sharp temperature
   (HIQL uses α = 3 on step-unit values); the one-switch value is FB π-Switch's own (paper §4).

## H2. Log-ratio proxy advantage: scale-free AWR weights

1. **Claim.** Training `π_h` with
   `A = log max(F(s,z_w)ᵀz + ρ̃·F(w,z)ᵀz, 1) − log max(F(s,z)ᵀz, 1)` for goal latents
   (normalized `z`, `ρ̃ = clip(ρ, 0, 0.99)`, mean over heads; the upstream proxy for Gaussian latents)
   beats both baseline rows on medium and on large.
2. **Mechanism.** FB values for a goal fall like `γ^d` with the distance (`V_1` median 77 at
   0–4 xy units, 9 at 20–40; §0). A linear advantage then has a spread that shrinks with distance,
   and the clip at 5 is reached mostly at the goal. In log space the advantage is a ratio, about
   `−(detour in steps)·log(1/γ)` for a correct value, at any distance. HIQL, QRL and CRL work in
   this space (step-unit values, quasimetrics, log density ratios). The log-ratio keeps the
   proxy's progress signal, the best of all definitions in every bin (§0 B), and gives the
   largest long-range selection at the upstream ESS (§0 C: +2.7 to +3.7 against +1.5 to +1.8).
   It moves the weight on goals ≥ 12 xy units away from 4 % to 32 %. Long goals decide task 4, the
   only task where the high level matters on medium (non-hier 8.0 % against 63.2 %, REPORT.md
   §8.2), and most of large.
3. **Evidence.** For: §0 A–C; sharper weighting helps (B seeds; P2 will show whether it helps the
   baseline). Against: there is no rollout evidence for a log advantage; FB values order far
   states weakly (rank correlation −0.35 in the far half, REPORT.md §8.5), so the log may amplify
   their noise; the floor at 1 is a free choice (2 % of states at 4–8 and 18 % at 20–40 xy units
   have `V_1 < 1`).
4. **Cheap falsification test** (CPU, done in §0; to repeat on the train split and for large,
   about 2 minutes each). Falsified if the log-ratio's rank correlation with progress at
   `d ≥ 12` falls below the upstream proxy's, or its weight share at `d ≥ 12` stays below 15 %.
   On medium it passes. Second check, before the GPU run: a floor of 0.5 and of 2 must give the
   same ranking (sensitivity).
5. **Full test.** An option `--advantage logratio` is added to `train_high` (the upstream loss with
   this advantage for goal latents). Temperatures α ∈ {1.25, 2.5, 5}, 5 seeds each: 3 GPU batches,
   2.8 GPU h, about $0.60. Select α on the held-out goals (design.md §8 rule; 15 × 600 episodes,
   about 14 CPU h), then evaluate the selected α on the 5 eval tasks (5 × 1250 episodes, 10 CPU h)
   and on large (10 CPU h; large needs held-out goals of its own, which exist in
   `results/heldout_goals/`). Gain rule: the pooled family and the selected seed, paired against
   `baseline_raw` and the normalized baseline. Attribution arm, only if it gains: the upstream
   advantage with at-goal samples (`‖xy(s) − xy(g)‖ ≤ 0.5`, the OGBench success test) dropped,
   one batch, which separates the weight allocation from the log transform.
6. **Constraint check.** F, B, `π_l` frozen; offline data; no generative model. The xy test uses
   the observation only, as A's xy cap does. It is a `π_h` tuning change, so the retrained-baseline
   family is its reference and the report names it as tuning, not chaining.
7. **Prior work.** HIQL (Park et al. 2023): AWR on value differences in step units, α = 3.
   QRL (Wang et al. 2023): quasimetric distances. CRL (Eysenbach et al. 2022): critics in log
   density-ratio space. None of them uses FB's intention value `F(s,z_w)ᵀz`.

## H3. Test-time ensemble of the `π_h` seeds of one family

1. **Claim.** Using `z = normalize(mean_k normalize(π_h^k(s, z_r)))` over the 5 seeds of a family
   at every step beats the pooled single seeds of that family and removes the seed spread.
2. **Mechanism.** The fit errors of the seeds are partly independent (pairwise cosine 0.955 on
   validation states). An average cancels them. For B (75.5–83.0 %),
   it also averages the over-sharp and under-sharp seeds. It is cheap and combines with the winner
   of H1 or H2.
3. **Evidence.** For: the seed spreads (retrained 3.1 pp, B 7.5 pp, B-wide 9.9 pp; REPORT.md §7.1,
   `results/medium/*/seed*.csv`). Against: a cosine of 0.955 leaves little to average; an average of
   latents in two different directions points between them (the actor-averaging risk of
   `docs/plans/approach-b.md` §9); the retrained family sits at 78.3 %, 2.5 pp below the bar.
4. **Cheap falsification test** (CPU, 5 minutes). On 2000 validation states × the 5 eval task
   latents, the cosine of each seed's latent with the ensemble latent. If all exceed 0.99, the
   ensemble behaves as one more seed: stop. Then one eval seed (250 episodes, about 0.4 CPU h)
   as a smoke test, not as a decision.
5. **Full test.** A controller `baseline_ensemble` that takes the 5 seeds of a family.
   Families: the retrained baselines, B-clean, and the H1/H2 winner. 1250 episodes each on medium,
   about 2 CPU h each (5 π_h passes per step add little to `π_l` and the physics). No GPU.
   Gain rule: paired against `baseline_raw` and the normalized baseline (no clusters: one controller).
   No selection among families on the eval tasks: the family is fixed before the run.
6. **Constraint check.** Nothing trained; F, B, `π_l` frozen.
7. **Prior work.** Policy ensembling and bagging; in offline RL, ensembles are mostly used for
   critics (SAC-N, EDAC); SoRB uses a critic ensemble for distances.

## H4. ESS-matched AWR temperature: B stable across seeds, sharpness matched for real

1. **Claim.** Calibrating the temperature of each family to a target ESS of its goal-latent AWR
   weights (at-goal samples excluded), from many batches and one value per family, makes the B
   seeds agree within the retrained baselines' spread, and is the right definition of "matched
   sharpness" for any B-against-baseline claim.
2. **Mechanism.** The std of the upstream advantage is set by its tails: the clip-reaching at-goal
   samples and the division by `F(w,z_w)ᵀB(w)` (`adv` from −600 to +640 per batch in the training
   logs, `runs/train_high/medium/seed*/train.csv`, not committed). So "same spread of `α·A`"
   gave B 3–6× fewer effective samples than the baseline (§0 C). ESS measures the sharpness of the
   regression directly, and it is stable across batches because it pools many samples.
3. **Evidence.** For: B's seed order equals its ESS order; per-seed α varied 2× (0.111–0.229); B-wide
   repeats it. Against: the 2× experiment already shares one temperature over all seeds, which
   should remove the split (P1); then H4 adds only the fairness part.
4. **Cheap falsification test** (CPU, 5 minutes). §0 C for each of the 5 `value-single` critics at
   its seed's α: ESS must follow the success order (83.0, 81.9, 80.5, 75.8, 75.5). After the 2×
   runs: ESS of the baseline-2× and the B-2× weights on the same batches. If they differ by less
   than 1.5×, the 2× comparison is matched and H4 adds nothing.
5. **Full test.** Only if the 2× ESS differ by more than 1.5×: B at the α that gives the baseline
   at 2×'s ESS (5 seeds, 0.9 GPU h), evaluated as the 2× families (15 CPU h). Reading: the seed spread
   (max − min of the 5 seeds) and the paired difference against the baseline at 2×.
6. **Constraint check.** Changes only the AWR temperature; everything frozen; offline.
7. **Prior work.** AWR and CRR practice of normalizing advantages (CRR's binary and exponential
   filters); ESS as a diagnostic of importance weights (Kong 1992).

## H5. B critic in distance space (a QRL-like shortest-path critic on FB `ρ`)

1. **Claim.** A multi-switch critic learned in steps instead of value units, with the target
   `d_H(s,g) = softmin_w [d_ρ(s,w) + d_H(w,g)]`, `d_ρ = log ρ / log γ` on in-trajectory edges, and
   the never-switch distance from `V_1` as its upper bound, orders far states better than `V_1`;
   and a `π_h` trained on the H2 advantage built on this critic gains over H2.
2. **Mechanism.** B's expectile in value space lifts far values (median 16.4 against 7.8 at 20–24
   xy units) and flattens the slope, so the switch advantage loses its sign at long range (§0 B).
   In distance space a target error is additive per hop and the slope stays one step per step;
   the advantage is minus the detour. This is the only hypothesis in my lens that adds chaining.
3. **Evidence.** For: §0 B (B's advantage −0.16 and −0.20 at 16–40 xy units, while `V_1`'s
   one-switch advantage keeps +0.13 to +0.18); A's graph value is this shortest path, and its
   problem on medium was execution time, not the route (REPORT.md §8.4). Against: FB `ρ̂` loses
   distance information at long range and has wormholes (REPORT.md §8.3); expectile 0.7 did not
   repair the value-space critic; the oracle leaves +0.5 pp on medium.
4. **Cheap falsification test** (CPU, about 10 minutes, approach A's code, no training). On landmark
   set 0 of medium, take A's graph distances `−log V_H(u)` (R̂ = 0, the defaults) for 50 landmark goals
   and compare their per-goal rank correlation with the maze geodesic against that of
   `−log V_1(u, B(g))`. If the graph does not order clearly better (≤ −0.70 against V_1's value,
   about −0.6), a critic that learns the same fixed point cannot either: stop.
5. **Full test.** A value mode `--space distance` (targets in steps, expectile 0.3 for
   a soft min, the same sampler and guards). Critic: 5 seeds, 0.9 GPU h. Gate: the critic check of
   REPORT.md §8.5 (per-goal rank correlation better than `V_1` in every seed) and §0 B (progress
   correlation ≥ +0.38 at 16–40 xy units). Only then a policy batch (0.9 GPU h) with the H2 advantage
   built on `d_H`, its temperature selected on held-out goals, and the evaluation of H2 (25 CPU h).
   Reading: paired against H2's selected family (chaining) and against both baselines (gain).
6. **Constraint check.** F, B, `π_l` frozen; offline in-trajectory edges, as B1; no generative model.
7. **Prior work.** QRL (quasimetric distances, triangle inequality by construction); SoRB (search
   on buffer distances); HIQL's value as a temporal distance. QRL's own policy extraction uses a
   learned latent dynamics model, which this constraint set excludes; only its distance idea is used.

## H6. HIQL value for the long-range term of the `π_h` advantage

1. **Claim.** Replacing `ρ·F(w,z)ᵀz − F(s,z)ᵀz` in the `π_h` advantage by `V_θ(w,g) − V_θ(s,g)`
   from a HIQL-style goal-conditioned value (IQL, expectile 0.7, reward −1, 30 % random goals),
   while keeping the FB term `F(s,z_w)ᵀz`, beats both baselines on large.
2. **Mechanism.** The weak part of the proxy at long range is the one-switch value through `w`
   (§0 B, +0.10 to +0.18). A TD value on dataset transitions stitches across trajectories and
   orders states by temporal distance; HIQL's high level reaches 93 % on medium (paper, with its own
   low level). The FB term still supplies the intention-specific signal for `π_l`.
3. **Evidence.** For: HIQL's numbers in the paper's table (design.md §1); §0 B. Against: a second
   critic is not FB chaining; HIQL's value has the same `γ^d` scale at long range (it needs H2's log
   form too); the oracle bound on medium.
4. **Cheap falsification test.** None on CPU without a trained value: the first gate is the critic
   check of REPORT.md §8.5 on the trained `V_θ` (per-goal rank correlation ≤ −0.8 against `V_1`'s
   −0.61). Training only the value: 0.9 GPU h for 5 seeds.
5. **Full test.** Value 5 seeds (upstream `GCValue`, HIQL recipe, 1M steps: about 1.4 GPU h), gate,
   then `π_h` 5 seeds (0.9 GPU h), held-out selection of α as in H2, evaluation on medium and large
   (25 CPU h). Gain rule as in H2.
6. **Constraint check.** F, B, `π_l` frozen; a new value network on offline data; no generative model.
   The report must say that the long-range value is not from FB.
7. **Prior work.** HIQL (Park et al. 2023), IQL (Kostrikov et al. 2022).

## Follow-ups that depend on the 2× result

| 2× outcome (design.md §8) | Next run, in order |
|---|---|
| B at 2× beats the baseline at 2× and `baseline_raw` | H1 (attribution) before the report credits chaining; H4 check; then large |
| both rise together | H2 (the sharpness lever with a balanced weighting), then H3 on the best family |
| B at 2× beats the baseline at 2× only | H1, then H2 |
| neither | H2; H1 only as a cheap check; H5 only if its CPU test passes |
| B at 2× still splits by seed | the split is not the temperature: report it; H3 for B; drop H4 |

## Answers to the lens questions

- **Temperature and its calibration.** The temperature matters through effective sharpness, and
  the clip at 5 makes `α` a poor measure of it (§0 C). Calibrate on ESS (H4), not on the std.
- **Advantage definition.** Keep `F(s,z_w)ᵀz`; it carries the long-range signal (§0 B). Use a
  scale-free form (H2). `R = 0` is right for a graph max (A), wrong for AWR.
- **Subgoal and goal distribution.** 17 % of goal draws sit at the goal and take 56 % of the weight
  (§0 A). H2 removes most of it; dropping at-goal samples is its attribution arm.
- **Seeds and B stability.** B splits by per-seed temperature; one shared temperature (the 2×
  design) should remove it (P1); ESS matching (H4) makes the comparison fair; an ensemble (H3)
  removes the rest.
- **Early stopping, distillation.** Not promising (rejected below).

## Rejected

- **B with the proxy term put back (`F(s,z_w)ᵀz + ρ·V_H(w) − V_H(s)`).** `V_H`'s gain cancels the
  term at long range: rank correlation +0.17 and +0.11 at 16–40 xy units, against +0.39 and +0.38
  for the clean proxy without `V_H` (§0 B).
- **Clip the subgoal at the goal (HIQL).** 28 % of goal samples have `w` past `g`, but near the goal
  a target past it acts as a look-ahead, which the oracle sweep does not penalize (REPORT.md §8.2);
  the sign of the effect is unclear and its size small.
- **Early stopping or checkpoint selection.** The training curves are flat after the first logs
  (actor loss 791 → 780, validation MSE 0.73 → 0.72, all 5 seeds alike).
- **Goal-latent-only `π_h` (no Gaussian latents).** Gaussian latents carry only 13 % of the AWR weight
  (§0), so little capacity is freed, and the FB zero-shot property is lost.
- **Mixture or discrete (codebook) `π_h`.** The regression MSE of 0.71 per dimension means that
  `π_h` outputs an average over very different `B(w)`. That average acts like a region latent, which
  `π_l` executes better than single-state latents (oracle sweep, REPORT.md §8.2); a multimodal
  `π_h` would output pose-specific latents.
- **Distill the oracle into `π_h`.** The oracle uses the maze map (privileged).
- **Distill A into `π_h` (B2).** A is 4–12 pp below the baseline; the B2 gate stays closed.
- **Critic-guided choice among `π_h` samples at test time (`argmax_k F(s,z_k)ᵀz_r`).** That value
  is close to 0 for far goals, and a max collects noise (A's `R̂` lesson); also h-exec's lens.
- **More training seeds.** Narrows the CI, does not move the mean.
- **Raw `z_r` for the retrained and B families.** The task latent is h-repr's lens. Any family of
  H1–H3 can also be evaluated with the raw `z_r` path, if h-repr shows that the +3.8 pp of
  `baseline_raw` transfers.
- **QRL as a method, CRL critic for `π_h`.** QRL's policy extraction needs a learned latent dynamics
  model (a generative model of latent states). CRL's critic learns the behaviour policy's occupancy,
  not that of the options `π_{z_w}`, which FB's `F(s,z_w)` already gives. Their distance and log-ratio
  ideas are in H2 and H5.
- **SoRB.** It is approach A (graph search over buffer states), 4–12 pp below the baseline.
