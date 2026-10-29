# Report: planning over sequences of intentions with frozen FB representations

Status: draft. All sections are complete except the B-wide and B-region rows and the A ablations.
Notation: `docs/design.md` §3; names that differ between the report and the code: Appendix. Every
number names the committed file it comes from. pp = percentage points; CI = 95 % confidence
interval.

**Platform.** All evaluation numbers come from one Linux (x86_64) machine, in `results/<env>/`.
The same commands on macOS (arm64) are a reproducibility check, in `results/macos/` (§6). Giant
has no CSV yet; §7.3 quotes preliminary numbers and marks them.

## 1. Summary

- **Question.** Can a high level that values a sequence of intentions beat the one-switch `π_h`
  of FB π-Switch, with `F`, `B` and `π_l` frozen and offline data only?
- **Theory (§3).** A chain of switches has an exact value: the chain formula and a graph Bellman
  equation, checked to 1e-8 on exact tabular mazes. Chaining can help only where `π_l(·∣s,z_r)`
  or the one-switch value fails at long range. With FB errors it needs `R̂ = 0` for goal reaching
  and guards. Then a guarded chain reaches 0.46 on far pairs of a tabular maze, against 0.26 for
  an idealised `π_h` (§3.1).
- **Central finding: with this frozen `π_l`, no high-level choice of intentions has measurable
  room on these mazes.** Two baseline rows differ by maze, and the stronger one is the bar (§6).
  Against it, the privileged oracle, which routes on the true maze map, gains +0.5 pp
  [−2.6, +3.5] on medium and +1.9 pp [−1.8, +5.6] on large, both not significant (§8.2). On giant
  the oracle reaches about 2 % (preliminary). On teleport, the time limit and a trap cap an agent
  that maximises a discounted value exactly at about 50 % (§7.3).
- **Approach A (graph planner, §4) is below the stronger baseline on every maze:** medium −8.3 to
  −15.9 pp, large −8.0 to −11.1 pp, teleport −3.6 to −9.1 pp (pooled paired differences of the
  A variants, all significant; §7). Its loss is mostly time: A reaches the goal later, and its
  extra failures are episodes still moving at the time limit (§8.4).
- **Approach B (learned multi-switch critic, §5) gives no gain.** B-clean changes only the critic
  of `π_h`. It gains +1.1 pp [−2.4, +4.5] over 5 retrained baselines and scores −1.5 pp
  [−5.1, +2.1] against the stronger baseline (§7.1). The learned `V_H` raises long-range values but
  orders states no better than the one-switch value, at expectile 0.9 or 0.7. The 5 training seeds
  split into two significantly different groups (75.5–75.8 % and 80.5–83.0 %). With +0.5 pp of
  room for the true route, a gain was unlikely (§8.5). B stops at its gate; B-wide and B-region
  *(pending)*.
- **Evaluation (§6).** 1,250 paired episodes per method and maze. The provided `π_h` behaves as a
  typical seed of 5 retrained ones. Mac and Linux runs agree on success rates within noise.

## 2. Setting

**Task.** OGBench AntMaze navigation: a quadruped ant must reach a goal position in a maze.
The main env is `antmaze-medium-navigate-v0`; large, giant and teleport (stochastic teleporters)
are bonus envs. Each env has 5 evaluation tasks, each a start cell and a goal cell. An episode
lasts at most 1000 steps and succeeds when the ant reaches the goal radius (OGBench's
`info["success"]`).

**Given and frozen.** One FB π-Switch checkpoint per env, training seed 0 (`third_party/ssm`,
arXiv 2605.13207):

| Module | Role | Detail |
|---|---|---|
| `B(s)` | backward representation | `d = 128`; latents are normalized to norm `√d` |
| `F(s,z)` | forward representation | 2 ensemble heads; `F(s,z)ᵀB(s') ≈ M^{π_z}_s(s')` |
| `π_l(a∣s,z)` | low-level policy | acts toward the intention `z` |
| `π_h(z∣s,z_r)` | high-level policy, **the baseline** | picks a new intention at every step |

The discount is γ = 0.99 (giant: 0.995). The task latent `z_r = mean(r·B(s))` over 100k relabeled
validation states is a mean of `B` over the goal ball. Upstream gives this latent to `π_h`
without normalizing it, and a double average in upstream `infer_latent` leaves its norm below
`√d`. Our baseline normalizes it to `√d`, the norm that `π_h` saw in training (§6).

**What the baseline already does.** `π_h` is trained with AWR on the proxy advantage

```
Â(s,w,z) = F(s,z_w)ᵀz + ρ(s,w)·F(w,z)ᵀz − F(s,z)ᵀz,     z_w = B(w)
```

for subgoals `w` on the trajectory of `s` at a geometric offset with mean `1/(1−γ) = 100` steps.
So `π_h` looks **one switch** ahead: "follow `π_{z_w}` until `w`, then follow `π_z` forever".
The proxy drops the term `−ρ(s,w)·F(w,z_w)ᵀz` of the paper's Eq. (5), on purpose (paper Sec. 6).
At test time `π_h` re-selects at every step, so the executed behaviour does chain many switches,
but each choice is valued as if it were the last switch.

**Goal.** A high-level mechanism that values a **sequence** of intentions and beats `π_h` on long
horizons. Constraints: no generative model of states or trajectories; no new environment
interaction for training (offline OGBench data only); `F`, `B` and `π_l` stay frozen.

**Two approaches.**

- **A** (§4), no training: a graph over dataset landmarks whose edges are FB option models; value
  iteration at test time; waypoint execution with `π_l`.
- **B** (§5), offline training: a learned multi-switch value `V_H(s,z)` replaces the one-switch
  value in the AWR advantage of `π_h`; the test-time controller stays the baseline's.

## 3. From one switch to a chain of switches

Derivations, assumptions and the exact checks: `docs/theory.md`. Each property below is checked
to 1e-8 on exact tabular copies of the medium and teleport mazes (`tests/test_tabular_theory.py`).

**Option model.** The option `o_w` is "follow `π_l(·∣s,z_w)` with `z_w = B(w)` fixed until `w` is
hit"; `H` is the hitting time. With `M ≈ FᵀB`:

```
ρ(s,w) = E[γ^H]                  ≈ F(s,z_w)ᵀB(w) / F(w,z_w)ᵀB(w)
R(s,w) = E[Σ_{t<H} γ^t r(s_t)]   ≈ F(s,z_w)ᵀz_r − ρ(s,w)·F(w,z_w)ᵀz_r
```

Both follow from the strong Markov property at `H` (`docs/theory.md` §2). Both are invariant to the
scale of `B(w)`. For `s ≠ w`, `H ≥ 1`, so `ρ(s,w) ≤ γ`.

**One switch** (paper, Theorem 1): `V(s; w → π) = R(s,w) + ρ(s,w)·V^π(w)`.

**A chain.** At `H` the agent is exactly at `w`, so the switch formula applies again at `w`
(Proposition 1, by induction):

```
V(s; w_1,…,w_k → π) = R(s,w_1) + ρ(s,w_1)·V(w_1; w_2,…,w_k → π)
```

**Graph Bellman equation** over a landmark set `L` (Proposition 2), with `V_d(u) = F(u,z_r)ᵀz_r`:

```
V_H(u) = max( V_d(u),  max_{w∈L, w≠u} R(u,w) + ρ(u,w)·V_H(w) )
```

| Property | Consequence for the methods |
|---|---|
| A self-loop (`ρ = 1`, `R = 0`) makes every larger `V_H(u)` a fixed point | exclude `w = u` |
| Without self-loops `ρ ≤ γ`, so the operator is a γ-contraction | clip `ρ̂` to `[0, γ]` for `u ≠ w`; value iteration converges from any start |
| `k` sweeps from `V_d` give the best chain of at most `k` switches | value iteration searches over chains, not over single switches |
| Open-loop chains are optimal among option policies | a route of landmarks is enough |
| If `π_{z_r}` is optimal, `V_H = V*` | **chaining can only help where FB values or `π_l(·∣s,z_r)` are inaccurate** |
| Goal reaching: `R = 0` off the goal | `V_H` is a shortest path with edge cost `−log ρ(u,w)` |
| With exact values, re-planning at every step is never worse than commitment (option interruption, Sutton, Precup and Singh 1999) | commitment is only a defence against noisy values |

**Continuous states.** A single state is hit with probability 0. The option then ends on a ball
around `w`, and the next option starts at the entry point, not at `w`. The chain formula becomes
approximate. Its error grows with the number of hops and with the change of option values inside
the ball (bound in `docs/theory.md` §7). So hops should be few and the reach radius small.

**The dropped term in a chain.** The proxy values the time after `w` twice: in `F(s,z_w)ᵀz` and
again in `ρ·V(w)`. For one switch this is one bias term. In a chain it repeats at every hop, and
two neighbouring landmarks inside a rewarding region form a 2-cycle that the proxy values at
`ρV/(1−ρ)`. On the tabular maze with region rewards and exact FB, the predicted value of the best
chain is wrong by 111.8 with the proxy and by 0.0 with the exact `R`; a single switch with the
proxy is wrong by 10.7 (`docs/figures/theory/study_reward.csv`). So the proxy must not be used in
a graph.

### 3.1 When does chaining help? The tabular study

`fbplan/tabular/` builds a fake FB model on the 104-state medium maze (γ = 0.9) with exact
successor measures and controlled errors: independent noise on `ρ̂` per head, and a controller
`π_l` that is either exact or greedy on the noisy `F(s,z)ᵀz`. The greedy controller is reliable
only at short range, as a learned `π_l` may be. Success means the goal is reached in 100 steps;
all start–goal pairs; 5 noise seeds (`docs/theory.md` §9).

Success on far pairs (11 steps or more), noise std 0.04 on `ρ̂`
(`docs/figures/theory/study_noise.csv`):

| Planner | greedy `π_l` | exact `π_l` |
|---|---|---|
| direct, `π_l(z_r)` | 0.16 | 1.00 |
| idealised one-switch `π_h` | 0.26 | 0.95 |
| single switch, re-planned | 0.24 | 0.91 |
| chained, FB `R̂` | 0.20 | 0.28 |
| chained, `R̂ = 0` | 0.37 | 1.00 |
| chained, `R̂ = 0`, guards (head min, `ρ_min` 0.4, commit) | **0.46** | 1.00 |

![Success against distance in the tabular study](docs/figures/theory/success_vs_distance.png)

Four results set the design of A and B:

1. **Chaining helps when `π_l` fails at long range.** With the greedy controller, the guarded
   chained planner reaches 0.46, against 0.26 for the idealised `π_h` and 0.16 for the direct
   policy. With an exact controller the direct policy is already optimal, and planning on noisy
   values can only lose.
2. **The main failure is a noisy `R̂`, not wormholes.** `R̂` is a difference of two noisy values,
   and for goal reaching the true `R` is about 0. The max over landmarks picks the largest noise
   at every hop. With the exact controller, `R̂ = 0` lifts the chained planner from 0.28 to 1.00.
3. **Guards** (far pairs, noise 0.04, `docs/figures/theory/study_guards.csv`): the min over the 2
   heads helps against independent errors in `R̂` (full `R̂`, exact `π_l`: 0.28 → 0.58).
   Commitment helps in every noisy case (`R̂ = 0`, greedy: 0.37 → 0.50). A hop discount (a factor
   on `ρ̂` per hop) hurts with `R̂ = 0` (0.37 → 0.11–0.13): it pushes the plan to fewer, longer hops
   that the controller cannot complete.
4. **Wormholes come from wall-blind features, not from independent noise.** Independent noise up
   to 0.04 puts no pair above `2γ^d`. A low-rank `B` with a wall-blind (cosine) basis does: 0.41 %
   of pairs at rank 16 and 0.24 % at rank 32 (`docs/figures/theory/study_wormholes.csv`). The two
   heads share such errors, so the head min does not remove them.

**Prediction for AntMaze.** A chained planner can beat `π_h` only if (a) `π_l` or the one-switch
value fails at long range, and (b) FB `ρ̂` is accurate at short range and free of wall-blind
errors. §8 measures both.

## 4. Approach A: landmark graph planner

No training. Code: `fbplan/landmarks.py`, `fbplan/option_model.py`, `fbplan/graph.py`,
`fbplan/controllers/planner.py`.

```
once per env:   training split ─► 300 landmarks ─► edge table ρ̂(u,w) ─► trusted edges
once per task:  z_r ─► V_d(u) = F(u,z_r)ᵀz_r ─► value iteration ─► V_H(u), route from each landmark
during the episode:
                s ─► w* = argmax over trusted edges s→w of ρ̂(s,w)·V_H(w), or the task policy
                commit to w* until reached ─► a = π_l(a | s, B(w*))
```

**Landmarks.** 300 states by farthest-point sampling on xy, from a random pool of 100k training
states. A landmark is a full observation, with the pose and the joint velocities of the ant.
Its latent is `B(w)` of that one state. Option: a region latent, the mean of `B` over the pool
states within 0.5 xy units.

**Edges.** For each landmark pair, `ρ̂(u,w)` per F head, then the min over the 2 heads. Values use
`ρ̂` clipped to `[0, γ]` for `u ≠ w` (contraction, §3). An edge `u → w` is **trusted** when:

- `0.3 ≤ raw ρ̂(u,w) < 1` (at γ = 0.99, `ρ = 0.3` is about 120 steps), and
- the xy distance from `u` to `w` is at most 6 units (1.5 maze cells).

**Values.** Goal reaching uses `R̂ = 0`. Value iteration then solves
`V_H(u) = max(V_d(u), max_w ρ̂(u,w)·V_H(w))` over trusted edges: a shortest path on `−log ρ̂` that
ends with the task policy.

**Execution.** At a re-plan the agent scores the trusted edges from its own state with the same
rules, `Q(s,w) = ρ̂(s,w)·V_H(w)`, and picks `w* = argmax_w Q(s,w)`, or the task policy if
`V_d(s)` is higher. It acts with `π_l(·∣s,B(w*))` and commits to `w*` until it is within 3 xy
units of `w*`, for at most 100 steps. Look-ahead: at each step it moves on to the next stop of the
route when `Q(s,next) ≥ Q(s,w*)` and the edge `s → next` is trusted. Variants: `planner_high`
gives `B(w*)` to `π_h` as its task; `replan_every_step` drops the commitment.

**Each guard and its evidence.**

| Guard | Reason | Evidence |
|---|---|---|
| `R̂ = 0` for goal reaching | `R̂` is a difference of two noisy values, and the max collects its noise | tabular §3.1 (0.28 → 1.00 with exact `π_l`); medium ablation *(pending)* |
| raw `ρ̂ ≥ 1` is untrusted | `ρ ≤ γ < 1` for `u ≠ w` (§3), so `ρ̂ ≥ 1` is an FB error and a free hop | theory; medium ablation *(pending)* |
| xy cap of 6 units | wormholes survive any `ρ_min` (§8.3). The maze distance is never shorter than the xy distance, so the cap removes only edges that are long in every metric, and it limits a wormhole hop to 6 units | at `ρ_min` 0.3 it removes 42 % of the wormholes on medium and 64 % on large, all the long ones (§8.3); medium ablation *(pending)* |
| min over the 2 F heads | pessimism against independent errors | tabular §3.1; lowers the AntMaze wormhole count by a fifth to a quarter (§8.3) |
| commit until within 3 xy units, with look-ahead | commitment stops dithering between near-equal noisy plans. A rule `ρ̂ ≥ ρ_reached` often does not fire, because FB `ρ̂` near a waypoint stays below about 0.9 | tabular §3.1 (0.37 → 0.50); medium ablation *(pending)* |
| single-state landmark latents | region latents make `ρ̂` higher and flatter, so it separates near and far landmarks less | 5 eval seeds × 3 landmark sets on medium: region −12.1 pp, single-state −8.6 pp against the baseline (§7.1) |

The xy position is part of the observation, so the cap uses no privileged information. Two FB-only
guards stay as ablations: `two_way`, which uses `min(ρ̂(u,w), ρ̂(w,u))` (maze dynamics are nearly
reversible, FB errors often are not), and a hop discount.

**Tuning.** The guards were chosen on medium, on start states outside the evaluation episodes but
on the same 5 tasks. Large uses the same settings without change.

## 5. Approach B: learned multi-switch value

Code: `fbplan/learned/`. Plan: `docs/plans/approach-b.md`.

**Idea.** The baseline's AWR advantage values a subgoal `w` with a one-switch value: "follow
`π_{z_w}` until `w`, then `π_z` forever". B learns `V_H(s,z)`, the value when the agent may switch
again at `w`, and again after that, and puts it into the same AWR recipe. The actor network, the
data, the spread of the AWR weights and the test-time controller stay as in the baseline. So
"B-clean against the retrained baseline" measures the effect of the multi-switch value and nothing
else. B needs no graph and no landmarks at test time, and it is defined for any task latent `z`.

```
offline data ─► (s, w, g, z) batches ─► ρ, R, V_1 from frozen F and B ─► targets y(s,w,z)
stage 1:  V_H(s,z)      ◄── expectile regression on y
stage 2:  π_h(z_w|s,z)  ◄── AWR with weight exp(min(α·[y − V_H(s,z)], 5))
test:     baseline controller: z_w = π_h(s, z_r) at every step, a = π_l(a | s, z_w)
```

**Critic.** Per sample, with the min over the 2 F heads:

```
ρ(s,w)   = min_h F_h(s,z_w)ᵀz_w / F_h(w,z_w)ᵀz_w,     capped at ρ_max = 0.9
V_1(s,z) = min_h F_h(s,z)ᵀz                            value of never switching again
V_H(s,z) = V_1(s,z) + softplus(Δ_θ(s,z))               so V_H ≥ V_1 by construction
y(s,w,z) = R(s,w;z) + ρ(s,w)·[V_1(w,z) + softplus(min_j Δ̄_j(w,z))]
```

`R = 0` for goal latents, as in A, and the exact `R` for Gaussian latents. `Δ_θ` is the upstream
`GCValue` network (3 × 512, LayerNorm, 2 heads), and `Δ̄` is its EMA target (rate 0.005). The loss
is the expectile loss with τ = 0.9 on `y − V_H(s,z)`: an in-sample soft max over the sampled `w`,
as in IQL.

**Sampling** (offline data only). `s` is uniform over the dataset. `w` is on the trajectory of `s`
at a geometric offset with mean 100 steps, as upstream. `g` is a future state of the same
trajectory (70 %) or a random dataset state (30 %). `z` is `B(g)` (50 %) or Gaussian (50 %), as
upstream. All latents are normalized to norm `√d`.

**Actor.** The upstream `GCActor`, trained with AWR toward `normalize(B(w))`, with the advantage
`A = y − V_H(s,z)` and `V_H` frozen. The AWR temperature α makes the spread of `α·A` equal to that
of the upstream run (upstream α = 0.1), so the weighting is as sharp as in the baseline.

**Guards against error growth over hops.**

| Guard | Effect |
|---|---|
| `ρ ≤ 0.9` | the hop-level Bellman operator is a 0.9-contraction: a target error ε grows to at most 10ε (a clip at γ would allow 100ε) |
| `w` only from the trajectory of `s` | the data shows every edge `(s,w)`, so no wormhole edge enters the targets |
| samples with raw `ρ̂ ≥ 1` are dropped | as in A: such edges are FB errors |
| min over the F heads, expectile instead of a hard max, clipped double target | pessimism against noise in `ρ̂` and in `V_H` |
| residual form `V_1 + softplus(Δ_θ)` | `V_H` starts at the one-switch value; the gain is learned only where switching helps in the data |

**Variants.** Each changes one thing against the row above it.

| Variant | Subgoal and goal latents | Goals for `π_h` | Measures |
|---|---|---|---|
| retrained baseline | `B(·)` of one state | upstream | reference |
| B-clean | as the baseline | as the baseline | the critic only |
| B-wide | as the baseline | + 30 % random dataset goals | goals across trajectories |
| B-region | mean of `B` over ±5 trajectory steps | as the baseline | the latent type (own `V_H`) |

**Limits by design.** Stitching across trajectories comes only from the generalization of
`V_H(w,z)`, as in IQL and HIQL: no training pair has a `w` from another trajectory. `π_h`
regresses onto `B(w)`, so if good subgoals lie in two directions, the mean lies between them.
A diagnostic controller, B-argmax (A's controller with `V_H` in place of graph value iteration),
would separate the quality of the critic from that of the actor.

**Gate.** B-clean had to gain over the retrained baselines before any further stage. It did not
(§7.1, §8.5), so B stops there: B-argmax, stage B2 (distilling A's routes into `π_h`,
`docs/plans/approach-b.md` §6) and a policy on the expectile 0.7 critic did not run. B-wide and
B-region were trained before the gate and are evaluated *(pending)*.

## 6. Evaluation protocol

**Episodes.** 5 tasks × 50 episodes × 5 eval seeds = 1,250 episodes per method and env. Episode
length 1000, deterministic actions (temperature 0), success as defined by OGBench at the last step.

**Paired start states.** The reset seed of an episode is
`SeedSequence([eval_seed, task_id, episode])`. OGBench draws the start noise from the global NumPy
RNG, and the goal also depends on the action-space RNG, so the loop seeds all three:
`np.random.seed(s); env.action_space.seed(s); env.reset(seed=s, options={"task_id": t})`.
All methods see the same 1,250 start and goal states. On one platform an episode replays exactly:
`analysis/failures.py` replays each failure and checks its outcome.

**Methods.**

| Row | What it is |
|---|---|
| baseline | the provided `π_h`, with `z_r` normalized to norm `√d` before `π_h`: the same input path as every other method |
| baseline_raw | the provided `π_h` through upstream `sample_actions`, with the raw `z_r`: a reference row |
| retrained baseline | 5 `π_h` seeds that we trained on the frozen `F`, `B`, `π_l` with the upstream recipe (500k steps), so the CI includes training variance |
| upstream | upstream `evaluate` on the provided checkpoint: raw `z_r` and unseeded resets; used only for the parity check, unpaired |
| non-hier | `π_l(·∣s,z_r)` alone, without a high level |
| oracle (privileged) | `π_l` toward a waypoint 2 cells ahead on the true shortest path, with a region-mean waypoint latent (radius 0.5); a reference, not a bound (§8.2) |
| A, B | §4 and §5; approach A over 3 landmark sets (`landmarks.seed` 0, 1, 2), each with all 1,250 episodes |

**Two baseline rows, and the rule for a gain.** `baseline` (normalized `z_r`) is the controlled
reference: it has the same input path as every other method, including B and the retrained
baselines. `baseline_raw` follows upstream. Neither row is the stronger one on every maze
(paired, `analysis/tables.py` on `results/<env>/`):

| Maze | `baseline_raw` − `baseline` | stronger row |
|---|---|---|
| medium | +3.8 [+0.7, +6.8], p = 0.02 | `baseline_raw` |
| large | −2.3 [−5.6, +1.0], p = 0.19 | not significant; `baseline` is higher |
| teleport | −3.5 [−6.2, −1.0], p = 0.009 | `baseline` |

Every table shows both rows. A method counts as better than the baseline only if its paired
difference is significant against both rows, so the stronger row is the bar.

**Statistics** (`fbplan/stats.py`, `analysis/tables.py`). Overall success is the mean over tasks of
the per-task success, as in OGBench. CIs are percentile intervals from 10,000 bootstrap resamples,
stratified by task. A paired comparison resamples episode pairs within each task and gives an exact
McNemar test on the discordant pairs. The landmark sets of approach A and the retrained baselines
are pooled with a cluster bootstrap over landmark sets or checkpoints and over episodes; their
paired differences are given pooled and per landmark set or checkpoint. A difference whose CI
contains 0 is called not significant.

**Parity with upstream.** Upstream `evaluate` on the provided checkpoint scores 77.9 %
[75.6, 80.2] on medium (`results/medium/upstream.csv`, 1,250 episodes with unseeded resets).
Against our loop (`analysis/tables.py`, unpaired): upstream − `baseline` = +0.9 pp [−2.3, +4.1],
and upstream − `baseline_raw` = −2.9 pp [−6.0, +0.2]. Both CIs contain 0: our loop reproduces the
upstream evaluation within noise.

**Physics.** MuJoCo is pinned to 3.1.6, the version that replays the dataset transitions exactly,
so `π_l` acts in the physics that produced its training data
(`tests/test_frozen.py::test_eval_env_reproduces_the_dataset_dynamics`).

**Platform check.** The same commands ran on macOS (arm64) and on the Linux (x86_64) machine
(`results/platform_check.csv`, made by `analysis/platform_check.py` from `results/` and
`results/macos/`; unpaired). 36 runs exist on both platforms. Linux − Mac lies between −4.6 and
+2.0 pp for the 24 runs of 1,250 episodes, and between −6.0 and +5.2 pp for the 12 oracle-sweep
runs of 250 episodes. The CI contains 0 for 35 of the 36; at 95 %, about 2 exclusions are expected
by chance. The one exclusion is planner set 2 on medium (−4.6 pp). All Mac planner runs used the
planner code before a tie-break fix (commit 06ea0a8), so the planner rows also differ in code. Float results differ between the platforms, so a single episode often
ends otherwise: only 55–75 % of the episodes have the same outcome on both. So the platforms agree
on success rates, not on single episodes, and every paired comparison stays within one platform.

## 7. Results *(B-wide and B-region pending)*

### 7.1 Medium (required)

Success (%) with 95 % CI, 1,250 episodes per row (`results/medium/`; `analysis/tables.py
--label-root results/medium`, with `--cluster config_hash` for approach A and `--cluster
checkpoint_seed` for the retrained baselines and B). Bold: the stronger baseline row.

| Method | task 1 | task 2 | task 3 | task 4 | task 5 | all |
|---|---|---|---|---|---|---|
| baseline | 72.4 | 85.2 | 79.6 | 63.2 | 84.8 | 77.0 [74.7, 79.3] |
| baseline_raw | 80.4 | 86.4 | 81.2 | 66.0 | 90.0 | **80.8** [78.6, 82.9] |
| retrained baseline (5 seeds) | 79.0 | 81.6 | 77.0 | 65.9 | 87.8 | 78.3 [76.9, 79.7] |
| non-hier | 73.6 | 91.6 | 72.8 | 8.0 | 81.2 | 65.4 [63.4, 67.5] |
| oracle (privileged) | 80.4 | 84.4 | 85.2 | 70.4 | 86.0 | 81.3 [79.1, 83.4] |
| A: planner | 64.7 | 75.7 | 74.9 | 52.5 | 74.1 | 68.4 [66.2, 70.5] |
| A: planner_high | 62.9 | 75.1 | 79.3 | 65.3 | 79.9 | 72.5 [70.7, 74.3] |
| A: planner, region latents | 49.7 | 72.3 | 78.7 | 65.2 | 58.7 | 64.9 [59.0, 71.9] |
| B-clean (5 seeds) | 81.7 | 85.6 | 76.0 | 66.7 | 86.7 | 79.3 [76.4, 82.2] |
| B-wide, B-region | | | | | | *(pending)* |

Per-task CIs are about ±5 pp for single runs and up to 25 pp wide for the pooled approach A
rows. The retrained and B-clean rows have only 5 clusters each, so their CIs are rough ones.

![Success on medium per task](docs/figures/report/success_antmaze-medium-navigate-v0.png)

Paired differences (pp) with CI and McNemar p, on the same 1,250 start states (`analysis/tables.py`
with `--reference baseline baseline_raw`). "Pooled" bootstraps over landmark sets or checkpoints
and over episodes:

| Method | against baseline | against baseline_raw |
|---|---|---|
| baseline_raw | +3.8 [+0.7, +6.8], p = 0.02 | – |
| retrained, pooled | +1.2 [−1.5, +4.1] | −2.5 [−5.2, +0.2] |
| retrained, per seed | −0.4 to +2.6, all p ≥ 0.12 | −4.2 to −1.1; seeds 2 and 3 significant (p = 0.04, 0.01) |
| non-hier | −11.6 [−14.7, −8.5], p = 3e-11 | −15.4 [−18.3, −12.4], p = 7e-20 |
| oracle (privileged) | +4.2 [+1.1, +7.5], p = 0.01 | +0.5 [−2.6, +3.5], p = 0.8 |
| planner, pooled | −8.6 [−12.0, −5.3] | −12.4 [−15.8, −9.2] |
| planner, sets 0 / 1 / 2 | −8.2 / −7.1 / −10.6, all p ≤ 4e-5 | −12.0 / −10.9 / −14.3, all p ≤ 2e-10 |
| planner_high, pooled | −4.5 [−7.7, −1.4] | −8.3 [−11.3, −5.2] |
| planner_high, sets 0 / 1 / 2 | −3.2 (p = 0.06) / −5.6 / −4.8 (p ≤ 0.006) | −7.0 / −9.4 / −8.6, all p ≤ 5e-5 |
| region latents, pooled | −12.1 [−18.8, −5.1] | −15.9 [−22.5, −8.9] |
| region latents, sets 0 / 1 / 2 | −18.3 / −4.9 / −13.2, all p ≤ 0.006 | −22.1 / −8.6 / −17.0, all p ≤ 4e-7 |
| B-clean, pooled | +2.3 [−1.5, +6.1] | −1.5 [−5.1, +2.1] |
| B-clean, seeds 0 / 1 / 2 / 3 / 4 | +4.9 / −1.5 / +5.9 / −1.2 / +3.4; seeds 0, 2, 4 significant (p ≤ 0.04) | +1.1 / −5.3 / +2.2 / −5.0 / −0.3; seeds 1 and 3 significantly below (p ≤ 0.003) |
| B-wide, B-region | *(pending)* | *(pending)* |

**The bar on medium is `baseline_raw`** (+3.8 pp over the normalized baseline, §6).

**Even the privileged oracle does not pass it:** +0.5 pp [−2.6, +3.5], p = 0.8. §8.2 discusses
what this means.

**The provided `π_h` is a typical seed of the upstream recipe.** The 5 retrained `π_h` score
76.6–79.7 % (`analysis/tables.py` on `results/medium/baseline_retrained/`), and none differs
significantly from the provided one (pooled +1.2 pp [−1.5, +4.1]). Two of the 5 are significantly
below `baseline_raw`. The choice between the provided and the retrained baseline does not change
any conclusion about approach A.

**Approach A is below both baselines on medium.** Every variant on every landmark set is
significantly below `baseline_raw`. Against the normalized baseline, only `planner_high` on set 0
is within noise (−3.2 pp, p = 0.06). §8.4 explains why.

**Approach B-clean shows no gain.** Its controlled reference is the retrained baselines: the same
recipe, input path and number of seeds, with only the critic changed (§5). Against them, B-clean
gains +1.1 pp [−2.4, +4.5], not significant (`analysis/tables.py --reference baseline_retrained
--cluster checkpoint_seed` on `results/medium/baseline_retrained/` and `results/medium/b_clean/`).
Against the bar, `baseline_raw`, it scores −1.5 pp [−5.1, +2.1]. On task 4, where a high level
matters most (§8.2), B-clean scores 66.7 % against 65.9 % for the retrained baselines.

**The B training seeds split into two groups.** Seeds 0, 2 and 4 score 80.5–83.0 %, seeds 1 and 3
score 75.5–75.8 %. Each seed of the first group beats each seed of the second by +4.6 to +7.4 pp
(paired, all p ≤ 0.005; `analysis/tables.py --reference b_clean/seed1 b_clean/seed3`). The
retrained baselines spread less (76.6–79.7 %): the largest gap between two of them, +3.0 pp
(seed 1 against seed 3), is not significant (p = 0.07; `--reference baseline_retrained/seed3`).
§8.5 discusses why B gives no gain.

### 7.2 Large

Success (%) with 95 % CI, 1,250 episodes per row (`results/large/`, same commands). Bold: the
higher baseline row.

| Method | task 1 | task 2 | task 3 | task 4 | task 5 | all |
|---|---|---|---|---|---|---|
| baseline | 24.0 | 35.2 | 78.8 | 22.8 | 26.4 | **37.4** [35.0, 39.9] |
| baseline_raw | 26.4 | 31.2 | 79.6 | 18.0 | 20.4 | 35.1 [32.8, 37.4] |
| oracle (privileged) | 30.8 | 43.2 | 54.8 | 29.6 | 38.4 | 39.4 [36.7, 42.0] |
| A: planner | 25.2 | 33.7 | 46.9 | 13.7 | 21.1 | 28.1 [24.7, 32.1] |
| A: planner_high | 25.2 | 34.0 | 54.0 | 12.9 | 21.2 | 29.5 [27.4, 31.4] |
| A: planner, region latents | 13.3 | 32.4 | 40.5 | 20.4 | 25.2 | 26.4 [24.7, 28.2] |
| retrained baseline (5 seeds) | | | | | | *(not evaluated yet)* |

![Success on large per task](docs/figures/report/success_antmaze-large-navigate-v0.png)

| Method | against baseline | against baseline_raw |
|---|---|---|
| baseline_raw | −2.3 [−5.6, +1.0], p = 0.19 | – |
| oracle (privileged) | +1.9 [−1.8, +5.6], p = 0.34 | +4.2 [+0.7, +7.7], p = 0.025 |
| planner, pooled | −9.3 [−13.8, −4.5] | −7.0 [−11.4, −2.2] |
| planner, sets 0 / 1 / 2 | −5.2 / −12.6 / −10.1, all p ≤ 0.005 | −2.9 (p = 0.12) / −10.3 / −7.8 (p ≤ 8e-6) |
| planner_high, pooled | −8.0 [−11.2, −4.6] | −5.7 [−9.0, −2.4] |
| planner_high, sets 0 / 1 / 2 | −6.6 / −9.6 / −7.8, all p ≤ 3e-4 | −4.2 / −7.3 / −5.4, all p ≤ 0.02 |
| region latents, pooled | −11.1 [−14.2, −7.8] | −8.7 [−11.8, −5.7] |
| region latents, sets 0 / 1 / 2 | −9.7 / −11.7 / −11.8, all p ≤ 8e-8 | −7.4 / −9.4 / −9.5, all p ≤ 4e-5 |

- **The bar on large is the normalized baseline.** The two rows do not differ significantly, and
  `baseline` is higher (+2.3 pp, p = 0.19).
- **The oracle does not pass it:** +1.9 pp [−1.8, +5.6]. The oracle gains 6.8–12.0 pp on tasks 1,
  2, 4 and 5, but loses 24.0 pp on task 3, where `π_h` succeeds in 79 % of the episodes.
- **Approach A is below both baselines.** Only the planner on landmark set 0 is within noise of
  `baseline_raw`. On task 3 every waypoint method loses 24–38 pp against the baseline, the oracle
  included, so following waypoints itself costs on that task (not analysed further).
- **Failures on large are mostly slow episodes, not flips** (`results/large/failures/`): of the
  baseline's 782 failures, 217 are flips (28 %) and 563 are still moving at the time limit
  (72 %); of the oracle's 758, 224 (30 %) and 534 (70 %). On medium most failures are flips
  (§8.2). On large the ant runs out of time, so the pace of `π_l` binds, as on giant (§7.3).

### 7.3 Giant and teleport

Two properties of these mazes limit any high-level method before it runs. They come from exact
cell-level computations and from the FB calibration (`docs/theory.md` §11).

**Teleport: a discounted value prefers a gamble.** An agent within 1.5 xy units of an entry jumps
to one of 3 exits, chosen uniformly, and one exit is a closed room (a trap). So a policy that jumps
at least once succeeds with probability at most 2/3, and a policy that jumps again after each bad
exit succeeds at most 1/2 of the time. Every goal can be reached on foot (`docs/theory.md` §11.1,
Proposition 4). FB values, `π_l`, `π_h` and both approaches maximise `E[γ^H]`, but success is
`P(H ≤ 1000)`. On the cell-level teleport maze, with 40 ant steps per cell move
(`docs/figures/theory/time_limit.csv`, `time_limit_thresholds.csv`):

| Task | walk (cells) | success: walk | success: risk-neutral policy, γ = 0.99 | γ at which the risk-neutral policy walks |
|---|---|---|---|---|
| 1 | 23 | 1.00 | 0.48 | 0.99856 |
| 2 | 15 | 1.00 | 0.50 | 0.99770 |
| 3 | 14 | 1.00 | 0.50 | 0.99850 |
| 4 | 25 | 1.00 | 0.50 | 0.99919 |
| 5 | 27 | 0.00 | 0.48 | 0.99933 |

![Success within the time limit against the discounted value](docs/figures/theory/time_limit.png)

The risk-neutral policy takes the teleporter on all 5 tasks at every pace from 20 to 60 ant steps
per cell. So an agent that maximises `E[γ^H]` at γ = 0.99 exactly scores about 50 % on teleport
at best, whatever its high level. `E[γ^H]` at one γ cannot tell "slow" from "sometimes never".
Approach A's xy cap removes the jump edges, because the exits are far from the entries in xy, so A
plans walks: the cap works against this risk by accident. But `π_l` can still walk into an entry on
its own (`docs/theory.md` §11.3).

**Giant: the time budget and `ρ` at long range.** The shortest walks of the 5 tasks are 30, 26,
30, 26 and 17 cells. In 1000 steps they fit only at 33, 38, 33, 38 and 58 ant steps per cell or
fewer. On the cell-level giant maze, no policy completes 4 of the 5 tasks at 40–55 steps per cell
(`docs/figures/theory/time_limit.csv`). So the pace of `π_l` bounds giant success, not the route.
The FB calibration on giant (γ = 0.995) shows the regime where chaining could help: the median
FB distance grows with the maze distance only up to about 45 xy units and then sits at the cap.
At `ρ_min` 0.3, 4,586 of 43,111 trusted pairs (10.6 %) are wormholes, the longest across 69.2 xy
units of maze distance; the xy cap leaves 801
(`docs/figures/theory/wormholes_antmaze-giant-navigate-v0_min.csv`).

![FB calibration on giant](docs/figures/theory/calibration_antmaze-giant-navigate-v0_min.png)

**What these limits imply.** On teleport, a method passes about 50 % only if it does not maximise
`E[γ^H]` exactly. On giant, a high level can supply a direction where the direct value has none,
but it cannot fix the time budget.

**Measured on teleport** (`results/teleport/`, 1,250 episodes per row; non-hier and a
teleport-aware oracle are pending). Bold: the stronger baseline row.

| Method | task 1 | task 2 | task 3 | task 4 | task 5 | all |
|---|---|---|---|---|---|---|
| baseline | 22.8 | 43.6 | 29.2 | 24.4 | 18.4 | **27.7** [25.3, 30.2] |
| baseline_raw | 18.8 | 39.6 | 26.4 | 21.2 | 14.8 | 24.2 [21.8, 26.6] |
| A: planner | 17.5 | 24.8 | 20.3 | 15.6 | 14.7 | 18.6 [16.5, 21.0] |
| A: planner_high | 17.1 | 33.6 | 28.1 | 21.7 | 19.9 | 24.1 [21.5, 26.4] |

| Method | against baseline | against baseline_raw |
|---|---|---|
| baseline_raw | −3.5 [−6.2, −1.0], p = 0.009 | – |
| planner, pooled | −9.1 [−12.2, −6.0] | −5.6 [−8.6, −2.5] |
| planner_high, pooled | −3.6 [−6.7, −0.5] | −0.1 [−3.3, +3.1] |

![Success on teleport per task](docs/figures/report/success_antmaze-teleport-navigate-v0.png)

All methods stay far below the about 50 % that an exact maximiser of `E[γ^H]` would reach.
Approach A is below the stronger baseline although its xy cap removes the jump edges, so its plans
are walks. `planner_high` does not differ from `baseline_raw` (−0.1 pp). *(pending: the share of failures that end in the trap, from
the failure classes.)*

**Giant** (no committed CSV yet). A preliminary Mac run gives the baseline 0.6 % and the privileged
oracle 2.2 %; in 1000 steps the oracle covers a median of only 5–15.5 of the 17–30 cells of a task
(`docs/theory.md` §11.3). So the pace of `π_l`, not the route, limits giant, as the time budget
above predicts.

## 8. Analysis

### 8.1 The gap to the paper's numbers

The paper reports 87 ± 6 % on medium and 66 ± 9 % on large (5 training seeds). With the provided
checkpoint we measure 77.0 % [74.7, 79.3] (`baseline`) and 80.8 % [78.6, 82.9] (`baseline_raw`) on
medium, and 37.4 % [35.0, 39.9] and 35.1 % [32.8, 37.4] on large (`results/{medium,large}/`).
Upstream `evaluate` gives 77.9 % [75.6, 80.2] on medium
(`results/medium/upstream.csv`).

Ruled out:

- **The restore.** The restored parameters equal the checkpoint arrays bit for bit
  (`tests/test_frozen.py::test_restored_parameters_equal_the_checkpoint_arrays`).
- **Our evaluation loop.** It matches upstream `evaluate` within noise (§6, parity).
- **The task latent, for most of the gap.** With the raw `z_r` of upstream, the provided `π_h`
  scores 80.8 % on medium, still 6 pp below the paper, and on large the raw `z_r` is not better.
- **The physics.** The evaluation env replays the dataset transitions exactly (§6).
- **The training seed of `π_h`.** 5 `π_h` that we retrained with the upstream recipe on the same
  frozen modules score 76.6–79.7 % on medium, and none differs significantly from the provided
  `π_h` (§7.1).

Still possible: (1) `F`, `B` and `π_l` of training seed 0 are weaker than the average of the
paper's 5 seeds (the paper's ± 6 and ± 9 show a wide spread between seeds); we cannot test this
without retraining FB, which the task forbids; (2) the paper's evaluation differs from upstream
`main.py`. The gap is larger on large (about 30 pp) than on medium. All our comparisons are paired
on the same frozen modules, so the gap does not bias them.

### 8.2 No measurable room for a high level

**The oracle is the test.** The privileged oracle routes on the true maze map: `π_l` toward a
waypoint 2 cells ahead on the shortest path, with a region-mean waypoint latent (radius 0.5). If
some choice of intentions could beat the baseline with this `π_l`, a choice with the true route
should come close (`results/{medium,large}/oracle.csv`, `analysis/tables.py`):

| Maze | oracle | against `baseline` | against `baseline_raw` | gain by the rule of §6 |
|---|---|---|---|---|
| medium | 81.3 % | +4.2 [+1.1, +7.5], p = 0.01 | +0.5 [−2.6, +3.5], p = 0.8 | none |
| large | 39.4 % | +1.9 [−1.8, +5.6], p = 0.34 | +4.2 [+0.7, +7.7], p = 0.025 | none |

**With this frozen `π_l`, no high-level choice of intentions has measurable room on these mazes.**
Against the stronger baseline, the oracle gains +0.5 pp on medium and +1.9 pp on large, both not
significant. On giant the oracle reaches about 2 % (preliminary, §7.3). On teleport the time limit
and the trap cap an exact maximiser of `E[γ^H]` at about 50 % (§7.3).

This does not say that chaining cannot help in principle. The tabular study shows that it helps
when `π_l` fails at long range and FB is accurate at short range (§3.1). It says that on these
mazes the frozen `π_l`, not the choice of intentions, limits success: on medium most failures are
flips (below), and on large and giant most failures are episodes still moving at the time limit
(§7.2, §7.3).

The oracle is a reference, not a bound. Another waypoint rule changes its success by a large amount
(next paragraph), so a better rule could find room that this one misses. In the sweep below, the
best cell (82.4 %) is within noise of the chosen one (80.4 %) on the same 250 episodes.

**The waypoint latent matters more than the route.** Oracle variants on the 250 episodes of eval
seed 0 (`results/medium/oracle_sweep/*.csv`, `analysis/tables.py`; on these episodes the baseline
scores 72.8 % and `baseline_raw` 79.2 %, `results/medium/baseline*.csv`; CIs about ±5 pp):

| Waypoint latent | 1 cell ahead | 2 cells ahead | 3 cells ahead |
|---|---|---|---|
| `B(w)` of one state | 50.0 | 57.2 | 63.2 |
| mean of `B` within 0.5 xy units | 78.4 | 80.4 | 82.4 |
| mean of `B` within 1.0 xy units | 81.6 | 76.8 | 79.6 |
| mean of `B` within 1.5 xy units | 76.8 | 80.8 | 77.2 |

With the true shortest path, single-state waypoint latents lose 10–23 pp against the baseline;
region-mean latents score 4–10 pp above it, and −2.4 to +3.2 pp against `baseline_raw`. No region setting
stands out from the others. The Mac sweep (`results/macos/medium/oracle_sweep/`) shows the same
pattern. *Hypothesis:* `B(w)` of one state encodes the
full observation of `w`, with the joint angles and the velocities, so `π_l(·∣s,B(w))` tries to
reach that exact pose, which is slower and less stable than passing near `w`. A region mean
averages the pose out and keeps mostly the position. The test-time `z_r` is also a mean over the
goal ball. This matters for approach A, whose default landmark latent is the single state (§4).

**A high level matters on medium mainly for task 4.** Without a high level, `π_l(z_r)` scores
−11.6 pp [−14.7, −8.5] against the baseline (p = 3e-11). Almost all of the loss is on task 4:
8.0 % against 63.2 %. On the other 4 tasks the direct policy is within 7 pp of `π_h`, and above it
on tasks 1 and 2. So the one-switch `π_h` already solves most of what a high level must solve on
medium.

**Failure classes on medium.** `analysis/failures.py` replays each failed episode and classifies
it from the observations: *flipped* (the torso points down for the last 50 steps; the ant cannot
turn back), *stuck* (it stays within 1 xy unit for the last 300 steps), or *other* (still moving at
the time limit). Counts out of 1,250 episodes (`results/medium/failures/<method>.csv`;
descriptive, no CIs):

| Method | failures | flipped | stuck | other | flipped, share of failures |
|---|---|---|---|---|---|
| baseline | 287 | 202 | 6 | 79 | 70 % |
| baseline_raw | 240 | 179 | 1 | 60 | 75 % |
| oracle (privileged) | 234 | 176 | 3 | 55 | 75 % |
| non-hier | 432 | 155 | 52 | 225 | 36 % |

- **Most failures with a high level are flips:** 70–75 % for the baseline, `baseline_raw` and the
  oracle. Once the ant has flipped, no high-level choice can repair the episode.
- **The true route does not lower the flips.** The oracle flips in 176 episodes, `baseline_raw` in
  179 and the baseline in 202. The oracle and `baseline_raw` fail in almost the same way (234 and
  240 failures). The Mac runs showed fewer oracle flips (142 against the baseline's 198,
  `results/macos/medium/failures/`); this difference does not replicate on Linux.
- **Without a high level, task 4 fails by getting lost, not by flipping.** Of the 230 non-hier
  failures on task 4, 135 are *other* and 50 *stuck*; only 45 are flips. This is the long-range
  failure of `π_l(z_r)` that a high level must repair, and `π_h` repairs most of it (§7.1).

### 8.3 FB distance calibration and wormholes

For 400 validation states (159,600 ordered pairs), the FB distance `−log ρ̂ / −log γ` against the
exact maze distance; the min over the 2 heads (`analysis/calibration.py`):

| Medium | Large |
|---|---|
| ![FB calibration on medium](results/figures/calibration_antmaze-medium-navigate-v0_min.png) | ![FB calibration on large](results/figures/calibration_antmaze-large-navigate-v0_min.png) |

- **`ρ̂` carries distance information at short and medium range.** On medium the median FB
  distance grows with the maze distance over the whole maze.
- **The spread is wide at every distance.** A band of pairs sits at the cap (`ρ̂` near 0) at all
  distances, short ones included: some short edges are missing from the graph.
- **On large, FB loses distance information at long range.** The median stops growing beyond
  about 30 xy units, and the upper quartile reaches the cap from about 20 units. This is the regime
  in which chaining short edges can help, if the short edges are reliable.

**Wormholes.** A wormhole is a trusted pair (`ρ_min ≤ raw ρ̂ < 1`) whose FB distance is shorter
than the time the ant needs to walk the maze distance at its top speed (99th percentile of the
step length in the data). This is not the tabular definition of §3.1 (Appendix). Counts for the
400 states, min over heads (`results/calibration/wormholes_<env>_min.csv`):

| Env | `ρ_min` | raw `ρ̂` ≥ 1 (untrusted) | trusted pairs (raw `ρ̂` in [`ρ_min`, 1)) | wormholes | share of trusted | within the xy cap of 6 |
|---|---|---|---|---|---|---|
| medium | 0.3 | 1,118 | 61,082 | 3,405 | 5.6 % | 1,965 |
| medium | 0.5 | 1,118 | 27,844 | 3,362 | 12.1 % | 1,965 |
| medium | 0.7 | 1,118 | 11,028 | 2,997 | 27.2 % | 1,965 |
| large | 0.3 | 1,170 | 37,189 | 3,992 | 10.7 % | 1,420 |
| large | 0.5 | 1,170 | 16,686 | 3,411 | 20.4 % | 1,420 |
| large | 0.7 | 1,170 | 7,173 | 2,707 | 37.7 % | 1,420 |

| Medium, 200 longest wormholes | Large, 200 longest wormholes |
|---|---|
| ![Wormholes on medium](results/figures/wormholes_antmaze-medium-navigate-v0_min.png) | ![Wormholes on large](results/figures/wormholes_antmaze-large-navigate-v0_min.png) |

- **Many wormholes cross the whole maze** (longest: 28.1 xy units of maze distance on medium,
  48.9 on large). A max over routes exploits exactly these pairs.
- **A higher `ρ_min` does not remove them.** From `ρ_min` 0.3 to 0.7 the trusted pairs drop by
  82 % on medium, but the wormholes drop by only 12 % (large: 81 % and 32 %). Their share of the
  trusted pairs grows from 5.6 % to 27.2 % (large: 10.7 % to 37.7 %). The wormholes sit at high
  `ρ̂`. This is why A uses an xy cap instead of a higher `ρ_min` (§4).
- **The xy cap removes the long wormholes, not all of them.** It leaves 1,965 of 3,405 on medium
  and 1,420 of 3,992 on large, all at most 6 xy units long, so a wormhole hop can no longer jump
  across the maze. *(pending: whether the remaining ones cross a wall.)*
- **The head min removes few.** With the mean over heads there are 4,319 (medium) and 5,380
  (large) wormholes at `ρ_min` 0.3 (`results/calibration/wormholes_<env>_mean.csv`). The min
  lowers the count by only a fifth to a quarter, so most of these errors are shared by the heads,
  as with the wall-blind low-rank features of the tabular study (§3.1, result 4), and unlike
  independent noise.
- *Hypothesis:* in the maps, many wormholes meet at a few target states. A target-side error would
  explain this. `ρ̂(s,w)` divides by `F(w,z_w)ᵀB(w)`, so an underestimated denominator inflates
  `ρ̂` from every source at once. If this holds, a guard on the target (distrust landmarks with a
  low `F(w,z_w)ᵀB(w)`) would remove many wormholes with one test per landmark. Not tested yet.

### 8.4 Why approach A is below the baseline on medium

Evidence from the evaluation CSVs: per-task success (§7.1), time to goal and where failures end
(`analysis/report_figures.py` on `results/medium/`), and failure classes (`analysis/failures.py`).
Time-to-goal numbers and failure counts are descriptive, without CIs.

**1. No room on medium.** The one-switch `π_h` already solves the long task 4 (§8.2), and perfect
routing gains only +0.5 pp against the stronger baseline. The tabular study shows the same pattern:
when the existing policy is already near the best that the controller allows, planning on noisy
values can only lose (§3.1, result 1).

**2. A does not get the routing gain where the oracle gets it, and loses elsewhere.** Against the
normalized baseline, the oracle gains most on tasks 1, 3 and 4 (+8.0, +5.6 and +7.2 pp). On tasks
3 and 4, `planner_high` (79.3 %, 65.3 %) and the region variant (78.7 %, 65.2 %) are only level with
the baseline (79.6 %, 63.2 %); the plain planner is lower (74.9 %, 52.5 %). On task 1 every variant
loses 8–23 pp. On tasks 2 and 5, where the oracle gains −0.8 and +1.2 pp, every variant loses
5–26 pp. So the planner misses the routing gain and pays a cost on tasks that `π_h` already solves.

**3. A reaches the goal later.** Successful episodes take a median of 442 (planner), 420
(`planner_high`) and 461 (region) steps, against 363 for the baseline, 368 for `baseline_raw` and
384 for the oracle, which also follows waypoints:

![Success within a step budget minus the baseline](docs/figures/report/time_to_goal_antmaze-medium-navigate-v0.png)

| Method | success within 400 steps | 600 | 800 | 1000 |
|---|---|---|---|---|
| baseline | 43.3 | 62.5 | 72.9 | 77.0 |
| planner | 29.8 | 49.6 | 61.2 | 68.4 |
| planner_high | 33.8 | 53.1 | 64.7 | 72.5 |
| planner, region latents | 26.1 | 44.9 | 56.1 | 64.9 |

The gap to the baseline is largest at 370–580 steps (−10.6 to −18.0 pp) and shrinks by 33–58 % by
step 1000 (to −4.5 to −12.1 pp). So a large part of the final gap is time: the planner reaches many
goals later, and the 1000-step limit ends some episodes before it arrives. The curves still rise at
1000 steps, so these data cannot say how much of the gap a longer episode would close.

**4. Failures end far from the goal, as the baseline's do.** The median final distance of a failed
episode is 10.2–11.4 xy units (baseline 8.6), and 7.6–9.7 % of A's failures end within 3 units of
the goal (baseline 11.5 %). So circling next to the goal is not a main failure mode on medium.

**5. A's extra failures are slow episodes, not flips.** Failure classes per landmark set, out of
1,250 episodes (`results/medium/failures/planner*_l<k>.csv`):

| Method | failures | flipped | stuck | other (moving at the time limit) |
|---|---|---|---|---|
| baseline | 287 | 202 | 6 | 79 |
| planner, sets 0 / 1 / 2 | 390 / 376 / 419 | 210 / 211 / 209 | 7 / 6 / 2 | 173 / 159 / 208 |
| `planner_high`, sets 0 / 1 / 2 | 327 / 357 / 347 | 173 / 185 / 171 | 4 / 3 / 7 | 150 / 169 / 169 |
| planner, region latents, sets 0 / 1 / 2 | 516 / 348 / 452 | 178 / 157 / 182 | 7 / 7 / 4 | 331 / 184 / 266 |

A flips about as often as the baseline (157–211 episodes against 202), but 2 to 4 times as many of
its episodes are still moving at the time limit (150–331 against 79). On tasks 1 and 5 these
failures are 35–118 (baseline 26) and 21–95 (baseline 17). So A loses time, not balance, which
agrees with point 3.

**6. The plan depends on the landmark set.** The paired difference moves by 3.5 pp (planner), 2.4 pp
(`planner_high`) and 13.4 pp (region) between the 3 landmark sets (§7.1). A graph of 300 sampled
states is itself a random object, and region latents make it more sensitive. On sets 0 and 2 the
region variant scores 40–54 % on tasks 1 and 5 (set 1: 63–74 %; `analysis/tables.py` on
`results/medium/planner_region/`).

**7. Handing the waypoint to `π_h` is the best variant.** `planner_high` gives `B(w*)` to `π_h`
as its task, so `π_h` chooses the intentions toward the waypoint, and it is the closest variant
to the baseline (pooled −4.5 pp [−7.7, −1.4]). With points 3 and 5, this suggests that the
execution of the waypoints is a main cost. The share of the route is open. An executor that acts
on a region latent ahead along the planned route tests it: a Mac probe by c-planner (held-out
start states, not committed) closed part of the gap to the oracle, not significantly. *(pending:
the Linux run of this executor, and route plots.)*

**Region latents: good for acting, bad for planning (hypothesis).** Region-mean waypoint latents
help the oracle by 14–32 pp over single-state ones at the same look-ahead (§8.2), but
they make the planner worse and slower. The oracle takes its route from the true maze and uses the
latent only to act. The planner uses the same latent for `ρ̂`, the planning signal, and for the
action. A region latent makes `ρ̂` higher and flatter between near and far landmarks (§4; ablation
pending), so the route gets worse even if each hop is easier to execute. Test: plan with
single-state `ρ̂` and act with region latents. A 100-episode probe of this split by c-planner
(held-out start states, not committed) showed no gain over single-state latents, so the hypothesis
is weak. A possible reason: the oracle's waypoints are 2 cells (about 8 xy units) ahead, the
planner's at most 6 xy units, and the planner switches at 3.

### 8.5 Why approach B gives no gain

B-clean changes only the critic, and it does not gain over the retrained baselines (+1.1 pp
[−2.4, +4.5], §7.1). Three findings explain this negative result.

**1. The learned `V_H` lifts long-range values but does not order states better than `V_1`.** The
critic check compares `V_1(s,B(g))` and `V_H(s,B(g))` with the maze distance from `s` to `g`, for
300 × 300 validation pairs and each of the 5 `V_H` seeds, without rollouts
(`fbplan/learned/diagnostics.py`; output in `docs/figures/report/b_critic_check_medium.txt`):

![V_1 and V_H against the maze distance on medium, expectile 0.9, seed 0](docs/figures/report/b_critic_medium_expectile0.9_seed0.png)

| Critic | median value at 20–24 xy units | rank correlation with distance: all pairs | far half | per goal | goals that `V_H` orders better than `V_1` |
|---|---|---|---|---|---|
| `V_1` (never switch again) | 7.8 | −0.586 | −0.346 | −0.613 | – |
| `V_H`, expectile 0.9 (B-clean) | 16.4 | −0.543 to −0.554 | −0.339 to −0.355 | −0.558 to −0.573 | 12–17 % |
| `V_H`, expectile 0.7 | 10.8 | −0.572 to −0.578 | −0.344 to −0.354 | −0.590 to −0.603 | 16–24 % |

Ranges are over the 5 seeds; medians are from seed 0. A more negative correlation means that the
value orders states better by their distance to the goal.

- **The lift is where chaining should act.** At 0–4 xy units, `V_H` equals `V_1` (median 53.3
  against 53.2). At 20–24 units it doubles (16.4 against 7.8), and 91 % of these pairs gain more
  than 1.
- **The order does not improve.** Over all pairs and per goal, `V_H` correlates less with the
  distance than `V_1` does, in every seed. In the far half, where chaining should help most, the
  two are level. `V_H` orders states better for only 12–17 % of the goals.
- **A lower expectile does not repair it.** Expectile 0.7 shrinks the lift (10.8 at 20–24 units)
  and moves the order toward that of `V_1`. Over all pairs and per goal it still orders worse than
  `V_1` in every seed; in the far half it is level (−0.344 to −0.354 against −0.346). So no policy
  was trained on it (§5, gate).
- **Why this matters for the actor.** AWR weights a subgoal `w` by its target, which contains
  `ρ(s,w)·V_H(w,z)` (§5). If `V_H` orders the states `w` no better than `V_1`, the weights get no
  better signal about which subgoal leads to the goal than the one-switch proxy has, so `π_h`
  cannot become much better.
  *Hypothesis:* the expectile over noisy targets collects their noise (a soft max over `w`), so the
  gain grows with the distance, not with the quality of the route. The smaller lift at expectile
  0.7 agrees with this.

**2. The result depends on the training seed.** The 5 B-clean seeds form two groups, 80.5–83.0 %
and 75.5–75.8 %, which differ significantly (§7.1). The retrained baselines, which differ from
B-clean only in the critic, spread 76.6–79.7 % without a significant gap. One seed alone could
show a significant gain over the normalized baseline (seed 2: +5.9 pp, p = 2e-4) or a significant
loss against `baseline_raw` (seed 1: −5.3 pp, p = 9e-4). The pooled row, which resamples the
training seeds, is the result. The split does not follow the critic check: the rank correlations
differ little between the seeds (−0.543 to −0.554 over all pairs), and the weakest policy (seed 1)
has the critic with the best order. *Hypothesis:* each `V_H` seed adds its own noise to the AWR
targets, and the policy follows that noise.

**3. With this `π_l`, the bar leaves almost no room.** With the true route, the privileged oracle
gains +0.5 pp [−2.6, +3.5] against `baseline_raw` (§8.2). B must pass the same bar, and it starts
below it: on B's input path (normalized `z_r`), the retrained baselines score −2.5 pp
[−5.2, +0.2] against `baseline_raw`. A learned critic knows less about the maze than the true
route, so a gain past the bar was unlikely even with a critic that orders states well. The oracle
is a reference, not a bound (§8.2), so this is an argument, not a proof. B-clean's CI against
`baseline_raw`, [−5.1, +2.1], does not exclude a small gain, but it does not show one.

So B gives a critic that changes the values but not the choice of subgoals, on a maze where even
the true route has no measurable room. The seed split also shows that a single-seed result of B
would have misled in either direction.

## 9. What worked and what did not *(B-wide, B-region and the A ablations pending)*

| Item | Outcome | Evidence |
|---|---|---|
| Chain formula, graph Bellman equation, contraction, option interruption | hold exactly on tabular mazes | `tests/test_tabular_theory.py`, to 1e-8 |
| `R̂ = 0` and guards in the tabular study | chaining beats the idealised `π_h` when `π_l` fails at long range (0.46 against 0.26) | §3.1 |
| Room for any high level with this `π_l` | **none measurable**: the privileged oracle gains +0.5 pp (medium) and +1.9 pp (large) against the stronger baseline, both not significant | §8.2 |
| Paired evaluation | paired tests resolve differences of about 3 pp; parity with upstream holds; Mac and Linux agree on success rates | §6, §7.1 |
| Two baseline rows | neither is the stronger one on every maze (raw +3.8 pp on medium, normalized +3.5 pp on teleport) | §6 |
| Retrained baselines | the provided `π_h` is a typical seed (retrained 76.6–79.7 %, pooled +1.2 pp against it) | §7.1, §8.1 |
| Approach A | **did not work**: below the stronger baseline on medium (−8.3 to −15.9 pp), large (−8.0 to −11.1 pp) and teleport (−3.6 to −9.1 pp) | §7 |
| A's loss on medium | mostly time: later arrival, and slow episodes instead of flips | §8.4 |
| Region landmark latents | **did not work** for the planner (worst variant, most sensitive to the landmark set), although region waypoint latents help the oracle | §7.1, §8.2, §8.4 |
| `planner_high` (waypoint as the task of `π_h`) | the best A variant; equal to `baseline_raw` on teleport, below the stronger baseline elsewhere | §7 |
| Giant and teleport | teleport: an exact maximiser of `E[γ^H]` takes the teleporter and succeeds about half the time, and every method stays far below that; giant: the walks fit in 1000 steps only at a fast pace (oracle about 2 %, preliminary) | §7.3 |
| Each guard of A on the real model | *(pending: guard and execution ablations)* | |
| Approach B (B-clean) | **did not work**: +1.1 pp [−2.4, +4.5] over the retrained baselines, −1.5 pp [−5.1, +2.1] against `baseline_raw`; B stops at the gate | §7.1, §8.5 |
| B's critic `V_H` | lifts long-range values (median 16.4 against 7.8 at 20–24 xy units) but orders states no better than `V_1`, at expectile 0.9 or 0.7 | §8.5 |
| B's training seeds | split into two significantly different groups (80.5–83.0 % and 75.5–75.8 %); the retrained baselines spread less | §7.1, §8.5 |
| B-wide, B-region | *(pending)* | |

## 10. Limitations and next steps

Limitations:

- **One FB model per env.** `F`, `B` and `π_l` come from training seed 0. The retrained `π_h`
  seeds add `π_h` training variance (76.6–79.7 % on medium), not the variance of the frozen
  modules. The central finding holds for this `π_l` only.
- **The oracle is a reference, not a bound** (§8.2). A better waypoint rule could find room that
  this oracle misses.
- **Flips** are 70–75 % of the failures with a high level on medium. Even the true route does not
  lower their count below `baseline_raw`'s, and no high level can repair a flipped episode (§8.2).
- **Tuning on the evaluation tasks.** A's guards were tuned on the 5 medium tasks with other start
  states (§4). This favours A on medium, so it does not explain A's loss.
- **3 landmark sets** estimate the landmark variance of A only roughly (§8.4, point 6).
- **5 training seeds for B.** They split into two groups (§8.5), so 5 seeds estimate B's training
  variance only roughly. B was tested on medium only.
- **B's critic was varied only offline.** The expectile (0.9, 0.7) was compared with the critic
  check; the `ρ` cap (0.9) was not varied in a policy run.
- **Giant** has no CSV yet (§7.3).

Next steps that the evidence suggests:

- A better executor before a better planner: with this `π_l`, even the true route gains nothing
  measurable (§8.2). A waypoint-reaching `π_l`, faster and less prone to flips, would change the
  frozen setting, so it is a separate study.
- For a learned critic: check the order of states against the maze distance before any policy
  run, as the gate of §5 did for expectile 0.7, and report the training-seed spread of the policy.
  A critic that orders far states better than `V_1` is the first requirement (§8.5).
- A target-side wormhole guard (§8.3, hypothesis).
- On teleport, an objective closer to `P(H ≤ T)` than `E[γ^H]` at one γ: FB at several γ, or a
  penalty on edges whose successor measure has mass in the trap (§7.3, `docs/theory.md` §11.1).
- Separate the planning latent from the acting latent (§8.4). A first probe showed no gain, so this
  test has a low priority.

## 11. Reproduction

Commands for every table and figure: `README.md`, section "Reproduce". Figures that only the report
uses:

- time to goal and the tables of §8.4: `uv run python analysis/report_figures.py`;
- success per task (§7): `uv run python analysis/plots.py results/<env>/*.csv
  results/<env>/planner*/*.csv --label-root results/<env> --cluster config_hash --out-dir
  docs/figures/report`;
- the Mac-vs-Linux check (§6): `uv run python analysis/platform_check.py` (writes
  `results/platform_check.csv`);
- B-clean against the retrained baselines and between its seeds (§7.1): `uv run python
  analysis/tables.py results/medium/baseline_retrained/*.csv results/medium/b_clean/*.csv
  --label-root results/medium --reference baseline_retrained --cluster checkpoint_seed`, and the
  same with `--reference b_clean/seed1 b_clean/seed3` and no `--cluster`;
- the critic check of B (§8.5): the two `fbplan.learned.diagnostics` commands at the top of
  `docs/figures/report/b_critic_check_medium.txt`; it needs the trained `V_H` folders
  `approach_b/medium/value-single*/`.

## Appendix: notation and names in the code

Symbols and parameters whose names differ between the report, approach A, approach B and the
tabular code:

| Report | Meaning | Approach A | Approach B | Tabular code |
|---|---|---|---|---|
| raw `ρ̂(s,w)`; `ρ̂` clipped | FB estimate of `E[γ^H]`; values use the clipped one | `OptionTable.rho_raw`; `.rho`, clipped to `[0, γ]` | `OptionTerms.rho_raw`; `.rho`, capped at 0.9 | `rho_hat` |
| raw `ρ̂ < 1` (trust bound, A) | an edge with raw `ρ̂ ≥ 1` is untrusted | `PlannerConfig.rho_max = 1.0` | samples with raw `ρ̂ ≥ 1` are dropped (`OptionTerms.valid`) | – |
| `ρ ≤ 0.9` (value cap, B) | cap on `ρ` in the targets: a 0.9-contraction | – | `OptionConfig.rho_cap = 0.9` (CLI `--rho-cap`) | – |
| `ρ_min` | lower trust bound | `PlannerConfig.rho_min = 0.3` | – | `PlannerConfig.rho_min` (0.4 in the guarded planner) |
| exact `R` | `F(s,z_w)ᵀz_r − ρ·F(w,z_w)ᵀz_r` | `edge_reward="exact"` | `edge_reward="exact"`; also Gaussian latents under the default `"goal_zero"` | `reward="full"` ("R=full"); `high_advantage="full"` |
| the proxy | `R` without `−ρ·F(w,z_w)ᵀz_r`, as upstream `π_h` | `edge_reward="paper"` | `edge_reward="paper"` | `reward="proxy"`; `high_advantage="proxy"` |
| `R̂ = 0` | goal reaching: shortest path on `−log ρ̂` | `edge_reward="zero"` (default) | `edge_reward="goal_zero"` (default): 0 for goal latents | `reward="zero"` ("R=zero", "chained R=0") |
| `V_d(u) = F(u,z_r)ᵀz_r` | direct value: switch to the task policy now | `direct`, `OptionModel.direct_value` | `V_1(s,z)`, `direct_value`, for any latent `z` | `v_direct`, `v_d`; `v_final` is the value of the last policy of a chain |
| `V_H(u)`, §3–4 | graph value over landmarks, for one task `z_r` | `graph.value_iteration` → `GraphSolution.values` | – | `v_graph`, `graph_value_iteration` |
| `V_H(s,z)`, §5 | learned multi-switch value, any state and latent | – | `ValueState.value` = `V_1` + min over heads of `ValueState.gain` | – |
| `Δ_θ`, `Δ̄` | residual network of `V_H` and its EMA target | – | `delta_network`, raw output `delta`; `ValueState.gain` returns softplus(delta) per head | – |
| wormhole, §8.3 | trusted pair, raw `ρ̂ < 1`, FB distance shorter than the walk at top speed | `analysis/calibration.py:wormhole_mask` | – | – |
| wormhole, §3.1 | `ρ̂ > 2γ^d`, `d` = shortest path in steps (the true `ρ ≤ γ^d`) | – | – | `fbplan/tabular/study.py:wormhole_mask` |

The two wormhole definitions both flag a `ρ̂` that the maze does not allow, but their counts are
not comparable.

Methods: name in the report, `--method` of `fbplan.evaluate`, and the `method` column of the CSV
(the `Controller.name`):

| Report | `--method` and options | CSV `method` | Results (`<env>`: medium, large, teleport, where run; Mac copies below `results/macos/`) |
|---|---|---|---|
| baseline | `baseline` | `baseline` | `results/<env>/baseline.csv` |
| baseline_raw | `baseline_raw` | `baseline_raw` | `results/<env>/baseline_raw.csv` |
| non-hier | `nonhier` | `nonhier` | `results/<env>/nonhier.csv` |
| oracle | `oracle` | `oracle` | `results/<env>/oracle.csv` |
| planner | `planner --opt landmarks.seed=k` | `graph_planner` | `results/<env>/planner/l<k>.csv` |
| planner_high | `planner_high --opt landmarks.seed=k` | `graph_plus_high` | `results/<env>/planner_high/l<k>.csv` |
| planner, region latents | `planner --opt landmarks.latent=region landmarks.seed=k` | `graph_planner` (another `config_hash`) | `results/<env>/planner_region/l<k>.csv` |
| retrained baseline | `baseline --checkpoint retrained:<seed>` | `baseline`, `checkpoint_kind` = `retrained` | `results/<env>/baseline_retrained/seed<k>.csv` |
| B-clean, B-wide, B-region | `baseline --checkpoint b:<variant>:<seed>` | `baseline`, `checkpoint_kind` = `b`, `checkpoint_variant` = the variant | `results/medium/b_clean/seed<k>.csv`; B-wide and B-region *(pending)* |
