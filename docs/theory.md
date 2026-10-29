# Theory: chaining switches with successor measures

Status: report appendix, owned by b-theory. Notation follows `docs/design.md` §3 and the paper
(arXiv 2605.13207, §3–5, App. B.1). Every claim marked ✓ has a numeric check in
`tests/test_tabular_theory.py` (exact to 1e-8, on the deterministic medium maze and on the
stochastic teleport maze).

## 1. Setting

- MDP with discount γ < 1 and a state reward r, collected from t = 0:
  `V^π(s; r) = E[Σ_t γ^t r(s_t)]`.
- Successor measure with the t = 0 term: `M^π_s(s') = Σ_t γ^t P(s_t = s' | s_0 = s)`, so
  `V^π(s; r) = ⟨M^π_s, r⟩` and `M^π_w(w) ≥ 1`.
- Hitting time `H = min{t ≥ 0 : s_t = w}` (H = 0 if s = w, H = ∞ if w is never reached).
- An **option** `o_w` = "follow the Markov policy π_w until w is hit". In FB π-Switch,
  `π_w = π_l(·|s, z_w)` with `z_w = B(w)` fixed, which is Markov.

## 2. Option model

```
ρ(s,w) = E[γ^H]                       R(s,w) = E[Σ_{t<H} γ^t r(s_t)]
```

**Lemma 1** (Myers et al., Lemma B.1). `M^{π_w}_s(w) = ρ(s,w) · M^{π_w}_w(w)`. ✓
*Proof.* No visit to w happens before H. At H the state is w, and by the strong Markov property
the process after H is the chain started at w under π_w. So the visits from H on contribute
`E[γ^H] · M^{π_w}_w(w)`. □

**Lemma 2.** `R(s,w) = V^{π_w}(s; r) − ρ(s,w) · V^{π_w}(w; r)`. ✓
*Proof.* Split `V^{π_w}(s)` at H. The part after H is `E[γ^H] V^{π_w}(w)` by the same argument. □

With `M ≈ Fᵀ B` this gives design.md's formulas:
`ρ(s,w) = F(s,z_w)ᵀB(w) / F(w,z_w)ᵀB(w)` and `R(s,w) = F(s,z_w)ᵀz_r − ρ(s,w) F(w,z_w)ᵀz_r`.
Both are invariant to a positive scale of `B(w)` (F depends only on the direction of z).
A data-density factor at w cancels in the ratio.

**Bounds.** `ρ(w,w) = 1` and `R(w,w) = 0`. For s ≠ w, H ≥ 1, so `ρ(s,w) ≤ γ`; more precisely
`ρ(s,w) ≤ γ^{d(s,w)}` with d the least number of steps from s to w. ✓
In a deterministic maze with a shortest-path π_w, `ρ = γ^d` exactly.

## 3. One switch (Theorem 1 of the paper)

For any policy π, possibly history-dependent, that starts fresh at the hitting time:

```
V(s; w → π) = R(s,w) + ρ(s,w) · V^π(w; r)
```

Equivalently, at the measure level, `M^{π_w→π}_s = M^{π_w}_s + ρ(s,w)(M^π_w − M^{π_w}_w)`. ✓
*Proof.* Split at H. Before H: Lemma 2. After H: the state is w, and the strong Markov property of
the environment makes the future the process started at w under π. The policy π can have memory,
because it only sees the history after H. Only π_w must be Markov (Lemmas 1–2 use it). □

## 4. Chains

**Proposition 1.** For subgoals w_1, …, w_k and a last policy π (w_0 = s):

```
V(s; w_1,…,w_k → π) = R(s,w_1) + ρ(s,w_1) · V(w_1; w_2,…,w_k → π)
                    = Σ_{i=1..k} (Π_{j<i} ρ(w_{j−1},w_j)) R(w_{i−1},w_i) + (Π_{j≤k} ρ(w_{j−1},w_j)) V^π(w_k)
```
✓ (4 hops, random rewards, detour option policies, both mazes).
*Proof.* Apply Theorem 1 with the last policy "w_2,…,w_k → π". This policy has memory, but it
starts fresh at the first hit of w_1, which Theorem 1 allows. Induct on k. □
If some w_i is never hit, `ρ = 0` and the rest of the chain drops out, which is also correct.

The test also shows that the check is not trivial: a random policy hits w only from w itself
(ρ = 0 everywhere else), so the option policies in the test are optimal policies for
"indicator of w plus small noise". The upstream proxy (no subtraction in R) misses the true
chain value by up to 0.56 (medium) and 0.36 (teleport) in the same test. ✓

## 5. Graph Bellman equation

Options: `o_w` for each landmark `w ∈ L`, and `o_∞` = "follow π_{z_r} forever" with value
`V_d(u) = F(u,z_r)ᵀz_r`.

**Proposition 2.**
```
V_H(u) = max( V_d(u),  max_{w ∈ L, w ≠ u} R(u,w) + ρ(u,w) · V_H(w) )
```
(a) **Self-loops must be excluded.** `o_u` at u is empty (H = 0, ρ = 1, R = 0). With it, any
`V(u)` above the other terms solves the equation, so the fixed point is not unique. ✓ (with
self-loops, `V_H + 1000` is a fixed point too.)
(b) **Contraction.** Without self-loops, `ρ(u,w) ≤ γ`, so the operator is a γ-contraction in
the sup norm: one fixed point, value iteration converges from any start. ✓ (20 random pairs of
value vectors.)
(c) **k sweeps = best chain of ≤ k switches.** `T^k V_d (u) = max over chains of length ≤ k` of
Proposition 1. ✓ (brute force over all chains of 5 landmarks, k ≤ 3.)
(d) **Open-loop chains are optimal among option policies.** Each option ends exactly at its
subgoal, so the state after an option is known in advance. Choosing the next option after
seeing the outcome gives no extra information. V_H is the optimal value over these options.
(e) **Exact planning cannot beat an exact direct policy.** If π_{z_r} is optimal, `V_H = V*`. ✓
Chaining can only help when FB values or the policy π_l(·|s, z_r) are inaccurate.
(f) **Goal reaching.** If π_w's path avoids the goal, `R(u,w) = 0`, and `log V_H` solves a
shortest-path problem with edge cost `−log ρ(u,w) ≥ −log γ > 0` (no zero-cost cycles).

**With FB estimates.** `ρ̂` can exceed γ (or 1) for u ≠ w. Then (b) fails, and a cycle with
`ρ̂ = 1` and `R̂ ≥ 0` has no unique value. Clip `ρ̂` to `[0, γ]` for u ≠ w; the planner also
distrusts every edge with a raw `ρ̂ ≥ 1` (`fbplan.graph.trusted_edges`).

## 6. Executing a plan: commit or re-plan

Let `Q_H(s,w) = R(s,w) + ρ(s,w) V_H(w)`, with the option at its own subgoal excluded.

**Proposition 3** (option interruption, Sutton, Precup & Singh 1999, Thm. 2). The Markov policy
that picks `argmax(V_d(s), max_w Q_H(s,w))` at every step has value `≥ max(V_d(s), max_w Q_H(s,w))`
at every state. ✓ It is strictly better at 7 of 26 states (medium) and 20 of 45 (teleport) in
the test: mid-route states can offer a better landmark than the planned one.

So with exact values, "keep w* until it is reached" is never better than re-planning. Commitment
is a defence against noisy values (see §9). The upstream π_h, which picks a new z every step,
approximates the interrupting version of the best single switch (with the proxy advantage).

## 7. Assumptions and what changes

**Markov option policies.** Lemmas 1–2 need π_w Markov. `π_l(·|s, z_w)` with a fixed `z_w` is.
A policy with memory (for example a recurrent π_l) breaks Lemma 1.

**Continuous states.** A single point is hit with probability 0, and `M_s(·)` is a density, so
`ρ̂ = m_s(w)/m_w(w)` is a density ratio. Two approximations replace exact hitting:

1. *Ball option.* Let the option end at the first entry `x_H` into a ball `B(w)`. By the strong
   Markov property, `M_s(B) = E[γ^{H_B} M_{x_H}(B)]`, so
   `M_s(B)/M_w(B) = E[γ^{H_B} κ(x_H)]` with `κ(x) = M_x(B)/M_w(B)`. This equals the ball hitting
   discount `ρ_B = E[γ^{H_B}]` only if `κ ≈ 1` on the entry points (π_w enters the ball and stays).
2. *Entry point ≠ w.* The next option starts at `x_H`, not at w. Let
   `ε_i = sup_{x ∈ B(w_i)} |V(x; rest) − V(w_i; rest)|` for the true chain values. Then
   ```
   |V_true(s; chain) − formula(s; chain)| ≤ Σ_i (Π_{j≤i} ρ_B(w_{j−1}, w_j)) ε_i ≤ ε ρ_max / (1 − ρ_max)
   ```
   *Proof.* `V_true(s) = R_B(s,w_1) + E[γ^{H_1} V_true(x_1; rest)]`. Replacing `x_1` by `w_1`
   costs at most `ρ_B ε_1`; the rest of the chain is handled by induction. □
   The error grows with the number of hops, and it is small when option values are smooth in the
   start state at the scale of the ball (ε ≈ Lipschitz constant × radius).

The bound `ρ ≤ γ` becomes `ρ_B(s,w) ≤ γ^{t_min(s, B(w))}` for s outside the ball.

**Stochastic environments (teleport).** All proofs use only the Markov property, and every ✓ test
also passes on the teleport maze. Two things change in practice:
- `ρ = E[γ^H]` averages over random hitting times. The planner is risk-neutral in `γ^H`, but
  the success metric is `P(H ≤ T)`: a route through a teleport with a 1/3 chance of a good exit
  can have a good `ρ` and a poor success rate.
- Re-planning gains more (20/45 states vs. 7/26), because the outcome of a teleport is new
  information that a committed option ignores.

**The dropped term (upstream proxy).** The proxy uses `R̃(s,w) = F(s,z_w)ᵀz_r = R(s,w) + ρ(s,w)
V^{π_w}(w; r)`. In a chain the bias is `Σ_i (Π_{j≤i} ρ_j) V^{π_{w_i}}(w_i; r)`: one term per
hop. For goal reaching through landmarks outside the goal, `V^{π_w}(w) = 0` and the proxy is
exact. Inside a rewarding region, two neighbouring landmarks u, w form a 2-cycle that the proxy
values at `ρ V/(1 − ρ)` (for example 45/(1 − 0.9) = 450 instead of 50 in the +5 goal cell).
Value iteration on a proxy graph therefore produces values that cannot be interpreted (§9.4).

## 8. Exact tabular testbed

`fbplan/tabular/` (tests: `tests/test_tabular_*.py`):

- `maze.py`: discrete copies of the OGBench medium, large, giant and teleport mazes (maps, tasks
  and teleporters checked against OGBench by a test). Each maze cell is split into `resolution²` states; medium at
  resolution 2 has 104 states, as in the paper's discrete maze. Observations are AntMaze (x, y);
  5 actions (stay, ±x, ±y). The teleport maze is stochastic, as in OGBench.
- `mdp.py`, `options.py`: exact `M^π = (I − γP_π)^{-1}`, optimal policies, option models from the
  true model, the augmented-chain ground truth, the re-planning policy. Graph value iteration
  and Q-values are the planner's own (`fbplan/graph.py`).
- `study.py`: the planners of §9. They get `ρ̂`, `R̂` and `V_d` from the planner's
  `fbplan/option_model.py` applied to the fake, and use its trust rule, value iteration and tie
  rules (`fbplan/graph.py`, `best_options`). The plan is the planner's; the execution is not:
  the study has no reach radius or look-ahead, and its commit rule keeps w* until it is reached
  or for twice its predicted duration plus 2 steps (the AntMaze planner: 100 steps). Code names:
  `v_direct` = V_d, `v_graph` = V_H, `r_before` = R.
- `fake_fb.py`: `TabularFB` implements `FBModel`. B has orthonormal columns times √n (upstream
  convention). A latent z means the reward `r_z = B ẑ / n`; `π*_z` is its exact optimal policy and
  `F(s,z) = M^{π*_z} B / n + noise`, so `F(s,z)ᵀB(s') = M^{π*_z}(s,s')` in full rank.
  Modes: exact (d = n); low rank (Laplacian basis respects walls, cosine "euclidean" basis ignores
  them); independent additive noise per head with std ≈ `rho_noise` on `ρ̂`. The controller
  π_l is greedy on the noisy value `F(s',z)ᵀz` (default) or the exact `π*_z`. π_h is idealised:
  the argmax of the proxy advantage over states `0 < d(s,w) ≤ high_radius`.
- `env.py`: an OGBench-like env and `rollout()` for any `Controller`.
- `time_limit.py`: success within a time limit vs. the discounted value on the cell mazes (§11).

Noise model note: the error on `F(s,z)ᵀz` is about `rho_noise/(1 − γ)`, while the value gap
between neighbours is about `γ^{d−1}`. So the greedy controller fails beyond
`d ≈ log(rho_noise/(1−γ)) / log γ` steps (about 9 steps at 0.04), earlier than `ρ̂` itself becomes
unreliable. This mirrors a controller that is reliable only at short range.

## 9. Study: single switch vs. chained switches under FB errors

Setup: medium, 104 states, γ = 0.9 (ρ from 0.9 for neighbours to 0.09 for the farthest pair),
landmarks = all states, every (start, goal) pair with a one-state goal, success = goal reached in
100 steps, 5 noise seeds, mean ± s.e. over seeds. "Far" = 11 steps or more (ρ ≤ 0.31).
Planners (`fbplan/tabular/study.py`) see only F, B, π_l, π_h; evaluation is exact. They use the
planner's own code (§8): `OptionModel` for `ρ̂`, `R̂`, `V_d`; `fbplan.graph` for trust (raw
`ρ̂` in `[ρ_min, 1)`), value iteration and Q-values. R̂ variants use the planner's names:
exact (`R` of §2), paper (the upstream proxy `F(s,z_w)ᵀz_r`, §7) and zero.

- **direct**: π_l(z_r). **π_h**: the idealised baseline (subgoals within 10 = 1/(1−γ) steps).
- **single switch**: best `R̂ + ρ̂ V_d(w)` over all landmarks, re-planned every step.
- **chained**: graph value iteration (§5), exact `R̂`. **chained R=0**: `R̂ = 0` (§5 f).
- **guarded**: R = 0, head min, `ρ_min = 0.4`, commit until reached (timeout: twice the
  predicted duration `log ρ̂ / log γ`, plus 2 steps).

Reproduce: `uv run python -m fbplan.tabular.run_study && uv run python -m fbplan.tabular.study_plots`
(14 min with 4 workers on the shared 12-core Mac; 10 CPU minutes). Tables:
`docs/figures/theory/study_*.csv`.

### 9.1 Chaining helps when the controller fails at long range

![success vs distance](figures/theory/success_vs_distance.png)
![far success vs noise](figures/theory/far_success_vs_noise.png)

Far-pair success:

| controller | planner | ρ noise 0.02 | 0.03 | 0.04 |
|---|---|---|---|---|
| greedy | direct | 0.70 | 0.36 | 0.16 |
| greedy | π_h (idealised) | 0.89 | 0.55 | 0.26 |
| greedy | single switch | 0.80 | 0.49 | 0.24 |
| greedy | chained | 0.89 | 0.55 | 0.20 |
| greedy | chained R=0 | 0.90 | 0.67 | 0.37 |
| greedy | chained R=0, guarded | **0.90** | **0.69** | **0.46** |
| exact | direct | 1.00 | 1.00 | 1.00 |
| exact | chained | 0.90 | 0.60 | 0.28 |
| exact | chained R=0 (± guards) | 1.00 | 1.00 | 1.00 |

- With the greedy controller, chained R=0 with guards almost triples direct at noise 0.04 and
  beats the idealised π_h (0.46 vs. 0.26). At noise 0.02 the gain over π_h is small.
- With an exact controller the direct policy is already optimal (§5 e). Planning on noisy values
  can only lose, and the planner with exact-mode `R̂` loses a lot.

### 9.2 The main failure is noisy R̂, not wormholes

`R̂ = F(s,z_w)ᵀz_r − ρ̂ F(w,z_w)ᵀz_r` is a difference of two noisy values. For goal reaching
the true R is 0 almost everywhere, so `R̂` is pure noise of size about `rho_noise/(1 − γ)`. The max
over about 100 landmarks per hop picks the largest noise, and cycles with positive `R̂` look
profitable. Setting `R̂ = 0` restores 1.00 with the exact controller.

Independent noise creates no gross wormholes: no pair has `ρ̂ > 2γ^d` at any noise level up to
0.04 (`study_wormholes.csv`). The harm comes from many small overestimates compounded by the max.
"Wormhole" here means `ρ̂ > 2γ^d` (`study.rho_bound_wormholes`, the true ρ is at most `γ^d`).
d-eval's AntMaze calibration uses another definition: a trusted pair whose FB distance is
shorter than the time to walk the geodesic at the ant's top speed (§11.2).

![calibration](figures/theory/calibration.png)

### 9.3 Guards (one change at a time from a plain chained planner)

![guards](figures/theory/guards.png)

Far-pair success at noise 0.04:

| change | R exact, greedy | R exact, exact | R=0, greedy | R=0, exact |
|---|---|---|---|---|
| plain | 0.20 | 0.28 | 0.37 | 1.00 |
| head min | 0.29 | 0.58 | 0.29 | 1.00 |
| ρ_min 0.6 | 0.21 | 0.28 | 0.42 | 1.00 |
| hop discount 0.95 | 0.27 | 0.50 | 0.11 | 1.00 |
| hop discount 0.8 | 0.16 | 0.83 | 0.13 | 1.00 |
| commit | 0.31 | 0.40 | 0.50 | 1.00 |
| all (min, ρ_min 0.4, commit) | 0.53 | 0.80 | 0.46 | 1.00 |

- **Head min** counters independent errors in `R̂` (0.28 → 0.58), not shared errors (§9.5).
- **ρ_min** matters only when hops must stay inside the controller's reliable range (R=0, greedy:
  0.37 → 0.42; at noise 0.02, 0.90 → 0.95). With exact-mode `R̂` the chosen edges are already
  short.
- **Hop discount** (a factor on ρ̂ per hop, as in the planner) helps against fake `R̂` when the
  controller is exact (0.28 → 0.83), but hurts with R = 0: it pushes the plan to fewer, longer
  hops that the greedy controller cannot complete (0.37 → 0.11–0.13).
- **Commitment** helps in every noisy case: it stops dithering between near-equal noisy plans.
  With exact values it can only lose (§6).

### 9.4 The dropped term over hops (general rewards)

Region rewards: +5 on the task's goal cell, +1 and −1 on two random cells; 5 tasks × 10 rewards;
exact controller; value from the task start (V* ≈ 11).

![dropped term](figures/theory/dropped_term.png)

| planner | regret, exact FB | predicted − true, exact | regret, noise 0.01 | predicted − true, 0.01 |
|---|---|---|---|---|
| single, R exact | 0.00 | 0.0 | 0.08 | 2.1 |
| single, R paper (proxy) | 0.19 | 10.7 | 0.19 | 11.3 |
| chained, R exact | 0.00 | 0.0 | 0.41 | 6.6 |
| chained, R paper (proxy) | 0.00 | 111.8 | 0.00 | 121.3 |
| chained, R = 0 | 0.03 | 0.2 | 0.11 | 1.3 |

- The proxy bias repeats per hop: the predicted value is 10× more wrong for chains than for one
  switch, mostly through the 2-cycles of §7. Actions suffer little here, because the inflation sits
  in the goal region, where the agent should go anyway.
- With noise, exact-mode `R̂` makes chained plans long (9.8 hops; with exact values 0, because the
  direct policy is optimal and wins ties, `graph.best_options`) and costly (regret 0.41):
  the same noisy-`R̂` effect as §9.2. R = 0 ignores the ±1 regions on the way and costs little.

### 9.5 Low rank: wall-blind features make wormholes

![low rank](figures/theory/low_rank.png)

- The wall-blind basis puts 0.24 % (rank 32) and 0.41 % (rank 16) of pairs above `2γ^d`; the
  wall-aware basis almost none. The wormholes cross thin walls (calibration figure, right).
- At rank ≤ 32 all methods lose together (far success 0.57–0.76 at rank 16, 0.86–0.97 at rank 32),
  and chaining gives no gain (direct is best at rank 16): the
  reward `P_B e_w` is blurred, so π_l(·|B(w)) goes near w, not to w. Errors shared by both heads
  are not removed by head min.

## 10. Consequences for the planner (approach A)

1. Exclude self-loops; clip `ρ̂` to `[0, γ]` for distinct landmarks.
2. For goal-reaching tasks use `R̂ = 0` (shortest path on `−log ρ̂`). Use exact-mode `R̂` only
   for general rewards, with head min and a hop discount. Never use the proxy ("paper") in a
   graph.
3. Choose `ρ_min` so that every hop stays in the range where π_l reliably reaches its subgoal;
   commit to w* until it is reached, with a timeout.
4. Expect gains where π_l fails at long range, not where the direct policy already works. The
   oracle-waypoint controller (d-eval) measures exactly this headroom on AntMaze.
5. In continuous states the chain value is approximate (§7): keep hops few and the reach radius
   small compared with the scale on which option values change.

## 11. Bonus mazes: teleport and giant

Runs: the protocol of design.md §5 (5 eval seeds × 5 tasks × 50 episodes, paired start states),
method defaults. The final runs are on the Linux evaluation instance (d-eval, CSVs in
`results/teleport/` and `results/giant/`, commands in `docs/status/b-theory.md`). §11.3 marks
the rows that still come from preliminary Mac runs (`runs/mac/`, same code, MuJoCo 3.1.6); the
Linux runs replace them.

### 11.1 Teleport: the value is risk-neutral, the success metric is not

**The env** (OGBench source and the train split):

- An agent within 1.5 xy units of an entry centre, (4,6) or (5,1), jumps to one of the exits
  (1,7), (6,1), (6,10), chosen uniformly. Exit (1,7) has walls on all four sides: a trap.
- Every task goal can be reached on foot, without a teleporter. The shortest such walks are
  23, 15, 14, 25 and 27 cells long (tasks 1–5). The plain BFS path goes through an entry on
  tasks 1, 3, 4 and 5, so the BFS oracle of `fbplan/controllers/oracle.py` is not valid here.
- The data route through the entries: trajectories that leave the start cell (5,6) of tasks 3
  and 5 go up into the entry 91 times and down 2 times; from (6,1), up into the entry 323 times
  and down 54 times. OGBench's `get_oracle_subgoal` runs BFS on the map without the teleporters.
  18 % of the train states lie in the trap, and 27 % of the 1,000 train trajectories end there.

**Proposition 4.** Each jump ends in the trap with probability 1/3, and the trap is absorbing.
So a policy that jumps at least once succeeds with probability ≤ 2/3. A policy that jumps again
after each bad exit succeeds with probability `Σ_k (1/3)^{k+1} = 1/2` at most. A walk that fits
in the time limit succeeds with probability 1. ✓ (`tests/test_tabular_time_limit.py`)

**Why a discounted value prefers the jump.** FB values estimate `E[γ^H]`, and π_l, π_h and the
graph planner all maximise it. Let one cell move take c ant steps, so `g = γ^c` per move. From
(5,6) to the goal (7,10) (task 3), the jump is worth `g²/(3 − g)` (exit (6,10) is next to the
goal, and exit (6,1) is next to the other entry, so a new jump costs one move), and the walk is
worth `g^14`. At γ = 0.99 and c = 40: 0.19 against 0.0036. The trap removes a third of the
value per jump; the walk loses a factor `γ^560`.

Exact cell-level computation (`fbplan/tabular/time_limit.py`; resolution 1, T = 1000 ant
steps, c = 40, the median pace of successful medium episodes):

| task | walk (cells) | P(success): walk | risk-neutral (max `E[γ^H]`, γ = 0.99) | best for the metric | γ at which the risk-neutral policy walks |
|---|---|---|---|---|---|
| 1 | 23 | 1.00 | 0.48 | 1.00 | 0.99856 |
| 2 | 15 | 1.00 | 0.50 | 1.00 | 0.99770 |
| 3 | 14 | 1.00 | 0.50 | 1.00 | 0.99850 |
| 4 | 25 | 1.00 | 0.50 | 1.00 | 0.99919 |
| 5 | 27 | 0.00 | 0.48 | 0.67 | 0.99933 |

![time limit](figures/theory/time_limit.png)

- The risk-neutral policy takes the teleporter on all 5 tasks at every pace from 20 to 60 ant
  steps per cell, and its success stays at 0.44–0.50. An exact FB agent with this γ would score
  below 50 % on teleport.
- It walks only for γ ≥ 0.9977–0.9993: an effective horizon `1/(1 − γ)` of 435–1,500 steps,
  4–15 times the checkpoint's 100 steps.
- The best policy for the metric walks whenever the walk fits. On task 5 the walk does not fit
  at 40 steps per cell; the best policy then jumps once and walks from the far exit (2/3).

**Consequences.** `E[γ^H]` at one γ is one point of the Laplace transform of H. It cannot tell
"slow" from "sometimes never", so no planner on FB values alone can be optimal for `P(H ≤ T)`
here. The graph planner with `R̂ = 0` maximises `Π ρ̂ = E[γ^H]` of the chain: risk-neutral too. Its
xy cap (6 units) removes every jump edge, because the exits are far from the entries in xy, so it
plans walks; this guard works against risk by accident. Remedies within the frozen setting: a
larger γ (it costs accuracy, see giant), FB at several γ to bound `P(H ≤ T)`, or an explicit
penalty on edges whose successor measure has mass on an absorbing region (the trap).

### 11.2 Giant: the time limit and ρ at long range

**The time budget.** The giant walks are 30, 26, 30, 26 and 17 cells (medium: 6–10). They fit in
1,000 steps only at ≤ 33, 38, 33, 38 and 59 ant steps per cell. Successful medium episodes take a
median of 41 (baseline) and 43 (oracle) steps per cell, and only 29–45 % of them are fast enough
for tasks 1–4 (70 % for task 5). So with this π_l even a perfect high level fails most episodes
of tasks 1–4.

The privileged oracle (true shortest path, waypoints 2 cells ahead) measures the pace on giant
itself. On eval seed 0 it fails 249 of 250 episodes: 19 % of the failures are flips, the rest
are still moving at the time limit. In 1,000 steps it covers a median of 14, 15.5, 6, 15 and 5
cells of the 30-, 26-, 30-, 26- and 17-cell walks (d-eval's replay, `cells_left`), so 65–200
steps per cell against about 40 on medium. On giant, π_l follows even perfect waypoints too
slowly for the time limit; the route is not the bottleneck.

**ρ at long range.** The checkpoint uses γ = 0.995 (horizon 200 steps). A 1,000-step route has
`ρ = 0.995^1000 = 0.0067`. FB calibration on 400 val states (d-eval's
`uv run python analysis/calibration.py --env antmaze-giant-navigate-v0 --out-dir docs/figures/theory
--table-dir docs/figures/theory`):

![giant calibration](figures/theory/calibration_antmaze-giant-navigate-v0_min.png)

- The median FB distance `−log ρ̂ / −log γ` grows with the geodesic only up to about 45 xy units
  (11 cells); beyond that it sits at the cap. The task walks are 17–30 cells (68–120 xy units
  between cell centres). The quartiles are 200–400 steps apart already at 15–20 units (medium:
  a narrow band up to 25).
- Caveat: the calibration's point geodesic passes between two walls that touch only at a
  corner (4 such places in giant, 2 in medium and in teleport), which the ant cannot do. It
  makes some pairs look closer than they are; the giant task 5 measures 21 xy units, but its
  walk is 17 cells.
- `ρ̂ ≤ 0` (clipped to 0) for 23.9 % of the pairs (medium 10.2 %). At ρ_min 0.3, 10.6 % of the
  trusted pairs are wormholes in d-eval's sense (FB distance shorter than the walk at the ant's
  top speed; medium 5.5 %), some across the whole maze (longest geodesic 69 xy units). The
  planner's 6-unit xy cap removes all but 801 of the 4,586 (table:
  `docs/figures/theory/wormholes_antmaze-giant-navigate-v0_min.csv`):

![giant wormholes](figures/theory/wormholes_antmaze-giant-navigate-v0_min.png)

So at the start of a task, `F(s,z_r)ᵀz_r` and the subgoal values `F(w,z_r)ᵀz_r` that drive π_h
carry no reliable direction. This is the regime of §9.1: the controller fails at long range, and
a graph of short trusted edges can supply the direction (the planner's edges are ≤ 6 xy units,
well inside the calibrated range). Chaining cannot fix the time budget or the flips.

### 11.3 Measured results

Success (%), 95 % stratified bootstrap CI, paired difference against the baseline on the same
episodes: `uv run python analysis/tables.py results/teleport/*.csv results/teleport/planner*/*.csv
--label-root results/teleport --cluster config_hash`. Planner rows pool landmark sets 0–2
(3,750 episodes, cluster bootstrap over the sets).

| teleport | task 1 | task 2 | task 3 | task 4 | task 5 | all | vs. baseline (pp) |
|---|---|---|---|---|---|---|---|
| baseline | 22.8 | 43.6 | 29.2 | 24.4 | 18.4 | 27.7 [25.3, 30.2] | |
| baseline_raw (upstream z_r) | 18.8 | 39.6 | 26.4 | 21.2 | 14.8 | 24.2 [21.8, 26.6] | −3.5 [−6.2, −1.0] |
| planner | 17.5 | 24.8 | 20.3 | 15.6 | 14.7 | 18.6 [16.5, 21.0] | −9.1 [−12.2, −6.0] |
| planner_high | 17.1 | 33.6 | 28.1 | 21.7 | 19.9 | 24.1 [21.5, 26.4] | −3.6 [−6.7, −0.5] |
| nonhier (Mac, preliminary) | 20.8 | 25.2 | 15.2 | 16.4 | 17.6 | 19.0 [16.9, 21.2] | −9.8 vs. the Mac baseline (28.8) |

Giant, preliminary Mac runs (planner not run):

| giant | task 1 | task 2 | task 3 | task 4 | task 5 | all | vs. baseline (pp) |
|---|---|---|---|---|---|---|---|
| baseline | 0.0 | 0.0 | 0.0 | 1.6 | 1.2 | 0.6 [0.2, 1.0] | |
| nonhier | 0.0 | 2.0 | 0.0 | 1.2 | 0.4 | 0.7 [0.3, 1.2] | +0.2 (p 0.8) |
| oracle (privileged) | 0.0 | 3.6 | 0.0 | 0.8 | 6.8 | 2.2 [1.5, 3.0] | +1.7 (p 3e-4) |

**Teleport: most failures end in the trap.** A trapped ant ends its episode inside cell (1,7), so
its final distance to the goal is within 2 xy units of the trap centre's (26.8 for goal (7,10),
33.9 for goal (7,1)). On eval seed 0 of the baseline this rule finds all 129 trapped episodes of
an exact replay (d-eval's `replay_failures`, trap = `cells_left = −1`) and adds 3 of the 48 other
failures (Mac CSV). Over all episodes:

| teleport | success | trap | other failures |
|---|---|---|---|
| baseline | 27.7 % | 54.7 % | 17.6 % |
| baseline_raw | 24.2 % | 57.4 % | 18.4 % |
| planner | 18.6 % | 64.2 % | 17.3 % |
| planner_high | 24.1 % | 53.9 % | 22.1 % |
| nonhier (Mac) | 19.0 % | 64.5 % | 16.5 % |

- A policy that jumps until it lands on a good exit ends in the trap half the time (Proposition
  4). All agents end there more often: they also jump when the walk is short, and they jump
  again after a good exit when π_l does not finish.
- The planner plans walks (its xy cap removes the jump edges), yet 64 % of its episodes end in
  the trap. π_h after the waypoint (planner_high) is as trapped as the baseline (54 %) and
  recovers most of the gap (−3.6 pp against −9.1 pp).
- π_l walks into the entries by itself: on dev episodes (eval seed 100, not used for any
  reported number) the teleport-aware oracle with waypoints *below* (5,6) still saw π_l walk up
  into entry (4,6) and jump to the trap. The data explain this: trajectories that leave (5,6)
  go up 91 times and down 2 times (§11.1). π_l learned the expert's route through the entries.
- On the same 30 dev episodes, the teleport-aware oracle succeeded 2/30 with π_l and 6/30 when
  π_h follows the waypoints, against 8/30 for the baseline. Even with the true risk-free route,
  waypoint following with this π_l fails on teleport (it also stalls at corners). So on
  teleport the frozen π_l, not the plan, sets the ceiling; Proposition 4 adds a second ceiling
  for any agent that maximises a discounted value.

**Giant: the executor is the bottleneck.** The baseline and nonhier are equal (0.6 % and 0.7 %).
The privileged oracle reaches only 2.2 %; its failures are mostly slow progress (§11.2:
5–15.5 cells covered in 1,000 steps, 19 % flips). A planner can at most supply the route, and
the oracle shows that the route alone is worth about +1.7 points here.
