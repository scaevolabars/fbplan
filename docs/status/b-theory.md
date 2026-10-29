# Status: b-theory

## Done

- `fbplan/tabular/maze.py`: discrete copies of the OGBench medium, large, giant and teleport mazes
  (maps checked against the OGBench source). Each maze cell splits into `resolution²` states;
  medium at resolution 2 has 104 states, as in the paper. Observations are AntMaze (x, y).
  The teleport maze is stochastic, as in OGBench.
- `fbplan/tabular/mdp.py`: exact successor measures `(I − γP_π)^{-1}`, exact optimal policies.
- `fbplan/tabular/fake_fb.py`: `TabularFB` implements `FBModel`. Config `FakeFBConfig`:
  - exact mode (default): `F(s,z)ᵀB(s') = M^{π_z}(s,s')` to 1e-9.
  - `rank=d`: low-rank `B` ("laplacian" respects walls, "euclidean" ignores walls → wormholes).
  - `rho_noise=σ`: independent additive noise per head; error std on `ρ̂` is about σ.
  - `low_policy="greedy"` (default, π_l greedy on noisy F) or `"optimal"` (exact π_z).
  - `high_latent` is an idealised `π_h`: exact argmax of the switching advantage
    (`high_advantage="proxy"` like upstream, or `"full"`).
  - Ground truth: `true_successor_measure(z)`, `policy(z)`, `reward_of_latent(z)`, `task_reward(k)`.
- `fbplan/tabular/env.py`: `TabularMazeEnv` (OGBench-like reset/step/success) and `rollout(...)`.
- Tests: `tests/test_tabular_{maze,mdp,fake_fb,env}.py`, 43 pass.
- `fbplan/tabular/options.py` + `tests/test_tabular_theory.py` (19 checks, medium and teleport):
  Lemma B.1, `R` = reward before the hit, Theorem 1 at the measure level, the chained formula
  (4 hops), and "k sweeps of graph VI = best chain of ≤ k switches" all hold to 1e-8 against
  ground truth built from definitions. Exact planning cannot beat the optimal direct value.

## Findings on design.md §3

User decision (2026-10-29): propose the changes to the coordinator, continue the work.
The formulas are correct for exact quantities. Extra conditions found:

1. Graph Bellman equation: the max must exclude `w = u`. A self-loop has `ρ = 1`, `R = 0`, so
   any `V(u)` above the other terms solves the equation (shown numerically: VI from a high start
   stays +1000). Without self-loops the operator is a γ-contraction because `ρ(u,w) ≤ γ` for `u ≠ w`.
2. Consequence for c-planner: clip `ρ̂` to `[0, γ]` for distinct landmarks, not `[0, 1]`.
   With `ρ̂ = 1` cycles and `R̂ ≥ hop cost`, value iteration has no unique fixed point.
3. Re-planning at every step (option interruption, Sutton–Precup–Singh 1999) is never worse than
   committing to `w*` until it is reached, with exact values. It is strictly better at 7/26 states
   (medium) and 20/45 (teleport) in the test. "Keep `w*`" is a defence against noise, not a
   consequence of the theory.
4. Expected assumptions, to be written in theory.md: option policies must be Markov (fixed `z_w`;
   the last policy can be any); continuous states make the chain approximate (entry point on the
   ball ≠ `w`, error per hop).

## Usage for c-planner

```python
from fbplan.tabular import FakeFBConfig, TabularFB, rollout
model = TabularFB(FakeFBConfig(rho_noise=0.02))   # an FBModel; model.discount = 0.9
obs = model.maze.xy                               # all states = the "dataset"
task = model.task_spec(1)                         # TaskSpec for OGBench task 1
result = rollout(my_controller, model, task_id=1, seed=0, max_steps=200)
```

Edge table for all 104 states: 0.3 s (medium), 5 s (giant).
First observations (5 tasks × 4 seeds, 200 steps): exact → direct and `π_h` 100 %.
`rho_noise=0.02` → direct 25 %, `π_h` 20 %. `rho_noise=0.05` → direct 20 %, `π_h` 0 %.

- Study (`fbplan/tabular/{study,run_study,study_plots}.py`, outputs in `docs/figures/theory/`,
  1 min on 12 cores). Medium, 104 states, all start/goal pairs, 5 seeds. Far = ≥ 11 steps:
  - Greedy controller (π_l degrades with distance): chaining helps. ρ noise 0.02: direct 0.70,
    chained 0.89, guarded chained 0.90; noise 0.04: direct 0.16, guarded chained 0.46.
    The idealised π_h (subgoals within 10 steps) is a strong baseline at low noise.
  - Exact controller: direct is perfect; chained with FB `R̂` collapses (0.28 at 0.04), because
    `R̂` is a difference of two noisy values and the max exploits it (also cycles).
    `R̂ = 0` (shortest path in −log ρ̂) gives 1.00. Recommendation for c-planner:
    goal-reaching → `R̂ = 0`, plus ensemble min, `ρ_min` ≈ 0.4–0.6 and commitment.
  - Wall-blind low-rank B creates wormholes (0.4 % of pairs at rank 16); the wall-aware basis
    almost none. Low rank hurts all methods about equally (blurred targets).
  - Dropped term: the proxy inflates predicted values about 10× more for chains than for one
    switch (2-cycles inside rewarding regions: ρ·V/(1−ρ)); regret stays small on these tasks.

## Next

- `docs/theory.md`: write the derivation (conditions above) and the study section with the
  figures. Not started.
- Check figures once more for layout (legend placement was just fixed).

## Blockers

- None.

## Requests to the coordinator

- Please merge `b-theory` into `main` now, so c-planner can use `fbplan/tabular/fake_fb.py`.
- Proposed design.md §3 edits (findings 1–3), approved by the user as a proposal:
  - Graph Bellman: `max_{w∈L, w≠u, edge (u,w) trusted}`. Add: "For u ≠ w, ρ(u,w) = E[γ^H] ≤ γ
    because H ≥ 1, so value iteration is a γ-contraction with a unique fixed point. A self-loop
    (ρ = 1, R = 0) makes every larger V(u) a fixed point. Clip ρ̂ to [0, γ] for u ≠ w."
  - §4.A: "keep w* until it is reached" → "commit to w* until reached, or re-plan every step
    (config switch). With exact values re-planning is never worse (option interruption,
    Sutton–Precup–Singh 1999); commitment guards against noisy values."
  - §3, new line: "The chain formula assumes exact hitting of w and Markov option policies
    (fixed z_w). In continuous states the option ends on a ball around w and the next option
    starts at the entry point, so the formula is approximate (error bound in docs/theory.md)."
  - c-planner brief: clip ρ to [0, γ] (not [0, 1]) for distinct landmarks.
