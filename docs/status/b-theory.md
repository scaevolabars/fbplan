# Status: b-theory

All items of the brief are done. Main document: `docs/theory.md`.

## Done

- `fbplan/tabular/maze.py`: discrete OGBench medium, large, giant and teleport mazes (maps checked
  against the OGBench source). Medium at resolution 2 has 104 states, as in the paper.
  Observations are AntMaze (x, y). The teleport maze is stochastic, as in OGBench.
- `fbplan/tabular/mdp.py`, `options.py`: exact successor measures, optimal policies, option models,
  graph value iteration, the re-planning policy, and ground truth from definitions.
- `fbplan/tabular/fake_fb.py`: `TabularFB` implements `FBModel` (config `FakeFBConfig`):
  - exact mode (default): `F(s,z)ᵀB(s') = M^{π_z}(s,s')` to 1e-9.
  - `rank=d`: low-rank `B` ("laplacian" respects walls, "euclidean" ignores walls → wormholes).
  - `rho_noise=σ`: independent additive noise per head; error std on `ρ̂` is about σ.
  - `low_policy="greedy"` (default, π_l greedy on noisy F) or `"optimal"` (exact π_z).
  - `high_latent`: idealised `π_h`, argmax of the switching advantage over `w ≠ s`
    (`high_advantage="proxy"` like upstream, or `"full"`; `high_radius` limits subgoals).
  - Ground truth: `true_successor_measure(z)`, `policy(z)`, `reward_of_latent(z)`, `task_reward(k)`.
- `fbplan/tabular/env.py`: `TabularMazeEnv` (OGBench-like) and `rollout(...)`.
- Theory checks (`tests/test_tabular_theory.py`, medium and teleport, exact to 1e-8): Lemma B.1,
  `R` = reward before the hit, Theorem 1 at the measure level, the chain formula (4 hops),
  "k sweeps of graph VI = best chain of ≤ k switches", unique fixed point without self-loops,
  exact planning = optimal direct value, re-planning ≥ plan.
- Study (`fbplan/tabular/{study,run_study,study_plots}.py`; CSVs and figures in
  `docs/figures/theory/`; 1 min on 12 cores). Results in `docs/theory.md` §9.
- `docs/theory.md`: derivation with assumptions (Markov options, ball hitting with an error bound,
  stochastic envs, the dropped term) and the study.
- Tests: 75 pass (`tests/test_tabular_*.py`). `ruff check` passes.

## Usage for c-planner

```python
from fbplan.tabular import FakeFBConfig, TabularFB, rollout
model = TabularFB(FakeFBConfig(rho_noise=0.02))   # an FBModel; model.discount = 0.9
obs = model.maze.xy                               # all states = the "dataset"
task = model.task_spec(1)                         # TaskSpec for OGBench task 1
result = rollout(my_controller, model, task_id=1, seed=0, max_steps=200)
```

Edge table for all 104 states: 0.3 s (medium), 5 s (giant).

## Main results (far pairs = 11 steps or more; details in theory.md §9)

- Greedy controller (π_l degrades with distance): chaining helps. ρ noise 0.04: direct 0.16,
  idealised π_h 0.26, chained R=0 with guards 0.46.
- Exact controller: direct is optimal; chained with FB `R̂` collapses (0.28 at 0.04). `R̂` is a
  difference of two noisy values and the max exploits it. `R̂ = 0` gives 1.00.
- Independent noise creates no gross wormholes (`ρ̂ > 2γ^d`); a wall-blind low-rank B does
  (0.4 % of pairs at rank 16). Low rank hurts all methods together.
- The proxy (dropped term) inflates predicted chain values 10× more than single-switch values
  (2-cycles in rewarding regions); actions suffer little on goal-like rewards.

## Next

- None in the brief. Possible follow-ups if the coordinator wants them: run the study on the
  large / giant / teleport mazes; landmark subsets (farthest-point sampling) instead of all states;
  a smooth (correlated in s) noise model closer to a network's errors.

## Blockers

- None.

## Requests to the coordinator

- Please merge `b-theory` into `main`, so c-planner can use `fbplan/tabular/fake_fb.py`.
- Proposed design.md edits (user approved them as a proposal on 2026-10-29):
  - §3 graph Bellman: `max_{w∈L, w≠u, edge (u,w) trusted}`. Add: "For u ≠ w, ρ(u,w) = E[γ^H] ≤ γ
    because H ≥ 1, so value iteration is a γ-contraction with a unique fixed point. A self-loop
    (ρ = 1, R = 0) makes every larger V(u) a fixed point. Clip ρ̂ to [0, γ] for u ≠ w."
  - §3 new line: "The chain formula assumes exact hitting of w and Markov option policies (fixed
    z_w). In continuous states the option ends on a ball around w and the next option starts at
    the entry point, so the formula is approximate (error bound in docs/theory.md §7)."
  - §4.A: "keep w* until it is reached" → "commit to w* until reached, or re-plan every step
    (config switch). With exact values re-planning is never worse (option interruption,
    Sutton–Precup–Singh 1999); commitment guards against noisy values."
  - §4.A, from the study: "For goal-reaching tasks use R̂ = 0 (shortest path on −log ρ̂); FB R̂
    is noise there and the max exploits it."
  - c-planner brief: clip ρ to [0, γ] (not [0, 1]) for distinct landmarks; add `R̂ = 0` as the
    default for goal reaching (theory.md §10).
