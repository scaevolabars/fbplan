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

## Next

- Theory: derivation of the chained formula and the graph Bellman equation; numeric tests.
- Study: single switch vs. chained planning under noise and low rank.

## Blockers

- None.

## Requests to the coordinator

- Please merge `b-theory` into `main` now, so c-planner can use `fbplan/tabular/fake_fb.py`.
