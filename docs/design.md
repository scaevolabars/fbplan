# Design: planning over sequences of intentions with frozen FB

Status: living document, owned by the coordinator tab. Last update: 2026-10-29.

## 1. Task

Source: the assignment "Тестовое FB июнь 2026" (Notion). Summary:

- Given: frozen FB pi-Switch representations `F`, `B`, a low-level policy `π_l(a|s,z)`, and a
  single-intention high-level policy `π_h(z|s,z_r)`. Upstream code and paper:
  `third_party/ssm` (github.com/stestoKTH/switching-successor-measures), arXiv 2605.13207.
- Build: a high-level mechanism that reasons over a **sequence** of intentions and beats `π_h`
  on long horizons. Main env: `antmaze-medium-navigate-v0`. Bonus: large, giant, teleport.
- Constraints: no generative model of states or trajectories; no extra environment interaction
  for training (offline data only). `F`, `B`, `π_l` stay frozen unless we explain why we change them.
- Deliverables: clean reproducible code with run instructions, a report (most important:
  analysis, not only numbers), results vs. baseline on medium with several seeds.
  We must defend every design choice on a call.

Paper numbers for the baseline (success %, 5 tasks × 50 episodes × 5 seeds):
medium 87 ± 6, large 66 ± 9, giant 1 ± 1, teleport 40 ± 6. HIQL: medium 93 ± 1.

## 2. Facts from the upstream code

- `π_h` picks a new intention at **every** step (`agents/fbpiswitch.py:sample_actions`).
- `F` has 2 ensemble heads. `B` has 1.
- Latents are normalized to norm `sqrt(d)` (`d = 128`) before they enter a network.
- `π_h` is trained with AWR on the proxy advantage
  `Â(s,w,z) = F(s,z_w)ᵀz + ρ(s,w)·F(w,z)ᵀz − F(s,z)ᵀz` with `z_w = B(w)`.
  It drops the term `−ρ(s,w)·F(w,z_w)ᵀz` of the paper's Eq. (5). The paper does this on purpose (Sec. 6).
- Training subgoals `w` are trajectory states at a geometric offset (mean `1/(1−γ)` = 100 steps).
- Task embedding: `z_r = mean(r · B(s))` over 100k relabeled validation samples, then normalized.
- One checkpoint per env (training seed 0): Drive folder `1dKYhaDJH9lUREo-kUV3AwmTLrxvKO7Ek`,
  subfolders `medium`, `large`, `giant`, `teleport`, each with `flags.json` and `params.pkl`.
- `params.pkl` (medium, checked): `{'agent': flax state dict}` with `rng`, `network/step`,
  `network/params/modules_{forward_repr, backward_repr, actor, high_actor}` and `network/opt_state`.
  It pickles plain numpy arrays only (`numpy._core`), so loading does not depend on the JAX version.

## 3. Math and notation

`M^π_s(s')` is the state successor measure. With FB: `M^{π_z}_s(s') ≈ F(s,z)ᵀB(s')`
and `V^{π_z}(s; r) = F(s,z)ᵀz_r`.

**Option model of "follow π_{z_w} until w is hit".** With `z_w = B(w)`:

```
ρ(s,w) = F(s,z_w)ᵀB(w) / F(w,z_w)ᵀB(w)        ≈ E[γ^H],  H = hitting time of w from s
R(s,w) = F(s,z_w)ᵀz_r − ρ(s,w)·F(w,z_w)ᵀz_r    reward collected before w is hit
```

`ρ` does not depend on the scale of `B(w)`. The paper's Theorem 1 gives the value of one switch:
`V(s; w → π) = R(s,w) + ρ(s,w)·V^π(w; r)`.

**Chained switches.** At the hitting time the agent is exactly at `w`, so by the strong
Markov property the switch formula applies again at `w`. By induction:

```
V(s; w1, …, wk → π_z) = R(s,w1) + ρ(s,w1)·[ R(w1,w2) + ρ(w1,w2)·[ … + ρ(w_{k−1},w_k)·F(w_k,z_r)ᵀz_r ] ]
```

**Graph Bellman equation** over a finite landmark set `L`:

```
V_H(u) = max( F(u,z_r)ᵀz_r ,  max_{w∈L, edge (u,w) trusted} R(u,w) + ρ(u,w)·V_H(w) )
```

This is SMDP value iteration over options. FB gives the option models for free.
For goal-reaching with the proxy (`R ≈ 0` away from the goal), it reduces to a shortest
path with edge cost `−log ρ(u,w)`.

**Why chaining can help.** FB values are accurate at short range and noisy at long range
(`γ^500 ≈ 0.007` at `γ = 0.99`). The graph joins trusted short edges into long routes.
The main risk is **wormholes**: overestimated `ρ` across walls, which the max operator exploits.
Guards: an edge threshold `ρ ≥ ρ_min`, the min over the 2 `F` heads, a cost per hop.

## 4. Approaches

- **A. Landmark graph planner (first).** No training. Once per env: landmarks from the dataset
  and the edge table `ρ(u,w)`. Once per task: value iteration for `V_H`. Every step:
  `w* = argmax_w R(s,w) + ρ(s,w)·V_H(w)` over trusted edges from `s`, keep `w*` until it is
  reached, act with `π_l(a|s,B(w*))`. Variant: give `B(w*)` to `π_h` as its task.
- **B. Learned multi-switch value (second).** Fit `V_H(s,z)` offline by expectile regression
  on `R(s,w) + ρ(s,w)·V̄_H(w,z)`, then train a new `π_h` with AWR on that advantage.
  Works for any reward and needs no graph at test time. B can distill A.

## 5. Evaluation protocol

- The baseline is re-evaluated by us on the same frozen `F`, `B`, `π_l`. We do not compare against
  the paper's numbers.
- `π_h` is retrained on the frozen modules with 5 seeds (500k steps, upstream recipe) so that the
  baseline variance includes training variance. The provided `π_h` is reported as a separate reference row.
- 5 tasks × 50 episodes × ≥ 5 eval seeds per method. The reset seed of each episode comes from
  `(eval_seed, task_id, episode)`, so all methods see the same start states (paired design).
- Eval temperature 0, episode length 1000, success as defined by OGBench.
- Statistics: mean success with 95 % stratified bootstrap CI; paired test over episodes.
- Diagnostics: oracle-waypoint controller (shortest path on the true maze map) gives the best
  result that any high-level method can get with this `π_l`; FB distance calibration
  (`−log ρ / −log γ` against maze geodesic distance); wormhole count; route plots.

## 6. Components

| Branch / tab | Owns | Provides |
|---|---|---|
| `a-frozen` | `fbplan/frozen.py`, `tests/test_frozen.py` | `load_model`, `make_env`, `load_dataset`, checkpoint download |
| `b-theory` | `fbplan/tabular/`, `docs/theory.md`, `tests/test_tabular*.py` | exact tabular maze with true `M`; proofs; wormhole study |
| `c-planner` | `fbplan/option_model.py`, `landmarks.py`, `graph.py`, `controllers/planner.py` | approach A controllers |
| `d-eval` | `fbplan/evaluate.py`, `stats.py`, `controllers/{baseline,oracle}.py`, `analysis/` | eval loop, CSVs, stats, diagnostics |
| `e-learner` | `fbplan/train_high.py`, `fbplan/learned/` | retrained baseline seeds; approach B |
| coordinator | `fbplan/contracts.py`, `docs/design.md`, `REPORT.md` | contracts, merges, report |

Shared contracts: `fbplan/contracts.py` (`FBModel`, `Controller`, `TaskSpec`, `EpisodeRecord`).

## 7. Working rules

- Each component works on its own branch in `.worktrees/<branch>`. Edit only the files you own.
  The coordinator merges into `main`. Merge `main` into your branch to get updates.
- Do not edit `fbplan/contracts.py` or `third_party/`. Request contract changes in your status file.
- Status: keep `docs/status/<branch>.md` current (done, next, blockers, requests). Commit it with your work.
- Dependencies: you may add packages with `uv add`. The coordinator re-locks on merge.
- Shared data outside the repo: checkpoints in `~/.cache/fbplan/checkpoints/<medium|large|giant|teleport>/`
  (override with `FBPLAN_CKPT_DIR`); OGBench data in `~/.ogbench/data`. Run outputs in `runs/` (git-ignored).
  Final result CSVs used by the report go to `results/` (committed).
- Code quality is a hard requirement of the assignment: small typed functions, docstrings that say
  what and why, comments only where the reason is not obvious, no dead code, no notebooks in `fbplan/`.
  `uv run ruff check .` and `uv run pytest` must pass before each commit.
- Commit messages: imperative mood, one topic per commit. No Co-Authored-By lines.
- Stop and ask the user before you start long GPU jobs, download more than 1 GB,
  or change a design decision in this document.

## 8. Open items

- Environment: `uv.lock` resolves JAX 0.10.2, flax 0.12.8, distrax 0.1.9, numpy 2.4. distrax 0.1.9
  depends on `tfp-nightly` (locked to one dated build). a-frozen decides whether to keep this
  stack or pin an older stable one, after the checkpoint loads. Upstream `agents/fbpiswitch*` and
  `utils/env_utils` already import with this stack on macOS CPU (needs `np.in1d = np.isin`).
