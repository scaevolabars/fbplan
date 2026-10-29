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
  The upstream `infer_latent` averages twice, so the raw latent is about 1e-7 and the `1e-8` in
  the normalization is not negligible: upstream `|z_r|` is 0.79–0.98 `sqrt(d)` (per task values in
  `docs/status/a-frozen.md`). `infer_task_latent` returns this upstream value; our `high_latent`
  normalizes it, as the contract says. Upstream gives it to `π_h` as it is.
- One checkpoint per env (training seed 0): Drive folder `1dKYhaDJH9lUREo-kUV3AwmTLrxvKO7Ek`,
  subfolders `medium`, `large`, `giant`, `teleport`, each with `flags.json` and `params.pkl`.
  `python -m fbplan.frozen` downloads them; file ids and sha256 are pinned in `fbplan/frozen.py`.
- `params.pkl` (all 4 checked): `{'agent': flax state dict}` with `rng`, `network/step`,
  `network/params/modules_{forward_repr, backward_repr, actor, high_actor}` and `network/opt_state`.
  It pickles plain numpy arrays only (`numpy._core`), so loading does not depend on the JAX version.
  `load_numpy_pickle` refuses every other class. All 4 have the same shapes; in `flags.json` only
  `env_name` differs, and giant has discount 0.995.
- The provided checkpoints score below the paper on our re-evaluation: medium 78–80 % (5 eval
  seeds), large 31–45 % (1 eval seed). The user accepted this gap (2026-10-29): our numbers are
  the baseline, and the report states the gap. Ruled out: the restore (bit-equal), our wrappers
  (equal to upstream `evaluate` within noise), the MuJoCo version.
- MuJoCo is pinned to 3.1.6. Only 3.1.6 replays the dataset transitions exactly (qpos error 3.8e-7
  against 8.4e-3 with 3.17), so it has the physics that `π_l` learned from. A test guards this.

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
V_H(u) = max( F(u,z_r)ᵀz_r ,  max_{w∈L, w≠u, edge (u,w) trusted} R(u,w) + ρ(u,w)·V_H(w) )
```

This is SMDP value iteration over options. FB gives the option models for free.
For goal-reaching with the proxy (`R ≈ 0` away from the goal), it reduces to a shortest
path with edge cost `−log ρ(u,w)`.

For `u ≠ w`, `ρ(u,w) = E[γ^H] ≤ γ` because `H ≥ 1`, so value iteration is a γ-contraction with a
unique fixed point. A self-loop (`ρ = 1`, `R = 0`) makes every larger `V(u)` a fixed point.
Clip `ρ̂` to `[0, γ]` for `u ≠ w`.

The chain formula assumes exact hitting of `w` and Markov option policies (fixed `z_w`). In
continuous states the option ends on a ball around `w`, and the next option starts at the entry
point, so the formula is approximate (error bound in `docs/theory.md` §7).

**Why chaining can help.** FB values are accurate at short range and noisy at long range
(`γ^500 ≈ 0.007` at `γ = 0.99`). The graph joins trusted short edges into long routes.
The main risk is **wormholes**: overestimated `ρ` across walls, which the max operator exploits.
Guards: an edge threshold `ρ ≥ ρ_min`, the min over the 2 `F` heads, a cost per hop.
On medium, 5–27 % of trusted pairs are wormholes, some across the whole maze, and a higher
`ρ_min` does not remove them (they concentrate at high `ρ`; d-eval calibration). The approved
guards of approach A are in §4.

## 4. Approaches

- **A. Landmark graph planner (first).** No training. Once per env: landmarks from the dataset
  and the edge table `ρ(u,w)`. Once per task: value iteration for `V_H`. Every step:
  `w* = argmax_w R(s,w) + ρ(s,w)·V_H(w)` over trusted edges from `s`, commit to `w*` until it is
  reached, or re-plan every step (config switch), and act with `π_l(a|s,B(w*))`. With exact values
  re-planning is never worse (option interruption, Sutton–Precup–Singh 1999); commitment guards
  against noisy values. Variant: give `B(w*)` to `π_h` as its task.
  Decisions approved by the user (2026-10-29), from the tabular study and the medium probe:
  - Goal-reaching uses `R̂ = 0` (shortest path on `−log ρ̂`). FB `R̂` is a difference of two noisy
    values, and the max collects its noise. Exact and paper `R` stay as ablations.
  - An edge with `ρ̂ ≥ 1` is untrusted.
  - An edge longer than `max_edge_xy = 6` xy units is untrusted (wormhole guard). The xy position
    is part of the observation, so this guard uses no privileged data. `two_way = True`
    (`min(ρ(u,w), ρ(w,u))`) is an FB-only alternative for the report.
  - A waypoint counts as reached within `reached_xy = 3.0` xy units, with look-ahead along the
    route: the agent moves on to the next route stop when `Q(s, next) ≥ Q(s, current)` and the
    edge `s → next` is trusted. Neither change alone helps much; together they lift the planner
    from 56/100 to 79/100 on held-out medium episodes. Near a waypoint FB `ρ` peaks at about 0.9,
    so a rule `ρ ≥ ρ_reached` often fails to fire.
  - Landmark latents are single-state `B(w)` by default. Region means (`landmarks.latent=region`)
    are an option: they did not help the planner (59/100 against 68/100), although they help the
    oracle. Both are evaluated with 5 eval seeds.
  - The same rules apply to the landmark graph and to the first hop from `s`.
- **B. Learned multi-switch value (second).** Fit `V_H(s,z)` offline by expectile regression
  on `R(s,w) + ρ(s,w)·V̄_H(w,z)`, then train a new `π_h` with AWR on that advantage.
  Works for any reward and needs no graph at test time. B can distill A.
  Plan: `docs/plans/approach-b.md`, approved by the user on 2026-10-29 with one change:
  B-clean uses single-state latents `B(w)` as upstream, so that only the critic changes;
  region latents are a separate variant. Distillation of A (stage B2) waits until B1 shows a gain.

## 5. Evaluation protocol

- The baseline is re-evaluated by us on the same frozen `F`, `B`, `π_l`. We do not compare against
  the paper's numbers.
- `π_h` is retrained on the frozen modules with 5 seeds (500k steps, upstream recipe) so that the
  baseline variance includes training variance. The provided `π_h` is reported as a separate reference row.
- 5 tasks × 50 episodes × ≥ 5 eval seeds per method. The reset seed of each episode comes from
  `(eval_seed, task_id, episode)`, so all methods see the same start states (paired design).
  OGBench draws the start noise from the global NumPy RNG, and the goal also depends on the
  action-space RNG. For a reset seed `s`, this gives equal start and goal observations:
  `np.random.seed(s); env.action_space.seed(s); env.reset(seed=s, options={"task_id": t})`.
- Eval temperature 0, episode length 1000, success as defined by OGBench.
- Statistics: mean success with 95 % stratified bootstrap CI; paired test over episodes.
- Diagnostics: oracle-waypoint controller (shortest path on the true maze map, privileged). It is
  a reference, not a strict upper bound: with single-state waypoints it scores below the baseline,
  and with region-mean waypoint latents it scores 84.0 % against 77.8 % for the baseline on
  medium (+6.2 points, 5 eval seeds, paired, p = 1e-4). On medium, 71 % of baseline failures are
  the ant flipping over (a `π_l` failure), so the headroom for any high-level method is small
  there; FB distance calibration
  (`−log ρ / −log γ` against maze geodesic distance); wormhole count; route plots.

## 6. Components

| Branch / tab | Owns | Provides |
|---|---|---|
| `a-frozen` | `fbplan/frozen.py`, `tests/test_frozen.py` | `load_model`, `make_env`, `load_dataset`, checkpoint download |
| `b-theory` | `fbplan/tabular/`, `docs/theory.md`, `tests/test_tabular*.py` | exact tabular maze with true `M`; proofs; wormhole study |
| `c-planner` | `fbplan/option_model.py`, `landmarks.py`, `graph.py`, `controllers/planner.py`, `analysis/planner_routes.py` | approach A controllers |
| `d-eval` | `fbplan/evaluate.py`, `stats.py`, `controllers/{__init__,baseline,oracle}.py`, `analysis/` (except `planner_routes.py`) | eval loop, CSVs, stats, diagnostics |
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

- Baseline definition (open, user decision): `π_h` with the normalized `z_r` (our contract) or with
  the raw upstream `z_r`. On large the normalized one scored 14 points higher (one eval seed); on
  medium the difference is not significant. Proposal: normalized as the main baseline, raw upstream
  as a reference row.

Closed:

- Environment (a-frozen, 2026-10-29): keep JAX 0.10.2, flax 0.12.8, distrax 0.1.9 and the locked
  `tfp-nightly` build. The restore is exact, PyPI keeps all tfp-nightly builds, and an older stack
  gives no benefit because the pickles hold numpy arrays only. MuJoCo is pinned to 3.1.6 (§2).
