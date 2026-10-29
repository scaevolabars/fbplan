# Plan: approach B, a learned multi-switch value

Status: draft for approval (e-learner, 2026-10-29). Notation: docs/design.md §3.

## 1. Idea in one paragraph

The baseline `π_h` is trained with AWR on a one-switch advantage: "follow `π_{z_w}` until `w`,
then follow `π_z` forever". The value after `w` is the single-intention value `F(w,z)ᵀz`.
Approach B learns a multi-switch value `V_H(s,z)`: the value when the agent may switch again
at `w`, and again after that. Then it trains a new `π_h` with the same AWR recipe, but with
`V_H` in the advantage. The actor, the data and the controller stay as in the baseline. Only
the critic changes, so "B vs. retrained baseline" measures the value of chaining and nothing else.
No graph and no landmarks at test time, and it works for any task latent `z`.

```
 offline data ──► (s, w, g) batches ──► option model from frozen F, B ──► targets y(s,w,z)
                                                                              │
 stage 1:  V_H(s,z) ◄── expectile regression on y ◄───────────────────────────┤
 stage 2:  π_h(z_w | s,z) ◄── AWR, weight exp(α·[y − V_H(s,z)]) ◄──────────────┘
 test:     same controller as baseline: z_w ~ π_h(s, z_r) every step, a ~ π_l(s, z_w)
```

## 2. What the critic computes

Option "follow `π_{z_w}` until `w` is hit", with `z_w = B(w)`, per `F` head `h`
(the same formulas and conventions as c-planner's `fbplan/option_model.py`):

```
ρ(s,w)   = agg_h F_h(s,z_w)ᵀB(w) / F_h(w,z_w)ᵀB(w)     clipped to [0, ρ_max]
R(s,w;z) = agg_h [F_h(s,z_w)ᵀz − ρ_h(s,w)·F_h(w,z_w)ᵀz]
V_1(s,z) = agg_h F_h(s,z)ᵀz                              never switch again
```

`agg` = min over the 2 heads (pessimistic). The multi-switch value solves the SMDP Bellman
equation over these options, with `w` restricted to states that the data shows are reachable:

```
V_H(s,z) = max( V_1(s,z),  E^τ_w [ R(s,w;z) + ρ(s,w)·V_H(w,z) − c ] )
```

`E^τ_w` is the τ-expectile over sampled `w` (an in-sample soft max, as in IQL). `c ≥ 0` is an
optional cost per switch.

**Parameterization.** `V_H(s,z) = V_1(s,z) + softplus(Δ_θ(s,z))`. So `V_H ≥ V_1` by
construction (the option "never switch" always exists). With `Δ_θ` initialized near 0, `V_H`
starts at the single-intention value, and the network learns only the gain from switching.
`Δ_θ`: MLP 3×512, GELU, LayerNorm, input `(s, z)`, 2 heads, as upstream `GCValue`.

**Target and loss (stage 1).** With a target network `Δ̄` (EMA, rate 0.005):

```
y(s,w,z) = R(s,w;z) + ρ(s,w)·[V_1(w,z) + softplus(min_j Δ̄_j(w,z))] − c
L_V      = Σ_j E[ |τ − 1(u_j < 0)| · u_j² ],   u_j = y − V_1(s,z) − softplus(Δ_θ,j(s,z))
```

**Exact R, not the paper's proxy.** The paper drops `−ρ·F(w,z_w)ᵀz` from the one-switch
advantage (Sec. 6: the local term is hard to learn). For a chain this is a problem. The dropped
term is the reward that `π_{z_w}` would collect after `w`. The agent does not collect it, because
it switches at `w`, and `ρ·V_H(w)` already values the time after `w`. So the proxy values that time
twice. For non-negative rewards this is a positive bias at every hop, and the max over `w` then
prefers chains with more hops. For goal tasks the dropped term is large only when `w` is near the
goal, so the bias sits on the last hop. Default: exact `R`. Ablation: the proxy. b-theory measures
both on the exact maze.

## 3. Sampling

All from offline data; same sampler as `train_high` (`fbplan/learned/sampling.py`), on the device.

| Variable | Distribution | Why |
|---|---|---|
| `s` | uniform over dataset states | as upstream |
| `w` | same trajectory as `s`, offset `k ~ Geometric(1−γ)`, clipped at the trajectory end (mean 100 steps) | as upstream `π_h` subgoals. Every edge `(s,w)` is shown by the data, so the targets contain no wormhole edges |
| `g` | 70 %: uniform future state of the same trajectory (upstream); 30 %: uniform dataset state | random goals need chaining across trajectories; the value has to cover them |
| `z` | 50 %: `B(g)`; 50 %: Gaussian (as upstream `critic_latent_mix_prob = 0.5`) | goals for the eval tasks; Gaussian `z` keeps the zero-shot property for any reward |

All `z` are normalized to norm `√d` before the dot products, because the test-time `z_r` is
normalized. (Upstream uses raw `B(g)` in the dot products, so its advantage scale varies with
`‖B(g)‖`.) Option, off by default: a "goal region" `z` = normalized mean of `B` over 5 neighbours
of `g` in its trajectory. This is closer to the test-time `z_r = mean(r·B(s))` over the goal ball.

Stitching across trajectories comes from generalization of `V_H(w,z)` at `w`, as in IQL/HIQL.
Cross-trajectory `w` comes in only through the distillation of A (§6).

## 4. The new π_h (stage 2)

Same network as upstream (`GCActor`, constant std, input `(s, z)`, output a latent in `R^d`).
AWR toward the normalized `B(w)`, with `V_H` frozen:

```
A(s,w,z) = y(s,w,z) − V_H(s,z)
L_π      = −E[ exp(min(α·A, 5)) · log π_h(normalize(B(w)) | s, z) ]
```

Start with `α = 0.1` (upstream `high_alpha`). `A` has the same units, but the scale can differ,
so we compare the spread of `α·A` with the upstream run before training (logged).
Two variants of the `g` distribution for `π_h`:
**B-clean** uses the upstream goal distribution, so only the critic changes.
**B-wide** adds the 30 % random goals. We report both.

The checkpoint has the layout of a provided one (`params.pkl` with a new `modules_high_actor`),
so d-eval evaluates it with the baseline controller and `load_model(ckpt_dir=...)`.

## 5. Guards against error growth over hops

1. **Contraction.** `ρ ≤ ρ_max = 0.9`. Then the hop-level Bellman operator is a contraction
   with factor 0.9, so a per-target error ε grows at most to ε/(1 − 0.9) = 10ε. One-step TD on
   the same value would allow ε/(1 − γ) = 100ε. The cap lowers the value of very short hops only;
   the "never switch" option and the per-step re-selection make short hops unnecessary.
2. **Data-supported edges.** `w` is always on the trajectory of `s`, so `ρ(s,w)` is never
   queried across a wall that the data does not cross.
3. **Pessimistic option model.** min over the 2 `F` heads for `ρ` and `R`; clip rates logged.
4. **Soft max.** Expectile `τ = 0.9`, not a hard max over `w`. Optional cost per switch `c`.
5. **Clipped double target.** 2 `Δ` heads, min in the target, EMA target network.
6. **Floor at the baseline.** The residual form keeps `V_H ≥ V_1`; the gain is learned only
   where switching helps on the data.
7. **Fallback if values still drift:** an upper clip of `y` at the largest value seen at the
   goal, `V_1(g,z_g)`. Not on by default.

Per-step re-selection at test time is safe: if the option values are correct, interrupting an
option for a better one never lowers the return (Sutton, Precup & Singh, 1999, interruption
theorem). The baseline controller also re-selects every step.

## 6. Distilling A's routes (stage B2, after B1 works)

A gives, for landmark pairs `(u, g)`, a graph value `V_A(u; g)` and a next waypoint `w*(u, g)`.
Two additions to B, each with its own weight:

- **Value:** `λ_V · E_{u,g}[(V_H(u, B(g)) − V_A(u; g))²]` on landmark pairs whose route uses only
  trusted edges. This gives `V_H` long-range supervision where trajectories rarely connect.
- **Routes:** add triples `(u, w*(u,g), B(g))` to the AWR batch of `π_h`. Their advantage is
  computed with `V_H` as for data triples. This lets `w` lie on another trajectory, which B1
  cannot do.

The risk is that B inherits A's wormholes. Guard: use only pairs whose route `ρ` values pass A's
`ρ_min` and where B1's `V_H` and `V_A` do not disagree by more than a set margin.
Needs from c-planner: landmark observations, `V_A` for landmark goals, `w*` table.

## 7. Checks before and after training

Unit tests (`tests/test_learned_*.py`):
- the JAX option targets equal c-planner's numpy `OptionModel` on b-theory's fake FB;
- `ρ(w,w) = 1` before the cap, invariance to the scale of `B(w)`, `V_H ≥ V_1`;
- on the exact tabular maze (b-theory), fitting a tabular `V_H` with these targets converges to
  the exact SMDP value over the same options;
- frozen modules stay bit-identical; AWR weights finite; checkpoint loads in the baseline layout.

Offline diagnostics, no rollouts (with d-eval's calibration tools):
- `V_H(s, B(g))` and `V_1(s, B(g))` against maze geodesic distance. Expected: `V_1` collapses
  at long range, `V_H` decays smoothly;
- the gain `V_H − V_1` against distance; `ρ` clip rates; agreement with A's `V_A` on landmarks.

Evaluation (d-eval, design.md §5): B-clean and B-wide against the 5 retrained baseline seeds
(paired), A, and the oracle ceiling, on medium and large.
Diagnostic controller **B-argmax**: A's controller with `V_H` in place of graph value iteration
(`w* = argmax_{w∈L} R + ρ·V_H(w)`). It separates "critic quality" from "actor distillation".
Ablations: exact vs. proxy `R`; `ρ_max ∈ {0.9, 1.0}`; `τ ∈ {0.7, 0.9}`; residual vs. plain `V_H`;
with and without distillation.

**Hyperparameter selection** uses held-out goals (dataset states far from the 5 eval goals,
success by OGBench's goal radius), never the 5 eval tasks. Their `z` is the mean of `B` over dataset
states within the goal radius, as `infer_task_latent` does. This needs a custom-goal option in
d-eval's loop (request after approval).

## 8. Cost, code, dependencies

- GPU: one stage step calls `F` about 4 times per batch, as the upstream `π_h` step does.
  So each stage costs about one baseline run, and B costs about 2 baseline runs per (env, seed).
  10 B runs (5 seeds × medium, large) ≈ 20 baseline runs. I ask before each GPU job.
- Code (all in my area): `fbplan/learned/option_targets.py` (JAX ρ, R, V_1 per pair),
  `fbplan/learned/value.py` (V_H, expectile loss), `fbplan/learned/high_policy.py` (AWR),
  `fbplan/learned/train_b.py` (CLI, both stages), `fbplan/learned/controller.py` (B-argmax),
  tests in `tests/test_learned_*.py`.
- Needs: a-frozen `load_agent` / `load_model(ckpt_dir)` (already requested); b-theory fake FB
  and exact maze (merged on its branch); c-planner landmarks and `V_A` / `w*` tables (B2 only).

## 9. Risks

- **Little headroom on medium.** Baseline 87 %, HIQL 93 %. The paper says FB + `π_h` gains are
  small because `π_l` limits them. The oracle-waypoint ceiling (d-eval) tells us the headroom.
  Large (66 %) has more, so B runs on large too.
- **Actor averaging.** `π_h` regresses onto `B(w)`; if good subgoals lie in two directions, the
  mean lies between them. B-argmax shows whether the critic or the actor is the limit.
- **Test-time `z_r` differs from training `z`.** `z_r` is an average of `B` over the goal ball.
  The "goal region" option in §3 addresses this.

## 10. Decisions to approve

1. Exact `R` by default; the paper's proxy only as an ablation (§2).
2. Residual `V_H = V_1 + softplus(Δ)` (§2).
3. `w` only from the trajectory of `s` in B1; other `w` only through A in B2 (§3, §6).
4. Two stages; same `π_h` network and controller as the baseline (§4).
5. Report B-clean (only the critic changes) and B-wide (§4).
6. Cap `ρ ≤ 0.9` (§5.1). This deviates from A, which clips to [0, 1].
7. Hyperparameters chosen on held-out goals, not on the eval tasks (§7).
