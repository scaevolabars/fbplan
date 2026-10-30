# Plan: approach B, a learned multi-switch value

Status: approved by the user on 2026-10-29, with one change: B-clean keeps the upstream
single-state latents, and region latents are a separate variant, B-region (§3, §10).
Stage B2 waits until B1 shows a gain. Notation: docs/design.md §3.

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
(the same formulas and conventions as approach A's `fbplan/option_model.py`):

```
ρ(s,w)   = agg_h F_h(s,z_w)ᵀB(w) / F_h(w,z_w)ᵀB(w)     clipped to [0, ρ_cap]
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

**Which R.** `fbplan/option_model.py` offers three: exact, the paper's proxy, and zero.

- The proxy is wrong for a chain. It drops `−ρ·F(w,z_w)ᵀz`, the reward that `π_{z_w}` would
  collect after `w`. The agent does not collect it, because it switches at `w`, and `ρ·V_H(w)`
  already values the time after `w`. So the proxy values that time twice. For non-negative
  rewards this is a positive bias at every hop, and the max over `w` then prefers more hops.
- Approach A uses `R̂ = 0` for goal-reaching (approved, design §4): exact `R̂` is a difference of
  two noisy FB values, and a max over `w` collects the noise. The expectile in B is a softer max,
  but the same noise enters every target. For a goal latent the true `R` is close to 0, unless
  the path to `w` passes the goal.
- For a Gaussian latent (a general reward), the reward before `w` is not small, so `R = 0` would
  make `V_H` wrong there.

Default: `R = 0` for goal latents, exact `R` for Gaussian latents (code: `edge_reward="goal_zero"`;
`"exact"` and `"paper"` are the ablations). The name differs from A's `zero` mode on purpose: A
plans for one goal at a time, so `zero` applies to every edge; B trains on goal and Gaussian
latents in one batch, and `goal_zero` sets `R = 0` only for the goal latents. The eval tasks have goal
latents, so for them B uses the same estimator as A. Ablations: exact `R` for all latents; the
proxy. The exact tabular maze (`fbplan/tabular/`) measures all three.

## 3. Sampling

All from offline data; same sampler as `train_high` (`fbplan/learned/sampling.py`), on the device.

| Variable | Distribution | Why |
|---|---|---|
| `s` | uniform over dataset states | as upstream |
| `w` | same trajectory as `s`, offset `k ~ Geometric(1−γ)`, clipped at the trajectory end (mean 100 steps) | as upstream `π_h` subgoals. Every edge `(s,w)` is shown by the data, so the targets contain no wormhole edges |
| `g` | 70 %: uniform future state of the same trajectory (upstream); 30 %: uniform dataset state | random goals need chaining across trajectories; the value has to cover them |
| `z` | 50 %: `B(g)`; 50 %: Gaussian (as upstream `critic_latent_mix_prob = 0.5`) | goals for the eval tasks; Gaussian `z` keeps the zero-shot property for any reward |

`B` is frozen, so we compute it once for all dataset states (`[N, 128]`, 512 MB for medium) and
look latents up; no `B` call per step.

**Variants.** Each variant changes one thing against the one before it:

| Variant | Subgoal and goal latents | Goal distribution of `π_h` | What it measures |
|---|---|---|---|
| retrained baseline | `B(·)` of one state | upstream | reference |
| **B-clean** | `B(·)` of one state, as upstream | upstream | only the critic changes: the gain from chaining |
| **B-wide** | as B-clean | + 30 % random goals | training on goals across trajectories |
| **B-region** | region latents | as B-clean | the latent type |

In the code the latent kinds are `--latents single` (one state's `B`) and `--latents region`
(the window below). Approach A's `fbplan/landmarks.py` calls the first kind `state`; the B names
stay, because they are part of the trained folder names (`value-single`, `value-region`), which
are the release keys.

The region latent of a state is the mean of `B` over the states within ±5 steps of it on its
trajectory (cumulative sums over the `B` table make this cheap). Reasons to try it: the
oracle controller (`fbplan/controllers/oracle.py`) scored about 4 points above the baseline
with region-mean waypoint latents and below it with single-state ones, with this `π_l` (design
§5, one eval seed); and the test-time `z_r = mean(r·B(s))` is itself a mean over the goal ball.
In B-region, `z̃_w` replaces `z_w` everywhere in §2, so `ρ` becomes the discount to reach the
region around `w`, and B-region needs its own `V_H`. B-clean and B-wide share one `V_H`.

All `z` are normalized to norm `√d` before the dot products, because `high_latent` normalizes the
test-time `z_r` (design §2). Upstream uses raw `B(g)` in the dot products, so its advantage scale
varies with `‖B(g)‖`.

Stitching across trajectories comes from generalization of `V_H(w,z)` at `w`, as in IQL/HIQL.
Cross-trajectory `w` comes in only through the distillation of A (§6).

## 4. The new π_h (stage 2)

Same network as upstream (`GCActor`, constant std, input `(s, z)`, output a latent in `R^d`).
AWR toward the normalized subgoal latent (`B(w)`, or `z̃_w` in B-region), with `V_H` frozen:

```
A(s,w,z) = y(s,w,z) − V_H(s,z)
L_π      = −E[ exp(min(α·A, 5)) · log π_h(normalize(B(w)) | s, z) ]
```

Start with `α = 0.1` (upstream `high_alpha`). `A` has the same units, but the scale can differ,
so we compare the spread of `α·A` with the upstream run before training (logged).
The variants are in §3: B-clean (only the critic changes), B-wide, B-region.

The checkpoint has the layout of a provided one (`params.pkl` with a new `modules_high_actor`),
so the evaluation (`fbplan/evaluate.py`) runs it with the baseline controller, loaded as
`fbplan.learned.checkpoint.load_baseline_model` loads a retrained baseline.

## 5. Guards against error growth over hops

1. **Contraction.** `ρ ≤ ρ_cap = 0.9` (code: `OptionConfig.rho_cap`, CLI `--rho-cap`). Then the hop-level Bellman operator is a contraction
   with factor 0.9, so a per-target error ε grows at most to ε/(1 − 0.9) = 10ε. Design §3 clips
   `ρ̂` to `[0, γ]` for A, which gives γ-contraction and allows ε/(1 − γ) = 100ε, the same as
   one-step TD. FB `ρ̂` near a waypoint peaks at about 0.9 on medium (design §4), so the tighter
   cap changes few targets; it makes the bound explicit. Samples with raw `ρ̂ ≥ 1` are dropped
   from the loss, as A treats such edges as untrusted. The cap lowers the value of very short
   hops only; the "never switch" option and the per-step re-selection make them unnecessary.
   Measured on medium after approval (validation split, 4096 in-trajectory pairs): `ρ` median
   0.78, raw `ρ ≥ 1` on 9.7 %, `ρ > 0.9` on 33 %. So the cap binds on a third of the pairs,
   mostly short hops, not on few targets as said above. It stayed as approved; the ablation
   `ρ_cap ∈ {0.9, 1.0}` (§7) was to test it (§11). Exact `R` is noisy (10–90 %: −8.8 to +29),
   which supports `R = 0` for goals (§2). The one-hop gain with `R = 0` is positive on 26 % of
   pairs.
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

## 6. Distilling A's routes (stage B2, only after B1 shows a gain)

A scores below the baseline at present, so B2 waits until B1 gives a gain over the retrained
baseline (user decision, 2026-10-29).

A gives, for landmark pairs `(u, g)`, a graph value `V_A(u; g)` and a next waypoint `w*(u, g)`.
Two additions to B, each with its own weight:

- **Value:** `λ_V · E_{u,g}[(V_H(u, B(g)) − V_A(u; g))²]` on landmark pairs whose route uses only
  trusted edges. This gives `V_H` long-range supervision where trajectories rarely connect.
- **Routes:** add triples `(u, w*(u,g), B(g))` to the AWR batch of `π_h`. Their advantage is
  computed with `V_H` as for data triples. This lets `w` lie on another trajectory, which B1
  cannot do.

The risk is that B inherits A's wormholes. Guard: use only pairs whose route `ρ` values pass A's
`ρ_min` and where B1's `V_H` and `V_A` do not disagree by more than a set margin.
Needs from approach A (`fbplan/landmarks.py`, `fbplan/graph.py`): landmark observations, `V_A`
for landmark goals, `w*` table.

## 7. Checks before and after training

Unit tests (`tests/test_learned_*.py`):
- the JAX option targets equal the numpy `OptionModel` (`fbplan/option_model.py`) on the fake FB
  of `fbplan/tabular/fake_fb.py`;
- `ρ(w,w) = 1` before the cap, invariance to the scale of `B(w)`, `V_H ≥ V_1`;
- on the exact tabular maze (`fbplan/tabular/`), fitting a tabular `V_H` with these targets
  converges to the exact SMDP value over the same options;
- frozen modules stay bit-identical; AWR weights finite; checkpoint loads in the baseline layout.

Offline diagnostics, no rollouts (with the calibration tools of `analysis/calibration.py`):
- `V_H(s, B(g))` and `V_1(s, B(g))` against maze geodesic distance. Expected: `V_1` collapses
  at long range, `V_H` decays smoothly;
- the gain `V_H − V_1` against distance; `ρ` clip rates; agreement with A's `V_A` on landmarks.

Evaluation (`fbplan/evaluate.py`, design.md §5): B-clean, B-wide and B-region against the 5
retrained baseline seeds (paired), A, and the oracle ceiling, on medium and large.
Diagnostic controller **B-argmax**: A's controller with `V_H` in place of graph value iteration
(`w* = argmax_{w∈L} R + ρ·V_H(w)`). It separates "critic quality" from "actor distillation".
Ablations: exact vs. proxy `R`; `ρ_cap ∈ {0.9, 1.0}`; `τ ∈ {0.7, 0.9}`; residual vs. plain `V_H`;
with and without distillation (B2 only).

**Hyperparameter selection** uses held-out goals (dataset states far from the 5 eval goals,
success by OGBench's goal radius), never the 5 eval tasks. Their `z` is the mean of `B` over dataset
states within the goal radius, as `infer_task_latent` does. This needs a custom-goal option in
the evaluation loop (`fbplan/evaluate.py`).

## 8. Cost, code, dependencies

- GPU: one stage step calls `F` about 4 times per batch, as the upstream `π_h` step does.
  So each stage costs about one baseline run (measured on an RTX 3090: 5 runs in parallel take
  about 50 min). Per env and 5 seeds: one shared `V_H` + B-clean + B-wide `π_h`, and
  `V_H` + `π_h` for B-region, about 5 baseline-run batches. I ask before each GPU job.
- Code (all in my area): `fbplan/learned/option_targets.py` (JAX ρ, R, V_1 per pair),
  `fbplan/learned/value.py` (V_H, expectile loss), `fbplan/learned/high_policy.py` (AWR),
  `fbplan/learned/train_b.py` (CLI, both stages), `fbplan/learned/controller.py` (B-argmax),
  tests in `tests/test_learned_*.py`.
- Needs: `load_model(env).agent` and `load_dataset` of `fbplan/frozen.py`; the fake FB and the
  exact maze of `fbplan/tabular/`; approach A's landmarks and `V_A` / `w*` tables (B2 only).

## 9. Risks

- **Little headroom on medium.** The provided checkpoint scores 78–80 % on medium in our
  re-evaluation, and 70–80 % of its failures are the ant flipping over, a `π_l` failure that no
  high-level method fixes (design §2, §5). Large (31–45 %) has more headroom, so the main
  evidence for B will likely come from large; medium is required and is reported.
- **Actor averaging.** `π_h` regresses onto `B(w)`; if good subgoals lie in two directions, the
  mean lies between them. B-argmax shows whether the critic or the actor is the limit.
- **Test-time `z_r` differs from training `z`.** `z_r` is an average of `B` over the goal ball.
  B-region (§3) tests whether region latents reduce the gap.

## 10. Decisions (approved 2026-10-29)

1. `R = 0` for goal latents (as A), exact `R` for Gaussian latents; the proxy only as an
   ablation (§2). Approved.
2. Residual `V_H = V_1 + softplus(Δ)` (§2). Approved.
3. `w` only from the trajectory of `s` in B1; other `w` only through A in B2 (§3, §6). Approved.
4. Changed by the user: B-clean keeps single-state latents `B(·)`, exactly as upstream, because
   region latents also change the target of `π_h`, and a gain could then not be assigned to
   chaining. Region latents (±5 trajectory steps) are the separate variant B-region (§3).
5. Two stages; same `π_h` network and controller as the baseline (§4). Approved.
6. Follows from 4: B-clean = upstream latents and upstream goal distribution, only the critic
   changes; B-wide as planned (§3).
7. Cap `ρ ≤ 0.9`, tighter than A's `γ`, and drop samples with raw `ρ̂ ≥ 1` (§5.1). Approved.
8. Hyperparameters chosen on held-out goals, not on the eval tasks (§7). Approved.
9. B2 (distillation of A) only after B1 shows a gain (§6).

## 11. Runs that did not take place

B stopped at the gate of decision 9 on 2026-10-29: B-clean gave no significant gain over the
retrained baselines (`REPORT.md` §5, §7.1). B-argmax and B2 (§6) did not run. Three planned runs
did not either:

- **`ρ_cap = 1.0`** (the §5.1 ablation; value stage, medium, 5 seeds, `--rho-cap 1.0 --name
  value-single-rhocap1.0`): queued, and cancelled at the gate before it trained.
- **A policy on the expectile 0.7 critic:** the critic trained (5 seeds,
  `value-single-expectile0.7`), but it orders states no better than `V_1` (`REPORT.md` §8.5), so
  the policy did not run.
- **The attribution control of H1 in `docs/hypotheses/h-learn.md`** (B-clean with `--critic v_1`
  at the shared 2× temperature, `scripts/gpu/awr2x_v1.sh`): started on 2026-10-30 and cancelled at
  02:12 MSK, after the 2× result (baseline 2× 80.9 %, B-clean 2× 80.2 %): there was no B gain to
  attribute. No model of it was pulled or evaluated.
