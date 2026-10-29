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
| `a-frozen` | `fbplan/frozen.py`, `tests/test_frozen.py` | `load_model`, `make_env`, `load_dataset`, checkpoint download (frozen and trained), clean-clone check |
| `b-theory` | `fbplan/tabular/`, `docs/theory.md`, `tests/test_tabular*.py`, `fbplan/controllers/oracle_teleport.py`, `results/{giant,teleport}/` | exact tabular maze with true `M`; proofs; wormhole study; giant and teleport runs |
| `c-planner` | `fbplan/option_model.py`, `landmarks.py`, `graph.py`, `controllers/planner.py`, `analysis/planner_routes.py`, `results/*/planner_ablations/` | approach A controllers and ablations |
| `d-eval` | `fbplan/evaluate.py`, `stats.py`, `maze.py` (maze geometry, new), `controllers/{__init__,baseline,oracle}.py`, `analysis/` (except `planner_routes.py`, `report_figures.py`), `tests/test_evaluate.py`, `test_stats.py`, `test_eval_controllers.py`, `test_analysis.py` | eval loop, CSVs, stats, diagnostics |
| `e-learner` | `fbplan/train_high.py`, `fbplan/learned/`, `scripts/gpu/` | retrained baseline seeds; approach B |
| `f-report` | `REPORT.md`, `docs/figures/report/`, `analysis/report_figures.py` | the report |
| `g-release` | `README.md`, `scripts/reproduce.sh`, `devbox.json`, `.github/`, `pyproject.toml` (except dependencies added by owners), `tests/conftest.py` | reproduce entry point, CI, code review, process files |
| `h-exec`, `h-learn`, `h-repr` | `docs/hypotheses/<branch>.md` | ranked hypotheses for a gain (brief `docs/briefs/hypotheses.md`) |
| coordinator | `fbplan/contracts.py`, `docs/design.md`, `docs/briefs/` | contracts, merges, publishing; no implementation work |

Tests belong to the owner of the code they test (`tests/test_planner.py`, `test_graph.py`: c-planner;
`tests/test_learned*.py`: e-learner). For review findings K1–K4 (2026-10-29), a-frozen may edit
`fbplan/contracts.py` and `tests/test_contracts.py`.

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
- Do not add a git remote and do not push. The coordinator publishes the repository.
- Evaluations share one 12-core Mac: use at most 4 workers per run unless the user approves more.
- Decisions go to the coordinator, not to the user (the user delegated them on 2026-10-29).
  Stop and ask the coordinator before you start a GPU job, download more than 1 GB, or change a
  design decision in this document. Write the question in your status file, end your message
  with a line that starts with `QUESTION FOR COORDINATOR:`, and end your turn. Do not open a
  question dialog: a dialog blocks the tab, and the coordinator cannot answer it. The
  coordinator answers with a message in your tab. Only public uploads (releases) and actions
  that the permission system reserves for the user still go to the user.

## 8. Open items

- Seed selection on held-out goals (plan B decision 8; rule fixed on 2026-10-29 at 22:40 MSK,
  before any held-out result exists). For each family with several training seeds (retrained
  baseline, B clean, B wide, B region): evaluate every seed on the 10 held-out goals of medium
  (`--heldout`, eval seeds 0 1 2, 20 episodes per goal and eval seed, Linux instance). Select the
  seed with the highest held-out success; ties go to the lower seed id. Report, on the 5 eval
  tasks, the selected seed of each family paired against the selected retrained baseline and
  against `baseline_raw`, next to the pooled rows. The rule applies to all families in the same
  way, and nothing else is tuned on the held-out goals. No selection on the eval tasks.
  Result: no gain by the rule (selected B clean vs `baseline_raw`: +2.2 pp [−1.0, +5.1]).
- AWR sharpness at 2× (fixed on 2026-10-29 at 23:00 MSK, before any result). Motivation: the B
  clean seeds split by their AWR temperature, which each seed calibrates from one noisy draw;
  sharper seeds score higher on the eval tasks and on the held-out goals. Experiment: retrain
  both families on medium with the AWR weighting twice as sharp as upstream, 5 seeds each: the
  baseline `π_h` with `high_alpha = 0.2`; B clean with its temperature calibrated, from one stable
  estimate shared by all seeds, to the same spread of `α·A` as that baseline. Evaluate on Linux,
  5 eval seeds, pooled over training seeds. Reading: B at 2× beats the baseline at 2× and
  `baseline_raw` → chaining adds value at matched sharpness; both rise together → the gain is a
  `π_h` tuning effect, reported as such and not as a gain of chaining; neither → B stays
  negative. No other change and no tuning on the eval tasks.
  Addition (2026-10-30, 00:20 MSK, before any result): the control H1 of
  `docs/hypotheses/h-learn.md` (B clean at the same 2× temperature with `V_H = V_1`). If B 2× ≈ H1,
  the critic adds nothing and any rise is a temperature effect.
- Wave 1 of the hypothesis tests (fixed on 2026-10-30 at 00:20 MSK, before any result; sources
  `docs/hypotheses/{h-exec,h-learn,h-repr}.md`).
  - **Final protocol for any candidate:** eval seeds 0–9 (10 seeds × 5 tasks × 50 episodes),
    paired against `baseline` and `baseline_raw` on the same 10 seeds; the gain rule of this
    section applies. Reason: h-exec estimates that 5 seeds need about +4 pp to pass; h-repr finds
    that `baseline_raw` does not beat `baseline` on held-out goals (−0.7 pp), so its +3.8 pp on the
    eval tasks may be noise. Both baseline rows and every family already evaluated get eval
    seeds 5–9 added.
  - **T1, test-time ensemble** (h-learn H3): `z = normalize(mean_k normalize(π_h^k(s, z_r)))` over
    the 5 seeds of a family (retrained, B clean, B region). No training, no tuning. Reading: an
    ensemble that passes the gain rule is reported as test-time aggregation of `π_h` seeds; it is
    a gain of chaining only if the B ensemble also beats the retrained ensemble.
  - **T2, log-ratio advantage** (h-learn H2): retrain the baseline `π_h` (5 seeds) with the
    log-ratio proxy advantage of h-learn H2, upstream recipe otherwise. Reading: a pass is a `π_h`
    training improvement, not a gain of chaining. Choices (00:40 MSK, before any result):
    temperature 2.45 for the log-ratio part (matches upstream's effective sample size at 0.1, so
    only the shape of the advantage changes); upstream's 0.1 for the Gaussian-latent part;
    `ρ` per head clipped to [0, 0.99], head mean taken before the log. No temperature grid.
  - **T3, time-aware trap avoidance on teleport** (h-exec H1): penalise the data-detected trap in
    the intention only where the remaining time fits a walk; radius and weight chosen on teleport
    held-out goals only. Reading: a pass is a gain of a sequence-aware high level over the
    single-switch baseline on teleport.
  - **T4, both input paths for every family** (fixed on 2026-10-30 at 01:20 MSK, after T1 and
    before any T4 result). T1 used the normalized `z_r` for the families; `baseline_raw` uses the
    raw upstream `z_r` (upstream `sample_actions`). T4 completes the 2×2 design (family × input
    path): the T1 ensembles of the retrained, B clean and B region families, evaluated through
    the raw input path, eval seeds 0–9. Reading: a gain if a B ensemble through the raw path
    beats both baseline rows (paired, significant); chaining if it also beats the retrained
    ensemble through the raw path. Robustness check, no selection: the T1 and T4 ensembles on the
    10 medium held-out goals. All T4 rows are reported, whatever they show.
  - Not in wave 1: an unfrozen `π_l` (h-exec H2, H5: low chance of a high-level gain, and every
    row would need it); the `z_r` norm knob (h-repr: c = 1 is best on medium); new critics
    (h-learn H5, H6) wait for the 2× result.

Closed:

- Baseline definition (2026-10-29, decided by the coordinator on the user's delegation): the main
  baseline gives `π_h` the normalized `z_r`, the same input path as every other method, including
  approach B and the retrained baselines. `baseline_raw` (upstream `sample_actions` with the raw
  `z_r`) is a reference row. Evidence, 5 eval seeds, paired: medium raw − normalized = +2.6 pp
  [−0.5, +5.8], p = 0.11; large 38.1 % against 38.0 %. The earlier 14-point gap on large came from
  one eval seed on the old MuJoCo. Rule for the report: a gain counts only if it also holds
  paired against `baseline_raw`.
  Update with the canonical Linux runs (2026-10-29): on medium `baseline_raw` is stronger,
  +3.8 pp [+0.7, +6.8] over the normalized baseline; on large +2.3 pp the other way (not
  significant); on teleport −3.5 pp [−6.2, −1.0]. So no single row is the strongest everywhere.
  The normalized baseline stays the controlled reference (same input path as B and the retrained
  baselines); the report shows both rows in every table, and the rule above makes the stronger one
  the bar. Against `baseline_raw` the privileged oracle gains only +0.5 pp on medium.
- Held-out goals (d-eval proposal H, accepted): `task_id` 101..110 identify held-out goals; only
  the `TaskSpec` docstring changes. Checkpoint fields in `EpisodeRecord` (proposal C): rejected;
  one CSV holds one checkpoint, and `write_csv` adds its columns.
- Environment (a-frozen, 2026-10-29): keep JAX 0.10.2, flax 0.12.8, distrax 0.1.9 and the locked
  `tfp-nightly` build. The restore is exact, PyPI keeps all tfp-nightly builds, and an older stack
  gives no benefit because the pickles hold numpy arrays only. MuJoCo is pinned to 3.1.6 (§2).
