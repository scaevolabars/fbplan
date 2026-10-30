# Design: planning over sequences of intentions with frozen FB

Status: the design record of the project: task, facts, math, approaches, protocol, and every
decision with the date when it was fixed. Last update: 2026-10-30.

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
  the normalization is not negligible: upstream `|z_r|` is 0.79–0.98 `sqrt(d)`. Per task, with
  `sqrt(128)` = 11.31: medium 11.09, 11.02, 10.83, 10.46, 10.91; large 10.70, 10.68, 10.34,
  10.18, 10.18; teleport 10.49, 10.96, 10.96, 10.96, 10.49; giant 9.18, 8.96, 10.33, 8.90, 10.91.
  `infer_task_latent` returns this upstream value; our `high_latent` normalizes it, as the
  contract says. Upstream gives it to `π_h` as it is.
- One checkpoint per env (training seed 0): Drive folder `1dKYhaDJH9lUREo-kUV3AwmTLrxvKO7Ek`,
  subfolders `medium`, `large`, `giant`, `teleport`, each with `flags.json` and `params.pkl`.
  `python -m fbplan.frozen` downloads them; file ids and sha256 are pinned in `fbplan/frozen.py`.
- `params.pkl` (all 4 checked): `{'agent': flax state dict}` with `rng`, `network/step`,
  `network/params/modules_{forward_repr, backward_repr, actor, high_actor}` and `network/opt_state`.
  It pickles plain numpy arrays only (`numpy._core`), so loading does not depend on the JAX version.
  `load_numpy_pickle` refuses every other class. All 4 have the same shapes; in `flags.json` only
  `env_name` differs, and giant has discount 0.995.
- The provided checkpoints score below the paper on our re-evaluation (Linux, eval seeds 0–9):
  medium 78.4 % (`results/medium_10seeds/baseline.csv`), large 37.7 %
  (`results/large_10seeds/baseline.csv`). The gap was accepted on 2026-10-29: our numbers are
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
`ρ_min` does not remove them (they concentrate at high `ρ`; `results/calibration/`). The approved
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

## 6. Repository

Layout: `README.md`, section "Layout"; commands for every result: `scripts/reproduce.sh`.
Shared contracts: `fbplan/contracts.py` (`FBModel`, `Controller`, `TaskSpec`, `EpisodeRecord`).
Data outside the repository: the frozen checkpoints in `~/.cache/fbplan/checkpoints/<maze>/`
(override with `FBPLAN_CKPT_DIR`), trained `π_h` and critics in `~/.cache/fbplan/baselines/` and
`~/.cache/fbplan/approach_b/` (`FBPLAN_BASELINE_DIR`, `FBPLAN_APPROACH_B_DIR`), OGBench data in
`~/.ogbench/data`. Run outputs go to `runs/` (git-ignored); the CSVs and figures that the report
uses go to `results/` and `docs/figures/` (committed).

## 7. Code rules

- Code quality is a requirement of the assignment: small typed functions, docstrings that say
  what and why, comments only where the reason is not obvious, no dead code, no notebooks in
  `fbplan/`. `uv run ruff check .` and `uv run pytest` pass before each commit; CI runs both on
  CPU without data.
- `third_party/` holds the upstream code unchanged.
- A refactor must not change results: after it, a rerun of eval seed 0 of one committed config
  gives identical records.
- CLI convention of the scripts:
  1. Inputs are positional paths: CSV files, or one folder.
  2. Outputs: `--out FILE` for one file, `--out-dir DIR` for one folder. A script that writes
     figures and tables to different folders uses `--fig-dir` and `--table-dir`. A training CLI
     uses `--out-dir` for the checkpoint and `--run-dir` for the logs.
  3. Every file that a script writes has a flag, so that `reproduce.sh --smoke` can send all
     output to `runs/smoke/`.
  4. A script that writes committed results defaults to their path in `results/` or
     `docs/figures/`; other output goes to `runs/<script name>/`.
  5. `--env` for one env (with choices), `--envs` for several; `--seeds` for eval seeds,
     `--seed` for one training or sampling seed.
  6. `--workers` defaults to 4 in every script that starts processes.
  7. Every argument has a help text, and `--help` shows the defaults
     (`argparse.ArgumentDefaultsHelpFormatter`).
  8. `--method` takes an alias of `fbplan.controllers.ALIASES` or a `module:attribute` path.

## 8. Decisions and hypothesis tests

`h-exec`, `h-learn`, `h-repr` and `h-papers` name the hypothesis files in `docs/hypotheses/`;
each file ranks its hypotheses and fixes their readings before any test ran.

- Seed selection on held-out goals (`docs/plans/approach-b.md` §10, decision 8; rule fixed on
  2026-10-29 at 22:40 MSK, before any held-out result exists). For each family with several
  training seeds (retrained baseline, B clean, B wide, B region): evaluate every seed on the 10
  held-out goals of medium (`--heldout`, eval seeds 0 1 2, 20 episodes per goal and eval seed,
  Linux instance). Select the seed with the highest held-out success; ties go to the lower seed
  id. Report, on the 5 eval tasks, the selected seed of each family paired against the selected
  retrained baseline and against `baseline_raw`, next to the pooled rows. The rule applies to all
  families in the same way, and nothing else is tuned on the held-out goals. No selection on the
  eval tasks. Result: no gain by the rule (selected B clean vs `baseline_raw`: +2.2 pp [−1.0,
  +5.1]).
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
  The shared temperature is 0.2099: 0.2 times the median upstream advantage spread (16.49) over
  the median switch advantage spread (15.72), from 200 batches with a new draw of task latents per
  batch and the critics of seeds 0–4 pooled. It was computed on the GPU; the CPU and TF32 give
  different spreads (`fbplan.learned.train_b calibrate`).
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
  - **Gate G0 and wave 2** (h-papers, `docs/hypotheses/h-papers.md` §2; applied on 2026-10-30
    at 02:35 MSK). Medium: the 10-seed oracle is +1.0 pp [−1.2, +3.2] over `baseline_raw`, so
    by the fixed reading (≤ +1 pp) no route method can pass on medium: H1–H3 are dropped there.
    Large, fixed now before any result: run the oracle, `baseline` and `baseline_raw` on large
    eval seeds 5–9; the bar is the stronger of the two baseline rows on 10 seeds; apply the same
    table (≤ +1 pp: stop; +1 to +3 not significant: H1's held-out check first; significant:
    H1 is the first test). The 2× gate (10 seeds): baseline 2× − retrained +2.2 pp [+0.0, +4.4];
    B clean 2× − baseline 2× −0.6 pp: a `π_h` tuning effect, no gain of chaining.
  - **H1 on large** (h-papers H1, `docs/hypotheses/h-papers.md`; fixed on 2026-10-30 at 02:55 MSK,
    before any H1 result). G0 on large: the 10-seed oracle is +1.2 pp [−1.3, +3.8] over the bar
    (`baseline`, 37.7 %), the middle band, so H1's held-out check comes first. Selection on the
    large held-out goals only, with the grid and tie rule of h-papers H1 item 5 (executor ×
    `K`). Held-out check, on the same held-out episodes: H1 is falsified if its selected config
    scores more than 2 pp below the oracle or not above `baseline`; then large route methods
    stop. Otherwise the selected config runs on large eval seeds 0–9, paired against both
    baseline rows, and the gain rule decides. The report states that the route comes from
    recorded trajectories, not from FB values.
  - **H1 result and wave 3** (fixed on 2026-10-30 at 04:20 MSK, before any wave-3 result).
    H1 on large passed the gain rule: 41.2 % against `baseline` 37.7 % (+3.5 pp [+0.9, +6.0],
    p 0.009) and `baseline_raw` 35.5 % (+5.6 pp), eval seeds 0–9. Because many tests ran, two
    confirmatory steps follow, with the same code and the selected config (`high`, K 150):
    (R) replication on large eval seeds 10–19 (new episodes), H1 and both baseline rows on one
    platform; it confirms if H1 beats both rows there, paired, p < 0.05; the report shows seeds
    0–9, 10–19 and 0–19. (M) medium: G0 had dropped route methods on medium because the oracle
    had no room, but H1 beats the oracle on large (+2.3 pp), so the oracle does not bound H1.
    H1 on medium follows h-papers H1 item 5: the same grid and tie rule on the medium held-out
    goals, the falsification rule of 'H1 on large' against the medium oracle and `baseline`,
    then the selected config on medium eval seeds 0–9 against both rows; the gain rule decides.
    All rows are reported, whatever they show.
  - **Wave 3 results and the medium replication** (fixed on 2026-10-30 at 05:20 MSK, before any
    replication result). (R) Large seeds 10–19: H1 +1.8 pp over `baseline` (p 0.16), so the
    replication does not confirm H1 on large; pooled seeds 0–19 +2.7 pp (p 0.004). (M) Medium:
    held-out check passed (window, K 150); seeds 0–9: 83.8 % against `baseline` 78.4 % (+5.3 pp)
    and `baseline_raw` 80.5 % (+3.3 pp [+1.2, +5.4], p 0.003): H1 passes on medium. Replication,
    as for large: the same config on medium eval seeds 10–19 with `baseline` and `baseline_raw`;
    it confirms if H1 beats both rows on seeds 10–19 alone, paired, p < 0.05. The report shows
    seeds 0–9, 10–19 and 0–19 for both mazes.
    Result: confirmed on medium (seeds 10–19: +4.7 and +4.8 pp, p < 0.001).
  - **H2, FB orders the recorded route** (`docs/hypotheses/h-papers.md` H2; fixed on 2026-10-30
    at 09:40 MSK, before any H2 result). The recorded-route controller of H1 with one change:
    the weight of a certified edge is the FB cost `−log ρ̂` (the planner's convention: head min,
    `ρ̂` clipped to [0, γ] for distinct landmarks) instead of its xy length. Everything else as
    the selected H1 config of each maze: medium `window`, K 150; large `high`, K 150. No new
    selection. Held-out check first, on the same held-out episodes as H1: H2 is falsified if it
    scores more than 2 pp below H1 or not above `baseline`. If not falsified: eval seeds 0–19 on
    medium and on large, paired against `baseline`, `baseline_raw` and H1. Readings, per maze:
    (a) FB orders the sequence: H2 passes the gain rule on seeds 0–9 and on seeds 10–19 alone;
    (b) FB ordering is as good as xy ordering: H2 − H1 over seeds 0–19 has a 95 % CI above
    −2 pp; (c) FB values hurt even on certified edges: H2 − H1 significantly below 0. All rows
    are reported, whatever they show.
  - Not in wave 1: an unfrozen `π_l` (h-exec H2, H5: low chance of a high-level gain, and every
    row would need it); the `z_r` norm knob (h-repr: c = 1 is best on medium); new critics
    (h-learn H5, H6) wait for the 2× result.
- **Code versions and compute of the committed results.** Before each group of runs, a rerun of
  eval seed 0 of a committed config on the same machine repeated its records.
  - Evaluation code (Linux): 84a8127 for the 5-seed runs, approach A and its ablations, and eval
    seeds 5–9 of the baseline rows; 0be1249 for T1; 6c603a3 for T4, the 2× families, T2 and T3;
    8a202e6 for act-ahead; 5d7e642 for the recorded route on large; 2626146 for wave 3 and the
    replications. The Mac runs of approach A in `results/macos/`: 0de6e0a. Later commits changed
    the planner config, so the approach A CSVs keep the `config_hash` of 84a8127; a rerun at a
    later commit repeats their records with another hash (README, "Code versions").
  - Training code: the retrained baselines 9557a55 (medium) and a189a49 (large); the critics
    `value-single`, `value-region`, `value-single-expectile0.7` and the B policies clean, wide,
    region a189a49; `awr2x` and `clean-awr2x` 97d9200; `logratio` 34b17a5. Each checkpoint's
    `flags.json` records its commit.
  - Compute: evaluation on one rented Linux machine (AMD EPYC 7662, 128 cores, 1 TB RAM, CPU
    only) for about 9.2 h, about $8.6 with the hosts that failed at start; training on one
    rented RTX 3090 for 13.9 h, about $2.90 (5 seeds in parallel, about 55 min per family; a B
    run holds about 2.5 GB of GPU memory).

Closed:

- Baseline definition (decided on 2026-10-29): the main
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
- Held-out goals (accepted on 2026-10-29): `task_id` 101..110 identify held-out goals; only
  the `TaskSpec` docstring changes. The goal files are `results/heldout_goals/<env>.csv`
  (`analysis/heldout_goals.py`, seeded draws). A goal cell is at least 2 maze moves from every
  eval goal, its goal ball holds at least 50 validation states and at most 50 % flipped states,
  and its start cell is at the distance of an eval task. The flip limit (2026-10-30) changed 6
  of 10 large goals and 3 of 10 giant goals, before any held-out run on them; medium did not
  change. No teleport goal lies in the trap cell. Checkpoint fields in `EpisodeRecord`:
  rejected; one CSV holds one checkpoint, and `write_csv` adds its columns.
- Environment (2026-10-29): keep JAX 0.10.2, flax 0.12.8, distrax 0.1.9 and the locked
  `tfp-nightly` build. The restore is exact, PyPI keeps all tfp-nightly builds, and an older stack
  gives no benefit because the pickles hold numpy arrays only. MuJoCo is pinned to 3.1.6 (§2).
  `uv.lock` pins the tfp-nightly build (0.26.0.dev20261028); distrax 0.1.9 needs tfp-nightly,
  because tfp 0.25 does not support the new JAX. The MuJoCo pin moves dm-control from 1.0.50 to
  1.0.20, which pins an exact MuJoCo version.
- Route following (`switch_steps`, 2026-10-29): approach A can follow its route at a reach
  instead of re-planning. It is an ablation, not a default: on held-out episodes (eval seed 200,
  not used for any choice) it changed success by −2.0 pp [−9.6, +5.6] on medium and +5.3 pp
  [−5.3, +16.0] on large against the defaults.
