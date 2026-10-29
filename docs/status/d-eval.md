# Status: d-eval

Last update: 2026-10-29, 16:30. Branch merged with `main` at 30b4b44 (all branches, e-learner B1).
Large runs and the retrained baselines are in progress (see Next).

## Done

- `fbplan/evaluate.py`: episode loop, `make_tasks`, `run_episodes`, `run_parallel` (spawn
  workers), `write_csv`, CLI `python -m fbplan.evaluate --method <alias|module:attr>
  [--opt key=value ...] [--checkpoint ...] [--heldout]`. Reset seed =
  `SeedSequence([eval_seed, task_id, episode])`, then the reset recipe of design.md §5.
- Checkpoints (new): `--checkpoint provided | retrained:<seed> | b:<variant>:<seed>` loads the
  pi_h through `load_model`, `train_high.load_baseline_model` or `train_b.load_b_model`. Every CSV
  row records it in `checkpoint_kind`, `checkpoint_variant`, `checkpoint_seed` (provided: `""`,
  `-1`). `load_results` fills these defaults for older CSVs. `row_labels` names rows
  `baseline[retrained]/seed3` or `baseline[b-clean]/seed0`; `tables.py --cluster checkpoint_seed`
  pools the training seeds as a cluster level.
- Held-out goals (new, plan B decision 8): `results/heldout_goals/<env>.csv` for medium, large and
  giant, 10 goals each, task ids 101..110, maps in `results/figures/heldout_goals_<env>.png`.
  Rule (`select_heldout_goals`, seeded, a test checks that the files equal its output): goal
  cells at least 2 maze moves from every eval goal; start cell drawn among the cells whose
  distance to the goal is in the range of the eval tasks (medium 6–10 cells, large 12–19,
  giant 17–30). The goal is the cell centre, as in OGBench tasks; success uses OGBench's goal
  radius (0.5). `z` = `goal_latent`: the same upstream relabel and `infer_latent` call as
  `infer_task_latent` (a test checks bit equality on an eval task). Reset uses OGBench's
  `options={"task_info": ...}`. Run: `--heldout` (all 10) or `--tasks 101 ...`.
- `fbplan/stats.py`: stratified bootstrap CI (optional cluster level), paired comparison
  (bootstrap + exact McNemar), unpaired comparison, tables; `load_results(paths, label_root)`
  names rows by CSV path; `row_labels`, `pool_label`. Fix: config hashes such as `58e23249` were
  read as floats; text columns are now read as text.
- `fbplan/controllers/`: `BaselineController`, `RawBaselineController` (new, `baseline_raw`:
  upstream `sample_actions` with the raw `z_r`), `NonHierController`, `OracleWaypointController`,
  `MazeGeometry`.
- `analysis/`: `tables.py`, `plots.py` (both with `--label-root`, `--cluster`), `parity.py`,
  `failures.py` (now replays the recorded checkpoint and held-out goals), `calibration.py`,
  `heldout_goals.py` (new).
- Tests: 295 pass, ruff clean. Real-model tests skip without the checkpoint, the val split or
  retrained seed 0.

## Results on medium (5 eval seeds × 5 tasks × 50 episodes = 1,250 episodes per run)

CSVs in `results/medium/`; planner variants in `planner/`, `planner_high/`, `planner_region/`,
one CSV per landmark set (`landmarks.seed` 0, 1, 2), each crossed with all 5 eval seeds.
Planner rows pool the 3 landmark sets (cluster bootstrap over config_hash). Figure:
`results/figures/success_antmaze-medium-navigate-v0.png`. Command:
`uv run python analysis/tables.py results/medium/{baseline,baseline_raw,nonhier,oracle}.csv
results/medium/planner*/*.csv --label-root results/medium --cluster config_hash`.

| method | task 1 | task 2 | task 3 | task 4 | task 5 | all [95 % CI] |
|---|---|---|---|---|---|---|
| baseline (provided π_h, normalized z_r) | 81.6 | 84.4 | 76.0 | 62.8 | 84.4 | 77.8 [75.6, 80.1] |
| baseline_raw (upstream sample_actions, raw z_r) | 81.6 | 90.4 | 77.2 | 64.4 | 88.8 | 80.5 [78.3, 82.6] |
| non-hier (π_l with z_r) | 76.4 | 85.6 | 74.8 | 6.4 | 80.8 | 64.8 [62.7, 66.8] |
| oracle (privileged; region r 0.5, lookahead 2) | 81.6 | 88.4 | 88.8 | 74.8 | 86.4 | 84.0 [82.0, 86.0] |
| planner (π_l, single-state latents) | 65.7 | 75.9 | 75.6 | 56.1 | 77.2 | 70.1 [67.8, 72.3] |
| planner_high (π_h after the waypoint) | 68.1 | 76.7 | 79.5 | 63.6 | 78.3 | 73.2 [70.4, 75.6] |
| planner_region (`landmarks.latent=region`) | 52.7 | 73.5 | 79.1 | 64.3 | 58.4 | 65.6 [61.0, 71.5] |

Paired against the baseline (same 1,250 episodes; McNemar p):

| run | landmark set 0 | set 1 | set 2 | mean |
|---|---|---|---|---|
| planner | −6.5 pp (p 2e-4) | −9.9 (7e-8) | −6.8 (1e-4) | −7.7 |
| planner_high | −3.3 (0.06) | −3.0 (0.08) | −7.5 (2e-5) | −4.6 |
| planner_region | −16.6 (5e-19) | −6.0 (5e-4) | −14.2 (6e-14) | −12.3 |

- baseline_raw − baseline = +2.6 pp [−0.5, +5.8], p = 0.11: not significant on medium. Its
  failures are 72 % flips, as the baseline's.
- non-hier −13.0 pp (p = 3e-14); oracle +6.2 pp [+3.1, +9.2] (p = 1e-4).
- Every planner variant is below the baseline on every landmark set; only planner_high on
  sets 0 and 1 is within noise. The landmark set moves planner_region by 10 points (61.2 to
  71.8): on sets 0 and 2 it fails tasks 1 and 5 (44–54 %).
- Region runs take 9–13 min instead of 4–5 (more replans).
- Parity: ours − upstream `evaluate` = −2.2 pp [−5.3, +1.0] (unpaired). Holds.
- The baseline rerun after the merges gives records identical to the committed CSV.

## Results on large (in progress)

baseline 38.0 % [35.5, 40.5], oracle 41.4 % [38.7, 44.0], baseline_raw 38.1 % [35.7, 40.6]
(5 eval seeds each). Planner variants are running. Paired comparisons and failure classes follow.

## Failure classes on medium (`analysis/failures.py`)

| method | failures | flipped | other (moving at the time limit) | stuck |
|---|---|---|---|---|
| baseline | 277 | 198 (71 %) | 72 | 7 |
| baseline_raw | 244 | 175 (72 %) | 65 | 4 |
| oracle | 200 | 142 (71 %) | 56 | 2 |
| non-hier | 440 | 177 (40 %) | 213 | 50 |

## Oracle ablation (medium, eval seed 0, 250 episodes per config)

| waypoint latent | lookahead 1 | 2 | 3 |
|---|---|---|---|
| single state (B of the nearest val state) | 44.8 | 56.4 | 65.2 |
| region, radius 0.5 | 80.8 | **86.4** | 78.0 |
| region, radius 1.0 | 87.2 | 76.0 | 74.8 |
| region, radius 1.5 | 80.8 | 80.0 | 80.0 |

## FB distance calibration (`analysis/calibration.py`, 400 val states, F heads min)

Trust as in the planner: raw ρ ≥ ρ_min; raw ρ ≥ 1 counted apart. Wormhole = trusted pair with
raw ρ < 1 whose FB distance (clipped ρ) is shorter than the geodesic walk at top speed.
Figures: `results/figures/{calibration,wormholes}_<env>_{min,mean}.png`.

| env | ρ_min | trusted pairs | raw ρ ≥ 1 | wormholes | share of trusted | within xy 6 |
|---|---|---|---|---|---|---|
| medium | 0.3 | 62,200 | 1,118 | 3,385 | 5.4 % | 1,965 |
| medium | 0.7 | 12,146 | 1,118 | 2,997 | 24.7 % | 1,965 |
| large | 0.3 | 38,359 | 1,170 | 3,992 | 10.4 % | 1,420 |
| large | 0.7 | 8,343 | 1,170 | 2,707 | 32.4 % | 1,420 |

The wormholes left by the planner's xy cap (6) cross no wall (geodesic ≤ 1.5 × xy + 0.5 for
all). They are short pairs (median xy 3.5, raw ρ 0.93) where FB distance is about 0.6 × the
fastest walk: FB is overconfident at short range, not across walls. Large is noisier: ρ = 0
for 23 % of pairs (medium 10 %).

## Next

1. Large: planner, planner_high, planner_region × 3 landmark sets (running, about 1.5 h more),
   then failure classes of baseline and oracle on large.
2. Then the 5 retrained medium baselines (`--checkpoint retrained:0..4`, 5 eval seeds each),
   paired with the provided π_h.

## Blockers

- None.

## Requests to the coordinator

- Contract proposal C (checkpoint), optional: the checkpoint lives in 3 CSV columns that
  `write_csv` adds, so no contract change is needed. Cleaner: add `checkpoint_kind: str =
  "provided"`, `checkpoint_variant: str = ""`, `checkpoint_seed: int = -1` to `EpisodeRecord`,
  so that records from `run_episodes` carry it in memory too. The CSV format stays the same.
- Contract proposal H (held-out goal ids), needs your decision: keep `task_id` as the only key
  and document it as "1..5: OGBench eval tasks; 101..110: held-out goal k = task_id − 100 of
  `results/heldout_goals/<env>.csv` (`fbplan.evaluate.HELDOUT_OFFSET`)". Only the docstring of
  `TaskSpec.task_id` (and `EpisodeRecord.task_id`) changes; strata (env × task), pairing keys
  and reset seeds work unchanged. The code already uses this. Alternative: a `goal_set` field in
  `TaskSpec` and `EpisodeRecord`; more explicit, but it changes the record, the strata and the
  pairing key.
- Held-out goals with few dataset states: large goal 105 has 23 val states within the goal
  radius (eval goals on large: 304–624), so its `z` has norm 0.52 √d. Giant eval goals have
  65–578. If you want a floor (for example 50 states), I add it to the selection rule.
- For e-learner: B on held-out goals:
  `python -m fbplan.evaluate --method baseline --checkpoint b:clean:0 --heldout --seeds 0
  --episodes 50 --workers 10 --out runs/...`.
- design.md §8 (raw vs normalized `z_r`): paired on medium, raw is +2.6 pp (p = 0.11), not
  significant. Large result follows.
- design.md §4.A: the planner is below the baseline on medium for all 3 variants and all
  3 landmark sets (pooled −7.7, −4.6, −12.3 pp). Region latents are the worst and the most
  sensitive to the landmark set.
- New files outside the brief list: `tests/test_eval_controllers.py`,
  `results/heldout_goals/` (inputs, not results; tell me if you want them elsewhere).
