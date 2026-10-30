# Status: c-planner

The earlier status file was removed with the process files (main caafdc2); its last version is
`git show caafdc2^:docs/status/c-planner.md`. This file holds only H2.

## H2, FB orders the recorded route: code and commands for d-eval (2026-10-30)

Reading fixed in `docs/design.md` §8, "H2, FB orders the recorded route" (main dc75984).

### Code

- `FBRouteController` in `fbplan/controllers/planner.py` (commit 140c261). It is
  `RecordedRouteController` with one change: every certified edge u → w, the first hop from s
  included, weighs `−log ρ̂(u, w)` instead of its xy length. `ρ̂` is approach A's
  (`fbplan.option_model`): option latent B(w) of the landmark state, head min, clipped to [0, γ]
  for distinct states. An edge with `ρ̂ = 0` has no weight, so the route drops it. Certification,
  goal landmark, walks and executors are H1's.
- The first hop weighs the same way as the edges. With an xy first hop and FB edges the route
  cost would add xy units to FB units.
- CLI option: `--opt edge_weight=fb` of `fbplan.controllers.planner:make_recorded_route`.
  Default `xy` (H1).
- The option picks the class, not a config field. `RecordedRouteConfig` is unchanged, so every H1
  CSV keeps its `config_hash` (README "Code versions": "All other CSVs keep their hash at the
  current commit" stays true; `scripts/reproduce.sh --smoke` compares the hashes of the 16 H1
  CSVs). H2 rows have the method name `recorded_route_fb` and their own hash: medium `window`,
  K 150: `19600de4`; large `high`, K 150: `13f454d8`.
- Per step, H2 calls F once, for the distinct first-hop candidates only (not all 300 landmarks).
- Tests (`tests/test_planner.py`, tabular fakes):
  - a maze with two corridors where the xy-shortest certified route is 16 steps on foot and
    another one 12: H1 takes the 16-step route, H2 the 12-step route, with and without a landmark
    at the start (then the first hop decides);
  - with exact `ρ = γ^d` the H2 route length is `−log γ` × the shortest walk (12 steps), and on
    the tabular medium maze every certified edge has the length of its geodesic;
  - an edge with `ρ̂ = 0` is dropped, without a numpy warning; with no first-hop candidate H2
    has no route, as H1;
  - H2 solves every task of the exact tabular maze with each executor;
  - `edge_weight` picks the class, an unknown value raises;
  - H1's medium and large configs keep the committed hashes 285be155 and 7422f2eb, H2's differ.
  - Mutation checks: an xy first hop in H2 fails 5 tests; a new config field fails the hash test.

### Check: H1 records do not change (Mac)

The same commands at main dc75984 (before the change) and at 140c261, eval seed 0, 10
episodes per task: medium `window`, K 150: 50 of 50 records equal (success, steps,
final_goal_dist, config_hash 285be155); large `high`, K 150: 50 of 50 equal (7422f2eb). A Mac run
gives other episodes than the Linux CSVs (0 of 50 equal), so step 0 below repeats the check on
Linux against the committed CSVs.

### Commands for d-eval (Linux, at most 4 workers, nice)

Code: main with 140c261 merged. Run from the repository root.

**Step 0, record check.** Rerun eval seed 0 of the committed H1 runs and compare. Both lines must
print `250 of 250 records equal` (success, steps, final_goal_dist and config_hash):

```
nice -n 10 uv run python -m fbplan.evaluate --method fbplan.controllers.planner:make_recorded_route --env antmaze-medium-navigate-v0 --seeds 0 --episodes 50 --workers 4 --opt executor=window max_steps=150 --out runs/h2_check/h1_medium_seed0.csv
nice -n 10 uv run python -m fbplan.evaluate --method fbplan.controllers.planner:make_recorded_route --env antmaze-large-navigate-v0 --seeds 0 --episodes 50 --workers 4 --opt executor=high max_steps=150 --out runs/h2_check/h1_large_seed0.csv
uv run python -c "import sys,pandas as pd;k=['task_id','eval_seed','episode'];v=['success','steps','final_goal_dist','config_hash'];r=lambda p:pd.read_csv(p,dtype={'config_hash':str}).set_index(k)[v];new,old=r(sys.argv[1]),r(sys.argv[2]);print((new==old.loc[new.index]).all(axis=1).sum(),'of',len(new),'records equal')" runs/h2_check/h1_medium_seed0.csv results/medium/planner_ablations/h1_eval/window_k150.csv
uv run python -c "import sys,pandas as pd;k=['task_id','eval_seed','episode'];v=['success','steps','final_goal_dist','config_hash'];r=lambda p:pd.read_csv(p,dtype={'config_hash':str}).set_index(k)[v];new,old=r(sys.argv[1]),r(sys.argv[2]);print((new==old.loc[new.index]).all(axis=1).sum(),'of',len(new),'records equal')" runs/h2_check/h1_large_seed0.csv results/large/planner_ablations/h1_eval/high_k150.csv
```

**Step 1, held-out check.** The same held-out episodes as H1 (held-out goals, eval seeds 0 1 2,
20 episodes per goal and seed: 600 per maze), H1's selected config with FB costs:

```
nice -n 10 uv run python -m fbplan.evaluate --method fbplan.controllers.planner:make_recorded_route --env antmaze-medium-navigate-v0 --heldout --seeds 0 1 2 --episodes 20 --workers 4 --opt executor=window max_steps=150 edge_weight=fb --out results/medium/planner_ablations/h2_heldout/window_k150.csv
nice -n 10 uv run python -m fbplan.evaluate --method fbplan.controllers.planner:make_recorded_route --env antmaze-large-navigate-v0 --heldout --seeds 0 1 2 --episodes 20 --workers 4 --opt executor=high max_steps=150 edge_weight=fb --out results/large/planner_ablations/h2_heldout/high_k150.csv
uv run python analysis/tables.py results/medium/planner_ablations/h1_heldout/baseline.csv results/medium/planner_ablations/h1_heldout/oracle.csv results/medium/planner_ablations/h1_heldout/window_k150.csv results/medium/planner_ablations/h2_heldout/window_k150.csv --reference baseline recorded_route
uv run python analysis/tables.py results/large/planner_ablations/h1_heldout/baseline.csv results/large/planner_ablations/h1_heldout/oracle.csv results/large/planner_ablations/h1_heldout/high_k150.csv results/large/planner_ablations/h2_heldout/high_k150.csv --reference baseline recorded_route
```

Rows: `recorded_route` is H1, `recorded_route_fb` is H2. Rule (design.md §8): H2 is falsified on
a maze if it scores more than 2 pp below H1 or not above `baseline` on these episodes. Then H2
stops on that maze; all rows are still reported.

**Step 2, final runs** (only on a maze where H2 is not falsified). Eval seeds 0–9 and 10–19, in
the layout of H1's runs:

```
nice -n 10 uv run python -m fbplan.evaluate --method fbplan.controllers.planner:make_recorded_route --env antmaze-medium-navigate-v0 --seeds 0 1 2 3 4 5 6 7 8 9 --episodes 50 --workers 4 --opt executor=window max_steps=150 edge_weight=fb --out results/medium/planner_ablations/h2_eval/window_k150.csv
nice -n 10 uv run python -m fbplan.evaluate --method fbplan.controllers.planner:make_recorded_route --env antmaze-medium-navigate-v0 --seeds 10 11 12 13 14 15 16 17 18 19 --episodes 50 --workers 4 --opt executor=window max_steps=150 edge_weight=fb --out results/medium_seeds10-19/h2_window_k150.csv
nice -n 10 uv run python -m fbplan.evaluate --method fbplan.controllers.planner:make_recorded_route --env antmaze-large-navigate-v0 --seeds 0 1 2 3 4 5 6 7 8 9 --episodes 50 --workers 4 --opt executor=high max_steps=150 edge_weight=fb --out results/large/planner_ablations/h2_eval/high_k150.csv
nice -n 10 uv run python -m fbplan.evaluate --method fbplan.controllers.planner:make_recorded_route --env antmaze-large-navigate-v0 --seeds 10 11 12 13 14 15 16 17 18 19 --episodes 50 --workers 4 --opt executor=high max_steps=150 edge_weight=fb --out results/large_seeds10-19/h2_high_k150.csv
```

Tables, paired against `baseline`, `baseline_raw` and H1 (`recorded_route`); per maze, seeds
0–9, 10–19, and 0–19 (both seed sets of each row together):

```
uv run python analysis/tables.py results/medium_10seeds/baseline.csv results/medium_10seeds/baseline_raw.csv results/medium/planner_ablations/h1_eval/window_k150.csv results/medium/planner_ablations/h2_eval/window_k150.csv --reference baseline baseline_raw recorded_route
uv run python analysis/tables.py results/medium_seeds10-19/baseline.csv results/medium_seeds10-19/baseline_raw.csv results/medium_seeds10-19/h1_window_k150.csv results/medium_seeds10-19/h2_window_k150.csv --reference baseline baseline_raw recorded_route
uv run python analysis/tables.py results/medium_10seeds/baseline.csv results/medium_10seeds/baseline_raw.csv results/medium/planner_ablations/h1_eval/window_k150.csv results/medium/planner_ablations/h2_eval/window_k150.csv results/medium_seeds10-19/baseline.csv results/medium_seeds10-19/baseline_raw.csv results/medium_seeds10-19/h1_window_k150.csv results/medium_seeds10-19/h2_window_k150.csv --reference baseline baseline_raw recorded_route
uv run python analysis/tables.py results/large_10seeds/baseline.csv results/large_10seeds/baseline_raw.csv results/large/planner_ablations/h1_eval/high_k150.csv results/large/planner_ablations/h2_eval/high_k150.csv --reference baseline baseline_raw recorded_route
uv run python analysis/tables.py results/large_seeds10-19/baseline.csv results/large_seeds10-19/baseline_raw.csv results/large_seeds10-19/h1_high_k150.csv results/large_seeds10-19/h2_high_k150.csv --reference baseline baseline_raw recorded_route
uv run python analysis/tables.py results/large_10seeds/baseline.csv results/large_10seeds/baseline_raw.csv results/large/planner_ablations/h1_eval/high_k150.csv results/large/planner_ablations/h2_eval/high_k150.csv results/large_seeds10-19/baseline.csv results/large_seeds10-19/baseline_raw.csv results/large_seeds10-19/h1_high_k150.csv results/large_seeds10-19/h2_high_k150.csv --reference baseline baseline_raw recorded_route
```

Readings per maze (design.md §8): (a) H2 passes the gain rule on seeds 0–9 and on seeds 10–19
alone; (b) H2 − H1 over seeds 0–19 has a 95 % CI above −2 pp; (c) H2 − H1 significantly below 0.
All rows are reported.

**Cost.** H2 calls F once per step, so it is slower than H1. Mac, 4 workers, eval seed 0:
medium `window` 91 steps/s per worker (H1: 141); large `high` 149 (H1: 268; 10 episodes, so
start-up counts). From H1's Linux times, about 1.6× to 1.8× H1 on Linux: the medium final run
about 3.5–4 h per seed set, the large one about 2 h per seed set; each held-out row under 1 h.
Smoke (Mac, eval seed 0, 2 episodes per task, both mazes): runs, no warning; not a result.

### For f-report and g-release

- Method table (README, REPORT appendix): `make_recorded_route --opt ... edge_weight=fb` gives
  H2; its rows are `recorded_route_fb`.
- The H1 CSVs keep their hashes (above), so the README "Code versions" text needs no change for
  H1.
