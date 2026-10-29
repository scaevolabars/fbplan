# fbplan

Planning over sequences of intentions with frozen forward-backward (FB) representations,
on OGBench AntMaze. Baseline: FB π-Switch ([paper](https://arxiv.org/abs/2605.13207),
code in `third_party/ssm`).

Status: work in progress. Design and notation: `docs/design.md`.

## Setup

```bash
git clone --recurse-submodules https://github.com/scaevolabars/fbplan.git && cd fbplan
uv sync                         # or: devbox run setup
uv run python -m fbplan.frozen  # downloads the 4 frozen checkpoints (73 MB), hash-checked
uv run pytest                   # or: devbox run test
```

On a Linux machine with CUDA, `uv sync` installs the CUDA build of JAX.
OGBench datasets download to `~/.ogbench/data` on first use.
Theory and the exact tabular study: `docs/theory.md`.

## Methods

| `--method` | What it is | Code |
|---|---|---|
| `baseline` | FB π-Switch: provided `π_h` picks an intention every step, `π_l` acts | `fbplan/controllers/baseline.py` |
| `baseline_raw` | the same with the raw upstream `z_r` (exact upstream path) | `fbplan/controllers/baseline.py` |
| `nonhier` | `π_l` with the task latent `z_r` only | `fbplan/controllers/baseline.py` |
| `oracle` | privileged waypoints on the true maze shortest path (diagnostic) | `fbplan/controllers/oracle.py` |
| `planner`, `planner_high` | approach A: landmark graph, SMDP value iteration over FB options | `fbplan/controllers/planner.py` |
| approach B | learned multi-switch value `V_H` and a new `π_h` (trained checkpoints) | `fbplan/learned/` |

## Reproduce

All commands run from the repository root. Evaluation runs on CPU; training needs a GPU.

**1. Evaluate** (5 tasks × 50 episodes × 5 eval seeds, paired start states; about 1–2 min per
method on medium with 10 workers):

```bash
uv run python -m fbplan.evaluate --method baseline --env antmaze-medium-navigate-v0 \
    --seeds 0 1 2 3 4 --episodes 50 --workers 10 --out results/medium/baseline.csv
uv run python -m fbplan.evaluate --method planner --env antmaze-medium-navigate-v0 \
    --seeds 0 1 2 3 4 --episodes 50 --workers 10 --opt landmarks.seed=0 \
    --out results/medium/planner/l0.csv
```

Controller options are `--opt key=value`; a dotted key sets a nested field.

**2. Train on a GPU** (500k steps per run, about 50 min on an RTX 3090 with 5 runs in parallel):

```bash
uv run python -m fbplan.train_high --env antmaze-medium-navigate-v0 --seed 0     # baseline π_h
uv run python -m fbplan.learned.train_b value  --env antmaze-medium-navigate-v0 --seed 0
uv run python -m fbplan.learned.train_b policy --env antmaze-medium-navigate-v0 --seed 0 \
    --variant clean                                                              # approach B
```

Checkpoints go to `~/.cache/fbplan/`, training curves to `runs/`. `scripts/gpu/vast.sh` rents a
vast.ai instance, pushes the committed code, runs the seeds in parallel, pulls the results and
destroys the instance (see the comments at its top).

**3. Tables and figures:**

```bash
uv run python analysis/tables.py results/medium/*.csv results/medium/planner/*.csv \
    --label-root results/medium --cluster config_hash     # success, CIs, paired tests
uv run python analysis/plots.py results/medium/*.csv      # success figure
uv run python analysis/calibration.py --env antmaze-medium-navigate-v0   # FB distance vs. maze
uv run python analysis/failures.py results/medium/baseline.csv --method baseline
uv run python analysis/planner_routes.py --env antmaze-medium-navigate-v0  # route plots
uv run python analysis/parity.py --ours results/medium/baseline.csv  # our loop vs. upstream
uv run python -m fbplan.tabular.run_study && uv run python -m fbplan.tabular.study_plots
```

## Layout

| Path | Contents |
|---|---|
| `fbplan/contracts.py` | shared interfaces: `FBModel`, `Controller`, `TaskSpec`, `EpisodeRecord` |
| `fbplan/frozen.py` | frozen checkpoints, environments, datasets |
| `fbplan/option_model.py`, `graph.py`, `landmarks.py` | FB option model and the landmark graph (A) |
| `fbplan/learned/` | approach B and the device-side batch sampler |
| `fbplan/train_high.py` | retraining of the baseline `π_h` |
| `fbplan/evaluate.py`, `stats.py` | evaluation loop, bootstrap CIs, paired tests |
| `fbplan/tabular/` | exact tabular mazes, fake FB model, theory checks |
| `analysis/` | tables, figures, diagnostics |
| `results/` | CSVs and figures used by the report |
| `third_party/ssm` | upstream FB π-Switch code (git submodule, unchanged) |
