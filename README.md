# fbplan

Planning over sequences of intentions with frozen forward-backward (FB) representations,
on OGBench AntMaze. Baseline: FB π-Switch ([paper](https://arxiv.org/abs/2605.13207),
code in `third_party/ssm`).

Status: work in progress. Design and notation: `docs/design.md`.

## Setup

```bash
git clone --recurse-submodules https://github.com/scaevolabars/fbplan.git && cd fbplan
uv sync
uv run python -m fbplan.frozen --trained   # checkpoints, hash-checked (see below)
uv run pytest
```

`fbplan.frozen` downloads the 4 provided frozen checkpoints (73 MB) and every
trained checkpoint that this repository publishes as a release asset (retrained `π_h`,
approach B), so a machine without a GPU can evaluate them. OGBench datasets download to
`~/.ogbench/data` on first use. Tests never download data: a test that needs a checkpoint or a
dataset skips, and `pytest` lists the skipped tests. The first evaluation on medium downloads
the medium validation split (24 MB); after it, only tests that need other files skip.
CI (`.github/workflows/ci.yml`) runs `ruff check` and `pytest` on CPU, without data.

On a Linux machine with CUDA, `uv sync` installs the CUDA build of JAX. uv is the primary way to
set up; `devbox.json` only provides the pinned uv (`devbox run setup`, `devbox run test`).
Theory and the exact tabular study: `docs/theory.md`.

## Methods

| `--method` | What it is | Code |
|---|---|---|
| `baseline` | FB π-Switch: provided `π_h` picks an intention every step, `π_l` acts | `fbplan/controllers/baseline.py` |
| `baseline_raw` | the same with the raw upstream `z_r` (exact upstream path) | `fbplan/controllers/baseline.py` |
| `nonhier` | `π_l` with the task latent `z_r` only | `fbplan/controllers/baseline.py` |
| `oracle` | privileged waypoints on the true maze shortest path (diagnostic) | `fbplan/controllers/oracle.py` |
| `planner`, `planner_high` | approach A: landmark graph, SMDP value iteration over FB options | `fbplan/controllers/planner.py` |
| `baseline --checkpoint retrained:<seed>` | the baseline with a `π_h` retrained by us (training variance) | `fbplan/train_high.py` |
| `baseline --checkpoint b:<variant>:<seed>` | approach B: learned multi-switch value `V_H` and a new `π_h` | `fbplan/learned/` |

## Reproduce

`scripts/reproduce.sh` runs the pipeline from the repository root and prints each command
that it runs:

```bash
scripts/reproduce.sh               # checkpoints, eval, tables, figures (CPU)
scripts/reproduce.sh --smoke       # every stage, training included, CPU only (about 10 min)
scripts/reproduce.sh eval tables   # selected stages
scripts/reproduce.sh train         # retrain pi_h and approach B (Linux with a GPU)
```

| Stage | What it does | Output |
|---|---|---|
| `checkpoints` | downloads the frozen and the published trained checkpoints (sha256-checked) | `~/.cache/fbplan/` |
| `train` | retrains `π_h` and approach B, 5 seeds each; not in the default (about 1 h per job on an RTX 3090) | `~/.cache/fbplan/`, `runs/train_*/` |
| `eval` | every run in the table at the top of the script: 5 tasks × 50 episodes per eval seed, paired start states | `results/<maze>/*.csv` |
| `tables` | success with 95 % CIs, paired tests against the baseline, failure classes; `analysis/report_figures.py` (time to goal, report §8.4) | `results/<maze>/tables.md`, `docs/figures/report/` |
| `figures` | success, FB calibration and wormholes, held-out goals, planner routes; the tabular study | `results/figures/`, `results/calibration/`, `docs/figures/theory/` |

The eval stage writes the committed CSVs in `results/`, so `git diff` shows any change.
One 5-seed run on medium (1,250 episodes) takes about 11 min with 4 workers on a 12-core
laptop; the runs with region latents take about twice as long. The whole eval stage (25 runs
and the parity check) takes about 4 h (estimate from these run times). `--workers N` sets the
number of evaluation processes (default 4). `--smoke` uses medium only, the first run of each
result group, one eval seed, two episodes per task and 10 training steps; it writes to
`runs/smoke/` and leaves `results/`, `docs/figures/` and the trained models unchanged. It also
prints how many of its episodes equal the committed ones: resets are seeded, so on the same
machine all of them do. With devbox: `devbox run reproduce` and `devbox run smoke`.

**Single runs.** One evaluation (all methods use this loop); `--out runs/...` keeps the
committed CSVs unchanged:

```bash
uv run python -m fbplan.evaluate --method planner --env antmaze-medium-navigate-v0 \
    --seeds 0 1 2 3 4 --episodes 50 --workers 4 --opt landmarks.seed=0 \
    --out runs/planner_l0.csv
```

Controller options are `--opt key=value`; a dotted key sets a nested field. Training of one
seed on a GPU (500k steps, about 50 min on an RTX 3090):

```bash
uv run python -m fbplan.train_high --env antmaze-medium-navigate-v0 --seed 0     # baseline π_h
uv run python -m fbplan.learned.train_b value  --env antmaze-medium-navigate-v0 --seed 0
uv run python -m fbplan.learned.train_b policy --env antmaze-medium-navigate-v0 --seed 0 \
    --variant clean                                                              # approach B
```

Checkpoints go to `~/.cache/fbplan/`, training curves to `runs/`. `scripts/gpu/vast.sh` rents a
vast.ai instance, pushes the committed code, runs the seeds in parallel, pulls the results and
destroys the instance (see the comments at its top).

## Layout

| Path | Contents |
|---|---|
| `fbplan/contracts.py` | shared interfaces: `FBModel`, `Controller`, `TaskSpec`, `EpisodeRecord` |
| `fbplan/frozen.py` | frozen checkpoints, environments, datasets |
| `fbplan/controllers/` | complete agents for the eval loop: baseline, non-hier, oracle, planner |
| `fbplan/option_model.py`, `graph.py`, `landmarks.py` | FB option model and the landmark graph (A) |
| `fbplan/learned/` | approach B and the device-side batch sampler |
| `fbplan/train_high.py` | retraining of the baseline `π_h` |
| `fbplan/evaluate.py`, `stats.py` | evaluation loop, bootstrap CIs, paired tests |
| `fbplan/tabular/` | exact tabular mazes, fake FB model, theory checks |
| `analysis/` | tables, figures, diagnostics |
| `results/` | CSVs and figures used by the report |
| `scripts/` | `reproduce.sh` (the pipeline), `gpu/` (training on a rented GPU) |
| `third_party/ssm` | upstream FB π-Switch code (git submodule, unchanged) |
