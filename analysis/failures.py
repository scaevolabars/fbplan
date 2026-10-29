"""Why episodes fail: replay the failed episodes of a result CSV and classify them.

    uv run python analysis/failures.py runs/<csv> --method baseline [--opt key=value ...]
        [--workers 10]

Resets are seeded, so a replay with the same controller config and checkpoint repeats the
episode exactly (the script checks the config hash and the outcome; it reads the checkpoint
from the CSV). Classes, from the ant's observations:

- flipped: the torso points down (body z-axis below the horizon) for the last 50 steps;
  the ant cannot turn back, so this is a pi_l failure that no high-level choice can repair;
- stuck: not flipped, and the xy position stays within 1 unit over the last 300 steps;
- other: still moving at the time limit (slow, lost or on a wrong route).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from fbplan.contracts import Controller, TaskSpec
from fbplan.controllers import resolve_controller
from fbplan.controllers.oracle import MazeGeometry
from fbplan.evaluate import (
    Checkpoint,
    config_hash,
    make_tasks,
    parse_options,
    run_episode,
    task_latent,
)
from fbplan.frozen import make_env
from fbplan.stats import CHECKPOINT_DEFAULTS, load_results

FLIP_WINDOW, STUCK_WINDOW, STUCK_RADIUS = 50, 300, 1.0


class RecordingController:
    """Passes every call to `inner` and keeps the observations of the current episode."""

    def __init__(self, inner: Controller) -> None:
        self.inner = inner
        self.name = inner.name
        self.observations: list[np.ndarray] = []

    def reset(self, task: TaskSpec, seed: int) -> None:
        self.observations = []
        self.inner.reset(task, seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        self.observations.append(obs.copy())
        return self.inner.act(obs)

    def step_info(self) -> dict[str, float | int | str]:
        return self.inner.step_info()


def upright(observations: np.ndarray) -> np.ndarray:
    """z-component of the torso's z-axis from the quaternion (w, x, y, z) in obs[3:7]."""
    q = observations[:, 3:7]
    return 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)


def classify(observations: np.ndarray) -> str:
    if (upright(observations[-FLIP_WINDOW:]) < 0).all():
        return "flipped"
    xy = observations[-STUCK_WINDOW:, :2]
    if np.linalg.norm(xy - xy.mean(axis=0), axis=1).max() < STUCK_RADIUS:
        return "stuck"
    return "other"


def replay_failures(
    method: str,
    env_name: str,
    options: dict[str, object],
    checkpoint: Checkpoint,
    expected_hash: str,
    rows: list[dict],
) -> list[dict]:
    """Replay the episodes in `rows` and return one classified row per episode."""
    model = checkpoint.load(env_name)
    inner = resolve_controller(method)(model, **options)
    if config_hash(inner) != expected_hash:
        raise ValueError(f"config hash {config_hash(inner)} differs from the CSV's {expected_hash}")
    controller = RecordingController(inner)
    env = make_env(env_name)
    maze = MazeGeometry.from_env(env)
    task_ids = sorted({r["task_id"] for r in rows})
    tasks = {t.task_id: t for t in make_tasks(env_name, task_ids, partial(task_latent, model))}
    out = []
    for row in rows:
        task = tasks[row["task_id"]]
        record = run_episode(env, controller, task, row["eval_seed"], row["episode"], "replay")
        if record.success or record.steps != row["steps"]:
            raise RuntimeError(f"replay of {row} differs: {record}")
        obs = np.stack(controller.observations)
        cells_left = maze.bfs_steps(maze.cell_of(task.goal_xy))[maze.cell_of(obs[-1, :2])]
        out.append({**row, "failure": classify(obs), "cells_left": int(cells_left)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument("--method", required=True, help="the --method of the evaluation run")
    parser.add_argument("--opt", nargs="*", default=[], help="the --opt of the evaluation run")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    df = load_results([args.csv])
    (env_name,), (expected_hash,) = df["env_name"].unique(), df["config_hash"].unique()
    checkpoints = df[list(CHECKPOINT_DEFAULTS)].drop_duplicates()
    if len(checkpoints) != 1:
        raise ValueError(f"{args.csv} holds {len(checkpoints)} checkpoints; replay needs one")
    kind, variant, seed = checkpoints.iloc[0]
    checkpoint = Checkpoint(str(kind), str(variant), int(seed))
    failed = df[~df["success"]].to_dict("records")
    chunks = [failed[i :: args.workers] for i in range(args.workers) if failed[i :: args.workers]]
    options = parse_options(args.opt)
    with ProcessPoolExecutor(len(chunks), mp_context=get_context("spawn")) as pool:
        futures = [
            pool.submit(replay_failures, args.method, env_name, options, checkpoint,
                        expected_hash, chunk)
            for chunk in chunks
        ]
        rows = pd.DataFrame([row for future in futures for row in future.result()])

    counts = rows.pivot_table(index="task_id", columns="failure", values="episode",
                              aggfunc="count", fill_value=0)
    counts.loc["all"] = counts.sum()
    print(f"{args.method} on {env_name}: {len(df)} episodes, {len(rows)} failures")
    print(counts.to_string())
    shares = rows["failure"].value_counts(normalize=True).round(3).to_dict()
    print(f"share of failures: {shares}; median cells left to the goal: "
          f"{rows['cells_left'].median():.0f}")


if __name__ == "__main__":
    main()
