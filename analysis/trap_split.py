"""Trap and other failures of teleport runs, from the CSVs only (the rule of docs/theory.md §11.3).

    uv run python analysis/trap_split.py results/teleport_10seeds/baseline.csv
        results/teleport_10seeds/trap_aware.csv [--reference results/teleport_10seeds/baseline.csv]
        [--low-trap 0.10]

A failed episode is trapped if its final distance to the goal is within 2 xy units of the trap
centre's distance to the goal (the trap is cell (1, 7), theory.md §11.1); the other failures run
to the time limit or flip. Goals: the OGBench eval tasks, and the held-out goals (task ids above
100). The script prints, per CSV:

- success, trap and other failures (%), the median steps of the successes and the median final
  distance of the other failures; the same per task;
- the share of episodes identical to the reference CSV (same task, eval seed and episode, same
  steps and final distance within 1e-6): with a closed gate the trap-aware controller makes the
  baseline's calls, so such an episode had its gate closed at every step;
- success and trap share on the tasks where at most `low-trap` of the reference's episodes are
  trapped, and on the others.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fbplan.evaluate import heldout_goals_path, load_heldout_goals
from fbplan.frozen import make_env
from fbplan.maze import MazeGeometry
from fbplan.stats import markdown_table

ENV = "antmaze-teleport-navigate-v0"
TRAP_CELL = (1, 7)
TRAP_TOLERANCE = 2.0  # xy units
KEY = ["task_id", "eval_seed", "episode"]


def goal_xy(env_name: str = ENV) -> dict[int, np.ndarray]:
    """Goal xy of every eval task and held-out goal of the env."""
    env = make_env(env_name)
    goals = {i + 1: np.asarray(t["goal_xy"], dtype=float)
             for i, t in enumerate(env.unwrapped.task_infos)}
    if Path(heldout_goals_path(env_name)).exists():
        goals |= {h.task_id: np.asarray(h.goal_xy, dtype=float)
                  for h in load_heldout_goals(env_name)}
    return goals


def trap_centre(env_name: str = ENV) -> np.ndarray:
    return MazeGeometry.from_env(make_env(env_name)).centre(TRAP_CELL)


def outcomes(df: pd.DataFrame, goals: dict[int, np.ndarray], trap: np.ndarray) -> pd.Series:
    """'success', 'trap' or 'other' per episode."""
    trap_dist = df.task_id.map(lambda t: float(np.linalg.norm(goals[t] - trap)))
    success = df.success.astype(bool)
    trapped = ~success & ((df.final_goal_dist - trap_dist).abs() < TRAP_TOLERANCE)
    return pd.Series(np.where(success, "success", np.where(trapped, "trap", "other")),
                     index=df.index)


def shares(df: pd.DataFrame, outcome: pd.Series) -> dict[str, float]:
    other = outcome == "other"
    return {
        "episodes": len(df),
        "success %": 100 * (outcome == "success").mean(),
        "trap %": 100 * (outcome == "trap").mean(),
        "other %": 100 * other.mean(),
        "median steps of successes": df.steps[outcome == "success"].median(),
        "median final dist of other": df.final_goal_dist[other].median(),
    }


def identical_share(df: pd.DataFrame, reference: pd.DataFrame) -> pd.Series:
    """Per task: share of episodes with the reference's steps and final distance (1e-6)."""
    a, b = df.set_index(KEY).sort_index(), reference.set_index(KEY).sort_index()
    a, b = a.align(b, join="inner", axis=0)
    same = (a.steps == b.steps) & np.isclose(a.final_goal_dist, b.final_goal_dist, rtol=0,
                                             atol=1e-6)
    return same.groupby(level="task_id").mean()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv", nargs="+", type=Path, help="teleport evaluation CSVs")
    parser.add_argument("--reference", type=Path, default=None,
                        help="CSV for identical episodes and the task groups (default: the first)")
    parser.add_argument("--low-trap", type=float, default=0.10,
                        help="a task is 'low trap' if at most this share of the reference's "
                             "episodes is trapped")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    goals, trap = goal_xy(), trap_centre()
    reference = pd.read_csv(args.reference or args.csv[0])
    ref_outcome = outcomes(reference, goals, trap)
    ref_trap = (ref_outcome == "trap").groupby(reference.task_id).mean()
    low = ref_trap.index[ref_trap <= args.low_trap]
    print(f"Trap centre {trap}; low-trap tasks of the reference: {list(low)}")
    rows, per_task = [], []
    for path in args.csv:
        df = pd.read_csv(path)
        outcome = outcomes(df, goals, trap)
        rows.append({"csv": str(path), **shares(df, outcome)})
        same = identical_share(df, reference)
        for task, group in df.groupby("task_id"):
            per_task.append({"csv": str(path), "task": task, **shares(group, outcome[group.index]),
                             "identical to reference %": 100 * same.get(task, np.nan)})
        is_low = df.task_id.isin(low)
        for name, mask in (("low trap", is_low), ("other tasks", ~is_low)):
            if mask.any():
                per_task.append({"csv": str(path), "task": name, **shares(df[mask], outcome[mask])})
    print(markdown_table(pd.DataFrame(rows).round(1)))
    print()
    print(markdown_table(pd.DataFrame(per_task).round(1)))


if __name__ == "__main__":
    main()
