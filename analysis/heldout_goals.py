"""Write the held-out goals for hyperparameter selection (plan B decision 8) and draw them.

    uv run python analysis/heldout_goals.py [--envs antmaze-medium-navigate-v0 ...]

For each env: results/heldout_goals/<env>.csv (fbplan.evaluate.select_heldout_goals with its
defaults and the validation split, so the file is reproducible) and
results/figures/heldout_goals_<env>.png. It prints the number of validation states within the
goal radius of each goal (the states that make z).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fbplan.evaluate import (  # noqa: E402
    HeldoutGoal,
    heldout_goals_path,
    select_heldout_goals,
    write_heldout_goals,
)
from fbplan.frozen import load_dataset, make_env  # noqa: E402
from fbplan.maze import MazeGeometry, draw_walls  # noqa: E402

ENVS = [f"antmaze-{maze}-navigate-v0" for maze in ("medium", "large", "giant", "teleport")]
INK, MUTED, WALL, EVAL, HELDOUT = "#0b0b0b", "#52514e", "#e4e3df", "#7a7974", "#2a78d6"


def plot_goals(
    maze: MazeGeometry, eval_goals: np.ndarray, goals: list[HeldoutGoal], title: str, path: Path
) -> None:
    rows, cols = maze.maze_map.shape
    fig, ax = plt.subplots(figsize=(1.0 + 0.45 * cols, 1.0 + 0.45 * rows), dpi=150)
    draw_walls(ax, maze, WALL)
    for goal in goals:
        start = maze.centre(goal.init_ij)
        ax.plot(*np.stack([start, goal.goal_xy]).T, color=HELDOUT, lw=0.8, alpha=0.35)
        ax.scatter(*start, s=18, facecolor="white", edgecolor=HELDOUT, lw=1, zorder=3)
        ax.scatter(*goal.goal_xy, s=60, color=HELDOUT, zorder=4)
        ax.annotate(str(goal.task_id), goal.goal_xy, xytext=(5, 5), textcoords="offset points",
                    fontsize=7, color=INK)
    ax.scatter(eval_goals[:, 0], eval_goals[:, 1], s=70, marker="s", color=EVAL, zorder=4)
    ax.set_title(title, color=INK, fontsize=9, loc="left")
    handles = [
        plt.Line2D([], [], marker="s", ls="", color=EVAL, label="eval goal"),
        plt.Line2D([], [], marker="o", ls="", color=HELDOUT, label="held-out goal"),
        plt.Line2D([], [], marker="o", ls="", mfc="white", mec=HELDOUT, label="its start cell"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, 0.0), ncols=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--envs", nargs="+", default=ENVS)
    parser.add_argument("--fig-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    for env_name in args.envs:
        env = make_env(env_name)
        obs = load_dataset(env_name, "val")["observations"]
        xy = obs[:, :2]
        goals = select_heldout_goals(env, obs)
        write_heldout_goals(goals, heldout_goals_path(env_name))
        maze = MazeGeometry.from_env(env)
        eval_goals = np.array([info["goal_xy"] for info in env.unwrapped.task_infos])
        plot_goals(maze, eval_goals, goals, f"{env_name}: held-out goals (ids) and eval goals",
                   args.fig_dir / f"heldout_goals_{env_name}.png")
        radius = env.unwrapped._goal_tol
        print(f"{env_name}: {len(goals)} goals -> {heldout_goals_path(env_name)}")
        for goal in goals:
            inside = int((np.linalg.norm(xy - goal.goal_xy, axis=1) <= radius).sum())
            print(f"  {goal.task_id}: start {goal.init_ij} goal {goal.goal_ij} "
                  f"xy {goal.goal_xy}, {inside} val states within {radius}")


if __name__ == "__main__":
    main()
