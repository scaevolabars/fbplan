"""Route plots and edge statistics of the graph planner on one AntMaze env.

    uv run python analysis/planner_routes.py --env antmaze-medium-navigate-v0 \
        [--method planner] [--eval-seed 0] [--episode 0] \
        [--opt landmarks.count=200 reached_xy=3.0] [--out-dir runs/planner_routes]

The planner is built as the evaluation CLI builds `--method planner` or `planner_high`
(landmarks from the training split), and each of the 5 tasks runs episode `--episode` of eval
seed `--eval-seed` with fbplan.evaluate: the same start state as that row of an evaluation CSV.
The plot shows the route planned at the first step and the executed path. Writes
routes_<maze>_<method>.png and .json: edge statistics (landmarks, trusted edges, rho clips)
and episode outcomes.
"""

from __future__ import annotations

import argparse
import functools
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from fbplan.contracts import TaskSpec
from fbplan.controllers.planner import (
    GraphPlannerController,
    make_planner,
    make_planner_high,
)
from fbplan.evaluate import (
    config_hash,
    make_tasks,
    parse_options,
    run_episode,
    task_latent,
)
from fbplan.frozen import load_model, make_env

N_TASKS = 5
FACTORIES = {"planner": make_planner, "planner_high": make_planner_high}
WALL, EDGE, LANDMARK = "#d6d5d1", "#9c9b97", "#52514e"
ROUTE, PATH, START, GOAL = "#eb6834", "#2a78d6", "#1baf7a", "#e34948"


@dataclass(frozen=True)
class Episode:
    """One episode from a task's start state, with the route planned at the start."""

    task_id: int
    route: list[int]  # landmark ids
    path: np.ndarray  # [T, 2] xy of the agent before each step
    goal_xy: np.ndarray
    success: bool
    steps: int
    replans: int
    advances: int


class PathRecorder:
    """Passes every call to the planner; keeps the agent's xy and the route of the first step."""

    def __init__(self, planner: GraphPlannerController) -> None:
        self.planner = planner
        self.name = planner.name
        self.path: list[np.ndarray] = []
        self.route: list[int] | None = None

    def reset(self, task: TaskSpec, seed: int) -> None:
        self.planner.reset(task, seed)
        self.path, self.route = [], None

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self.route is None:
            self.route = self.planner.route(obs)
        self.path.append(obs[:2].copy())
        return self.planner.act(obs)

    def step_info(self) -> dict[str, float | int | str]:
        return self.planner.step_info()


def run_task(
    env: gym.Env, recorder: PathRecorder, task: TaskSpec, eval_seed: int, episode: int
) -> Episode:
    record = run_episode(env, recorder, task, eval_seed, episode, config_hash(recorder.planner))
    stats = recorder.step_info()
    return Episode(
        task_id=task.task_id,
        route=recorder.route or [],
        path=np.array(recorder.path),
        goal_xy=task.goal_xy,
        success=record.success,
        steps=record.steps,
        replans=int(stats["replans"]),
        advances=int(stats["advances"]),
    )


def edge_stats(ctrl: GraphPlannerController) -> dict[str, float | int]:
    """Size of the landmark graph and how often FB's rho needed clipping."""
    edges, n = ctrl.edges, len(ctrl.landmarks)
    off_diagonal = ~np.eye(n, dtype=bool)
    return {
        "landmarks": n,
        "trusted_edges": int(ctrl.trusted.sum()),
        "trusted_per_landmark": float(ctrl.trusted.sum() / n),
        "pairs": n * n,
        "rho_below_0": edges.n_clipped_low,
        "rho_above_bound": edges.n_clipped_high,
        "rho_at_least_1_off_diagonal": int((edges.rho_raw[off_diagonal] >= 1).sum()),
    }


def draw_maze(ax: plt.Axes, env: gym.Env) -> None:
    maze = env.unwrapped
    unit, offset = maze._maze_unit, np.array([maze._offset_x, maze._offset_y])
    for i, j in np.argwhere(maze.maze_map == 1):
        centre = np.array([j, i]) * unit - offset
        ax.add_patch(plt.Rectangle(centre - unit / 2, unit, unit, color=WALL, lw=0))
    rows, cols = maze.maze_map.shape
    ax.set_xlim(-offset[0] - unit / 2, (cols - 0.5) * unit - offset[0])
    ax.set_ylim(-offset[1] - unit / 2, (rows - 0.5) * unit - offset[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_episode(ax: plt.Axes, ctrl: GraphPlannerController, episode: Episode) -> None:
    xy = ctrl.landmarks[:, :2]
    u, w = np.nonzero(ctrl.trusted)
    edges = LineCollection(np.stack([xy[u], xy[w]], 1), colors=EDGE, linewidths=0.2, alpha=0.35)
    ax.add_collection(edges)
    ax.scatter(*xy.T, s=4, c=LANDMARK, zorder=2)
    ax.plot(*episode.path.T, color=PATH, lw=1.1, label="executed path", zorder=3)
    route = np.vstack([episode.path[:1], xy[episode.route]])
    ax.plot(*route.T, "-o", color=ROUTE, ms=4, lw=1.8, label="planned route", zorder=4)
    ax.scatter(*episode.path[0], marker="s", s=60, c=START, zorder=5, label="start")
    ax.scatter(*episode.goal_xy, marker="*", s=200, c=GOAL, zorder=5, label="goal")
    outcome = "success" if episode.success else "fail"
    ax.set_title(
        f"task {episode.task_id}: {outcome} in {episode.steps} steps, {len(episode.route)} hops",
        fontsize=9,
    )


def plot(
    env: gym.Env, ctrl: GraphPlannerController, episodes: list[Episode], title: str, path: Path
) -> None:
    fig, axes = plt.subplots(1, len(episodes), figsize=(4.6 * len(episodes), 4.4))
    for ax, episode in zip(np.atleast_1d(axes), episodes, strict=True):
        draw_maze(ax, env)
        draw_episode(ax, ctrl, episode)
    np.atleast_1d(axes)[0].legend(loc="lower left", fontsize=7, framealpha=0.9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def summary(episode: Episode, ctrl: GraphPlannerController) -> dict[str, object]:
    return {
        "task_id": episode.task_id,
        "success": episode.success,
        "steps": episode.steps,
        "replans": episode.replans,
        "advances": episode.advances,
        "route_hops": len(episode.route),
        "route_xy": ctrl.landmarks[episode.route, :2].round(2).tolist(),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", default="antmaze-medium-navigate-v0", help="OGBench env name")
    parser.add_argument("--method", default="planner", choices=sorted(FACTORIES), help="planner")
    parser.add_argument(
        "--eval-seed", type=int, default=0, help="eval seed of the reset, as in fbplan.evaluate"
    )
    parser.add_argument(
        "--episode", type=int, default=0, help="episode index of the reset, as in fbplan.evaluate"
    )
    parser.add_argument(
        "--opt", nargs="*", default=[], help="planner options key=value, as in fbplan.evaluate"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("runs/planner_routes"), help="output folder"
    )
    args = parser.parse_args(argv)

    model, env = load_model(args.env), make_env(args.env)
    planner = FACTORIES[args.method](model, **parse_options(args.opt))
    recorder = PathRecorder(planner)
    tasks = make_tasks(args.env, range(1, N_TASKS + 1), functools.partial(task_latent, model))
    episodes = [run_task(env, recorder, task, args.eval_seed, args.episode) for task in tasks]

    stats = edge_stats(planner)
    maze = args.env.split("-")[1]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    title = (
        f"{args.method}, {maze}: {stats['landmarks']} landmarks, "
        f"{stats['trusted_per_landmark']:.1f} trusted edges per landmark, "
        f"rho < 0 in {stats['rho_below_0']} / rho >= 1 in "
        f"{stats['rho_at_least_1_off_diagonal']} of {stats['pairs']} pairs"
    )
    plot(env, planner, episodes, title, args.out_dir / f"routes_{maze}_{args.method}.png")
    report = {
        "env": args.env,
        "method": args.method,
        "eval_seed": args.eval_seed,
        "episode": args.episode,
        "config": asdict(planner.config),
        "edges": stats,
        "episodes": [summary(e, planner) for e in episodes],
    }
    out = args.out_dir / f"routes_{maze}_{args.method}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"edges": stats, "success": [e.success for e in episodes]}, indent=2))


if __name__ == "__main__":
    main()
