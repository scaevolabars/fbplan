"""Route plots and edge statistics of the graph planner on one AntMaze env.

    uv run python analysis/planner_routes.py --env antmaze-medium-navigate-v0 \
        [--split train] [--seed 0] [--opt landmarks.count=200 reached_xy=3.0] \
        [--out-dir runs/c-planner]

For each of the 5 tasks: the route that the planner plans from the start state, and the path
of one episode (reset as in docs/design.md §5). Writes routes_<maze>.png and
routes_<maze>.json: edge statistics (landmarks, trusted edges, rho clips) and episode outcomes.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from fbplan.contracts import TaskSpec  # noqa: E402
from fbplan.controllers.planner import GraphPlannerController, config_from_options  # noqa: E402
from fbplan.frozen import load_dataset, load_model, make_env  # noqa: E402

N_TASKS = 5
WALL, EDGE, LANDMARK = "#d6d5d1", "#9c9b97", "#52514e"
ROUTE, PATH, START, GOAL = "#eb6834", "#2a78d6", "#1baf7a", "#e34948"


@dataclass(frozen=True)
class Episode:
    """One episode from a task's start state, with the route planned at the start."""

    task_id: int
    route: list[int]  # landmark ids
    path: np.ndarray  # [T + 1, 2] xy of the agent
    goal_xy: np.ndarray
    success: bool
    steps: int
    replans: int
    advances: int


def parse_options(items: Sequence[str]) -> dict[str, object]:
    """Planner options from "key=value" items; a value is a Python literal or a bare string."""
    options: dict[str, object] = {}
    for item in items:
        key, _, raw = item.partition("=")
        try:
            options[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            options[key] = raw
    return options


def reset(env: gym.Env, task_id: int, seed: int) -> tuple[np.ndarray, dict]:
    """Reset so that the start and goal depend only on the seed (design.md §5)."""
    np.random.seed(seed)
    env.action_space.seed(seed)
    return env.reset(seed=seed, options={"task_id": task_id})


def run_episode(
    env: gym.Env,
    ctrl: GraphPlannerController,
    env_name: str,
    task_id: int,
    z_r: np.ndarray,
    seed: int,
) -> Episode:
    obs, info = reset(env, task_id, seed)
    task = TaskSpec(env_name, task_id, z_r, np.asarray(info["goal"])[:2])
    ctrl.reset(task, seed)
    route, path, done = ctrl.route(obs), [obs[:2]], False
    while not done:
        obs, _, terminated, truncated, info = env.step(ctrl.act(obs))
        path.append(obs[:2])
        done = terminated or truncated
    stats = ctrl.step_info()
    return Episode(
        task_id=task.task_id,
        route=route,
        path=np.array(path),
        goal_xy=task.goal_xy,
        success=bool(info.get("success", terminated)),
        steps=len(path) - 1,
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
    parser.add_argument("--env", default="antmaze-medium-navigate-v0")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opt", nargs="*", default=[], help="planner options key=value")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/c-planner"))
    args = parser.parse_args(argv)

    model, env = load_model(args.env), make_env(args.env)
    observations = load_dataset(args.env, args.split)["observations"]
    config = config_from_options(parse_options(args.opt))
    ctrl = GraphPlannerController.from_dataset(model, observations, config)
    episodes = []
    for task_id in range(1, N_TASKS + 1):
        z_r = model.infer_task_latent(task_id)
        seed = int(np.random.SeedSequence([args.seed, task_id]).generate_state(1)[0])
        episodes.append(run_episode(env, ctrl, args.env, task_id, z_r, seed))

    stats = edge_stats(ctrl)
    maze = args.env.split("-")[1]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    title = (
        f"Graph planner, {maze} ({args.split} split): {stats['landmarks']} landmarks, "
        f"{stats['trusted_per_landmark']:.1f} trusted edges per landmark, "
        f"rho < 0 in {stats['rho_below_0']} / rho >= 1 in "
        f"{stats['rho_at_least_1_off_diagonal']} of {stats['pairs']} pairs"
    )
    plot(env, ctrl, episodes, title, args.out_dir / f"routes_{maze}.png")
    report = {
        "env": args.env,
        "split": args.split,
        "seed": args.seed,
        "config": asdict(config),
        "edges": stats,
        "episodes": [summary(e, ctrl) for e in episodes],
    }
    (args.out_dir / f"routes_{maze}.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"edges": stats, "success": [e.success for e in episodes]}, indent=2))


if __name__ == "__main__":
    main()
