"""Success within a time limit vs. the discounted value, on cell-level mazes.

OGBench counts an episode as a success when the goal is reached within T = 1000 ant steps, so
the metric is P(H <= T). FB values, and the policies trained on them, maximise E[gamma^H]: they
are risk-neutral in gamma^H. On a GridMaze at resolution 1 (one state per maze cell), with the
ant's speed modelled as `steps_per_cell` ant steps per cell move (N = T // steps_per_cell cell
moves, gamma_cell = gamma^steps_per_cell), this module compares three policies for one task:

- walk: the shortest route that enters no teleporter;
- risk-neutral: the optimal policy for E[gamma_cell^H];
- best: the policy that maximises P(H <= N) (finite-horizon dynamic programming; it may depend
  on the time left, so it is not one stationary policy).

It also finds, per teleport task, the smallest per-step discount at which the risk-neutral
policy walks (`walk_threshold`).

    uv run python -m fbplan.tabular.time_limit
    # --out-dir docs/figures/theory: time_limit.{csv,png} and time_limit_thresholds.csv
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fbplan.tabular.maze import GridMaze
from fbplan.tabular.mdp import action_values, optimal_policy, policy_matrix

EPISODE_STEPS = 1000  # OGBench AntMaze time limit
DISCOUNTS = {"giant": 0.995, "teleport": 0.99}  # flags.json of the provided checkpoints
STEPS_PER_CELL = (20, 25, 30, 35, 40, 45, 50, 55, 60)
OUT_DIR = Path("docs/figures/theory")


def walk_policy(maze: GridMaze, goal: int) -> np.ndarray:
    """Action per state of the shortest walk to `goal` that enters no teleporter.

    Ties go to the lowest action id. States that cannot walk to the goal (the trap) stay.
    """
    blocked = set(maze.entry_states.tolist())
    adjacency = maze.neighbours()
    dist = np.full(maze.n_states, np.inf)
    dist[goal] = 0
    queue = deque([goal])
    while queue:
        u = queue.popleft()
        for v in adjacency[u]:
            if v not in blocked and dist[v] == np.inf:
                dist[v] = dist[u] + 1
                queue.append(v)
    policy = np.zeros(maze.n_states, dtype=np.int64)
    for s in range(maze.n_states):
        if s == goal or dist[s] == np.inf:
            continue
        for action in range(1, maze.next_states.shape[1]):
            target = maze.next_states[s, action, 0]
            if maze.probs[s, action, 0] == 1.0 and dist[target] == dist[s] - 1:
                policy[s] = action
                break
    return policy


def discounted_policy(maze: GridMaze, goal: int, gamma_cell: float) -> np.ndarray:
    """The optimal policy for E[gamma_cell^H] (reward 1 at the goal, where the agent stays)."""
    reward = np.zeros(maze.n_states)
    reward[goal] = 1.0
    return optimal_policy(maze, reward, gamma_cell)[0]


def hitting_cdf(
    maze: GridMaze, policy: np.ndarray, start: int, goal: int, n_moves: int
) -> np.ndarray:
    """P(H <= k) for k = 0..n_moves under a stationary policy: [n_moves + 1]."""
    transitions = policy_matrix(maze, policy)
    transitions[goal] = 0.0
    transitions[goal, goal] = 1.0  # count each episode once: the goal ends it
    dist = np.zeros(maze.n_states)
    dist[start] = 1.0
    cdf = [dist[goal]]
    for _ in range(n_moves):
        dist = dist @ transitions
        cdf.append(dist[goal])
    return np.array(cdf)


def best_success(maze: GridMaze, start: int, goal: int, n_moves: int) -> float:
    """max P(H <= n_moves) over all policies, by backward induction on the moves left."""
    values = np.zeros(maze.n_states)
    values[goal] = 1.0
    for _ in range(n_moves):
        values = action_values(maze, values).max(axis=1)
        values[goal] = 1.0
    return float(values[start])


def enters_teleporter(maze: GridMaze, policy: np.ndarray, start: int, goal: int) -> bool:
    """Whether the policy's route from `start` reaches a random transition before the goal."""
    state = start
    for _ in range(maze.n_states):
        if state == goal:
            return False
        action = policy[state]
        if maze.probs[state, action, 0] < 1.0:
            return True
        state = maze.next_states[state, action, 0]
    return False


def walk_threshold(
    maze: GridMaze, task_id: int, steps_per_cell: float, low: float = 0.99, high: float = 1 - 1e-6
) -> float:
    """Smallest per-step discount at which the risk-neutral policy walks from the task start.

    Bisection on gamma; the policy jumps at `low` and walks at `high` (checked).
    """
    start_ij, goal_ij = maze.tasks[task_id - 1]
    start, goal = maze.centre_state(start_ij), maze.centre_state(goal_ij)

    def jumps(gamma: float) -> bool:
        return enters_teleporter(
            maze, discounted_policy(maze, goal, gamma**steps_per_cell), start, goal
        )

    if not jumps(low) or jumps(high):
        raise ValueError(f"no switch from jumping to walking between {low} and {high}")
    for _ in range(50):
        mid = 0.5 * (low + high)
        low, high = (mid, high) if jumps(mid) else (low, mid)
    return high


def task_rows(maze: GridMaze, gamma: float, task_id: int) -> list[dict]:
    """P(success within EPISODE_STEPS) and E[gamma^H] of each policy, per steps_per_cell."""
    start_ij, goal_ij = maze.tasks[task_id - 1]
    start, goal = maze.centre_state(start_ij), maze.centre_state(goal_ij)
    walk = walk_policy(maze, goal)
    walk_cells = int(np.argmax(hitting_cdf(maze, walk, start, goal, maze.n_states) == 1.0))
    rows = []
    for steps_per_cell in STEPS_PER_CELL:
        n_moves = EPISODE_STEPS // steps_per_cell
        gamma_cell = gamma**steps_per_cell
        risk_neutral = discounted_policy(maze, goal, gamma_cell)
        common = {
            "maze": maze.name,
            "task_id": task_id,
            "walk_cells": walk_cells,
            "steps_per_cell": steps_per_cell,
            "gamma_cell": gamma_cell,
        }
        for name, policy in (("walk", walk), ("risk-neutral", risk_neutral)):
            # E[gamma^H] from the CDF over a long horizon (gamma_cell^moves is then negligible).
            cdf = hitting_cdf(maze, policy, start, goal, 400)
            discounted = float(np.sum(np.diff(cdf, prepend=0.0) * gamma_cell ** np.arange(401)))
            rows.append(
                {
                    **common,
                    "policy": name,
                    "success": cdf[n_moves],
                    "discounted_value": discounted,
                    "gambles": enters_teleporter(maze, policy, start, goal),
                }
            )
        best = best_success(maze, start, goal, n_moves)
        rows.append(
            {
                **common,
                "policy": "best",
                "success": best,
                "discounted_value": np.nan,
                "gambles": np.nan,
            }
        )
    return rows


SERIES = {"walk": "#2a78d6", "risk-neutral": "#eb6834", "best": "#1baf7a"}  # validated palette
MARKERS = {"walk": "o", "risk-neutral": "s", "best": "^"}


def plot(table: pd.DataFrame, path: Path) -> None:
    """Success within 1000 steps vs. ant steps per cell, one panel per teleport task.

    Risk-neutral: max E[gamma^H] with gamma = 0.99. Dotted line: the walk just fits in time.
    """
    tele = table[table["maze"] == "teleport"]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4), sharey=True)
    for ax, (task_id, rows) in zip(axes, tele.groupby("task_id"), strict=True):
        for name, color in SERIES.items():
            part = rows[rows["policy"] == name]
            ax.plot(
                part["steps_per_cell"],
                part["success"],
                color=color,
                lw=2,
                marker=MARKERS[name],
                ms=6,
                label=name,
            )
        walk_cells = int(rows["walk_cells"].iloc[0])
        ax.axvline(EPISODE_STEPS / walk_cells, color="#8a8984", lw=1, ls=":")
        ax.set_title(f"task {task_id}: walk {walk_cells} cells", fontsize=10)
        ax.set_xlabel("ant steps per cell")
        ax.grid(color="#e4e3df", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("P(goal within 1000 steps)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out-dir", type=Path, default=OUT_DIR, help="folder for the CSV tables and the figure"
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, gamma in DISCOUNTS.items():
        maze = GridMaze(name, resolution=1)
        for task_id in range(1, len(maze.tasks) + 1):
            rows += task_rows(maze, gamma, task_id)
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "time_limit.csv", index=False)
    plot(table, args.out_dir / "time_limit.png")
    teleport = GridMaze("teleport", resolution=1)
    thresholds = pd.DataFrame(
        [
            {"task_id": t, "steps_per_cell": 40, "walk_threshold": walk_threshold(teleport, t, 40)}
            for t in range(1, len(teleport.tasks) + 1)
        ]
    )
    thresholds["horizon_steps"] = 1 / (1 - thresholds["walk_threshold"])
    thresholds.to_csv(args.out_dir / "time_limit_thresholds.csv", index=False)
    print(thresholds.round(5).to_string(index=False))
    at40 = table[table["steps_per_cell"] == 40]
    print(
        at40.pivot_table(
            index=["maze", "task_id", "walk_cells"], columns="policy", values="success"
        )
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
