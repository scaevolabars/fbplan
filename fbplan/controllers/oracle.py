"""Privileged oracle-waypoint controller.

The oracle knows the true maze map and the goal position. It follows the shortest cell path
to the goal and gives pi_l the intention of a waypoint a few cells ahead. It is a reference,
not an upper bound for high-level methods (docs/design.md §5): with single-state waypoints it
scores below the baseline. It is a diagnostic, not a method: tables label it privileged
(fbplan.controllers.PRIVILEGED).

Waypoint intentions: "state" is B(w) of the one dataset state w nearest to the cell centre;
"region" is the mean of B(s) over the dataset states near the centre, the way upstream builds
the task latent z_r. "state" is far weaker: B(w) carries the pose and the velocity of w, and
pi_l chases them (medium, eval seed 0, 250 episodes, lookahead 1/2/3: state 44.8/56.4/65.2 %,
region with radius 0.5: 80.8/86.4/78.0 %; results/medium/oracle_sweep/). Defaults: region,
radius 0.5 (the goal radius of the task reward that defines z_r), lookahead 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from fbplan.contracts import FBModel, TaskSpec
from fbplan.frozen import load_dataset, make_env
from fbplan.maze import Cell, MazeGeometry


def nearest_observations(observations: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    """For each point, the observation whose xy (obs[:2]) is nearest. One pass per point
    keeps memory at one distance vector, also for 1M-row datasets."""
    xy = observations[:, :2]
    idx = [int(np.argmin(np.sum((xy - p) ** 2, axis=1))) for p in points_xy]
    return observations[idx]


def region_latents(
    model: FBModel,
    observations: np.ndarray,
    centres_xy: np.ndarray,
    radius: float,
    batch_size: int = 8192,
) -> np.ndarray:
    """[C, d] mean B(s) over the dataset states within `radius` (in xy) of each centre."""
    b = np.concatenate([
        np.asarray(model.backward(observations[i : i + batch_size]))
        for i in range(0, len(observations), batch_size)
    ])
    latents = []
    for centre in centres_xy:
        near = np.linalg.norm(observations[:, :2] - centre, axis=1) <= radius
        if not near.any():
            raise ValueError(f"no dataset state within {radius} of {centre}")
        latents.append(b[near].mean(axis=0))
    return np.stack(latents)


@dataclass(frozen=True)
class OracleConfig:
    """lookahead: the waypoint is this many cells ahead on the shortest path.
    waypoint: "region" or "state" (see the module docstring).
    radius: xy radius of a region waypoint.
    """

    lookahead: int = 2
    waypoint: Literal["region", "state"] = "region"
    radius: float = 0.5

    def __post_init__(self) -> None:
        if self.waypoint not in ("region", "state"):
            raise ValueError(f"unknown waypoint kind {self.waypoint!r}")


class OracleWaypointController:
    """Privileged: pi_l(a | s, z) with z the waypoint intention of the cell `lookahead` moves
    ahead on the true shortest path to the goal."""

    name = "oracle"

    def __init__(
        self,
        model: FBModel,
        maze: MazeGeometry,
        dataset_obs: np.ndarray,
        config: OracleConfig | None = None,
    ) -> None:
        self.config = config or OracleConfig()
        self._model = model
        self._maze = maze
        cells = maze.free_cells()
        centres = np.stack([maze.centre(c) for c in cells])
        if self.config.waypoint == "region":
            latents = region_latents(model, dataset_obs, centres, self.config.radius)
        else:
            latents = np.asarray(model.backward(nearest_observations(dataset_obs, centres)))
        self._latents = dict(zip(cells, latents, strict=True))
        self._steps = np.zeros_like(maze.maze_map)
        self._target: Cell = cells[0]

    def reset(self, task: TaskSpec, seed: int) -> None:
        self._steps = self._maze.bfs_steps(self._maze.cell_of(task.goal_xy))

    def act(self, obs: np.ndarray) -> np.ndarray:
        cell = self._maze.cell_of(obs[:2])
        for _ in range(self.config.lookahead):
            cell = self._maze.next_cell(cell, self._steps)
        self._target = cell
        z = self._latents[cell]
        return np.asarray(self._model.low_action(obs[None], z[None])[0])

    def step_info(self) -> dict[str, float | int | str]:
        return {"target_i": self._target[0], "target_j": self._target[1]}


def make_oracle(model: FBModel, **options: object) -> OracleWaypointController:
    """Oracle on the maze of `model.env_name`, waypoints from the validation split.

    The validation split is always present (z_r inference reads it too), and its 100k
    states cover every cell, so the waypoints need no download of the training split.
    """
    maze = MazeGeometry.from_env(make_env(model.env_name))
    observations = load_dataset(model.env_name, "val")["observations"]
    return OracleWaypointController(model, maze, observations, OracleConfig(**options))
