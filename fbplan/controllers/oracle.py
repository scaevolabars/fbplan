"""Privileged oracle-waypoint controller and the maze geometry it uses.

The oracle knows the true maze map and the goal position. It follows the shortest cell path
to the goal and gives pi_l the intention z = B(w) of a waypoint w a few cells ahead.
Its success rate estimates the best that a high-level method can get with this frozen pi_l.
It is a diagnostic, not a method: label it privileged in every table.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from fbplan.contracts import FBModel, TaskSpec

Cell = tuple[int, int]
_NEIGHBOURS = ((-1, 0), (0, -1), (1, 0), (0, 1))  # OGBench get_oracle_subgoal order


@dataclass(frozen=True)
class MazeGeometry:
    """Cell grid of an OGBench maze (1 = wall). Cell (i, j) has its centre at
    (j * unit - offset_x, i * unit - offset_y), as in OGBench `ij_to_xy`."""

    maze_map: np.ndarray
    unit: float
    offset_xy: tuple[float, float]

    @classmethod
    def from_env(cls, env: gym.Env) -> MazeGeometry:
        maze = env.unwrapped
        return cls(maze.maze_map.copy(), float(maze._maze_unit), (maze._offset_x, maze._offset_y))

    def cell_of(self, xy: np.ndarray) -> Cell:
        """The cell that contains xy (OGBench `xy_to_ij`)."""
        half = 0.5 * self.unit
        i = int((xy[1] + self.offset_xy[1] + half) / self.unit)
        j = int((xy[0] + self.offset_xy[0] + half) / self.unit)
        return i, j

    def centre(self, cell: Cell) -> np.ndarray:
        i, j = cell
        return np.array([j * self.unit - self.offset_xy[0], i * self.unit - self.offset_xy[1]])

    def free_cells(self) -> list[Cell]:
        return [(int(i), int(j)) for i, j in zip(*np.nonzero(self.maze_map == 0), strict=True)]

    def _free_neighbours(self, cell: Cell) -> list[Cell]:
        rows, cols = self.maze_map.shape
        candidates = [(cell[0] + di, cell[1] + dj) for di, dj in _NEIGHBOURS]
        return [
            (i, j) for i, j in candidates
            if 0 <= i < rows and 0 <= j < cols and self.maze_map[i, j] == 0
        ]

    def bfs_steps(self, goal: Cell) -> np.ndarray:
        """Cell moves from every cell to `goal` (4-connected). -1: wall or unreachable."""
        steps = np.full(self.maze_map.shape, -1)
        steps[goal] = 0
        frontier = [goal]
        while frontier:
            next_frontier = []
            for cell in frontier:
                for nb in self._free_neighbours(cell):
                    if steps[nb] == -1:
                        steps[nb] = steps[cell] + 1
                        next_frontier.append(nb)
            frontier = next_frontier
        return steps

    def next_cell(self, cell: Cell, steps: np.ndarray) -> Cell:
        """One move along a shortest path, given `steps` = bfs_steps(goal)."""
        if steps[cell] < 0:
            raise ValueError(f"cell {cell} cannot reach the goal")
        if steps[cell] == 0:
            return cell
        return next(nb for nb in self._free_neighbours(cell) if steps[nb] == steps[cell] - 1)

    def path(self, start: Cell, goal: Cell) -> list[Cell]:
        """Shortest cell path from start to goal, both included."""
        steps = self.bfs_steps(goal)
        path = [start]
        while path[-1] != goal:
            path.append(self.next_cell(path[-1], steps))
        return path


def nearest_observations(observations: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    """For each point, the observation whose xy (obs[:2]) is nearest. One pass per point
    keeps memory at one distance vector, also for 1M-row datasets."""
    xy = observations[:, :2]
    idx = [int(np.argmin(np.sum((xy - p) ** 2, axis=1))) for p in points_xy]
    return observations[idx]


@dataclass(frozen=True)
class OracleConfig:
    lookahead: int = 1  # the waypoint is this many cells ahead on the shortest path


class OracleWaypointController:
    """Privileged: pi_l(a | s, B(w)), w = dataset state nearest to the centre of the cell
    `lookahead` moves ahead on the true shortest path to the goal."""

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
        waypoints = nearest_observations(dataset_obs, np.stack([maze.centre(c) for c in cells]))
        latents = np.asarray(model.backward(waypoints))
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
