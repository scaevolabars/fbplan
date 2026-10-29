"""Maze geometry of an OGBench AntMaze: cells, shortest cell paths, exact geodesic distances.

The privileged oracle, the held-out goal selection, the failure classes, the FB distance
calibration and the figures all use it, so it lives outside any controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import gymnasium as gym
import numpy as np

Cell = tuple[int, int]
_NEIGHBOURS = ((-1, 0), (0, -1), (1, 0), (0, 1))  # OGBench get_oracle_subgoal order
_WALL_SHRINK = 1e-6  # lets a path touch a wall edge or corner without entering the wall
_CHUNK_ELEMENTS = 2_000_000  # memory bound for the broadcasted pair computations


def _pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a[:, None] - b[None], axis=-1)


def _row_chunks(n_rows: int, row_size: int):
    step = max(1, _CHUNK_ELEMENTS // max(row_size, 1))
    return (slice(i, i + step) for i in range(0, n_rows, step))


def _min_plus(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(x ⊗ y)[i, j] = min_k x[i, k] + y[k, j]: path lengths through one intermediate node."""
    out = np.empty((x.shape[0], y.shape[1]))
    for rows in _row_chunks(x.shape[0], y.size):
        out[rows] = np.min(x[rows, :, None] + y[None], axis=1)
    return out


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

    @cached_property
    def _wall_boxes(self) -> np.ndarray:
        """[W, 4] (x_min, y_min, x_max, y_max) of each wall cell, shrunk by _WALL_SHRINK."""
        centres = np.stack([self.centre((i, j)) for i, j in np.argwhere(self.maze_map == 1)])
        half = 0.5 * self.unit - _WALL_SHRINK
        return np.concatenate([centres - half, centres + half], axis=1)

    def _visible(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """[N, M]: True where the segment from p_i to q_j enters no wall (slab test)."""
        lo, hi = self._wall_boxes[:, :2], self._wall_boxes[:, 2:]
        out = np.empty((len(p), len(q)), dtype=bool)
        for rows in _row_chunks(len(p), len(q) * len(lo) * 2):
            start = p[rows, None, None, :]  # [n, 1, 1, 2]
            delta = (q[None] - p[rows, None])[:, :, None, :]  # [n, M, 1, 2]
            parallel = delta == 0
            inside = (start > lo) & (start < hi)  # [n, 1, W, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                t_lo, t_hi = (lo - start) / delta, (hi - start) / delta
            t_in = np.where(parallel, np.where(inside, -np.inf, np.inf), np.minimum(t_lo, t_hi))
            t_out = np.where(parallel, np.where(inside, np.inf, -np.inf), np.maximum(t_lo, t_hi))
            enter, leave = t_in.max(-1), t_out.min(-1)  # [n, M, W]
            out[rows] = ~((enter < leave) & (leave > 0) & (enter < 1)).any(-1)
        return out

    @cached_property
    def _corner_graph(self) -> tuple[np.ndarray, np.ndarray]:
        """Bend corners and the shortest distances between them (Floyd-Warshall).

        A shortest path among square walls bends only at grid vertices where exactly one of
        the four cells around the vertex is a wall (a convex wall corner).
        """
        walls = np.pad(self.maze_map == 1, 1, constant_values=True).astype(int)
        n_walls = walls[:-1, :-1] + walls[:-1, 1:] + walls[1:, :-1] + walls[1:, 1:]
        vi, vj = np.nonzero(n_walls == 1)  # vertex = (min x, min y) corner of cell (vi, vj)
        corners = np.stack([(vj - 0.5) * self.unit - self.offset_xy[0],
                            (vi - 0.5) * self.unit - self.offset_xy[1]], axis=1)
        dist = np.where(self._visible(corners, corners), _pairwise(corners, corners), np.inf)
        for k in range(len(corners)):
            dist = np.minimum(dist, dist[:, k : k + 1] + dist[k : k + 1, :])
        return corners, dist

    def geodesic_distances(self, a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
        """[N, M] length of the shortest path in free space (walls solid, agent a point).

        Exact for square walls: the path is straight or bends at convex wall corners.
        Unreachable pairs get inf.
        """
        corners, between = self._corner_graph
        a_to_c = np.where(self._visible(a_xy, corners), _pairwise(a_xy, corners), np.inf)
        b_to_c = np.where(self._visible(b_xy, corners), _pairwise(b_xy, corners), np.inf)
        via_corners = _min_plus(_min_plus(a_to_c, between), b_to_c.T)
        direct = np.where(self._visible(a_xy, b_xy), _pairwise(a_xy, b_xy), np.inf)
        return np.minimum(direct, via_corners)

    def path(self, start: Cell, goal: Cell) -> list[Cell]:
        """Shortest cell path from start to goal, both included."""
        steps = self.bfs_steps(goal)
        path = [start]
        while path[-1] != goal:
            path.append(self.next_cell(path[-1], steps))
        return path


def draw_walls(ax: Any, maze: MazeGeometry, color: str = "#e4e3df") -> None:
    """Draw the wall cells on a matplotlib Axes, fit the limits to the maze, hide the axes."""
    from matplotlib.patches import Rectangle  # only figures need matplotlib

    for i, j in np.argwhere(maze.maze_map == 1):
        x, y = maze.centre((i, j)) - maze.unit / 2
        ax.add_patch(Rectangle((x, y), maze.unit, maze.unit, color=color, lw=0))
    rows, cols = maze.maze_map.shape
    lo = maze.centre((0, 0)) - maze.unit / 2
    hi = maze.centre((rows - 1, cols - 1)) + maze.unit / 2
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.axis("off")
