"""Discrete copies of the OGBench AntMaze layouts.

Each maze cell is split into `resolution x resolution` fine cells. Every free fine cell is a state.
Observations are the (x, y) centres of fine cells in AntMaze world coordinates, so `obs[:2]` means
the same thing as in the real environment and plots line up. There are 5 actions: stay and the
4 unit moves. A move into a wall keeps the agent in place. In the teleport maze, entering a
teleport-entry maze cell sends the agent to one of the exits with equal probability, as in OGBench.
"""

from __future__ import annotations

from collections import deque
from functools import cached_property

import numpy as np

MAZE_UNIT = 4.0  # OGBench maze cell size in world units
_OFFSET = 4.0  # OGBench: x = j * unit - offset

# Action ids: 0 stay, 1 +x, 2 -x, 3 +y, 4 -y. Continuous actions are these vectors.
ACTION_VECTORS = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.float64)
_FINE_MOVES = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]  # (d_row, d_col) for each action id

_MAPS = {
    "medium": [
        "########",
        "#..##..#",
        "#..#...#",
        "##...###",
        "#..#...#",
        "#.#..#.#",
        "#...#..#",
        "########",
    ],
    "large": [
        "############",
        "#....#.....#",
        "#.##.#.#.#.#",
        "#......#...#",
        "#.####.###.#",
        "#..#.#.....#",
        "##.#.#.#.###",
        "#..#...#...#",
        "############",
    ],
    "giant": [
        "################",
        "#.#......##....#",
        "#.#.##.#.#..##.#",
        "#...#..#...#...#",
        "#.###.######.#.#",
        "#...#...#....#.#",
        "###.#.#..#.#.###",
        "#...#..#...#...#",
        "#.#.#.######.#.#",
        "#.###...#...##.#",
        "#.....#...#....#",
        "################",
    ],
    "teleport": [
        "############",
        "#.....#.#..#",
        "##.#...#..##",
        "##.###.....#",
        "#....#.#.#.#",
        "#.##.#.#.#.#",
        "#.##.#.#.#.#",
        "#....#...#.#",
        "############",
    ],
}

# (init_ij, goal_ij) for the 5 evaluation tasks, copied from ogbench/locomaze/maze.py.
_TASKS = {
    "medium": [
        ((1, 1), (6, 6)),
        ((6, 1), (1, 6)),
        ((5, 3), (4, 2)),
        ((6, 5), (6, 1)),
        ((2, 6), (1, 1)),
    ],
    "large": [
        ((1, 1), (7, 10)),
        ((5, 4), (7, 1)),
        ((7, 4), (1, 10)),
        ((3, 8), (5, 4)),
        ((1, 1), (5, 4)),
    ],
    "giant": [
        ((1, 1), (10, 14)),
        ((1, 14), (10, 1)),
        ((8, 14), (1, 1)),
        ((8, 3), (5, 12)),
        ((5, 9), (3, 8)),
    ],
    "teleport": [
        ((1, 10), (7, 1)),
        ((1, 1), (7, 10)),
        ((5, 6), (7, 10)),
        ((7, 1), (7, 10)),
        ((5, 6), (7, 1)),
    ],
}

_TELEPORT_ENTRIES = {"teleport": [(4, 6), (5, 1)]}
_TELEPORT_EXITS = {"teleport": [(1, 7), (6, 1), (6, 10)]}


class GridMaze:
    """A discrete OGBench maze with its exact transition model.

    Attributes:
        n_states: number of free fine cells.
        xy: [n_states, 2] observation of each state.
        fine_rc: [n_states, 2] (row, col) of each state on the fine grid; rows go along +y.
        next_states, probs: [n_states, 5, K] possible next states of each (state, action) and their
            probabilities. K = 1 for deterministic mazes and 3 for the teleport maze.
        tasks: list of (init_ij, goal_ij) maze cells of the OGBench evaluation tasks.
        entry_states: states inside teleport-entry maze cells (empty without teleports). A move
            into one of them lands on an exit, so the agent never stays there.
    """

    def __init__(self, name: str, resolution: int = 2) -> None:
        if name not in _MAPS:
            raise ValueError(f"unknown maze {name!r}; choose from {sorted(_MAPS)}")
        if resolution < 1:
            raise ValueError("resolution must be >= 1")
        self.name = name
        self.resolution = resolution
        self.maze_map = np.array([[c == "#" for c in row] for row in _MAPS[name]], dtype=bool)
        self.tasks = list(_TASKS[name])

        fine_free = np.kron(~self.maze_map, np.ones((resolution, resolution), dtype=bool))
        self.fine_rc = np.argwhere(fine_free)  # row-major order
        self.n_states = len(self.fine_rc)
        self._state_of_fine = np.full(fine_free.shape, -1, dtype=np.int64)
        self._state_of_fine[self.fine_rc[:, 0], self.fine_rc[:, 1]] = np.arange(self.n_states)

        self.fine_unit = MAZE_UNIT / resolution
        rows, cols = self.fine_rc[:, 0], self.fine_rc[:, 1]
        self.xy = np.stack([self._fine_to_world(cols), self._fine_to_world(rows)], axis=-1)
        self.next_states, self.probs = self._build_transitions()
        self.entry_states = np.array(
            [s for ij in _TELEPORT_ENTRIES.get(name, []) for s in self.cell_states(ij)],
            dtype=np.int64,
        )

    def _fine_to_world(self, index: np.ndarray) -> np.ndarray:
        return (index + 0.5) * self.fine_unit - MAZE_UNIT / 2 - _OFFSET

    def _world_to_fine(self, coord: np.ndarray) -> np.ndarray:
        return np.rint((coord + MAZE_UNIT / 2 + _OFFSET) / self.fine_unit - 0.5).astype(np.int64)

    def cell_states(self, ij: tuple[int, int]) -> np.ndarray:
        """States inside maze cell `ij` (row, col), in row-major order."""
        k = self.resolution
        block = self._state_of_fine[ij[0] * k : (ij[0] + 1) * k, ij[1] * k : (ij[1] + 1) * k]
        if np.any(block < 0):
            raise ValueError(f"maze cell {ij} is a wall")
        return block.ravel()

    def cell_xy(self, ij: tuple[int, int]) -> np.ndarray:
        """World (x, y) of the centre of maze cell `ij`, as OGBench defines it."""
        return np.array([ij[1] * MAZE_UNIT - _OFFSET, ij[0] * MAZE_UNIT - _OFFSET])

    def centre_state(self, ij: tuple[int, int]) -> int:
        """The state of maze cell `ij` that contains its centre (lower-left of the 4 for even k)."""
        k = self.resolution
        return int(self._state_of_fine[ij[0] * k + k // 2, ij[1] * k + k // 2])

    def state_index(self, obs: np.ndarray) -> np.ndarray:
        """Map observations [N, >=2] to state ids [N] by their (x, y). Raises on walls."""
        obs = np.asarray(obs, dtype=np.float64)
        rows, cols = self._world_to_fine(obs[:, 1]), self._world_to_fine(obs[:, 0])
        shape = self._state_of_fine.shape
        inside = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
        states = np.full(len(obs), -1, dtype=np.int64)
        states[inside] = self._state_of_fine[rows[inside], cols[inside]]
        if np.any(states < 0):
            raise ValueError(f"observation in a wall or outside the maze: {obs[states < 0][0, :2]}")
        return states

    def action_from_vector(self, actions: np.ndarray) -> np.ndarray:
        """Map continuous actions [N, 2] to action ids [N]: the dominant axis, or stay if small."""
        actions = np.asarray(actions, dtype=np.float64)
        axis = np.argmax(np.abs(actions), axis=-1)
        value = np.take_along_axis(actions, axis[:, None], axis=-1)[:, 0]
        ids = np.where(axis == 0, np.where(value > 0, 1, 2), np.where(value > 0, 3, 4))
        return np.where(np.abs(value) < 0.5, 0, ids)

    def _build_transitions(self) -> tuple[np.ndarray, np.ndarray]:
        n, n_actions = self.n_states, len(_FINE_MOVES)
        exits = [self.centre_state(ij) for ij in _TELEPORT_EXITS.get(self.name, [])]
        entry_cells = set(_TELEPORT_ENTRIES.get(self.name, []))
        width = max(len(exits), 1)
        next_states = np.zeros((n, n_actions, width), dtype=np.int64)
        probs = np.zeros((n, n_actions, width))
        for s, (r, c) in enumerate(self.fine_rc):
            for a, (dr, dc) in enumerate(_FINE_MOVES):
                target = self._state_of_fine[r + dr, c + dc]  # outer wall keeps indices in range
                target = s if target < 0 else target
                tr, tc = self.fine_rc[target] // self.resolution
                if (tr, tc) in entry_cells:
                    next_states[s, a], probs[s, a] = exits, 1.0 / len(exits)
                else:
                    next_states[s, a, :], probs[s, a, 0] = target, 1.0
        return next_states, probs

    def neighbours(self) -> list[list[int]]:
        """Adjacency lists of the walking graph (unit moves only, no teleports)."""
        out: list[list[int]] = []
        for r, c in self.fine_rc:
            adj = [self._state_of_fine[r + dr, c + dc] for dr, dc in _FINE_MOVES[1:]]
            out.append([int(t) for t in adj if t >= 0])
        return out

    def geodesic_distances(self) -> np.ndarray:
        """[n, n] shortest walking distance in fine steps (BFS, ignores teleports)."""
        return self._geodesic_distances

    @cached_property
    def _geodesic_distances(self) -> np.ndarray:
        adjacency = self.neighbours()
        dist = np.full((self.n_states, self.n_states), np.inf)
        for source in range(self.n_states):
            dist[source, source] = 0
            queue = deque([source])
            while queue:
                u = queue.popleft()
                for v in adjacency[u]:
                    if dist[source, v] == np.inf:
                        dist[source, v] = dist[source, u] + 1
                        queue.append(v)
        return dist
