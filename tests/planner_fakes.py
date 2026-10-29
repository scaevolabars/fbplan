"""Exact tabular FBModel on a small grid maze, for the planner tests.

A latent z selects the target cell t(z) = argmax_j z_j. The policy pi_z walks a shortest
path to t(z) and then stays there. B(s) is the one-hot of the cell of s, so
F(s, z)^T B(s') is the exact successor measure M^{pi_z}(s, s') and rho(s, w) = gamma^d(s, w).
Two knobs imitate FB errors: a wormhole overrides the measure on one F head for one
(source, target) pair; a horizon makes FB and pi_z know nothing about targets farther away
than `horizon` steps (zero measure, and pi_z stands still).
This fake stands in for fbplan/tabular/fake_fb.py until b-theory merges it.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from fbplan.contracts import normalize_z

# A U-shaped corridor: the two bottom ends are 4 cells apart in xy but 10 steps apart on foot.
U_MAZE = (
    "#######",
    "#.....#",
    "#.###.#",
    "#.###.#",
    "#.###.#",
    "#######",
)
MOVES = ((1, 0), (-1, 0), (0, 1), (0, -1))


class GridFB:
    """Implements fbplan.contracts.FBModel exactly on a grid maze. Observations are (x, y)."""

    env_name = "fake-grid"

    def __init__(
        self,
        layout: tuple[str, ...] = U_MAZE,
        discount: float = 0.9,
        horizon: int | None = None,
        wormholes: dict[tuple[int, tuple[int, int], tuple[int, int]], float] | None = None,
        goals: dict[int, tuple[int, int]] | None = None,
    ) -> None:
        self.discount = discount
        self.horizon = horizon
        self.cells = [
            (x, y) for y, row in enumerate(layout) for x, ch in enumerate(row) if ch == "."
        ]
        self.index = {c: i for i, c in enumerate(self.cells)}
        self.latent_dim = len(self.cells)
        self.dist = self._all_distances()
        self.goals = goals or {1: (5, 4)}
        self.measure = np.stack([self._exact_measure()] * 2)  # [head, target, source, cell]
        for (head, src, dst), rho in (wormholes or {}).items():
            j = self.index[dst]
            self.measure[head, j, self.index[src], j] = rho / (1 - discount)
        self.high_calls = 0

    def obs(self, cell: tuple[int, int]) -> np.ndarray:
        return np.asarray(cell, dtype=np.float32)

    def cell_ids(self, obs: np.ndarray) -> np.ndarray:
        xy = np.rint(np.asarray(obs)[:, :2]).astype(int)
        return np.array([self.index[(int(x), int(y))] for x, y in xy])

    def forward(self, obs: np.ndarray, z: np.ndarray) -> np.ndarray:
        targets = np.argmax(np.asarray(normalize_z(np.asarray(z), self.latent_dim)), axis=-1)
        return self.measure[:, targets, self.cell_ids(obs), :]

    def backward(self, obs: np.ndarray) -> np.ndarray:
        return np.eye(self.latent_dim)[self.cell_ids(obs)]

    def low_action(self, obs, z, temperature=0.0, seed=None) -> np.ndarray:
        targets = np.argmax(np.asarray(z), axis=-1)
        return np.array(
            [self._first_move(s, t) for s, t in zip(self.cell_ids(obs), targets, strict=True)],
            dtype=np.float32,
        )

    def high_latent(self, obs, z_task, temperature=0.0, seed=None) -> np.ndarray:
        self.high_calls += 1
        return np.asarray(normalize_z(np.asarray(z_task, dtype=np.float64), self.latent_dim))

    def infer_task_latent(self, task_id: int) -> np.ndarray:
        return 3.0 * np.eye(self.latent_dim)[self.index[self.goals[task_id]]]

    def step(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Grid dynamics for closed-loop tests: move one cell unless a wall blocks it."""
        nxt = (int(round(obs[0] + action[0])), int(round(obs[1] + action[1])))
        return self.obs(nxt) if nxt in self.index else obs

    def _neighbours(self, i: int) -> list[int]:
        x, y = self.cells[i]
        return [self.index[(x + dx, y + dy)] for dx, dy in MOVES if (x + dx, y + dy) in self.index]

    def _all_distances(self) -> np.ndarray:
        n = len(self.cells)
        dist = np.full((n, n), -1, dtype=int)
        for src in range(n):
            dist[src, src] = 0
            queue = deque([src])
            while queue:
                i = queue.popleft()
                for j in self._neighbours(i):
                    if dist[src, j] < 0:
                        dist[src, j] = dist[src, i] + 1
                        queue.append(j)
        return dist

    def _next_cell(self, s: int, t: int) -> int:
        return min(j for j in self._neighbours(s) if self.dist[j, t] == self.dist[s, t] - 1)

    def _in_reach(self, s: int, t: int) -> bool:
        return self.horizon is None or self.dist[s, t] <= self.horizon

    def _first_move(self, s: int, t: int) -> tuple[float, float]:
        if s == t or not self._in_reach(s, t):
            return (0.0, 0.0)
        (x0, y0), (x1, y1) = self.cells[s], self.cells[self._next_cell(s, t)]
        return (float(x1 - x0), float(y1 - y0))

    def _exact_measure(self) -> np.ndarray:
        """M[t, s, :] = sum_k gamma^k 1{s_k = .} for the walk s -> t that then stays at t."""
        n, g = len(self.cells), self.discount
        m = np.zeros((n, n, n))
        for t in range(n):
            for s in filter(lambda s: self._in_reach(s, t), range(n)):
                cur = s
                for k in range(self.dist[s, t]):
                    m[t, s, cur] += g**k
                    cur = self._next_cell(cur, t)
                m[t, s, t] += g ** self.dist[s, t] / (1 - g)
        return m
