"""A fake FB pi-Switch model on a discrete maze, with known ground truth.

It implements the `FBModel` protocol, so planners and controllers can run on it unchanged.

Construction (n states, latent dimension d, counting measure over states):

- `B` is [n, d] with orthonormal columns times sqrt(n) (B^T B = n I, the upstream orthonormality
  convention). Full rank (d = n) makes every quantity exact. With d < n the basis keeps the d
  smoothest functions: "laplacian" eigenvectors respect walls; "euclidean" cosines on (x, y)
  ignore walls, like a network that is smooth in its input, and so create wormholes.
- A latent z defines the reward r_z = B z_hat / n, with z_hat = z normalised to norm sqrt(d)
  (so B^T r_z = z_hat). `pi*_z` is the exact optimal policy for r_z.
- F(s, z) = M^{pi*_z} B / n + noise, so F(s, z)^T B(s') = (M^{pi*_z} P_B)(s, s') with P_B the
  projection on span(B). In full rank this is M^{pi*_z} exactly.
- Noise: each head adds N_h B / n, with N_h an [n, n] matrix of i.i.d. N(0, sigma^2) entries,
  sigma = rho_noise / (1 - gamma). Then the error on rho(s, w) for nearby pairs has std about
  `rho_noise` (M(w, w) is about 1 / (1 - gamma)). N_h is a fixed function of (seed, head, pi*_z),
  like the fixed error of a trained network. Additive noise is small next to large (short-range)
  values and large next to small (long-range) values, as the design assumes for real FB.
- The low-level policy pi_l(a | s, z) is greedy on the head-mean value F(s', z)^T z of the next
  state, as AWR with the advantage F(s_{t+1}, z)^T z - F(s_t, z)^T z would learn. With noise it
  differs from pi*_z where the value gap between neighbours (about gamma^dist) is below the value
  noise rho_noise / (1 - gamma). `low_policy="optimal"` executes pi*_z instead, which isolates
  errors of the planning values from errors of the controller. `true_successor_measure` gives
  the exact M of the executed policy.
- The baseline pi_h(z | s, z_r) is idealised: the exact argmax over states w != s of the upstream
  proxy switching advantage (it drops -rho F(w, z_w)^T z, as upstream training does), computed
  from this model's F and B, returning normalised B(w*). `high_radius` limits
  w to states within that walking distance of s, like the support of the training subgoals
  (a geometric offset along trajectories). Without it, the proxy advantage counts the goal twice
  and pi_h picks the goal itself, which makes it the same as acting with z_r.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Literal

import numpy as np

from fbplan.contracts import Array, TaskSpec, normalize_z
from fbplan.tabular.maze import ACTION_VECTORS, GridMaze
from fbplan.tabular.mdp import greedy_policy, optimal_policy, successor_measure


@dataclass(frozen=True)
class FakeFBConfig:
    """Settings of the fake model. The defaults give the exact model on the paper's 104-state maze.

    With `discount` 0.9 per fine step, rho ranges from 0.9 (neighbours) to 0.09 (the farthest
    pair of the medium maze, 23 steps).
    """

    maze: str = "medium"
    resolution: int = 2
    discount: float = 0.9
    rank: int | None = None  # None: d = n (exact)
    basis: Literal["laplacian", "euclidean"] = "laplacian"
    rho_noise: float = 0.0
    n_heads: int = 2
    low_policy: Literal["greedy", "optimal"] = "greedy"  # greedy on noisy F, or exact pi*_z
    high_radius: int | None = None  # pi_h subgoals within this walking distance; None: all
    seed: int = 0

    def __post_init__(self) -> None:
        if self.basis not in ("laplacian", "euclidean"):
            raise ValueError(f"unknown basis {self.basis!r}")
        if self.low_policy not in ("greedy", "optimal"):
            raise ValueError(f"unknown low_policy {self.low_policy!r}")


@dataclass(frozen=True)
class _Intention:
    """Everything the fake knows about one normalised latent."""

    forward: np.ndarray  # [n_heads, n, d] F(s, z) for every state
    policy: np.ndarray  # [n] executed low-level action ids


class TabularFB:
    """Fake `FBModel` on a GridMaze; see the module docstring for the construction."""

    def __init__(self, config: FakeFBConfig | None = None) -> None:
        config = FakeFBConfig() if config is None else config
        self.config = config
        self.maze = GridMaze(config.maze, config.resolution)
        n = self.maze.n_states
        d = n if config.rank is None else config.rank
        if not 1 <= d <= n:
            raise ValueError(f"rank must be in [1, {n}], got {d}")
        self.env_name = f"tabular-{config.maze}"
        self.latent_dim = d
        self.discount = config.discount
        self.basis = np.sqrt(n) * _orthonormal_basis(self.maze, config.basis, d)  # B, [n, d]
        self._intentions: dict[bytes, _Intention] = {}
        self._probe = np.random.default_rng(0).standard_normal(d)  # see _unique_latents
        self._subgoals: dict[bytes, np.ndarray] = {}

    # ----- FBModel protocol -----

    def forward(self, obs: Array, z: Array) -> np.ndarray:
        """F(s, z): [n_heads, N, d]."""
        states = self.maze.state_index(np.asarray(obs))
        out = np.empty((self.config.n_heads, len(states), self.latent_dim))
        for z_hat, rows in self._unique_latents(z):
            out[:, rows] = self._intention(z_hat).forward[:, states[rows]]
        return out

    def backward(self, obs: Array) -> np.ndarray:
        """B(s): [N, d], raw."""
        return self.basis[self.maze.state_index(np.asarray(obs))]

    def low_action(self, obs: Array, z: Array) -> np.ndarray:
        """Greedy low-level action as a unit move vector: [N, 2]."""
        states = self.maze.state_index(np.asarray(obs))
        actions = np.empty(len(states), dtype=np.int64)
        for z_hat, rows in self._unique_latents(z):
            actions[rows] = self._intention(z_hat).policy[states[rows]]
        return ACTION_VECTORS[actions]

    def high_latent(self, obs: Array, z_task: Array) -> np.ndarray:
        """Idealised baseline pi_h: normalised B(w*) of the best single switch. [N, d]."""
        states = self.maze.state_index(np.asarray(obs))
        best = np.empty(len(states), dtype=np.int64)
        for z_hat, rows in self._unique_latents(z_task):
            best[rows] = self._best_subgoals(z_hat)[states[rows]]
        return np.asarray(normalize_z(self.basis[best], self.latent_dim))

    def infer_task_latent(self, task_id: int) -> np.ndarray:
        """z_r = B^T r normalised to norm sqrt(d), as upstream infer_latent with uniform data."""
        return np.asarray(normalize_z(self.basis.T @ self.task_reward(task_id), self.latent_dim))

    # ----- ground truth, not part of FBModel -----

    def task_reward(self, task_id: int) -> np.ndarray:
        """Indicator of the fine cells of the task's OGBench goal cell: [n]."""
        reward = np.zeros(self.maze.n_states)
        reward[self.maze.cell_states(self.maze.tasks[task_id - 1][1])] = 1.0
        return reward

    def task_spec(self, task_id: int) -> TaskSpec:
        """TaskSpec of OGBench task `task_id` on this maze, with z_r from infer_task_latent."""
        goal_ij = self.maze.tasks[task_id - 1][1]
        return TaskSpec(
            env_name=self.env_name,
            task_id=task_id,
            z_r=self.infer_task_latent(task_id),
            goal_xy=self.maze.cell_xy(goal_ij),
        )

    def reward_of_latent(self, z: np.ndarray) -> np.ndarray:
        """r_z = B z_hat / n, the reward that pi*_z maximises: [n]."""
        return self.basis @ self._normalize(z) / self.maze.n_states

    def policy(self, z: np.ndarray) -> np.ndarray:
        """Executed low-level action ids for latent z at every state: [n]."""
        return self._intention(self._normalize(z)).policy

    def true_successor_measure(self, z: np.ndarray) -> np.ndarray:
        """Exact M of the executed low-level policy for latent z: [n, n]."""
        return successor_measure(self.maze, self.policy(z), self.discount)

    # ----- internals -----

    def _normalize(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(normalize_z(np.asarray(z, dtype=np.float64), self.latent_dim))

    def _unique_latents(self, z: Array) -> list[tuple[np.ndarray, np.ndarray]]:
        """Group a batch of latents: (normalised latent, row indices) per distinct direction.

        Rows are grouped by a random projection of the rounded latent (np.unique on whole rows
        sorts them and dominates the study's run time); the check below rules out collisions.
        """
        z_hat = self._normalize(np.asarray(z))
        rounded = np.round(z_hat, 9)
        _, first, inverse = np.unique(rounded @ self._probe, return_index=True, return_inverse=True)
        if not np.array_equal(rounded, rounded[first][inverse]):
            raise RuntimeError("two different latents share a projection")
        return [(z_hat[first[i]], np.flatnonzero(inverse == i)) for i in range(len(first))]

    def _intention(self, z_hat: np.ndarray) -> _Intention:
        key = np.round(z_hat, 9).tobytes()
        if key not in self._intentions:
            self._intentions[key] = self._build_intention(z_hat)
        return self._intentions[key]

    def _build_intention(self, z_hat: np.ndarray) -> _Intention:
        n = self.maze.n_states
        pi_star, _ = optimal_policy(self.maze, self.basis @ z_hat / n, self.discount)
        clean = successor_measure(self.maze, pi_star, self.discount) @ self.basis / n
        heads = np.stack([clean + self._noise(pi_star, h) for h in range(self.config.n_heads)])
        if self.config.low_policy == "optimal":
            return _Intention(forward=heads, policy=pi_star)
        greedy = greedy_policy(self.maze, heads.mean(axis=0) @ z_hat)
        return _Intention(forward=heads, policy=greedy)

    def _noise(self, pi_star: np.ndarray, head: int) -> np.ndarray | float:
        if self.config.rho_noise == 0.0:
            return 0.0
        n = self.maze.n_states
        policy_key = zlib.crc32(pi_star.astype(np.int64).tobytes())
        rng = np.random.default_rng([self.config.seed, head, policy_key])
        sigma = self.config.rho_noise / (1.0 - self.discount)
        return sigma * rng.standard_normal((n, n)) @ self.basis / n

    def _best_subgoals(self, z_hat: np.ndarray) -> np.ndarray:
        """argmax_w of the switching advantage for every state s: [n] state ids."""
        key = np.round(z_hat, 9).tobytes()
        if key not in self._subgoals:
            # Training subgoals are at offset >= 1. With exact values, w = s ties with every
            # subgoal on a shortest path, and choosing it would mean "stay".
            dist = self.maze.geodesic_distances()
            radius = np.inf if self.config.high_radius is None else self.config.high_radius
            allowed = (dist > 0) & (dist <= radius)
            self._subgoals[key] = np.argmax(
                np.where(allowed, self._switch_advantages(z_hat), -np.inf), axis=1
            )
        return self._subgoals[key]

    def _switch_advantages(self, z_hat: np.ndarray) -> np.ndarray:
        """Upstream proxy advantage A[s, w] from this model's F and B, with head-mean F."""
        v_task = self._intention(z_hat).forward.mean(axis=0) @ z_hat  # F(s, z)^T z
        adv = np.empty((self.maze.n_states, self.maze.n_states))
        for w in range(self.maze.n_states):
            z_w = self.basis[w]
            f_w = self._intention(self._normalize(z_w)).forward.mean(axis=0)  # F(., z_w)
            m_sw = f_w @ z_w
            rho = m_sw / m_sw[w]
            v_sw = f_w @ z_hat  # F(s, z_w)^T z
            adv[:, w] = v_sw + rho * v_task[w] - v_task
        return adv


def _orthonormal_basis(maze: GridMaze, kind: str, d: int) -> np.ndarray:
    """[n, d] orthonormal columns, smoothest first."""
    if kind == "laplacian":
        return _laplacian_eigenvectors(maze)[:, :d]
    if kind == "euclidean":
        return _cosine_basis(maze, d)
    raise ValueError(f"unknown basis {kind!r}")


def _laplacian_eigenvectors(maze: GridMaze) -> np.ndarray:
    """Eigenvectors of the walking-graph Laplacian, by increasing eigenvalue."""
    n = maze.n_states
    adjacency = np.zeros((n, n))
    for s, adj in enumerate(maze.neighbours()):
        adjacency[s, adj] = 1.0
    laplacian = np.diag(adjacency.sum(1)) - adjacency
    return np.linalg.eigh(laplacian)[1]


def _cosine_basis(maze: GridMaze, d: int) -> np.ndarray:
    """Gram-Schmidt on 2D cosines over the maze bounding box, by increasing spatial frequency.

    The cosines do not see walls: two states on either side of a thin wall get similar features.
    """
    height, width = (np.array(maze.maze_map.shape) * maze.resolution).tolist()
    rows, cols = maze.fine_rc[:, 0] + 0.5, maze.fine_rc[:, 1] + 0.5
    freqs = sorted(
        ((p, q) for p in range(width) for q in range(height)),
        key=lambda pq: ((pq[0] / width) ** 2 + (pq[1] / height) ** 2, pq),
    )
    columns: list[np.ndarray] = []
    for p, q in freqs:
        v = np.cos(np.pi * p * cols / width) * np.cos(np.pi * q * rows / height)
        for u in columns:
            v = v - (u @ v) * u
        norm = np.linalg.norm(v)
        if norm > 1e-8:
            columns.append(v / norm)
        if len(columns) == d:
            return np.stack(columns, axis=1)
    raise ValueError(f"cosine basis has rank {len(columns)} < {d}")
