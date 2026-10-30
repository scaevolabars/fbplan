"""Privileged oracle-waypoint controller for the stochastic teleport maze.

In OGBench teleport, an agent that comes within `trigger_xy` (1.5) xy units of an entry cell
centre jumps to one of the exit cells, chosen uniformly at random. Exit (1, 7) has walls on
all four sides: an agent that lands there can never leave. The oracle of
`fbplan.controllers.oracle` plans with BFS on the map, so its path goes through an entry on
tasks 1, 3, 4 and 5, and it has no plan in the trap. This oracle plans on the cell MDP with the
jumps (entering an entry cell = landing on a uniform exit):

- objective "reach" (default): never enter a teleporter. Each jump ends in the trap with
  probability 1/3, and every cell except the trap reaches every goal on foot, so walking
  maximises the probability to reach the goal; among walks it takes the shortest.
- objective "discounted": maximise E[gamma_cell^H] over cell moves, with
  gamma_cell = gamma ** steps_per_cell. This is the risk-neutral value that FB estimates.
  At gamma = 0.99 and 40 ant steps per cell it takes the teleport gamble whenever the walk is
  long, although the gamble fails with probability >= 1/2 (docs/theory.md §11).

Waypoint intentions are region means of B, as in the plain oracle. An entry cell has almost no
dataset states within 0.5 of its centre (the agent jumps away), so its region is the trigger
disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import numpy as np

from fbplan.contracts import FBModel, TaskSpec
from fbplan.controllers.oracle import region_latents
from fbplan.frozen import load_dataset, make_env
from fbplan.maze import Cell, MazeGeometry

_MOVES = ((-1, 0), (0, -1), (1, 0), (0, 1))  # OGBench order, as in fbplan.controllers.oracle
_VI_TOL = 1e-13


@dataclass(frozen=True)
class Teleports:
    """Entry and exit cells of the teleporters and the xy radius that triggers a jump."""

    entries: tuple[Cell, ...]
    exits: tuple[Cell, ...]
    trigger_xy: float

    @classmethod
    def from_env(cls, env: gym.Env) -> Teleports:
        info = env.unwrapped._teleport_info
        if info is None:
            raise ValueError("this maze has no teleporters")
        return cls(
            entries=tuple(tuple(ij) for ij in info["teleport_in_ijs"]),
            exits=tuple(tuple(ij) for ij in info["teleport_out_ijs"]),
            trigger_xy=1.5 * info["teleport_radius"],  # the check in OGBench's step()
        )


def _neighbours(maze_map: np.ndarray, cell: Cell) -> list[Cell]:
    """Free 4-neighbours of `cell` in `maze_map` (the cell itself may be a wall there)."""
    candidates = [(cell[0] + di, cell[1] + dj) for di, dj in _MOVES]
    return [c for c in candidates if maze_map[c] == 0]  # the outer wall keeps c in range


def reach_plan(maze: MazeGeometry, teleports: Teleports, goal: Cell) -> dict[Cell, Cell]:
    """Next cell of the shortest walk to `goal` that enters no teleporter, for every cell that
    has one. An agent in an entry cell (outside the trigger disk) steps out of it."""
    safe_map = maze.maze_map.copy()
    for entry in teleports.entries:
        safe_map[entry] = 1
    steps = MazeGeometry(safe_map, maze.unit, maze.offset_xy).bfs_steps(goal)
    plan = {goal: goal}
    for cell in maze.free_cells():
        options = [n for n in _neighbours(safe_map, cell) if steps[n] >= 0]
        if cell != goal and options:
            plan[cell] = min(options, key=lambda n: steps[n])  # first in _MOVES order on ties
    return plan


def _landing_values(values: np.ndarray, maze_map: np.ndarray, teleports: Teleports) -> np.ndarray:
    """Expected value after a move into each cell: an entry cell lands on a uniform exit."""
    landing = np.where(maze_map == 0, values, 0.0)
    exits = np.mean([values[e] for e in teleports.exits])
    for entry in teleports.entries:
        landing[entry] = exits
    return landing


def discounted_values(
    maze: MazeGeometry,
    teleports: Teleports,
    goal: Cell,
    cell_discount: float,
    max_iters: int = 100_000,
) -> np.ndarray:
    """E[cell_discount^H] of the best policy, H = cell moves to `goal` (value iteration).

    [rows, cols] array; 0 on walls and in cells that cannot reach the goal (the trap).
    Raises if the values do not converge in `max_iters` sweeps (cell_discount must be < 1).
    """
    cells = [c for c in maze.free_cells() if c != goal]
    values = np.zeros(maze.maze_map.shape)
    values[goal] = 1.0
    for _ in range(max_iters):
        landing = _landing_values(values, maze.maze_map, teleports)
        new = values.copy()
        for cell in cells:
            new[cell] = cell_discount * max(
                (landing[n] for n in _neighbours(maze.maze_map, cell)), default=0.0
            )
        if np.max(np.abs(new - values)) < _VI_TOL:
            return new
        values = new
    raise RuntimeError(f"value iteration did not converge in {max_iters} sweeps")


def discounted_plan(
    maze: MazeGeometry, teleports: Teleports, goal: Cell, cell_discount: float
) -> dict[Cell, Cell]:
    """Next cell of the policy that maximises E[cell_discount^H]; no entry for the trap."""
    values = discounted_values(maze, teleports, goal, cell_discount)
    landing = _landing_values(values, maze.maze_map, teleports)
    plan = {goal: goal}
    for cell in maze.free_cells():
        options = _neighbours(maze.maze_map, cell)
        if cell != goal and values[cell] > 0:
            plan[cell] = max(options, key=lambda n: landing[n])  # first in _MOVES order on ties
    return plan


@dataclass(frozen=True)
class TeleportOracleConfig:
    """objective: "reach" or "discounted" (see the module docstring).
    lookahead: the waypoint is this many cells ahead; it stops at an entry cell, because the
        cell after a jump is random.
    radius: xy radius of a waypoint region (entry cells use the trigger radius).
    steps_per_cell: ant steps per cell move for gamma_cell. 40 is the median of successful
        medium episodes (baseline and oracle: 29-56 per task).
    follow: "pi_l" acts with pi_l(a | s, z_w); "pi_h" gives z_w to pi_h as its task and acts
        with pi_l on the intention that pi_h picks (as planner_high in controllers/planner.py).
    """

    objective: Literal["reach", "discounted"] = "reach"
    lookahead: int = 2
    radius: float = 0.5
    steps_per_cell: float = 40.0
    follow: Literal["pi_l", "pi_h"] = "pi_l"

    def __post_init__(self) -> None:
        if self.objective not in ("reach", "discounted"):
            raise ValueError(f"unknown objective {self.objective!r}")
        if self.follow not in ("pi_l", "pi_h"):
            raise ValueError(f"unknown follow {self.follow!r}")


class TeleportOracleController:
    """Privileged: pi_l(a | s, z) with z the region intention of the cell `lookahead` moves
    ahead on the teleport-aware plan to the goal."""

    name = "oracle_teleport"

    def __init__(
        self,
        model: FBModel,
        maze: MazeGeometry,
        teleports: Teleports,
        dataset_obs: np.ndarray,
        config: TeleportOracleConfig | None = None,
    ) -> None:
        self.config = config or TeleportOracleConfig()
        self._model = model
        self._maze = maze
        self._teleports = teleports
        cells = [c for c in maze.free_cells() if c not in teleports.entries]
        latents = region_latents(
            model, dataset_obs, np.stack([maze.centre(c) for c in cells]), self.config.radius
        )
        entry_latents = region_latents(
            model,
            dataset_obs,
            np.stack([maze.centre(e) for e in teleports.entries]),
            teleports.trigger_xy,
        )
        self._latents = dict(zip(cells, latents, strict=True))
        self._latents.update(zip(teleports.entries, entry_latents, strict=True))
        self._plan: dict[Cell, Cell] = {}
        self._target: Cell = cells[0]

    def reset(self, task: TaskSpec, seed: int) -> None:
        goal = self._maze.cell_of(task.goal_xy)
        if self.config.objective == "reach":
            self._plan = reach_plan(self._maze, self._teleports, goal)
        else:
            gamma_cell = self._model.discount**self.config.steps_per_cell
            self._plan = discounted_plan(self._maze, self._teleports, goal, gamma_cell)

    def act(self, obs: np.ndarray) -> np.ndarray:
        target = self._maze.cell_of(obs[:2])
        for _ in range(self.config.lookahead):
            if target not in self._plan:  # the trap: no plan, stay
                break
            target = self._plan[target]
            if target in self._teleports.entries:
                break
        self._target = target
        z = self._latents[target][None]
        if self.config.follow == "pi_h":
            z = self._model.high_latent(obs[None], z)
        return np.asarray(self._model.low_action(obs[None], z)[0])

    def step_info(self) -> dict[str, float | int | str]:
        return {"target_i": self._target[0], "target_j": self._target[1]}


def make_teleport_oracle(model: FBModel, **options) -> TeleportOracleController:
    """Teleport-aware oracle on the maze of `model.env_name`, waypoints from the val split."""
    env = make_env(model.env_name)
    observations = load_dataset(model.env_name, "val")["observations"]
    return TeleportOracleController(
        model,
        MazeGeometry.from_env(env),
        Teleports.from_env(env),
        observations,
        TeleportOracleConfig(**options),
    )
