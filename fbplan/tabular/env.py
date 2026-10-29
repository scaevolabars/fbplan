"""A gymnasium environment and an episode loop for the discrete maze.

The API follows the OGBench navigate envs: `reset(options={"task_id": k})` starts in the task's
init cell (a random fine cell of it, like OGBench's start noise), `info["success"]` is 1.0 when the
agent is in the goal cell, and the episode terminates there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from fbplan.contracts import Controller
from fbplan.tabular.fake_fb import TabularFB
from fbplan.tabular.maze import GridMaze


class TabularMazeEnv(gym.Env):
    """OGBench-like navigation env on a GridMaze. Observations are (x, y), actions are 2D."""

    def __init__(self, maze: GridMaze, max_episode_steps: int = 1000) -> None:
        self.maze = maze
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float64)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float64)
        self._goal_states: np.ndarray = np.empty(0, dtype=np.int64)
        self._state = 0
        self._steps = 0

    @property
    def state(self) -> int:
        """Current state id (privileged)."""
        return self._state

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        task_id = (options or {}).get("task_id", 1)
        init_ij, goal_ij = self.maze.tasks[task_id - 1]
        self._goal_states = self.maze.cell_states(goal_ij)
        self._state = int(self.np_random.choice(self.maze.cell_states(init_ij)))
        self._steps = 0
        return self.maze.xy[self._state].copy(), {"goal": self.maze.cell_xy(goal_ij)}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = self.maze.action_from_vector(np.asarray(action, dtype=np.float64)[None])[0]
        outcome = self.np_random.choice(
            self.maze.probs.shape[-1], p=self.maze.probs[self._state, a]
        )
        self._state = int(self.maze.next_states[self._state, a, outcome])
        self._steps += 1
        success = bool(np.isin(self._state, self._goal_states))
        truncated = not success and self._steps >= self.max_episode_steps
        obs = self.maze.xy[self._state].copy()
        return obs, float(success), success, truncated, {"success": float(success)}


@dataclass(frozen=True)
class Rollout:
    """One episode: visited states (including the start), success and step count."""

    states: list[int]
    success: bool
    steps: int


def rollout(
    controller: Controller, model: TabularFB, task_id: int, seed: int, max_steps: int = 1000
) -> Rollout:
    """Run one episode of `controller` on the model's maze for an OGBench task."""
    env = TabularMazeEnv(model.maze, max_episode_steps=max_steps)
    obs, _ = env.reset(seed=seed, options={"task_id": task_id})
    controller.reset(model.task_spec(task_id), seed)
    states = [env.state]
    while True:
        obs, _, terminated, truncated, _ = env.step(controller.act(obs))
        states.append(env.state)
        if terminated or truncated:
            return Rollout(states=states, success=terminated, steps=len(states) - 1)
