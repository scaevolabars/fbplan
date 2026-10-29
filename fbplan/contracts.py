"""Shared contracts between components.

Every component codes against the types in this module. The coordinator owns this file:
if you need a change, write the request in your docs/status/<component>.md and tell the user.
Math and notation are in docs/design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

Array = np.ndarray | jax.Array


def normalize_z(z: Array, latent_dim: int) -> Array:
    """Scale latents to norm sqrt(d) along the last axis, as upstream FB pi-Switch does."""
    xp = np if isinstance(z, np.ndarray) else jnp
    norm = xp.linalg.norm(z, axis=-1, keepdims=True)
    return z / (norm + 1e-8) * np.sqrt(latent_dim)


@dataclass(frozen=True)
class TaskSpec:
    """One evaluation task of an OGBench environment.

    `task_id` 1..5 are the OGBench eval tasks. Ids above `fbplan.evaluate.HELDOUT_OFFSET` (100)
    are held-out goals for hyperparameter selection: goal k = task_id - 100 of
    `results/heldout_goals/<env>.csv`. Strata, pairing keys and reset seeds use `task_id` as is.
    """

    env_name: str  # e.g. "antmaze-medium-navigate-v0"
    task_id: int  # 1..5 eval tasks; 101.. held-out goals
    z_r: np.ndarray  # [d] raw task embedding from FBModel.infer_task_latent
    goal_xy: np.ndarray  # [2] privileged: only oracle controllers and diagnostics may read it


class FBModel(Protocol):
    """A frozen FB pi-Switch checkpoint.

    All methods take a batch on the leading axis. Latent inputs `z` can have any scale:
    the implementation normalizes them to norm sqrt(d) before it calls a network.
    Dot products such as F(s, z_w)^T B(w) use the raw vector that the caller passes.
    """

    env_name: str
    latent_dim: int  # 128 in the provided checkpoints
    discount: float  # gamma used in training (0.99 medium/large/teleport, 0.995 giant)

    def forward(self, obs: Array, z: Array) -> Array:
        """F(s, z). obs [N, obs_dim], z [N, d] -> [n_heads, N, d] (n_heads = 2)."""
        ...

    def backward(self, obs: Array) -> Array:
        """B(s). obs [N, obs_dim] -> [N, d], raw (not normalized)."""
        ...

    def low_action(
        self, obs: Array, z: Array, temperature: float = 0.0, seed: jax.Array | None = None
    ) -> Array:
        """Low-level policy pi_l(a | s, z). Returns actions [N, act_dim] clipped to [-1, 1]."""
        ...

    def high_latent(
        self, obs: Array, z_task: Array, temperature: float = 0.0, seed: jax.Array | None = None
    ) -> Array:
        """Baseline high-level policy pi_h(z | s, z_task). Returns normalized intentions [N, d]."""
        ...

    def infer_task_latent(self, task_id: int) -> np.ndarray:
        """Task embedding z_r [d] for an evaluation task, computed as upstream main.py does."""
        ...


class Controller(Protocol):
    """A complete agent for one episode: maps observations to actions.

    The evaluation loop calls reset() once per episode, then act() once per step.
    """

    name: str

    def reset(self, task: TaskSpec, seed: int) -> None:
        """Prepare for a new episode. `seed` makes any internal randomness reproducible."""
        ...

    def act(self, obs: np.ndarray) -> np.ndarray:
        """obs [obs_dim] -> action [act_dim] in [-1, 1]."""
        ...

    def step_info(self) -> dict[str, float | int | str]:
        """Optional per-step debug values (for example the current waypoint id). Can be {}."""
        ...


@dataclass(frozen=True)
class EpisodeRecord:
    """One row of the evaluation CSV. Results from all components use this schema."""

    method: str  # Controller.name
    config_hash: str  # short hash of the controller config
    env_name: str
    task_id: int  # see TaskSpec.task_id
    eval_seed: int  # outer seed; reset seed = f(eval_seed, task_id, episode)
    episode: int
    success: bool
    steps: int  # steps until success or the time limit
    final_goal_dist: float  # xy distance to the goal at episode end
    wall_time_s: float
