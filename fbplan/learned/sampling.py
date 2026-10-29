"""Offline batch sampling on the accelerator.

Upstream samples every batch with numpy on the host (`utils/datasets.py:HGCDataset`). We draw
from the same distribution in JAX, so that a whole chunk of training steps runs inside one
`lax.scan` without a host round trip per step.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np


@flax.struct.dataclass
class TrajectoryData:
    """Dataset states on the device, with the trajectory end of every state."""

    observations: jax.Array  # [N, obs_dim]
    final_idx: jax.Array  # [N] int32: index of the last state of the trajectory that holds i

    @property
    def size(self) -> int:
        return self.observations.shape[0]


def make_trajectory_data(observations: np.ndarray, terminals: np.ndarray) -> TrajectoryData:
    """Build `TrajectoryData` from flat OGBench arrays. `terminals[i] > 0` ends a trajectory."""
    (terminal_locs,) = np.nonzero(terminals > 0)
    if len(terminal_locs) == 0 or terminal_locs[-1] != len(observations) - 1:
        raise ValueError("the last state of the dataset must be a trajectory end")
    final_idx = terminal_locs[np.searchsorted(terminal_locs, np.arange(len(observations)))]
    return TrajectoryData(
        observations=jnp.asarray(observations, dtype=jnp.float32),
        final_idx=jnp.asarray(final_idx, dtype=jnp.int32),
    )


@dataclass(frozen=True)
class HighBatchSpec:
    """The agent config keys that define the high-level batch distribution."""

    batch_size: int
    discount: float
    actor_geom_sample: bool
    actor_p_randomgoal: float

    @classmethod
    def from_agent_config(cls, config: Mapping[str, Any]) -> HighBatchSpec:
        return cls(
            batch_size=int(config["batch_size"]),
            discount=float(config["discount"]),
            actor_geom_sample=bool(config["actor_geom_sample"]),
            actor_p_randomgoal=float(config["actor_p_randomgoal"]),
        )


def geometric_future_idx(
    key: jax.Array, idxs: jax.Array, final_idx: jax.Array, discount: float
) -> jax.Array:
    """State at offset k ~ Geometric(1 - discount), k >= 1, clipped at the trajectory end."""
    offsets = jax.random.geometric(key, 1.0 - discount, idxs.shape, dtype=jnp.int32)
    return jnp.minimum(idxs + offsets, final_idx)


def uniform_future_idx(key: jax.Array, idxs: jax.Array, final_idx: jax.Array) -> jax.Array:
    """Uniform state in [idx + 1, end] of the same trajectory (the end state itself at the end).

    Upstream rounds `start * u + end * (1 - u)`. We round the equal `end - (end - start) * u`,
    because float32 cannot hold fractions of indices near 1e6.
    """
    start = jnp.minimum(idxs + 1, final_idx)
    u = jax.random.uniform(key, idxs.shape)
    return final_idx - jnp.round((final_idx - start) * u).astype(jnp.int32)


def sample_high_batch(data: TrajectoryData, key: jax.Array, spec: HighBatchSpec) -> dict:
    """The keys that `FBpiSwitchAgent.high_actor_loss` reads, distributed as `HGCDataset.sample`.

    observations: uniform dataset states s.
    high_actor_goals: the task goal g, a future state of the same trajectory (uniform or
        geometric offset), or a uniform dataset state with probability actor_p_randomgoal.
    high_actor_targets: the subgoal w, at a geometric offset from s in the same trajectory.
        It is not clipped at g, as upstream.
    """
    k_idx, k_goal, k_random, k_pick, k_target = jax.random.split(key, 5)
    shape = (spec.batch_size,)
    idxs = jax.random.randint(k_idx, shape, 0, data.size, dtype=jnp.int32)
    final = data.final_idx[idxs]

    if spec.actor_geom_sample:
        traj_goal_idxs = geometric_future_idx(k_goal, idxs, final, spec.discount)
    else:
        traj_goal_idxs = uniform_future_idx(k_goal, idxs, final)
    random_goal_idxs = jax.random.randint(k_random, shape, 0, data.size, dtype=jnp.int32)
    pick_random = jax.random.uniform(k_pick, shape) < spec.actor_p_randomgoal
    goal_idxs = jnp.where(pick_random, random_goal_idxs, traj_goal_idxs)
    target_idxs = geometric_future_idx(k_target, idxs, final, spec.discount)

    obs = data.observations
    return {
        "observations": obs[idxs],
        "high_actor_goals": obs[goal_idxs],
        "high_actor_targets": obs[target_idxs],
    }
