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

from fbplan.contracts import normalize_z


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


def sample_high_indices(
    data: TrajectoryData, key: jax.Array, spec: HighBatchSpec
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Dataset indices of (s, g, w), distributed as upstream `HGCDataset.sample`.

    s: uniform dataset states.
    g: the task goal, a future state of the same trajectory (uniform or geometric offset), or a
        uniform dataset state with probability actor_p_randomgoal.
    w: the subgoal, at a geometric offset from s in the same trajectory. It is not clipped at g,
        as upstream.
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
    return idxs, goal_idxs, target_idxs


def sample_high_batch(data: TrajectoryData, key: jax.Array, spec: HighBatchSpec) -> dict:
    """The keys that upstream `high_actor_loss` reads: s, g and w of `sample_high_indices`."""
    idxs, goal_idxs, target_idxs = sample_high_indices(data, key, spec)
    obs = data.observations
    return {
        "observations": obs[idxs],
        "high_actor_goals": obs[goal_idxs],
        "high_actor_targets": obs[target_idxs],
    }


@flax.struct.dataclass
class SwitchBatch:
    """One batch for approach B: a state, a subgoal on its trajectory and a task latent."""

    observations: jax.Array  # [N, obs_dim] s
    subgoals: jax.Array  # [N, obs_dim] w
    subgoal_latents: jax.Array  # [N, d] z_w: B(w) or its region mean, raw
    task_latents: jax.Array  # [N, d] z, norm sqrt(d)
    is_goal: jax.Array  # [N] bool: z is the latent of a goal state, not a Gaussian sample


def sample_switch_batch(
    data: TrajectoryData,
    latents: jax.Array,
    key: jax.Array,
    spec: HighBatchSpec,
    gaussian_prob: float,
) -> SwitchBatch:
    """(s, w, z) with s, g, w from `sample_high_indices` and `latents` [N, d] looked up by index.

    With probability gaussian_prob, z is a Gaussian sample instead of the latent of g, as
    upstream `sample_latents` with critic_latent_mix_prob.
    """
    k_idx, k_gauss, k_pick = jax.random.split(key, 3)
    idxs, goal_idxs, target_idxs = sample_high_indices(data, k_idx, spec)
    d = latents.shape[-1]
    gaussian = jax.random.normal(k_gauss, (spec.batch_size, d))
    use_gaussian = jax.random.uniform(k_pick, (spec.batch_size,)) < gaussian_prob
    task = jnp.where(use_gaussian[:, None], gaussian, latents[goal_idxs])
    return SwitchBatch(
        observations=data.observations[idxs],
        subgoals=data.observations[target_idxs],
        subgoal_latents=latents[target_idxs],
        task_latents=normalize_z(task, d),
        is_goal=~use_gaussian,
    )


def trajectory_window_latents(
    latents: np.ndarray, terminals: np.ndarray, half_width: int
) -> np.ndarray:
    """Mean latent over the states within +-half_width steps on the same trajectory. [N, d].

    The latents of the B-region variant. (fbplan.landmarks.region_latents is another average:
    over the states within an xy radius.)
    """
    (ends,) = np.nonzero(terminals > 0)
    starts = np.concatenate([[0], ends[:-1] + 1])
    out = np.empty(latents.shape, dtype=np.float32)
    for start, end in zip(starts, ends, strict=True):
        segment = latents[start : end + 1].astype(np.float64)
        sums = np.concatenate([np.zeros((1, segment.shape[1])), np.cumsum(segment, axis=0)])
        t = np.arange(len(segment))
        lo, hi = np.maximum(t - half_width, 0), np.minimum(t + half_width, len(segment) - 1)
        out[start : end + 1] = (sums[hi + 1] - sums[lo]) / (hi - lo + 1)[:, None]
    return out
