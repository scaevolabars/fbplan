"""Trap-averse intentions with a time gate on teleport (design.md §8 T3).

Hypothesis: H1 of docs/hypotheses/h-exec.md.

The baseline maximises a discounted value, so it takes the teleporters even when a walk would
reach the goal in time; a third of the jumps end in a closed trap (docs/theory.md §11.1). This
controller penalises the trap in the intention, but only while a walk still fits in the steps
left; otherwise it acts exactly as the baseline:

    z_t = normalize(pi_h(s, z_r) - weight * g(s, t) * z_trap)

- Trap, from the offline data only (no maze map; the rule of docs/hypotheses/h-exec.md M10):
  a trajectory whose states after its last jump (an xy step above `jump_xy`) all stay within
  `trap_radius` of their mean xy ends in a trap, and those states are trap states.
  `z_trap = normalize(mean((1[trap] - p) B(s)))` over a seeded pool of train states (p = trap
  share of the pool). FB is linear in the reward, so `-z_trap` adds "avoid the trap" to the
  reward of the intention.
- Gate, from F, B and the step counter: `z_c = normalize(z_hat_r - weight * z_trap)` is the
  trap-averse task latent. `rho_walk(s) = min_h F_h(s, z_c)^T z_hat_r / F_goal`, where F_goal is
  the median of the same value over the `goal_states` pool states with the highest
  `B(s)^T z_hat_r` (the goal region as the data sees it; the controller never reads goal_xy).
  As M(s, w) / M(w, w) = E[gamma^H] (theory.md Lemma B1), `walk_steps(s) = log rho_walk / log gamma`
  is the length of the trap-free walk in steps (exact for a deterministic walk). The gate is open
  while walk_steps(s) <= time_limit - t. `gate="none"` keeps it open (the ungated control).

With weight 0, or while the gate is closed, the actions are those of `BaselineController`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from fbplan.contracts import FBModel, TaskSpec, normalize_z
from fbplan.frozen import load_dataset, make_env

_BATCH = 8192  # states per B call


@dataclass(frozen=True)
class TrapAwareConfig:
    """weight: lambda on z_trap (z_trap and pi_h's intention both have norm sqrt(d)).
    trap_radius, jump_xy: the data rule, in xy units. time_limit: steps per episode (the env's).
    pool_size, seed: the train states that give z_trap and the goal states. goal_states: how many
    pool states define F_goal."""

    weight: float = 1.0
    trap_radius: float = 2.5
    gate: Literal["time", "none"] = "time"
    time_limit: int = 1000
    jump_xy: float = 2.0
    pool_size: int = 200_000
    goal_states: int = 64
    seed: int = 0

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("weight must be >= 0")
        if self.gate not in ("time", "none"):
            raise ValueError(f"unknown gate {self.gate!r}")
        for name in ("trap_radius", "jump_xy", "time_limit", "pool_size", "goal_states"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")


def trap_mask(
    xy: np.ndarray, next_xy: np.ndarray, terminals: np.ndarray, jump_xy: float, radius: float
) -> np.ndarray:
    """Trap states of a flat offline dataset (OGBench layout, `terminals` ends each trajectory).

    The states after the last jump of a trajectory are trap states if they all lie within
    `radius` of their mean xy. [N] bool.
    """
    xy, next_xy = np.asarray(xy)[:, :2], np.asarray(next_xy)[:, :2]
    ends = np.flatnonzero(np.asarray(terminals) > 0)
    if len(ends) == 0 or ends[-1] != len(xy) - 1:
        raise ValueError("the last state must end a trajectory (terminals)")
    jumps = np.linalg.norm(next_xy - xy, axis=1) > jump_xy
    mask = np.zeros(len(xy), dtype=bool)
    start = 0
    for end in ends:
        last = np.flatnonzero(jumps[start : end + 1])
        if len(last) and start + last[-1] < end:
            first = start + last[-1] + 1
            post = xy[first : end + 1]
            if np.all(np.linalg.norm(post - post.mean(axis=0), axis=1) <= radius):
                mask[first : end + 1] = True
        start = end + 1
    return mask


def trap_latent(model: FBModel, observations: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """normalize(mean((1[trap] - p) B(s))) over the given states: [d]."""
    return _centred_latent(_backward(model, observations), mask, model.latent_dim)


def _backward(model: FBModel, observations: np.ndarray) -> np.ndarray:
    parts = [np.asarray(model.backward(observations[i : i + _BATCH]), dtype=np.float32)
             for i in range(0, len(observations), _BATCH)]
    return np.concatenate(parts)


def _centred_latent(backward: np.ndarray, mask: np.ndarray, latent_dim: int) -> np.ndarray:
    if not mask.any():
        raise ValueError("no trap states in the data")
    weights = mask.astype(np.float64) - mask.mean()
    return np.asarray(normalize_z(weights @ backward / len(mask), latent_dim))


class TrapAwareController:
    """See the module docstring."""

    name = "trap_aware"

    def __init__(
        self,
        model: FBModel,
        pool_obs: np.ndarray,
        pool_backward: np.ndarray,
        z_trap: np.ndarray,
        config: TrapAwareConfig,
    ) -> None:
        self.config = config
        self._model = model
        self._pool_obs = pool_obs
        self._pool_backward = pool_backward
        self._z_trap = z_trap
        self._log_gamma = math.log(model.discount)
        self._z_r = np.zeros(model.latent_dim)
        self._z_hat = np.zeros(model.latent_dim)
        self._z_c = np.zeros(model.latent_dim)
        self._f_goal = 0.0
        self._t = 0
        self._info: dict[str, float | int | str] = {}

    @classmethod
    def from_data(
        cls, model: FBModel, data: dict[str, np.ndarray], config: TrapAwareConfig
    ) -> TrapAwareController:
        """Apply the trap rule to an offline dataset and keep a seeded pool of its states."""
        observations = data["observations"]
        mask = trap_mask(observations, data["next_observations"], data["terminals"],
                         config.jump_xy, config.trap_radius)
        rng = np.random.default_rng(config.seed)
        size = min(config.pool_size, len(observations))
        pool = np.sort(rng.choice(len(observations), size=size, replace=False))
        pool_obs = np.asarray(observations[pool])
        pool_backward = _backward(model, pool_obs)
        z_trap = _centred_latent(pool_backward, mask[pool], model.latent_dim)
        return cls(model, pool_obs, pool_backward, z_trap, config)

    @property
    def z_trap(self) -> np.ndarray:
        """The trap latent [d] (norm sqrt(d))."""
        return self._z_trap

    @property
    def z_c(self) -> np.ndarray:
        """The trap-averse task latent of the current task [d], as the gate uses it."""
        return self._z_c

    def reset(self, task: TaskSpec, seed: int) -> None:
        self._z_r = task.z_r  # passed to pi_h as the baseline passes it
        self._z_hat = np.asarray(normalize_z(np.asarray(task.z_r, dtype=np.float64),
                                             self._model.latent_dim))
        self._z_c = np.asarray(normalize_z(self._z_hat - self.config.weight * self._z_trap,
                                           self._model.latent_dim))
        self._t = 0
        self._info = {}
        if self.config.gate == "time":
            k = min(self.config.goal_states, len(self._pool_obs))
            goal = np.argpartition(-(self._pool_backward @ self._z_hat), k - 1)[:k]
            self._f_goal = float(np.median(self._walk_values(self._pool_obs[goal])))

    def walk_steps(self, obs: np.ndarray) -> float:
        """Estimated steps of the trap-free walk from obs to the goal: log rho_walk / log gamma."""
        if self._f_goal <= 0:
            return math.inf
        rho = float(self._walk_values(obs[None])[0]) / self._f_goal
        if rho <= 0:
            return math.inf
        return max(0.0, math.log(rho) / self._log_gamma)

    def act(self, obs: np.ndarray) -> np.ndarray:
        intention = self._model.high_latent(obs[None], self._z_r[None])
        gate = self._gate(obs)
        if gate:
            penalised = np.asarray(intention, dtype=np.float64) - self.config.weight * self._z_trap
            intention = np.asarray(normalize_z(penalised, self._model.latent_dim))
        self._t += 1
        return np.asarray(self._model.low_action(obs[None], intention)[0])

    def step_info(self) -> dict[str, float | int | str]:
        return dict(self._info)

    def _gate(self, obs: np.ndarray) -> bool:
        if self.config.weight == 0:
            return False
        if self.config.gate == "none":
            return True
        steps = self.walk_steps(obs)
        self._info = {"walk_steps": steps, "gate": int(steps <= self.config.time_limit - self._t)}
        return bool(self._info["gate"])

    def _walk_values(self, obs: np.ndarray) -> np.ndarray:
        """min_h F_h(s, z_c)^T z_hat_r for a batch of states: [N]."""
        z = np.tile(self._z_c, (len(obs), 1))
        return (np.asarray(self._model.forward(obs, z)) @ self._z_hat).min(axis=0)


def make_trap_aware(model: FBModel, **options) -> TrapAwareController:
    """Trap-aware controller on the train split of `model.env_name` (teleport)."""
    config = TrapAwareConfig(**options)
    limit = make_env(model.env_name).spec.max_episode_steps
    if config.time_limit != limit:
        raise ValueError(f"time_limit {config.time_limit} differs from the env's {limit}")
    return TrapAwareController.from_data(model, load_dataset(model.env_name, "train"), config)
