"""Reference controllers that use the frozen checkpoint as upstream does.

`BaselineController` is FB pi-Switch at evaluation: pi_h picks a new intention at every step
(upstream `agents/fbpiswitch.py:sample_actions`), then pi_l acts on it. It is also the parity
reference against upstream `utils/evaluation.evaluate`.
`RawBaselineController` is the same agent through upstream code: it calls
`FBpiSwitchAgent.sample_actions` with z_r as upstream infers it, so pi_h gets z_r without
normalization (a reference row, docs/design.md §8).
`NonHierController` drops pi_h and gives the task latent z_r directly to pi_l.
`EnsembleBaselineController` averages the intentions of the pi_h of several training seeds
(a test-time ensemble); `RawEnsembleBaselineController` does the same through baseline_raw's
input path.
All act at temperature 0 (deterministic), as the upstream evaluation default.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

from fbplan.contracts import Controller, FBModel, TaskSpec, normalize_z
from fbplan.evaluate import Checkpoint
from fbplan.frozen import FrozenFBModel


class _TaskLatentController:
    """Keeps the task latent z_r of the current episode; subclasses act on it."""

    name = ""

    def __init__(self, model: FBModel) -> None:
        self._model = model
        self._z_r = np.zeros(model.latent_dim)

    def reset(self, task: TaskSpec, seed: int) -> None:
        self._z_r = task.z_r

    def step_info(self) -> dict[str, float | int | str]:
        return {}


class BaselineController(_TaskLatentController):
    """pi_h(z | s, z_r) at every step, then pi_l(a | s, z)."""

    name = "baseline"

    def act(self, obs: np.ndarray) -> np.ndarray:
        intention = self._model.high_latent(obs[None], self._z_r[None])
        return np.asarray(self._model.low_action(obs[None], intention)[0])


class RawBaselineController(_TaskLatentController):
    """Upstream sample_actions(s, z_r): pi_h with the raw z_r, then pi_l.

    It needs the upstream agent, so it takes a FrozenFBModel. The key is fixed: at
    temperature 0 both policies return their mean, so the key does not matter.
    """

    name = "baseline_raw"

    def __init__(self, model: FrozenFBModel) -> None:
        super().__init__(model)
        self._agent = model.agent
        self._key = jax.random.PRNGKey(0)

    def act(self, obs: np.ndarray) -> np.ndarray:
        action = self._agent.sample_actions(obs, self._z_r, seed=self._key, temperature=0.0)
        return np.asarray(action)


class NonHierController(_TaskLatentController):
    """pi_l(a | s, z_r): the low-level policy with the task latent as its intention."""

    name = "nonhier"

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.low_action(obs[None], self._z_r[None])[0])


@dataclass(frozen=True)
class EnsembleConfig:
    """family: the trained pi_h family whose seeds vote, as in --checkpoint without the seed
    ("retrained", "b:clean", "b:region"); seeds: the first `seeds` training seeds."""

    family: str = "retrained"
    seeds: int = 5


class EnsembleBaselineController(_TaskLatentController):
    """pi_l(a | s, z) with z = normalize(mean_k pi_h^k(s, z_r)).

    Each member's pi_h returns a normalized intention; their mean is normalized again. F, B and
    pi_l are the shared frozen modules, so only `model` acts.
    """

    name = "baseline_ensemble"

    def __init__(
        self, model: FBModel, members: Sequence[FBModel], config: EnsembleConfig
    ) -> None:
        super().__init__(model)
        self._members = list(members)
        self.config = config

    def act(self, obs: np.ndarray) -> np.ndarray:
        intentions = [self._intention(member, obs) for member in self._members]
        z = normalize_z(np.mean(intentions, axis=0), self._model.latent_dim)
        return np.asarray(self._model.low_action(obs[None], z[None])[0])

    def _intention(self, member: FBModel, obs: np.ndarray) -> np.ndarray:
        """The member's normalized intention, pi_h given the normalized z_r (FBModel)."""
        return np.asarray(member.high_latent(obs[None], self._z_r[None]))[0]


class RawEnsembleBaselineController(EnsembleBaselineController):
    """As EnsembleBaselineController, but each member's pi_h gets the raw z_r, as upstream
    sample_actions does (baseline_raw's input path). Members must be FrozenFBModel."""

    name = "baseline_ensemble_raw"

    def __init__(
        self, model: FBModel, members: Sequence[FrozenFBModel], config: EnsembleConfig
    ) -> None:
        super().__init__(model, members, config)
        self._key = jax.random.PRNGKey(0)  # temperature 0: the key does not matter

    def _intention(self, member: FBModel, obs: np.ndarray) -> np.ndarray:
        agent = member.agent  # type: ignore[attr-defined]
        return np.asarray(_raw_intention(agent, obs[None], self._z_r[None], self._key))[0]


@jax.jit
def _raw_intention(agent: Any, obs: jax.Array, z_r: jax.Array, key: jax.Array) -> jax.Array:
    """The pi_h half of upstream FBpiSwitchAgent.sample_actions at temperature 0."""
    high_key, _ = jax.random.split(key)
    dist = agent.network.select("high_actor")(obs, z_r, goal_encoded=True, temperature=0.0)
    return agent.normalize_z(dist.sample(seed=high_key))


def make_ensemble(
    model: FBModel, family: str = "retrained", seeds: int = 5, z_input: str = "normalized"
) -> Controller:
    """Ensemble of the pi_h of seeds 0..seeds-1 of `family`, loaded like --checkpoint.

    z_input: "normalized" (the FBModel path) or "raw" (baseline_raw's path).
    """
    if z_input not in ("normalized", "raw"):
        raise ValueError(f"z_input must be normalized or raw, not {z_input!r}")
    config = EnsembleConfig(family, seeds)
    members = [Checkpoint.parse(f"{family}:{k}").load(model.env_name) for k in range(seeds)]
    cls = RawEnsembleBaselineController if z_input == "raw" else EnsembleBaselineController
    return cls(model, members, config)
