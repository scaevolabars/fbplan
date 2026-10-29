"""Reference controllers that use the frozen checkpoint as upstream does.

`BaselineController` is FB pi-Switch at evaluation: pi_h picks a new intention at every step
(upstream `agents/fbpiswitch.py:sample_actions`), then pi_l acts on it. It is also the parity
reference against upstream `utils/evaluation.evaluate`.
`RawBaselineController` is the same agent through upstream code: it calls
`FBpiSwitchAgent.sample_actions` with z_r as upstream infers it, so pi_h gets z_r without
normalization (a reference row, docs/design.md §8).
`NonHierController` drops pi_h and gives the task latent z_r directly to pi_l.
All act at temperature 0 (deterministic), as the upstream evaluation default.
"""

from __future__ import annotations

import jax
import numpy as np

from fbplan.contracts import FBModel, TaskSpec
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
