"""Reference controllers that use the frozen checkpoint as upstream does.

`BaselineController` is FB pi-Switch at evaluation: pi_h picks a new intention at every step
(upstream `agents/fbpiswitch.py:sample_actions`), then pi_l acts on it. It is also the parity
reference against upstream `utils/evaluation.evaluate`.
`NonHierController` drops pi_h and gives the task latent z_r directly to pi_l.
Both act at temperature 0 (deterministic), as the upstream evaluation default.
"""

from __future__ import annotations

import numpy as np

from fbplan.contracts import FBModel, TaskSpec


class BaselineController:
    """pi_h(z | s, z_r) at every step, then pi_l(a | s, z)."""

    name = "baseline"

    def __init__(self, model: FBModel) -> None:
        self._model = model
        self._z_r = np.zeros(model.latent_dim)

    def reset(self, task: TaskSpec, seed: int) -> None:
        self._z_r = task.z_r

    def act(self, obs: np.ndarray) -> np.ndarray:
        intention = self._model.high_latent(obs[None], self._z_r[None])
        return np.asarray(self._model.low_action(obs[None], intention)[0])

    def step_info(self) -> dict[str, float | int | str]:
        return {}


class NonHierController:
    """pi_l(a | s, z_r): the low-level policy with the task latent as its intention."""

    name = "nonhier"

    def __init__(self, model: FBModel) -> None:
        self._model = model
        self._z_r = np.zeros(model.latent_dim)

    def reset(self, task: TaskSpec, seed: int) -> None:
        self._z_r = task.z_r

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.low_action(obs[None], self._z_r[None])[0])

    def step_info(self) -> dict[str, float | int | str]:
        return {}
