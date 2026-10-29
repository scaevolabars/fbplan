"""Fixtures shared by the test files: skips for absent files, the real medium model, and a
small upstream agent.

Tests never download data. A test that needs a checkpoint or an OGBench split skips when the
file is absent, so `pytest` passes on a fresh clone and in CI, which downloads no data.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from fbplan import frozen

MEDIUM = "antmaze-medium-navigate-v0"


def _skip_without_checkpoint(env_name: str) -> None:
    if not (frozen.checkpoint_dir(env_name) / "params.pkl").exists():
        pytest.skip(f"{env_name} checkpoint absent: run `uv run python -m fbplan.frozen`")


def _skip_without_dataset(env_name: str, split: str) -> None:
    path = frozen.dataset_path(env_name, split)
    if not path.exists():
        pytest.skip(f"{path.name} absent: fbplan.frozen.load_dataset downloads it")


class RequireFiles:
    """Skips the calling test when a file it needs is absent."""

    def checkpoint(self, env_name: str = MEDIUM) -> None:
        """The provided checkpoint of `env_name`."""
        _skip_without_checkpoint(env_name)

    def dataset(self, env_name: str = MEDIUM, split: str = "val") -> None:
        """One OGBench split of `env_name`."""
        _skip_without_dataset(env_name, split)


@pytest.fixture
def require_files() -> RequireFiles:
    """`require_files.checkpoint(env)`, `require_files.dataset(env, split)` inside a test."""
    return RequireFiles()


@pytest.fixture(scope="session")
def medium_model() -> frozen.FrozenFBModel:
    """The provided medium checkpoint, loaded once per session (z_r needs the val split)."""
    _skip_without_checkpoint(MEDIUM)
    _skip_without_dataset(MEDIUM, "val")
    return frozen.load_model(MEDIUM)


@pytest.fixture(scope="session")
def medium_val() -> dict[str, np.ndarray]:
    """The medium validation split (24 MB), shared by all tests: do not modify its arrays."""
    _skip_without_dataset(MEDIUM, "val")
    return frozen.load_dataset(MEDIUM, "val")


@dataclass(frozen=True)
class TinyAgent:
    """An upstream FBpiSwitchAgent with small networks and its shapes.

    Its F, B and pi_l stand in for the frozen modules in tests that train, save or reload pi_h.
    The agent is immutable: derive a variant with `tiny.agent.replace(...)`. An example batch:
    `fbplan.learned.checkpoint.example_batch(tiny.agent, tiny.obs_dim)`.
    """

    agent: Any  # upstream agents.fbpiswitch.FBpiSwitchAgent
    config: Mapping[str, Any]  # read-only; dict(tiny.config) gives a copy to change
    obs_dim: int
    action_dim: int

    @property
    def latent_dim(self) -> int:
        """Size of z, from the config."""
        return int(self.config["latent_dim"])


@pytest.fixture(scope="session")
def tiny_agent() -> TinyAgent:
    """Upstream agent config with 16x16 networks and latent_dim 4; obs dim 5, action dim 2."""
    fbpiswitch = frozen.upstream_module("agents.fbpiswitch")
    config = fbpiswitch.get_config().to_dict() | {
        "actor_hidden_dims": (16, 16),
        "forward_repr_hidden_dims": (16, 16),
        "backward_repr_hidden_dims": (16, 16),
        "latent_dim": 4,
        "batch_size": 32,
    }
    obs_dim, action_dim = 5, 2
    example = {  # one transition: upstream `create` reads the shapes from it
        "observations": np.zeros((1, obs_dim), np.float32),
        "actions": np.zeros((1, action_dim), np.float32),
    }
    agent = fbpiswitch.FBpiSwitchAgent.create(123, example, config)
    return TinyAgent(agent, MappingProxyType(config), obs_dim, action_dim)
