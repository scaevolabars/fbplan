"""Fixtures shared by the test files: skips for absent files and the real medium model.

Tests never download data. A test that needs a checkpoint or an OGBench split skips when the
file is absent, so `pytest` passes on a fresh clone and in CI, which downloads no data.
"""

from __future__ import annotations

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
