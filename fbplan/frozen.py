"""Frozen checkpoints, environments and offline data.

Owner: component a-frozen. Other components import only the functions below.
The bodies are stubs until a-frozen replaces them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import gymnasium as gym
import numpy as np

from fbplan.contracts import FBModel

ENV_NAMES = (
    "antmaze-medium-navigate-v0",
    "antmaze-large-navigate-v0",
    "antmaze-giant-navigate-v0",
    "antmaze-teleport-navigate-v0",
)


def checkpoint_dir(env_name: str) -> Path:
    """Folder with flags.json and params.pkl for one env. Shared by all worktrees."""
    default = Path.home() / ".cache" / "fbplan" / "checkpoints"
    root = Path(os.environ.get("FBPLAN_CKPT_DIR", default))
    return root / env_name.split("-")[1]  # medium, large, giant, teleport


def load_model(env_name: str) -> FBModel:
    """Load the frozen F, B, pi_l and baseline pi_h for one env."""
    raise NotImplementedError("a-frozen")


def make_env(env_name: str) -> gym.Env:
    """Create the evaluation env exactly as upstream main.py does."""
    raise NotImplementedError("a-frozen")


def load_dataset(env_name: str, split: Literal["train", "val"]) -> dict[str, np.ndarray]:
    """Offline OGBench data: keys observations, actions, next_observations, terminals."""
    raise NotImplementedError("a-frozen")
