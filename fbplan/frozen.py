"""Frozen checkpoints, environments and offline data.

Owner: component a-frozen. Other components import only the public names below.
All access to upstream code (third_party/ssm) goes through `_upstream`, so the rest of fbplan
does not depend on the upstream sys.path layout.

Download the checkpoints with `uv run python -m fbplan.frozen [env_name ...]`.
OGBench dataset files download on the first `load_dataset` call.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import pickle
import shutil
import sys
import urllib.request
from functools import cached_property
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import flax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import ogbench.utils

from fbplan.contracts import Array

ENV_NAMES = (
    "antmaze-medium-navigate-v0",
    "antmaze-large-navigate-v0",
    "antmaze-giant-navigate-v0",
    "antmaze-teleport-navigate-v0",
)

_UPSTREAM_DIR = Path(__file__).resolve().parents[1] / "third_party" / "ssm"

# Checkpoints of the paper's authors (training seed 0) in Google Drive folder
# 1dKYhaDJH9lUREo-kUV3AwmTLrxvKO7Ek. Per maze and file: (Drive file id, sha256 of the content).
_CHECKPOINT_FILES: dict[str, dict[str, tuple[str, str]]] = {
    "medium": {
        "flags.json": (
            "1r0wR4il8LIbrPKRMakEckCR4MOs7romJ",
            "a6ad11ce0af0c1cda425bdaff3c4af1a3580ef54f65218a129acc60f8d15916b",
        ),
        "params.pkl": (
            "1683zzDk5v4m2Otad6I2E0TsoYBcGypqB",
            "c7efb93cf2caba0d311b87a1c73313b5fe6acda93e7122f47eea5db665858bfe",
        ),
    },
    "large": {
        "flags.json": (
            "19nh-KCENZDYrOa_ME5du4PqNi3BQVlru",
            "722290cdff30b81bd639a1755c82b03332bbd7a1626f088b66228e6ca645267c",
        ),
        "params.pkl": (
            "1gV70CRaJudM_lkqi3FFYwLTg82DAyodi",
            "36208d32d869b3d4fb11608934139767d95f8c9754143cac0df44c7691727cf2",
        ),
    },
    "giant": {
        "flags.json": (
            "1ZzEc7ujezMj2U2aUDeeVFp1qUOi1t0Zx",
            "d8104aae44f0b82a51a9f3c026f610a7568cd4b3b80b03cfdc26b656240a0fa2",
        ),
        "params.pkl": (
            "1HOOIW3uQkcwEYEZwLTsXWu-tU7lshkt0",
            "cb6c0e25e8ee504ffa13355a9288cecae91b55fd2577c76c01c2f1137380b77b",
        ),
    },
    "teleport": {
        "flags.json": (
            "1OrHUH4kPwUdQI067sFsq6twMe1oEAE28",
            "035a924033090781d0a8853cddd43168efe1550f1f2a45a6c8ba4393e15eedd5",
        ),
        "params.pkl": (
            "1FCNHFuMlrLJYlNvqJaoaHP2nw0vreWOR",
            "413780d58400a3c1af9bb55b2f62c7c10d0e5ad91453327c6e18e10fb062288d",
        ),
    },
}
_DRIVE_URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"

# The only classes a checkpoint pickle may reference: together they rebuild numpy arrays.
_PICKLE_ALLOWED = frozenset(
    {("numpy._core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype")}
)

# Upstream make_env_and_datasets clips dataset actions to [-1 + eps, 1 - eps].
_ACTION_CLIP_EPS = 1e-5

_log = logging.getLogger(__name__)


def _maze_name(env_name: str) -> str:
    """medium, large, giant or teleport. Refuses other envs: no checkpoint exists for them."""
    if env_name not in ENV_NAMES:
        raise ValueError(f"no frozen checkpoint for {env_name!r}; known envs: {ENV_NAMES}")
    return env_name.split("-")[1]


def checkpoint_dir(env_name: str) -> Path:
    """Folder with flags.json and params.pkl for one env. Shared by all worktrees."""
    default = Path.home() / ".cache" / "fbplan" / "checkpoints"
    root = Path(os.environ.get("FBPLAN_CKPT_DIR", default))
    return root / _maze_name(env_name)


def download_checkpoint(env_name: str) -> Path:
    """Download flags.json and params.pkl of one env into checkpoint_dir(env_name).

    Idempotent: a file that is present is only hash-checked. A present file with a wrong hash
    raises instead of being replaced, because then the file was edited or the source changed.
    """
    target = checkpoint_dir(env_name)
    for name, (file_id, sha256) in _CHECKPOINT_FILES[_maze_name(env_name)].items():
        path = target / name
        if path.exists():
            _check_sha256(path, sha256)
        else:
            _download(_DRIVE_URL.format(file_id), path, sha256)
    return target


def _download(url: str, path: Path, sha256: str | None = None) -> None:
    """Fetch url into path. Writes a .part file first, so path never holds a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    _log.info("downloading %s -> %s", url, path)
    try:
        with urllib.request.urlopen(url) as response, part.open("wb") as file:
            shutil.copyfileobj(response, file)
        if sha256 is not None:
            _check_sha256(part, sha256)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(path)


def _check_sha256(path: Path, expected: str) -> None:
    with path.open("rb") as file:
        actual = hashlib.file_digest(file, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(
            f"{path}: sha256 {actual} differs from the expected {expected}. "
            "Delete the file to download it again."
        )


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Unpickler that can only rebuild numpy arrays, so a downloaded pickle cannot run code."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _PICKLE_ALLOWED:
            raise pickle.UnpicklingError(
                f"pickle references {module}.{name}; only numpy arrays are allowed"
            )
        return super().find_class(module, name)


def load_numpy_pickle(path: Path) -> Any:
    """Load a pickle of nested dicts of numpy arrays, such as an upstream checkpoint."""
    with path.open("rb") as file:
        return _NumpyOnlyUnpickler(file).load()


def _upstream(module: str) -> ModuleType:
    """Import a module of the upstream repo, for example "agents.fbpiswitch".

    Upstream code imports its top-level packages `agents` and `utils` as if main.py runs from
    third_party/ssm. That folder goes first on sys.path, and the check below makes sure that a
    different package named `utils` did not win.
    """
    if not (_UPSTREAM_DIR / "agents").is_dir():
        raise ImportError(f"{_UPSTREAM_DIR} is empty: run `git submodule update --init`")
    if str(_UPSTREAM_DIR) not in sys.path:
        sys.path.insert(0, str(_UPSTREAM_DIR))
    imported = importlib.import_module(module)
    if not Path(imported.__file__).resolve().is_relative_to(_UPSTREAM_DIR):
        raise ImportError(f"{module} resolved to {imported.__file__}, not to {_UPSTREAM_DIR}")
    return imported


def load_model(env_name: str) -> FrozenFBModel:
    """Load the frozen F, B, pi_l and baseline pi_h for one env. Downloads the checkpoint if absent.

    Builds the upstream FBpiSwitchAgent from get_config() updated by flags.json['agent'], as the
    authors trained it, and restores the whole saved state into it with from_state_dict.
    """
    ckpt = download_checkpoint(env_name)
    flags = json.loads((ckpt / "flags.json").read_text())
    if flags["env_name"] != f"ogbench-{env_name}":
        raise ValueError(f"{ckpt} holds a checkpoint for {flags['env_name']}, not for {env_name}")
    state = load_numpy_pickle(ckpt / "params.pkl")["agent"]

    fbpiswitch = _upstream("agents.fbpiswitch")
    config = fbpiswitch.get_config()
    config.update(flags["agent"])
    agent = fbpiswitch.FBpiSwitchAgent.create(0, _example_batch(state), config)
    agent = flax.serialization.from_state_dict(agent, state)
    # On device once, so the jitted calls below do not copy the 18 MB of weights every step.
    return FrozenFBModel(env_name, jax.device_put(agent))


def _example_batch(state: dict[str, Any]) -> dict[str, np.ndarray]:
    """Zero batch with the observation and action sizes that the saved weights expect."""
    params = state["network"]["params"]
    obs_dim = params["modules_backward_repr"]["value_net"]["Dense_0"]["kernel"].shape[0]
    action_dim = params["modules_actor"]["mean_net"]["bias"].shape[0]
    return {
        "observations": np.zeros((1, obs_dim), np.float32),
        "actions": np.zeros((1, action_dim), np.float32),
    }


class FrozenFBModel:
    """FBModel (see fbplan.contracts) on top of the restored upstream FBpiSwitchAgent.

    Each method calls the same upstream network module and normalize_z as FBpiSwitchAgent does.
    One deviation, required by the contract: high_latent normalizes z_task, while upstream
    passes z_r to pi_h as it is. The real z_r has norm 0.92-0.98 sqrt(d) (see
    infer_task_latent), so actions differ from upstream by about 1e-3.
    `agent` is public for code that needs upstream methods, for example retraining pi_h or
    upstream utils.evaluation.evaluate.
    """

    def __init__(self, env_name: str, agent: Any) -> None:
        self.env_name = env_name
        self.agent = agent  # upstream agents.fbpiswitch.FBpiSwitchAgent
        self.latent_dim = int(agent.config["latent_dim"])
        self.discount = float(agent.config["discount"])

    def forward(self, obs: Array, z: Array) -> jax.Array:
        """F(s, z) of both heads. obs [N, obs_dim], z [N, d] -> [2, N, d]."""
        return _forward(self.agent, obs, z)

    def backward(self, obs: Array) -> jax.Array:
        """B(s), raw. obs [N, obs_dim] -> [N, d]."""
        return _backward(self.agent, obs)

    def low_action(
        self, obs: Array, z: Array, temperature: float = 0.0, seed: jax.Array | None = None
    ) -> jax.Array:
        """pi_l(a | s, z). Returns actions [N, act_dim] clipped to [-1, 1]."""
        return _low_action(self.agent, obs, z, temperature, _sampling_key(seed, temperature))

    def high_latent(
        self, obs: Array, z_task: Array, temperature: float = 0.0, seed: jax.Array | None = None
    ) -> jax.Array:
        """Baseline pi_h(z | s, z_task). Returns normalized intentions [N, d]."""
        key = _sampling_key(seed, temperature)
        return _high_latent(self.agent, obs, z_task, temperature, key)

    def infer_task_latent(self, task_id: int) -> np.ndarray:
        """Task embedding z_r [d] for evaluation task `task_id` (1..5), as upstream main.py does.

        Upstream relabels the validation split with the task reward (1 within the goal radius,
        else 0) and calls FBpiSwitchAgent.infer_latent on the first num_zero_shot_samples
        (100k) transitions: z_r = normalize_z(mean(r * B(s))). Upstream draws them with
        HGCDataset.sample(idxs=arange(n), relabeling=False), which returns exactly these rows.
        The norm of z_r is a little below sqrt(d): upstream averages twice (uniform softmax
        weights 1/n, then a mean), so the raw vector is about 1e-7 and the 1e-8 in normalize_z
        is not negligible. We keep the upstream value.
        """
        env = make_env(self.env_name)
        env.reset(options={"task_id": task_id})
        relabel = _upstream("utils.env_utils").relabel_dataset
        relabeled = relabel(f"ogbench-{self.env_name}", env, self._val_dataset)
        n = self.agent.config["num_zero_shot_samples"]
        if relabeled.size < n:
            raise ValueError(f"validation split has {relabeled.size} rows, upstream uses {n}")
        batch = {"observations": relabeled["observations"][:n], "rewards": relabeled["rewards"][:n]}
        return np.asarray(self.agent.infer_latent(batch))

    @cached_property
    def _val_dataset(self) -> Any:
        """Validation split as an upstream Dataset, the type that upstream relabel_dataset takes."""
        return _upstream("utils.datasets").Dataset.create(**load_dataset(self.env_name, "val"))


def _sampling_key(seed: jax.Array | None, temperature: float) -> jax.Array:
    """PRNG key for a policy sample. At temperature 0 the sample is the mean, so any key works."""
    if seed is not None:
        return seed
    if temperature != 0.0:
        raise ValueError("a seed is required when temperature > 0")
    return jax.random.PRNGKey(0)


@jax.jit
def _forward(agent: Any, obs: Array, z: Array) -> jax.Array:
    return agent.network.select("forward_repr")(obs, agent.normalize_z(z), goal_encoded=True)


@jax.jit
def _backward(agent: Any, obs: Array) -> jax.Array:
    return agent.network.select("backward_repr")(obs)


@jax.jit
def _low_action(agent: Any, obs: Array, z: Array, temperature: float, seed: jax.Array) -> jax.Array:
    dist = agent.network.select("actor")(
        obs, agent.normalize_z(z), goal_encoded=True, temperature=temperature
    )
    return jnp.clip(dist.sample(seed=seed), -1, 1)


@jax.jit
def _high_latent(
    agent: Any, obs: Array, z_task: Array, temperature: float, seed: jax.Array
) -> jax.Array:
    # pi_h was trained on task latents of norm sqrt(d). Its sampled intention is normalized
    # before it goes to pi_l, as in upstream sample_actions.
    dist = agent.network.select("high_actor")(
        obs, agent.normalize_z(z_task), goal_encoded=True, temperature=temperature
    )
    return agent.normalize_z(dist.sample(seed=seed))


def make_env(env_name: str) -> gym.Env:
    """Create the evaluation env exactly as upstream main.py does.

    Upstream builds it with utils.env_utils.make_env_and_datasets (the OGBench env in an
    EpisodeMonitor wrapper, no frame stack) and then turns goal noise off, so all episodes of a
    task share one goal. env_only=True builds the same wrappers without loading the datasets.
    """
    env = _upstream("utils.env_utils").make_env_and_datasets(f"ogbench-{env_name}", env_only=True)
    env.unwrapped._add_noise_to_goal = False
    return env


def dataset_path(env_name: str, split: Literal["train", "val"]) -> Path:
    """Local file of one OGBench dataset split, in OGBench's default data folder."""
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', not {split!r}")
    suffix = "" if split == "train" else "-val"
    return Path(ogbench.utils.DEFAULT_DATASET_DIR).expanduser() / f"{env_name}{suffix}.npz"


def load_dataset(env_name: str, split: Literal["train", "val"]) -> dict[str, np.ndarray]:
    """Offline OGBench data: keys observations, actions, next_observations, terminals, qpos, qvel.

    The same arrays as the split that upstream make_env_and_datasets returns: the regular
    (not compact) OGBench layout, with qpos and qvel (add_info=True, needed for relabeling) and
    actions clipped to [-1 + 1e-5, 1 - 1e-5]. Upstream always loads both splits; this function
    loads, and downloads if absent, only the requested one.
    """
    path = dataset_path(env_name, split)
    if not path.exists():
        _download(f"{ogbench.utils.DATASET_URL}/{path.name}", path)
    data = ogbench.utils.load_dataset(str(path), compact_dataset=False, add_info=True)
    data["actions"] = np.clip(data["actions"], -1 + _ACTION_CLIP_EPS, 1 - _ACTION_CLIP_EPS)
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in sys.argv[1:] or ENV_NAMES:
        print(download_checkpoint(name))
