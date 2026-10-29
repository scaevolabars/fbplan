"""Frozen checkpoints, environments and offline data.

Owner: component a-frozen. Other components import only the public names below.
All access to upstream code (third_party/ssm) goes through `upstream_module`, so the rest of
fbplan does not depend on the upstream sys.path layout.

Download the provided checkpoints with `uv run python -m fbplan.frozen [ENV ...]`. `--trained`
also gets our published trained checkpoints, and `--datasets ENV ...` the OGBench train and val
splits; without it, an OGBench split downloads on the first `load_dataset` call.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import pickle
import shutil
import sys
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from functools import cached_property
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, get_args

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

# Our trained checkpoints (fbplan.train_high, fbplan.learned.train_b), published as assets of
# GitHub releases of the public repo. Key: as in trained_dir, for example
# "baselines/medium/seed0". Per file: (release tag, sha256 of the content).
# The asset name is the key with "/" replaced by "-", then "-" and the file name.
# Release baselines-v1: the retrained baseline pi_h (fbplan.train_high), medium and large.
_TRAINED_FILES: dict[str, dict[str, tuple[str, str]]] = {
    "baselines/medium/seed0": {
        "flags.json": (
            "baselines-v1",
            "6e7fa4de8942c6b7f20e708458455ce782d39fb63f5ea2423abd5a984868280a",
        ),
        "high_actor.pkl": (
            "baselines-v1",
            "bcca90f5fae93bf1afddc4b169db4a669ea1037dbe27aa39067d272ad9a06dfe",
        ),
    },
    "baselines/medium/seed1": {
        "flags.json": (
            "baselines-v1",
            "16657f4d6a776181e1c1a6d13a1be44d395837ae47666803a809d3f17a671a1c",
        ),
        "high_actor.pkl": (
            "baselines-v1",
            "4454869d65df8235164c30c1ea03db3efb7a81d5f7b156a2afff9453ea11442a",
        ),
    },
    "baselines/medium/seed2": {
        "flags.json": (
            "baselines-v1",
            "c95747dc8e828407c956c799ab3faf03fe634a0a771085c346e121ceb29f8ee1",
        ),
        "high_actor.pkl": (
            "baselines-v1",
            "4e4bd0418d71c40dbe375c5da01b21434eff4ed8a81528cc2e172b65092e86e0",
        ),
    },
    "baselines/medium/seed3": {
        "flags.json": (
            "baselines-v1",
            "daa46b9710fc91e978ed171a67165be83adadec39e13275205fb4d3ec3540171",
        ),
        "high_actor.pkl": (
            "baselines-v1",
            "650d7d2bcd1ea7d0bba2b25e9cf8d1683d2239284fc462c66b384e08414327a2",
        ),
    },
    "baselines/medium/seed4": {
        "flags.json": (
            "baselines-v1",
            "a5896f84f4290dc36301c7de460d6b91118f803cebfb833dcd6a59307cf7427f",
        ),
        "high_actor.pkl": (
            "baselines-v1",
            "60e9eb3be00d40704ff278e9823e2ac955938f9025e0cb3de4fbdb33fcf287ab",
        ),
    },
    "baselines/large/seed0": {
        "flags.json": (
            "baselines-v1",
            "4a2527032eb7cb474165c6a772fd6123b3166946a0211cad481f3edee9eff32c",
        ),
        "high_actor.pkl": (
            "baselines-v1",
            "dca1bf2047af0ade301f97dec585620593bceb072d8cb88872cdd26be155892b",
        ),
    },
    "baselines/large/seed1": {
        "flags.json": (
            "baselines-v1",
            "cae25803c91f774fe4126bf7e1c915930e8d3d5e372962de4e8c46bc258f619b",
        ),
        "high_actor.pkl": (
            "baselines-v1",
            "578a7e69411734c52577f8187a924c61e546f9814c2c3c74d9dd93f52934666c",
        ),
    },
    "baselines/large/seed2": {
        "flags.json": (
            "baselines-v1",
            "0c0230acb5e5081bee2c891da332dc765dc96169c9a4d6c1d309991f4c00826a",
        ),
        "high_actor.pkl": (
            "baselines-v1",
            "af959953ee8da04d29ad959a4f93647138c11f8f15149d6bccea71602d39d13c",
        ),
    },
    "baselines/large/seed3": {
        "flags.json": (
            "baselines-v1",
            "b02d710c652e7b18990218d176a3104819d5aab1f70c07d17a7667e1e4c9da68",
        ),
        "high_actor.pkl": (
            "baselines-v1",
            "345b07403b31b6195c038ab57b305d84d4f14ad088e41cdec1e4504a40cabe25",
        ),
    },
    "baselines/large/seed4": {
        "flags.json": (
            "baselines-v1",
            "c9fbaf9512eab129ef38c8fb83b2743f22f9808fbd4ecce00b95e53153c698fc",
        ),
        "high_actor.pkl": (
            "baselines-v1",
            "d592d75708c918fc7d080872e94d0af925c690fe98c8701510df6eaa5b53f6d6",
        ),
    },
}
_RELEASE_URL = "https://github.com/scaevolabars/fbplan/releases/download/{tag}/{asset}"

# Shared folders below ~/.cache/fbplan, one per kind of file. The env variable of a kind moves
# its folder, for example so that a smoke run does not replace the real models.
TrainedKind = Literal["baselines", "approach_b"]  # fbplan.train_high, fbplan.learned.train_b
CacheKind = Literal["checkpoints", TrainedKind]
_CACHE_ENV_VARS: dict[str, str] = {
    "checkpoints": "FBPLAN_CKPT_DIR",
    "baselines": "FBPLAN_BASELINE_DIR",
    "approach_b": "FBPLAN_APPROACH_B_DIR",
}

# A trained checkpoint stores only the retrained pi_h; F, B and pi_l stay the provided ones.
HIGH_ACTOR_FILE = "high_actor.pkl"
_TRAINED_MODULE = "modules_high_actor"

# The only classes a checkpoint pickle may reference: together they rebuild numpy arrays.
_PICKLE_ALLOWED = frozenset(
    {("numpy._core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype")}
)

# Upstream make_env_and_datasets clips dataset actions to [-1 + eps, 1 - eps].
_ACTION_CLIP_EPS = 1e-5

_log = logging.getLogger(__name__)


def maze_name(env_name: str) -> str:
    """medium, large, giant or teleport. Refuses other envs: no checkpoint exists for them."""
    if env_name not in ENV_NAMES:
        raise ValueError(f"no frozen checkpoint for {env_name!r}; known envs: {ENV_NAMES}")
    return env_name.split("-")[1]


def cache_dir(kind: CacheKind) -> Path:
    """Folder for one kind of file, shared by all worktrees: ~/.cache/fbplan/<kind>.

    The env variable of the kind (FBPLAN_CKPT_DIR, FBPLAN_BASELINE_DIR, FBPLAN_APPROACH_B_DIR)
    replaces the folder.
    """
    if kind not in _CACHE_ENV_VARS:
        raise ValueError(f"unknown cache kind {kind!r}; kinds: {tuple(_CACHE_ENV_VARS)}")
    default = Path.home() / ".cache" / "fbplan" / kind
    return Path(os.environ.get(_CACHE_ENV_VARS[kind], default))


def checkpoint_dir(env_name: str) -> Path:
    """Folder with flags.json and params.pkl of the provided checkpoint of one env."""
    return cache_dir("checkpoints") / maze_name(env_name)


def trained_dir(key: str) -> Path:
    """Folder of one trained checkpoint, from its key "<kind>/<path>".

    For example "baselines/medium/seed0" (fbplan.train_high) or "approach_b/medium/clean/seed0"
    (fbplan.learned.train_b). The key also names the published checkpoint (download_trained).
    """
    kind, _, rest = key.partition("/")
    if kind not in get_args(TrainedKind) or not rest:
        raise ValueError(f"trained checkpoint key {key!r} is not baselines/... or approach_b/...")
    return cache_dir(kind) / rest


def trained_key(kind: TrainedKind, env_name: str, seed: int, name: str | None = None) -> str:
    """Key of one trained checkpoint: "<kind>/<maze>[/<name>]/seed<seed>".

    `name` is the approach B folder (a variant such as "clean", or "value-single"); a baseline
    has none.
    """
    if kind not in get_args(TrainedKind):
        raise ValueError(f"unknown trained kind {kind!r}; kinds: {get_args(TrainedKind)}")
    return "/".join([kind, maze_name(env_name), *([name] if name else []), f"seed{seed}"])


def download_checkpoint(env_name: str) -> Path:
    """Download flags.json and params.pkl of one env into checkpoint_dir(env_name).

    Idempotent: a file that is present is only hash-checked (see _fetch_pinned).
    """
    target = checkpoint_dir(env_name)
    files = _CHECKPOINT_FILES[maze_name(env_name)]
    _fetch_pinned(
        {target / name: (_DRIVE_URL.format(id_), sha256) for name, (id_, sha256) in files.items()}
    )
    return target


def _asset_name(key: str, file_name: str) -> str:
    """Release asset name of one file of a trained checkpoint (a release has no folders)."""
    return f"{key.replace('/', '-')}-{file_name}"


def download_trained(key: str) -> Path:
    """Download the published files of one trained checkpoint into trained_dir(key).

    Idempotent and hash-checked like download_checkpoint. Returns the folder.
    """
    if key not in _TRAINED_FILES:
        published = ", ".join(sorted(_TRAINED_FILES)) or "none"
        raise ValueError(f"trained checkpoint {key} is not published; published: {published}")
    target = trained_dir(key)
    _fetch_pinned(
        {
            target / name: (_RELEASE_URL.format(tag=tag, asset=_asset_name(key, name)), sha256)
            for name, (tag, sha256) in _TRAINED_FILES[key].items()
        }
    )
    return target


def _fetch_pinned(files: dict[Path, tuple[str, str]]) -> None:
    """For each path -> (url, sha256): hash-check the file if present, else download it.

    A present file with a wrong hash raises instead of being replaced, because then the file
    was edited or the source changed.
    """
    for path, (url, sha256) in files.items():
        if path.exists():
            _check_sha256(path, sha256)
        else:
            _download(url, path, sha256)


def _download(url: str, path: Path, sha256: str | None = None) -> None:
    """Fetch url into path through a .part file of its own, then rename it to path.

    So path never holds a partial file, and parallel eval workers that download the same
    missing file at the same time do not write into one .part file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.{uuid.uuid4().hex}.part")
    _log.info("downloading %s -> %s", url, path)
    try:
        with urllib.request.urlopen(url) as response, part.open("wb") as file:
            shutil.copyfileobj(response, file)
        if sha256 is not None and (actual := _sha256(part)) != sha256:
            raise ValueError(f"{url}: sha256 {actual} differs from the pinned {sha256}")
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(path)


def _check_sha256(path: Path, expected: str) -> None:
    if (actual := _sha256(path)) != expected:
        raise ValueError(
            f"{path}: sha256 {actual} differs from the pinned {expected}. "
            "Delete the file to download it again."
        )


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Unpickler that can only rebuild numpy arrays, so a downloaded pickle cannot run code."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _PICKLE_ALLOWED:
            raise pickle.UnpicklingError(
                f"pickle references {module}.{name}; only numpy arrays are allowed"
            )
        return super().find_class(module, name)


def load_numpy_pickle(path: Path) -> Any:
    """Load a pickle of nested dicts of numpy arrays, such as an upstream checkpoint.

    Refuses a pickle that references any other class, so loading cannot run code from the file.
    """
    with path.open("rb") as file:
        return _NumpyOnlyUnpickler(file).load()


def save_numpy_pickle(tree: Any, path: Path) -> None:
    """Write nested dicts of numpy or JAX arrays as numpy arrays, for load_numpy_pickle.

    Numpy arrays make the file independent of the JAX version. Refuses object arrays, because
    load_numpy_pickle could not read them back.
    """
    arrays = jax.tree_util.tree_map(np.asarray, tree)
    if any(leaf.dtype == object for leaf in jax.tree_util.tree_leaves(arrays)):
        raise TypeError(f"{path}: only numeric arrays can be saved")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(arrays, file)


def upstream_module(module: str) -> ModuleType:
    """Import a module of the upstream repo, for example "agents.fbpiswitch".

    Other components use this function for upstream code (for example utils.evaluation or the
    pi_h training recipe) instead of their own sys.path changes. Upstream code imports its
    top-level packages `agents` and `utils` as if main.py runs from third_party/ssm. That folder
    goes first on sys.path, and the check below makes sure that a different package named
    `utils` did not win.
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

    fbpiswitch = upstream_module("agents.fbpiswitch")
    config = fbpiswitch.get_config()
    config.update(flags["agent"])
    agent = fbpiswitch.FBpiSwitchAgent.create(0, example_batch(state["network"]["params"]), config)
    return FrozenFBModel(env_name, flax.serialization.from_state_dict(agent, state))


def example_batch(params: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Zero batch with the observation and action sizes of the network params of an agent.

    Upstream FBpiSwitchAgent.create reads only these sizes. `params` is agent.network.params or
    the network/params part of a saved state.
    """
    obs_dim = params["modules_backward_repr"]["value_net"]["Dense_0"]["kernel"].shape[0]
    action_dim = params["modules_actor"]["mean_net"]["bias"].shape[0]
    return {
        "observations": np.zeros((1, obs_dim), np.float32),
        "actions": np.zeros((1, action_dim), np.float32),
    }


def save_flags(flags: Mapping[str, Any], out_dir: Path) -> None:
    """Write out_dir/flags.json, the record of how a trained checkpoint was made."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "flags.json").write_text(json.dumps(flags, indent=2))


def load_trained_file(key: str, name: str) -> Any:
    """The numpy pickle `name` of trained checkpoint `key`, from trained_dir(key).

    Downloads the published checkpoint when `name` or flags.json is missing. A present file is
    used as it is, without a hash check, so that a model trained on this machine loads before
    it is published.
    """
    folder = trained_dir(key)
    if not all((folder / file).exists() for file in (name, "flags.json")):
        download_trained(key)
    return load_numpy_pickle(folder / name)


def save_trained_checkpoint(agent: Any, flags: Mapping[str, Any], out_dir: Path) -> None:
    """Write the retrained pi_h of `agent` to out_dir: high_actor.pkl and flags.json.

    Only pi_h is stored (2.7 MB instead of 18.3 MB for the full state): F, B and pi_l stay the
    provided ones, and load_trained_model takes them from the provided checkpoint. `flags` must
    hold "env_name" as the provided flags.json does ("ogbench-<env>").
    """
    pi_h = flax.serialization.to_state_dict(agent.network.params[_TRAINED_MODULE])
    save_numpy_pickle(pi_h, out_dir / HIGH_ACTOR_FILE)
    save_flags(flags, out_dir)


def load_trained_model(env_name: str, key: str) -> FrozenFBModel:
    """The provided model of env_name with the pi_h of the trained checkpoint `key`.

    `key` as in trained_dir, for example trained_key("baselines", env_name, 0). Downloads a
    missing published checkpoint (see load_trained_file). Raises if the checkpoint belongs to
    another env.
    """
    saved = load_trained_file(key, HIGH_ACTOR_FILE)
    flags = json.loads((trained_dir(key) / "flags.json").read_text())
    if flags["env_name"] != f"ogbench-{env_name}":
        raise ValueError(f"{key} holds a checkpoint for {flags['env_name']}, not for {env_name}")
    agent = load_model(env_name).agent
    params = flax.core.unfreeze(agent.network.params)
    params[_TRAINED_MODULE] = flax.serialization.from_state_dict(params[_TRAINED_MODULE], saved)
    return FrozenFBModel(env_name, agent.replace(network=agent.network.replace(params=params)))


class FrozenFBModel:
    """FBModel (see fbplan.contracts) on top of an upstream FBpiSwitchAgent.

    Each method calls the same upstream network module and normalize_z as FBpiSwitchAgent does,
    at temperature 0: the policies return the mean action or intention. One deviation, required
    by the contract: high_latent normalizes z_task, while upstream passes z_r to pi_h as it is
    (see infer_task_latent). `agent` is public for code that needs upstream methods, for example
    retraining pi_h or upstream utils.evaluation.evaluate.
    """

    def __init__(self, env_name: str, agent: Any) -> None:
        self.env_name = env_name
        # On device once, so the jitted calls below do not copy the 18 MB of weights every step.
        self.agent = jax.device_put(agent)  # upstream agents.fbpiswitch.FBpiSwitchAgent
        self.latent_dim = int(agent.config["latent_dim"])
        self.discount = float(agent.config["discount"])

    def forward(self, obs: Array, z: Array) -> jax.Array:
        """F(s, z) of both heads. obs [N, obs_dim], z [N, d] -> [2, N, d]."""
        return _forward(self.agent, obs, z)

    def backward(self, obs: Array) -> jax.Array:
        """B(s), raw. obs [N, obs_dim] -> [N, d]."""
        return _backward(self.agent, obs)

    def low_action(self, obs: Array, z: Array) -> jax.Array:
        """pi_l(s, z): the mean action [N, act_dim], clipped to [-1, 1]."""
        return _low_action(self.agent, obs, z)

    def high_latent(self, obs: Array, z_task: Array) -> jax.Array:
        """Baseline pi_h(s, z_task): the mean intention, normalized [N, d]."""
        return _high_latent(self.agent, obs, z_task)

    def infer_task_latent(self, task_id: int) -> np.ndarray:
        """Task embedding z_r [d] for evaluation task `task_id` (1..5), as upstream main.py does.

        Upstream relabels the validation split with the task reward (1 within the goal radius,
        else 0) and calls FBpiSwitchAgent.infer_latent on the first num_zero_shot_samples
        (100k) transitions: z_r = normalize_z(mean(r * B(s))). Upstream draws them with
        HGCDataset.sample(idxs=arange(n), relabeling=False), which returns exactly these rows.

        The norm of z_r is 0.79-0.98 sqrt(d) over the tasks of the 4 mazes, not sqrt(d):
        upstream averages twice (uniform softmax weights 1/n, then a mean), so the raw vector is
        about 1e-7 and the 1e-8 in normalize_z is not negligible. We return the upstream value;
        high_latent normalizes it, upstream does not (effect on success: design.md §2 and §8).
        """
        return self._goal_reward_latent({"task_id": task_id}, f"task {task_id}")

    def infer_goal_latent(self, goal_ij: tuple[int, int]) -> np.ndarray:
        """Latent [d] of a goal at the centre of maze cell goal_ij, as infer_task_latent computes
        z_r: for goals that are not OGBench tasks, such as the held-out goals of fbplan.evaluate.
        """
        options = {"task_info": {"init_ij": goal_ij, "goal_ij": goal_ij}}
        return self._goal_reward_latent(options, f"cell {tuple(goal_ij)}")

    def _goal_reward_latent(self, reset_options: dict[str, Any], goal: str) -> np.ndarray:
        """Upstream infer_latent on the val split relabeled for the goal that the reset sets."""
        env = make_env(self.env_name)
        env.reset(options=reset_options)
        relabel = upstream_module("utils.env_utils").relabel_dataset
        relabeled = relabel(f"ogbench-{self.env_name}", env, self._val_dataset)
        n = self.agent.config["num_zero_shot_samples"]
        if relabeled.size < n:
            raise ValueError(f"validation split has {relabeled.size} rows, upstream uses {n}")
        rewards = relabeled["rewards"][:n]
        if not rewards.any():
            raise ValueError(f"no validation state within the goal radius of {goal}")
        batch = {"observations": relabeled["observations"][:n], "rewards": rewards}
        return np.asarray(self.agent.infer_latent(batch))

    @cached_property
    def _val_dataset(self) -> Any:
        """Validation split as an upstream Dataset, the type that upstream relabel_dataset takes."""
        return upstream_module("utils.datasets").Dataset.create(
            **load_dataset(self.env_name, "val")
        )


def forward(agent: Any, obs: Array, z: Array) -> jax.Array:
    """F(s, z) of both heads [2, N, d] of an upstream agent, as upstream calls it.

    A plain function: FrozenFBModel.forward jits it, and code of other components (the V_H
    of approach B) uses it inside its own jitted functions.
    """
    return agent.network.select("forward_repr")(obs, agent.normalize_z(z), goal_encoded=True)


_forward = jax.jit(forward)


@jax.jit
def _backward(agent: Any, obs: Array) -> jax.Array:
    return agent.network.select("backward_repr")(obs)


@jax.jit
def _low_action(agent: Any, obs: Array, z: Array) -> jax.Array:
    dist = agent.network.select("actor")(obs, agent.normalize_z(z), goal_encoded=True)
    return jnp.clip(dist.mode(), -1, 1)


@jax.jit
def _high_latent(agent: Any, obs: Array, z_task: Array) -> jax.Array:
    # The intention goes to pi_l normalized, as in upstream sample_actions.
    dist = agent.network.select("high_actor")(obs, agent.normalize_z(z_task), goal_encoded=True)
    return agent.normalize_z(dist.mode())


def make_env(env_name: str) -> gym.Env:
    """Create the evaluation env exactly as upstream main.py does.

    Upstream builds it with utils.env_utils.make_env_and_datasets (the OGBench env in an
    EpisodeMonitor wrapper, no frame stack) and then turns goal noise off, so all episodes of a
    task share one goal. env_only=True builds the same wrappers without loading the datasets.
    """
    env = upstream_module("utils.env_utils").make_env_and_datasets(
        f"ogbench-{env_name}", env_only=True
    )
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
    path = _download_dataset(env_name, split)
    data = ogbench.utils.load_dataset(str(path), compact_dataset=False, add_info=True)
    data["actions"] = np.clip(data["actions"], -1 + _ACTION_CLIP_EPS, 1 - _ACTION_CLIP_EPS)
    return data


def _download_dataset(env_name: str, split: Literal["train", "val"]) -> Path:
    """Download one OGBench split if it is absent, and return its path."""
    path = dataset_path(env_name, split)
    if not path.exists():
        _download(f"{ogbench.utils.DATASET_URL}/{path.name}", path)
    return path


def _known_env(name: str) -> str:
    """argparse type: an env with a provided checkpoint (maze_name checks it)."""
    try:
        maze_name(name)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None
    return name


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: download checkpoints and, on request, OGBench data. See the module docstring."""
    parser = argparse.ArgumentParser(
        description="Download the provided checkpoints into cache_dir('checkpoints')."
    )
    parser.add_argument(
        "env_names",
        nargs="*",
        type=_known_env,
        metavar="ENV",
        help=f"provided checkpoints to get (default: all of {', '.join(ENV_NAMES)})",
    )
    parser.add_argument(
        "--trained",
        action="store_true",
        help="also get every published trained checkpoint (retrained baselines, approach B)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=_known_env,
        default=[],
        metavar="ENV",
        help="also get the OGBench train and val splits of these envs into ~/.ogbench/data",
    )
    args = parser.parse_args(argv)
    for name in args.env_names or ENV_NAMES:
        print(download_checkpoint(name))
    if args.trained:
        for key in _TRAINED_FILES:
            print(download_trained(key))
    for name in args.datasets:
        for split in ("train", "val"):
            print(_download_dataset(name, split))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
