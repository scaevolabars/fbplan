"""Trained models of train_high and approach B: folders, file format, loaders, a fresh pi_h.

A trained folder holds only what training changed (user decision, 2026-10-29):

    high_actor.pkl   params of pi_h (upstream modules_high_actor), numpy arrays only
    value.pkl        params of the V_H network (approach B stage 1), numpy arrays only
    flags.json       how the folder was made; paths are relative to ~/.cache/fbplan

F, B and pi_l always come from the provided checkpoint (fbplan.frozen.load_model), so a trained
folder cannot change them. A missing file is fetched with fbplan.frozen.download_trained.

    uv run python -m fbplan.learned.checkpoint export <folder> ...
converts folders written by earlier code (full agent state in params.pkl) to this format.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import flax
import jax
import numpy as np

from fbplan import frozen

Agent = Any  # upstream agents.fbpiswitch.FBpiSwitchAgent
Kind = Literal["baselines", "approach_b"]
FROZEN_MODULES = ("modules_forward_repr", "modules_backward_repr", "modules_actor")
SAMPLER_STREAM = 1  # fold_in tag: keeps batch sampling independent of the network init key
CACHE = Path.home() / ".cache" / "fbplan"
_ROOT_VARS: dict[Kind, str] = {
    "baselines": "FBPLAN_BASELINE_DIR",
    "approach_b": "FBPLAN_APPROACH_B_DIR",
}
_OLD_KEYS = {"rho_max": "rho_cap"}  # renamed config fields, for export


def maze_name(env_name: str) -> str:
    """ "medium" for "antmaze-medium-navigate-v0"."""
    return env_name.split("-")[1]


def trained_key(kind: Kind, env_name: str, seed: int, name: str | None = None) -> str:
    """Folder of one trained model below ~/.cache/fbplan, the key of fbplan.frozen.download_trained.

    For example baselines/medium/seed0 or approach_b/medium/clean/seed0.
    """
    return "/".join([kind, maze_name(env_name), *([name] if name else []), f"seed{seed}"])


def trained_dir(kind: Kind, env_name: str, seed: int, name: str | None = None) -> Path:
    """Local folder of trained_key. FBPLAN_BASELINE_DIR or FBPLAN_APPROACH_B_DIR move its root."""
    root = Path(os.environ.get(_ROOT_VARS[kind], CACHE / kind))
    return root / trained_key(kind, env_name, seed, name).split("/", 1)[1]


def save_numpy_pickle(tree: Any, path: Path) -> None:
    """Pickle a tree of arrays as numpy arrays, which fbplan.frozen.load_numpy_pickle reads back."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, tree), f)


def save_flags(flags: Mapping[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "flags.json").write_text(json.dumps(flags, indent=2))


def load_trained_file(key: str, folder: Path, name: str) -> Any:
    """folder/name, downloaded with fbplan.frozen.download_trained(key) when it is absent."""
    if not (folder / name).exists():
        frozen.download_trained(key, folder)
    return frozen.load_numpy_pickle(folder / name)


def save_high_actor(agent: Agent, flags: Mapping[str, Any], out_dir: Path) -> None:
    """high_actor.pkl and flags.json of a trained pi_h."""
    high_actor = flax.serialization.to_state_dict(agent.network.params["modules_high_actor"])
    save_numpy_pickle(high_actor, out_dir / "high_actor.pkl")
    save_flags(flags, out_dir)


def with_high_actor(agent: Agent, high_actor: Any) -> Agent:
    """The agent with its pi_h params replaced by a saved state dict."""
    params = flax.core.unfreeze(agent.network.params)
    params["modules_high_actor"] = flax.serialization.from_state_dict(
        params["modules_high_actor"], high_actor
    )
    return agent.replace(network=agent.network.replace(params=params))


def load_high_actor_model(
    env_name: str, kind: Kind, seed: int, name: str | None = None
) -> frozen.FrozenFBModel:
    """The provided model with a trained pi_h, for evaluation."""
    key = trained_key(kind, env_name, seed, name)
    high_actor = load_trained_file(key, trained_dir(kind, env_name, seed, name), "high_actor.pkl")
    agent = with_high_actor(frozen.load_model(env_name).agent, high_actor)
    return frozen.FrozenFBModel(env_name, jax.device_put(agent))


def load_baseline_model(env_name: str, seed: int) -> frozen.FrozenFBModel:
    """The pi_h that train_high retrained with `seed`: a baseline row for d-eval."""
    return load_high_actor_model(env_name, "baselines", seed)


def load_b_model(env_name: str, variant: str, seed: int) -> frozen.FrozenFBModel:
    """The pi_h of an approach B variant (clean, wide, region) for d-eval."""
    return load_high_actor_model(env_name, "approach_b", seed, variant)


def hashable_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Agent config with lists turned into tuples.

    jit hashes the agent config (a static field). flags.json stores tuples as lists.
    """
    return {k: tuple(v) if isinstance(v, list) else v for k, v in config.items()}


def example_batch(agent: Agent, obs_dim: int) -> dict[str, np.ndarray]:
    """Zero batch with the observation and action sizes that upstream `create` reads."""
    action_dim = agent.network.params["modules_actor"]["mean_net"]["bias"].shape[-1]
    return {
        "observations": np.zeros((1, obs_dim), np.float32),
        "actions": np.zeros((1, action_dim), np.float32),
    }


def with_fresh_high_actor(frozen_agent: Agent, seed: int, obs_dim: int) -> Agent:
    """New agent: F, B and pi_l of `frozen_agent`, and a new pi_h initialized from `seed`.

    Same as upstream `load_agent_from_frozen`: `create(seed)` builds all modules and the
    optimizer state, then the frozen modules replace their fresh copies.
    """
    config = hashable_config(frozen_agent.config)
    agent = type(frozen_agent).create(seed, example_batch(frozen_agent, obs_dim), config)
    params = flax.core.unfreeze(agent.network.params)
    for name in FROZEN_MODULES:
        params[name] = flax.core.unfreeze(frozen_agent.network.params[name])
    return agent.replace(network=agent.network.replace(params=params))


def git_commit() -> str:
    """Commit of this code, recorded with each trained model for reproducibility.

    On a GPU instance the code arrives without .git: scripts/gpu/vast.sh writes REVISION.
    """
    repo = Path(__file__).resolve().parents[2]
    if (repo / "REVISION").exists():
        return (repo / "REVISION").read_text().strip()
    cmd = ["git", "-C", str(repo), "rev-parse", "HEAD"]
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip() or "unknown"


def cache_relative(path: Path | str) -> str:
    """A path below some .cache/fbplan, written relative to it (it may be another machine's)."""
    text = str(path)
    marker = "/.cache/fbplan/"
    return text.split(marker, 1)[1] if marker in text else text


def portable_flags(flags: Any) -> Any:
    """flags.json of earlier code: cache paths made relative, renamed config fields updated."""
    if isinstance(flags, dict):
        return {_OLD_KEYS.get(k, k): portable_flags(v) for k, v in flags.items()}
    if isinstance(flags, list):
        return [portable_flags(v) for v in flags]
    return cache_relative(flags) if isinstance(flags, str) else flags


def export_folder(folder: Path) -> None:
    """Rewrite a folder of earlier code in this format, in place.

    params.pkl (full agent state) gives high_actor.pkl; value.pkl is already in this format.
    flags.json is rewritten by portable_flags. params.pkl stays; it is no longer read.
    """
    if (folder / "params.pkl").exists():
        state = frozen.load_numpy_pickle(folder / "params.pkl")["agent"]
        high_actor = state["network"]["params"]["modules_high_actor"]
        save_numpy_pickle(high_actor, folder / "high_actor.pkl")
    flags = json.loads((folder / "flags.json").read_text())
    save_flags(portable_flags(flags), folder)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert trained folders of earlier code.")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="params.pkl -> high_actor.pkl; portable flags.json")
    export.add_argument("folders", type=Path, nargs="+", help="trained folders, e.g. .../seed0")
    args = parser.parse_args(argv)
    for folder in args.folders:
        export_folder(folder)
        print(f"exported {folder}")


if __name__ == "__main__":
    main()
