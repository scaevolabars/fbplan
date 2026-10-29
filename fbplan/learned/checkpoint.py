"""Trained models of train_high and approach B: keys, a fresh pi_h, file helpers, export.

A trained folder holds only what training changed (user decision, 2026-10-29). fbplan.frozen
owns the folders, the pi_h format and the download:

    high_actor.pkl   params of pi_h: frozen.save_trained_checkpoint, frozen.load_trained_model
    value.pkl        params of the V_H network (approach B stage 1): fbplan.learned.value
    flags.json       how the folder was made; paths are relative to ~/.cache/fbplan

F, B and pi_l always come from the provided checkpoint, so a trained folder cannot change them.

    uv run python -m fbplan.learned.checkpoint export <folder> ...
converts folders written by earlier code (full agent state in params.pkl) to this format.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import flax

from fbplan import frozen

Agent = Any  # upstream agents.fbpiswitch.FBpiSwitchAgent
Kind = Literal["baselines", "approach_b"]
FROZEN_MODULES = ("modules_forward_repr", "modules_backward_repr", "modules_actor")
SAMPLER_STREAM = 1  # fold_in tag: keeps batch sampling independent of the network init key
_OLD_KEYS = {"rho_max": "rho_cap"}  # renamed config fields, for export


def trained_key(kind: Kind, env_name: str, seed: int, name: str | None = None) -> str:
    """Key of one trained model for fbplan.frozen.trained_dir and download_trained.

    For example baselines/medium/seed0 or approach_b/medium/clean/seed0.
    """
    parts = [kind, frozen.maze_name(env_name), *([name] if name else []), f"seed{seed}"]
    return "/".join(parts)


def trained_dir(kind: Kind, env_name: str, seed: int, name: str | None = None) -> Path:
    """frozen.trained_dir(trained_key(...)). Kept for tests/test_evaluate.py until it calls that."""
    return frozen.trained_dir(trained_key(kind, env_name, seed, name))


def save_flags(flags: Mapping[str, Any], out_dir: Path) -> None:
    """Write out_dir/flags.json (indented), creating the folder."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "flags.json").write_text(json.dumps(flags, indent=2))


def load_trained_file(key: str, name: str) -> Any:
    """File `name` of the trained folder `key`; frozen.download_trained(key) fetches a missing one.

    The folder is frozen.trained_dir(key), the folder that download_trained writes.
    """
    folder = frozen.trained_dir(key)
    if not all((folder / f).exists() for f in (name, "flags.json")):
        frozen.download_trained(key)
    return frozen.load_numpy_pickle(folder / name)


def load_baseline_model(env_name: str, seed: int, name: str | None = None) -> frozen.FrozenFBModel:
    """The provided model with the pi_h that train_high retrained with `seed`.

    `name` selects a variant of the recipe (train_high --name), for example awr2x.
    """
    return frozen.load_trained_model(env_name, trained_key("baselines", env_name, seed, name))


def load_b_model(env_name: str, variant: str, seed: int) -> frozen.FrozenFBModel:
    """The provided model with the pi_h of an approach B variant (clean, wide, region)."""
    return frozen.load_trained_model(env_name, trained_key("approach_b", env_name, seed, variant))


def hashable_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Agent config with lists turned into tuples.

    jit hashes the agent config (a static field). flags.json stores tuples as lists.
    """
    return {k: tuple(v) if isinstance(v, list) else v for k, v in config.items()}


def with_fresh_high_actor(frozen_agent: Agent, seed: int) -> Agent:
    """New agent: F, B and pi_l of `frozen_agent`, and a new pi_h initialized from `seed`.

    Same as upstream `load_agent_from_frozen`: `create(seed)` builds all modules and the
    optimizer state, then the frozen modules replace their fresh copies.
    """
    config = hashable_config(frozen_agent.config)
    batch = frozen.example_batch(frozen_agent.network.params)
    agent = type(frozen_agent).create(seed, batch, config)
    params = flax.core.unfreeze(agent.network.params)
    for name in FROZEN_MODULES:
        params[name] = flax.core.unfreeze(frozen_agent.network.params[name])
    return agent.replace(network=agent.network.replace(params=params))


def run_flags(env_name: str, seed: int, steps: int) -> dict[str, Any]:
    """flags.json of a training run: the provided checkpoint's flags, with seed, steps, commit."""
    flags = json.loads((frozen.checkpoint_dir(env_name) / "flags.json").read_text())
    return flags | {"seed": seed, "train_steps": steps, "commit": git_commit()}


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
        frozen.save_numpy_pickle(high_actor, folder / frozen.HIGH_ACTOR_FILE)
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
