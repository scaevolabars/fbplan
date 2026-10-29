"""Retrain the baseline high-level policy pi_h on the frozen F, B and pi_l.

Recipe: upstream `main.py` with `agents/fbpiswitch.py` and the checkpoint's own agent config
(500k steps, batch 1024, Adam 3e-4, AWR with high_alpha 0.1 on the proxy advantage of
docs/design.md §2). We call the upstream `create` and `update`, so the networks, the loss and
the optimizer are the upstream code. Differences from upstream `main.py`, none of which changes
the objective:
- batches come from `fbplan.learned.sampling` (same distribution, sampled on the device);
- no wandb and no evaluation during training (d-eval evaluates the saved checkpoints);
- logged training metrics are means over each log interval, not values of one step.

Usage:
    uv run python -m fbplan.train_high --env antmaze-medium-navigate-v0 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import time
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from fbplan import frozen
from fbplan.learned.sampling import (
    HighBatchSpec,
    TrajectoryData,
    make_trajectory_data,
    sample_high_batch,
)

Agent = Any  # upstream agents.fbpiswitch.FBpiSwitchAgent
FROZEN_MODULES = ("modules_forward_repr", "modules_backward_repr", "modules_actor")
SAMPLER_STREAM = 1  # fold_in tag: keeps batch sampling independent of the network init key


def baseline_dir(env_name: str, seed: int) -> Path:
    """Folder of the retrained pi_h for one env and seed. Shared by all worktrees."""
    default = Path.home() / ".cache" / "fbplan" / "baselines"
    root = Path(os.environ.get("FBPLAN_BASELINE_DIR", default))
    return root / env_name.split("-")[1] / f"seed{seed}"


def hashable_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Agent config with lists turned into tuples.

    jit hashes the agent config (a static field). flags.json stores tuples as lists.
    """
    return {k: tuple(v) if isinstance(v, list) else v for k, v in config.items()}


def with_fresh_high_actor(frozen_agent: Agent, seed: int, example_batch: dict) -> Agent:
    """New agent: F, B and pi_l of `frozen_agent`, and a new pi_h initialized from `seed`.

    Same as upstream `load_agent_from_frozen`: `create(seed)` builds all modules and the
    optimizer state, then the frozen modules replace their fresh copies.
    """
    config = hashable_config(frozen_agent.config)
    agent = type(frozen_agent).create(seed, example_batch, config)
    params = flax.core.unfreeze(agent.network.params)
    for name in FROZEN_MODULES:
        params[name] = flax.core.unfreeze(frozen_agent.network.params[name])
    return agent.replace(network=agent.network.replace(params=params))


@partial(jax.jit, static_argnames=("n_steps", "spec"))
def train_chunk(
    agent: Agent, data: TrajectoryData, key: jax.Array, n_steps: int, spec: HighBatchSpec
) -> tuple[Agent, dict[str, jax.Array]]:
    """Run `n_steps` upstream updates, each on a new batch. Returns the agent and mean metrics."""

    def step(agent: Agent, step_key: jax.Array) -> tuple[Agent, dict[str, jax.Array]]:
        return agent.update(sample_high_batch(data, step_key, spec))

    agent, infos = jax.lax.scan(step, agent, jax.random.split(key, n_steps))
    return agent, jax.tree_util.tree_map(jnp.mean, infos)


@partial(jax.jit, static_argnames=("spec",))
def validation_metrics(
    agent: Agent, data: TrajectoryData, key: jax.Array, spec: HighBatchSpec
) -> dict[str, jax.Array]:
    """Upstream loss metrics on one batch of held-out data, without an update."""
    _, info = agent.total_loss(sample_high_batch(data, key, spec), grad_params=None)
    return info


def train(
    agent: Agent,
    train_data: TrajectoryData,
    val_data: TrajectoryData,
    steps: int,
    log_every: int,
    key: jax.Array,
    log_path: Path,
) -> Agent:
    """Train for `steps` updates. After every `log_every` updates, rewrite the CSV at log_path."""
    if steps % log_every != 0:
        raise ValueError(f"steps={steps} is not a multiple of log_every={log_every}")
    spec = HighBatchSpec.from_agent_config(agent.config)
    rows: list[dict[str, float]] = []
    start = time.time()
    for chunk in range(1, steps // log_every + 1):
        key, train_key, val_key = jax.random.split(key, 3)
        agent, train_info = train_chunk(agent, train_data, train_key, log_every, spec)
        val_info = validation_metrics(agent, val_data, val_key, spec)
        row = {"step": chunk * log_every, "time_s": time.time() - start}
        row |= {f"training/{k}": float(v) for k, v in train_info.items()}
        row |= {f"validation/{k}": float(v) for k, v in val_info.items()}
        rows.append(row)
        pd.DataFrame(rows).to_csv(log_path, index=False)
        print(
            f"step {row['step']:>7d}  loss {row['training/high_actor/actor_loss']:.4f}"
            f"  val {row['validation/high_actor/actor_loss']:.4f}  {row['time_s']:.0f}s",
            flush=True,
        )
    return agent


def save_checkpoint(agent: Agent, flags: dict[str, Any], out_dir: Path) -> None:
    """Write params.pkl and flags.json in the layout of the provided checkpoints.

    fbplan.frozen can then load the folder as it loads a provided one. The arrays are numpy,
    so the pickle does not depend on the JAX version.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    state = jax.tree_util.tree_map(np.asarray, flax.serialization.to_state_dict(agent))
    with open(out_dir / "params.pkl", "wb") as f:
        pickle.dump({"agent": state}, f)
    (out_dir / "flags.json").write_text(json.dumps(flags, indent=2))


def load_baseline_model(env_name: str, seed: int) -> frozen.FrozenFBModel:
    """The frozen model with the pi_h that `seed` retrained: a baseline row for d-eval.

    Raises if the saved F, B or pi_l differ from the provided checkpoint, because then the file
    belongs to another env or to modified modules.
    """
    provided = frozen.load_model(env_name).agent
    state = frozen.load_numpy_pickle(baseline_dir(env_name, seed) / "params.pkl")["agent"]
    agent = flax.serialization.from_state_dict(provided, state)
    for name in FROZEN_MODULES:
        saved = jax.tree_util.tree_leaves(agent.network.params[name])
        expected = jax.tree_util.tree_leaves(provided.network.params[name])
        if not all(np.array_equal(a, b) for a, b in zip(saved, expected, strict=True)):
            raise ValueError(f"{name} in {baseline_dir(env_name, seed)} is not the provided one")
    return frozen.FrozenFBModel(env_name, jax.device_put(agent))


def git_commit() -> str:
    """Commit of this code, recorded with each checkpoint for reproducibility.

    On a GPU instance the code arrives without .git: scripts/gpu/vast.sh writes REVISION.
    """
    repo = Path(__file__).resolve().parents[1]
    if (repo / "REVISION").exists():
        return (repo / "REVISION").read_text().strip()
    cmd = ["git", "-C", str(repo), "rev-parse", "HEAD"]
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip() or "unknown"


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--env", required=True, choices=frozen.ENV_NAMES)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--log-every", type=int, default=5_000)
    parser.add_argument("--out", type=Path, help="default: baseline_dir(env, seed)")
    parser.add_argument("--run-dir", type=Path, help="default: runs/train_high/<env>/seed<k>")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    short_env = args.env.split("-")[1]
    out_dir = args.out or baseline_dir(args.env, args.seed)
    run_dir = args.run_dir or Path("runs") / "train_high" / short_env / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_set = frozen.load_dataset(args.env, "train")
    val_set = frozen.load_dataset(args.env, "val")
    example_batch = {k: train_set[k][:1] for k in ("observations", "actions")}
    agent = with_fresh_high_actor(frozen.load_model(args.env).agent, args.seed, example_batch)
    print(json.dumps({"env": args.env, "seed": args.seed, "agent": agent.config}, default=str))

    agent = train(
        agent,
        make_trajectory_data(train_set["observations"], train_set["terminals"]),
        make_trajectory_data(val_set["observations"], val_set["terminals"]),
        steps=args.steps,
        log_every=args.log_every,
        key=jax.random.fold_in(jax.random.PRNGKey(args.seed), SAMPLER_STREAM),
        log_path=run_dir / "train.csv",
    )

    source = frozen.checkpoint_dir(args.env)
    flags = json.loads((source / "flags.json").read_text())
    flags |= {
        "seed": args.seed,
        "train_steps": args.steps,
        "agent": dict(agent.config),
        "retrained_high_actor": {"from": str(source), "commit": git_commit()},
    }
    save_checkpoint(agent, flags, out_dir)
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()
