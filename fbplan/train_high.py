"""Retrain the baseline high-level policy pi_h on the frozen F, B and pi_l.

Recipe: upstream `main.py` with `agents/fbpiswitch.py` and the checkpoint's own agent config
(500k steps, batch 1024, Adam 3e-4, AWR with high_alpha 0.1 on the proxy advantage of
docs/design.md §2). We call the upstream `create` and `update`, so the networks, the loss and
the optimizer are the upstream code. Differences from upstream `main.py`, none of which changes
the objective:
- batches come from `fbplan.learned.sampling` (same distribution, sampled on the device);
- no wandb and no evaluation during training (fbplan.evaluate evaluates the saved pi_h);
- logged training metrics are means over each log interval, not values of one step.

The result is saved as high_actor.pkl (fbplan.frozen.save_trained_checkpoint); load it with
`fbplan.learned.checkpoint.load_baseline_model(env, seed)`.

Usage:
    uv run python -m fbplan.train_high --env antmaze-medium-navigate-v0 --seed 0
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import jax

from fbplan import frozen
from fbplan.learned.checkpoint import (
    SAMPLER_STREAM,
    Agent,
    git_commit,
    load_baseline_model,  # noqa: F401  fbplan.evaluate imports it from here (until it switches)
    trained_dir,
    with_fresh_high_actor,
)
from fbplan.learned.loop import run_logged, scan_updates
from fbplan.learned.sampling import (
    HighBatchSpec,
    TrajectoryData,
    make_trajectory_data,
    sample_high_batch,
)


def baseline_dir(env_name: str, seed: int) -> Path:
    """Folder of a retrained pi_h. tests/test_evaluate.py calls it from here (until it switches
    to fbplan.learned.checkpoint.trained_dir)."""
    return trained_dir("baselines", env_name, seed)


@partial(jax.jit, static_argnames=("n_steps", "spec"))
def train_chunk(
    agent: Agent, data: TrajectoryData, key: jax.Array, n_steps: int, spec: HighBatchSpec
) -> tuple[Agent, dict[str, jax.Array]]:
    """Run `n_steps` upstream updates, each on a new batch. Returns the agent and mean metrics."""
    return scan_updates(
        lambda a, k: a.update(sample_high_batch(data, k, spec)), agent, key, n_steps
    )


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
    spec = HighBatchSpec.from_agent_config(agent.config)
    return run_logged(
        agent,
        chunk=lambda agent, key, n: train_chunk(agent, train_data, key, n, spec),
        validate=lambda agent, key: validation_metrics(agent, val_data, key, spec),
        steps=steps,
        log_every=log_every,
        key=key,
        log_path=log_path,
        progress_key="high_actor/actor_loss",
    )


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--env", required=True, choices=frozen.ENV_NAMES)
    parser.add_argument("--seed", type=int, required=True, help="pi_h init and batch sampling")
    parser.add_argument("--steps", type=int, default=500_000, help="number of updates")
    parser.add_argument("--log-every", type=int, default=5_000, help="updates per CSV row")
    parser.add_argument("--out-dir", type=Path, help="default: <cache>/baselines/<maze>/seed<k>")
    parser.add_argument("--run-dir", type=Path, help="default: runs/train_high/<maze>/seed<k>")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    maze = frozen.maze_name(args.env)
    out_dir = args.out_dir or trained_dir("baselines", args.env, args.seed)
    run_dir = args.run_dir or Path("runs/train_high") / maze / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_set = frozen.load_dataset(args.env, "train")
    val_set = frozen.load_dataset(args.env, "val")
    agent = with_fresh_high_actor(frozen.load_model(args.env).agent, args.seed)
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

    flags = json.loads((frozen.checkpoint_dir(args.env) / "flags.json").read_text())
    flags |= {
        "seed": args.seed,
        "train_steps": args.steps,
        "agent": dict(agent.config),
        "retrained_high_actor": {"from": f"checkpoints/{maze}", "commit": git_commit()},
    }
    frozen.save_trained_checkpoint(agent, flags, out_dir)
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()
