"""Train approach B (docs/plans/approach-b.md): stage 1 fits V_H, stage 2 trains a new pi_h.

Variants (plan §3). Each changes one thing against the one before it:
    clean   single-state latents B(.), upstream goal distribution: only the critic changes
    wide    as clean, plus 30 % random goals for pi_h
    region  latents averaged over +-5 trajectory steps; needs its own V_H

Usage:
    uv run python -m fbplan.learned.train_b value  --env antmaze-medium-navigate-v0 --seed 0
    uv run python -m fbplan.learned.train_b policy --env antmaze-medium-navigate-v0 --seed 0 \\
        --variant clean
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any, Literal, get_args

import flax
import jax
import jax.numpy as jnp
import numpy as np

from fbplan import frozen
from fbplan.learned.checkpoint import (
    SAMPLER_STREAM,
    Agent,
    git_commit,
    trained_key,
    with_fresh_high_actor,
)
from fbplan.learned.high_policy import (
    masked_mean_std,
    matched_alpha,
    policy_loss,
    policy_update,
    switch_advantage,
)
from fbplan.learned.loop import run_logged, scan_updates
from fbplan.learned.option_targets import EdgeReward, OptionConfig
from fbplan.learned.sampling import (
    HighBatchSpec,
    SwitchBatch,
    TrajectoryData,
    make_trajectory_data,
    sample_high_batch,
    sample_switch_batch,
    trajectory_window_latents,
)
from fbplan.learned.value import (
    ValueConfig,
    ValueState,
    load_value,
    save_value,
    value_loss,
    value_update,
)

LatentKind = Literal["single", "region"]
# variant -> (latents, probability of a random dataset goal for pi_h)
VARIANTS: dict[str, tuple[LatentKind, float]] = {
    "clean": ("single", 0.0),
    "wide": ("single", 0.3),
    "region": ("region", 0.0),
}
VALUE_RANDOM_GOALS = 0.3  # V_H must cover goals across trajectories, in every variant
WINDOW_HALF_WIDTH = 5
CALIBRATION_BATCHES = 10


def latent_table(
    model: frozen.FrozenFBModel, observations: np.ndarray, terminals: np.ndarray, kind: LatentKind
) -> jax.Array:
    """B of every dataset state, or its trajectory-window mean. B is frozen: one pass serves all.

    TODO(e-learner): use c-planner's public batched-B helper when it exists (review X4).
    """
    starts = range(0, len(observations), 65536)
    table = np.concatenate(
        [np.asarray(model.backward(observations[i : i + 65536])) for i in starts]
    )
    if kind == "region":
        table = trajectory_window_latents(table, terminals, WINDOW_HALF_WIDTH)
    return jnp.asarray(table, dtype=jnp.float32)


@dataclasses.dataclass(frozen=True)
class Stage:
    """The settings of a jitted step that are not arrays (static under jit)."""

    spec: HighBatchSpec
    gaussian_prob: float  # share of Gaussian task latents, as upstream critic_latent_mix_prob
    value: ValueConfig


@flax.struct.dataclass
class Split:
    """One dataset split on the device, with its latent table. A pytree: jitted code takes it."""

    data: TrajectoryData
    latents: jax.Array

    @classmethod
    def load(cls, model: frozen.FrozenFBModel, split: str, kind: LatentKind) -> Split:
        raw = frozen.load_dataset(model.env_name, split)
        data = make_trajectory_data(raw["observations"], raw["terminals"])
        return cls(data, latent_table(model, raw["observations"], raw["terminals"], kind))

    @property
    def obs_dim(self) -> int:
        return self.data.observations.shape[-1]

    def batch(self, key: jax.Array, stage: Stage) -> SwitchBatch:
        return sample_switch_batch(self.data, self.latents, key, stage.spec, stage.gaussian_prob)


# ----- stage 1: V_H -----


@partial(jax.jit, static_argnames=("n_steps", "stage"))
def value_chunk(
    value: ValueState, agent: Agent, split: Split, key: jax.Array, n_steps: int, stage: Stage
) -> tuple[ValueState, dict[str, jax.Array]]:
    """n_steps updates of V_H on new batches of `split`; returns mean metrics."""

    def step(value: ValueState, k: jax.Array) -> tuple[ValueState, dict[str, jax.Array]]:
        return value_update(value, agent, split.batch(k, stage), stage.value)

    return scan_updates(step, value, key, n_steps)


@partial(jax.jit, static_argnames=("stage",))
def value_validation(
    value: ValueState, agent: Agent, split: Split, key: jax.Array, stage: Stage
) -> dict[str, jax.Array]:
    """Stage 1 metrics on one batch of held-out data, without an update."""
    return value_loss(value, value.params, agent, split.batch(key, stage), stage.value)[1]


def train_value(args: argparse.Namespace, model: frozen.FrozenFBModel, run_dir: Path) -> None:
    """Stage 1 of approach B for one seed; saves value.pkl in approach_b/<maze>/<name>/."""
    agent = model.agent
    train_split, val_split = (Split.load(model, s, args.latents) for s in ("train", "val"))
    config = ValueConfig(
        option=OptionConfig(edge_reward=args.edge_reward, rho_cap=args.rho_cap),
        expectile=args.expectile,
        hop_cost=args.hop_cost,
    )
    stage = Stage(
        spec=dataclasses.replace(
            HighBatchSpec.from_agent_config(agent.config), actor_p_randomgoal=VALUE_RANDOM_GOALS
        ),
        gaussian_prob=float(agent.config["critic_latent_mix_prob"]),
        value=config,
    )
    init_key = jax.random.PRNGKey(args.seed)
    value = ValueState.create(init_key, train_split.obs_dim, model.latent_dim, config)
    value = run_logged(
        value,
        chunk=lambda v, k, n: value_chunk(v, agent, train_split, k, n, stage),
        validate=lambda v, k: value_validation(v, agent, val_split, k, stage),
        steps=args.steps,
        log_every=args.log_every,
        key=jax.random.fold_in(init_key, SAMPLER_STREAM),
        log_path=run_dir / "train.csv",
        progress_key="loss",
    )
    flags = base_flags(args) | {
        "latents": args.latents,
        "value_spec": dataclasses.asdict(stage.spec),
    }
    out_dir = frozen.trained_dir(trained_key("approach_b", args.env, args.seed, args.name))
    save_value(value, config, flags, out_dir)
    print(f"saved {out_dir}")


# ----- stage 2: pi_h -----


@partial(jax.jit, static_argnames=("n_steps", "stage"))
def policy_chunk(
    agent: Agent,
    value: ValueState,
    split: Split,
    key: jax.Array,
    alpha: jax.Array,
    n_steps: int,
    stage: Stage,
) -> tuple[Agent, dict[str, jax.Array]]:
    """n_steps AWR updates of pi_h on new batches of `split`; returns mean metrics."""

    def step(agent: Agent, k: jax.Array) -> tuple[Agent, dict[str, jax.Array]]:
        return policy_update(agent, value, split.batch(k, stage), alpha, stage.value)

    return scan_updates(step, agent, key, n_steps)


@partial(jax.jit, static_argnames=("stage",))
def policy_validation(
    agent: Agent, value: ValueState, split: Split, key: jax.Array, alpha: jax.Array, stage: Stage
) -> dict[str, jax.Array]:
    """Stage 2 metrics on one batch of held-out data, without an update."""
    return policy_loss(None, agent, value, split.batch(key, stage), alpha, stage.value)[1]


@partial(jax.jit, static_argnames=("stage",))
def advantage_stds(
    agent: Agent, value: ValueState, split: Split, key: jax.Array, stage: Stage
) -> tuple[jax.Array, jax.Array]:
    """std of upstream's one-switch advantage and of the switch advantage, one batch each."""
    upstream_spec = HighBatchSpec.from_agent_config(agent.config)
    k_up, k_b = jax.random.split(key)
    upstream_batch = sample_high_batch(split.data, k_up, upstream_spec)
    _, info = agent.total_loss(upstream_batch, grad_params=None)
    adv, valid = switch_advantage(agent, value, split.batch(k_b, stage), stage.value)
    return info["high_actor/adv_std"], masked_mean_std(adv, valid)[1]


def calibrate_alpha(
    agent: Agent, value: ValueState, split: Split, key: jax.Array, stage: Stage
) -> tuple[float, dict[str, float]]:
    """AWR temperature with the spread of upstream's alpha * A (see matched_alpha)."""
    stds = [
        advantage_stds(agent, value, split, k, stage)
        for k in jax.random.split(key, CALIBRATION_BATCHES)
    ]
    upstream_std = float(np.mean([s[0] for s in stds]))
    switch_std = float(np.mean([s[1] for s in stds]))
    alpha = matched_alpha(upstream_std, switch_std, float(agent.config["high_alpha"]))
    return alpha, {"upstream_adv_std": upstream_std, "switch_adv_std": switch_std, "alpha": alpha}


def policy_stage(agent: Agent, variant: str, config: ValueConfig) -> Stage:
    """Batches for pi_h: upstream's goal distribution, plus random goals in the wide variant."""
    spec = dataclasses.replace(
        HighBatchSpec.from_agent_config(agent.config), actor_p_randomgoal=VARIANTS[variant][1]
    )
    return Stage(spec, float(agent.config["critic_latent_mix_prob"]), config)


def train_policy(args: argparse.Namespace, model: frozen.FrozenFBModel, run_dir: Path) -> None:
    """Stage 2 of approach B for one seed; saves high_actor.pkl in approach_b/<maze>/<name>/."""
    latents = VARIANTS[args.variant][0]
    train_split, val_split = (Split.load(model, s, latents) for s in ("train", "val"))
    value, config = load_value(
        args.env, args.value_name, args.seed, train_split.obs_dim, model.latent_dim
    )
    agent = with_fresh_high_actor(model.agent, args.seed)
    stage = policy_stage(agent, args.variant, config)

    key = jax.random.fold_in(jax.random.PRNGKey(args.seed), SAMPLER_STREAM)
    calibration_key, key = jax.random.split(key)
    alpha, calibration = calibrate_alpha(agent, value, train_split, calibration_key, stage)
    if args.alpha is not None:
        alpha = args.alpha
    print(json.dumps(calibration | {"alpha_used": alpha}), flush=True)

    alpha_array = jnp.asarray(alpha, dtype=jnp.float32)
    agent = run_logged(
        agent,
        chunk=lambda a, k, n: policy_chunk(a, value, train_split, k, alpha_array, n, stage),
        validate=lambda a, k: policy_validation(a, value, val_split, k, alpha_array, stage),
        steps=args.steps,
        log_every=args.log_every,
        key=key,
        log_path=run_dir / "train.csv",
        progress_key="actor_loss",
    )
    flags = base_flags(args) | {
        "agent": dict(agent.config),
        "approach_b": {
            "variant": args.variant,
            "value": trained_key("approach_b", args.env, args.seed, args.value_name),
            "value_config": dataclasses.asdict(config),
            "policy_spec": dataclasses.asdict(stage.spec),
            "calibration": calibration,
            "alpha_used": alpha,
        },
    }
    out_dir = frozen.trained_dir(trained_key("approach_b", args.env, args.seed, args.name))
    frozen.save_trained_checkpoint(agent, flags, out_dir)
    print(f"saved {out_dir}")


# ----- CLI -----


def base_flags(args: argparse.Namespace) -> dict[str, Any]:
    """The provided checkpoint's flags.json, with this run's seed, steps and commit."""
    flags = json.loads((frozen.checkpoint_dir(args.env) / "flags.json").read_text())
    return flags | {"seed": args.seed, "train_steps": args.steps, "commit": git_commit()}


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    stages = {
        "value": sub.add_parser("value", help="stage 1: fit V_H"),
        "policy": sub.add_parser("policy", help="stage 2: train pi_h with the switch advantage"),
    }
    for p in stages.values():
        p.add_argument("--env", required=True, choices=frozen.ENV_NAMES)
        p.add_argument("--seed", type=int, required=True, help="network init and batch sampling")
        p.add_argument("--steps", type=int, default=500_000, help="number of updates")
        p.add_argument("--log-every", type=int, default=5_000, help="updates per CSV row")
        p.add_argument("--run-dir", type=Path, help="default: runs/train_b/<maze>/<name>/seed<k>")
        p.add_argument("--name", help="output folder below approach_b/<maze>/; default below")
    value = stages["value"]
    value.add_argument("--latents", choices=get_args(LatentKind), default="single")
    value.add_argument("--expectile", type=float, default=ValueConfig.expectile, help="tau")
    value.add_argument("--rho-cap", type=float, default=OptionConfig.rho_cap, help="cap on rho")
    value.add_argument("--edge-reward", choices=get_args(EdgeReward), default="goal_zero")
    value.add_argument("--hop-cost", type=float, default=ValueConfig.hop_cost, help="per switch")
    policy = stages["policy"]
    policy.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    policy.add_argument("--value-name", help="V_H folder to load; default value-<latents>")
    policy.add_argument("--alpha", type=float, help="AWR temperature; default: upstream spread")
    args = parser.parse_args(argv)
    # Default folders: value-<latents> for V_H, the variant for pi_h. Give ablations their own
    # --name, so they do not replace the V_H that the policy stage loads.
    if args.name is None:
        args.name = f"value-{args.latents}" if args.stage == "value" else args.variant
    if args.stage == "policy" and args.value_name is None:
        args.value_name = f"value-{VARIANTS[args.variant][0]}"
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = (
        args.run_dir
        or Path("runs/train_b") / frozen.maze_name(args.env) / args.name / f"seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    model = frozen.load_model(args.env)
    (train_value if args.stage == "value" else train_policy)(args, model, run_dir)


if __name__ == "__main__":
    main()
