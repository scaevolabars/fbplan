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
import os
import pickle
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any, Literal

import flax
import jax
import jax.numpy as jnp
import numpy as np

from fbplan import frozen
from fbplan.learned.high_policy import (
    matched_alpha,
    policy_loss,
    policy_update,
    switch_advantage,
)
from fbplan.learned.loop import run_logged
from fbplan.learned.option_targets import OptionConfig
from fbplan.learned.sampling import (
    HighBatchSpec,
    TrajectoryData,
    make_trajectory_data,
    region_latents,
    sample_high_batch,
    sample_switch_batch,
)
from fbplan.learned.value import Agent, ValueConfig, ValueState, value_loss, value_update
from fbplan.train_high import (
    git_commit,
    load_retrained_model,
    save_checkpoint,
    with_fresh_high_actor,
)

LatentKind = Literal["single", "region"]
# variant -> (latents, probability of a random dataset goal for pi_h)
VARIANTS: dict[str, tuple[LatentKind, float]] = {
    "clean": ("single", 0.0),
    "wide": ("single", 0.3),
    "region": ("region", 0.0),
}
VALUE_RANDOM_GOALS = 0.3  # V_H must cover goals across trajectories, in every variant
REGION_HALF_WIDTH = 5
CALIBRATION_BATCHES = 10
SAMPLER_STREAM = 1  # fold_in tag: keeps batch sampling independent of the network init key


def approach_b_dir(env_name: str, name: str, seed: int) -> Path:
    """Folder of one approach B model (value-<latents> or a variant name). Shared by worktrees."""
    default = Path.home() / ".cache" / "fbplan" / "approach_b"
    root = Path(os.environ.get("FBPLAN_APPROACH_B_DIR", default))
    return root / env_name.split("-")[1] / name / f"seed{seed}"


def latent_table(
    model: frozen.FrozenFBModel, observations: np.ndarray, terminals: np.ndarray, kind: LatentKind
) -> jax.Array:
    """B for every dataset state, or its region mean. B is frozen, so one pass serves all steps."""
    starts = range(0, len(observations), 65536)
    table = np.concatenate(
        [np.asarray(model.backward(observations[i : i + 65536])) for i in starts]
    )
    if kind == "region":
        table = region_latents(table, terminals, REGION_HALF_WIDTH)
    return jnp.asarray(table, dtype=jnp.float32)


@dataclasses.dataclass(frozen=True)
class Split:
    """One dataset split on the device, with its latent table."""

    data: TrajectoryData
    latents: jax.Array

    @classmethod
    def load(cls, model: frozen.FrozenFBModel, split: str, kind: LatentKind) -> Split:
        raw = frozen.load_dataset(model.env_name, split)
        data = make_trajectory_data(raw["observations"], raw["terminals"])
        return cls(data, latent_table(model, raw["observations"], raw["terminals"], kind))


# ----- stage 1: V_H -----


@partial(jax.jit, static_argnames=("n_steps", "spec", "gaussian_prob", "config"))
def value_chunk(
    value: ValueState,
    agent: Agent,
    data: TrajectoryData,
    latents: jax.Array,
    key: jax.Array,
    n_steps: int,
    spec: HighBatchSpec,
    gaussian_prob: float,
    config: ValueConfig,
) -> tuple[ValueState, dict[str, jax.Array]]:
    def step(value: ValueState, k: jax.Array) -> tuple[ValueState, dict[str, jax.Array]]:
        batch = sample_switch_batch(data, latents, k, spec, gaussian_prob)
        return value_update(value, agent, batch, config)

    value, infos = jax.lax.scan(step, value, jax.random.split(key, n_steps))
    return value, jax.tree_util.tree_map(jnp.mean, infos)


@partial(jax.jit, static_argnames=("spec", "gaussian_prob", "config"))
def value_validation(
    value: ValueState,
    agent: Agent,
    data: TrajectoryData,
    latents: jax.Array,
    key: jax.Array,
    spec: HighBatchSpec,
    gaussian_prob: float,
    config: ValueConfig,
) -> dict[str, jax.Array]:
    batch = sample_switch_batch(data, latents, key, spec, gaussian_prob)
    return value_loss(value, value.params, agent, batch, config)[1]


def save_value(value: ValueState, flags: dict[str, Any], out_dir: Path) -> None:
    """value.pkl (numpy arrays only, like the provided checkpoints) and flags.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {"params": value.params, "target_params": value.target_params, "step": value.step}
    with open(out_dir / "value.pkl", "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, state), f)
    (out_dir / "flags.json").write_text(json.dumps(flags, indent=2))


def config_from_dict(d: dict[str, Any]) -> ValueConfig:
    return ValueConfig(
        **(d | {"option": OptionConfig(**d["option"]), "hidden_dims": tuple(d["hidden_dims"])})
    )


def load_value(out_dir: Path, obs_dim: int, latent_dim: int) -> tuple[ValueState, ValueConfig]:
    """The V_H saved by save_value, with the config it was trained with."""
    config = config_from_dict(json.loads((out_dir / "flags.json").read_text())["value_config"])
    template = ValueState.create(jax.random.PRNGKey(0), obs_dim, latent_dim, config)
    state = frozen.load_numpy_pickle(out_dir / "value.pkl")
    value = template.replace(
        params=flax.serialization.from_state_dict(template.params, state["params"]),
        target_params=flax.serialization.from_state_dict(template.params, state["target_params"]),
        step=jnp.asarray(state["step"]),
    )
    return value, config


# ----- stage 2: pi_h -----


@partial(jax.jit, static_argnames=("n_steps", "spec", "gaussian_prob", "config"))
def policy_chunk(
    agent: Agent,
    value: ValueState,
    data: TrajectoryData,
    latents: jax.Array,
    key: jax.Array,
    alpha: jax.Array,
    n_steps: int,
    spec: HighBatchSpec,
    gaussian_prob: float,
    config: ValueConfig,
) -> tuple[Agent, dict[str, jax.Array]]:
    def step(agent: Agent, k: jax.Array) -> tuple[Agent, dict[str, jax.Array]]:
        batch = sample_switch_batch(data, latents, k, spec, gaussian_prob)
        return policy_update(agent, value, batch, alpha, config)

    agent, infos = jax.lax.scan(step, agent, jax.random.split(key, n_steps))
    return agent, jax.tree_util.tree_map(jnp.mean, infos)


@partial(jax.jit, static_argnames=("spec", "gaussian_prob", "config"))
def policy_validation(
    agent: Agent,
    value: ValueState,
    data: TrajectoryData,
    latents: jax.Array,
    key: jax.Array,
    alpha: jax.Array,
    spec: HighBatchSpec,
    gaussian_prob: float,
    config: ValueConfig,
) -> dict[str, jax.Array]:
    batch = sample_switch_batch(data, latents, key, spec, gaussian_prob)
    return policy_loss(None, agent, value, batch, alpha, config)[1]


@partial(jax.jit, static_argnames=("spec", "gaussian_prob", "config"))
def advantage_stds(
    agent: Agent,
    value: ValueState,
    data: TrajectoryData,
    latents: jax.Array,
    key: jax.Array,
    spec: HighBatchSpec,
    gaussian_prob: float,
    config: ValueConfig,
) -> tuple[jax.Array, jax.Array]:
    """std of upstream's one-switch advantage and of the switch advantage, on one batch each."""
    upstream_spec = HighBatchSpec.from_agent_config(agent.config)
    k_up, k_b = jax.random.split(key)
    _, info = agent.total_loss(sample_high_batch(data, k_up, upstream_spec), grad_params=None)
    adv, valid = switch_advantage(
        agent, value, sample_switch_batch(data, latents, k_b, spec, gaussian_prob), config
    )
    mean = (adv * valid).sum() / valid.sum()
    std = jnp.sqrt((((adv - mean) * valid) ** 2).sum() / valid.sum())
    return info["high_actor/adv_std"], std


def calibrate_alpha(
    agent: Agent,
    value: ValueState,
    split: Split,
    key: jax.Array,
    spec: HighBatchSpec,
    gaussian_prob: float,
    config: ValueConfig,
) -> tuple[float, dict[str, float]]:
    """AWR temperature with the spread of upstream's alpha * A (see matched_alpha)."""
    stds = [
        advantage_stds(agent, value, split.data, split.latents, k, spec, gaussian_prob, config)
        for k in jax.random.split(key, CALIBRATION_BATCHES)
    ]
    upstream_std = float(np.mean([s[0] for s in stds]))
    switch_std = float(np.mean([s[1] for s in stds]))
    alpha = matched_alpha(upstream_std, switch_std, float(agent.config["high_alpha"]))
    return alpha, {"upstream_adv_std": upstream_std, "switch_adv_std": switch_std, "alpha": alpha}


def example_batch(agent: Agent, obs_dim: int) -> dict[str, np.ndarray]:
    """Zero batch with the sizes that upstream `create` reads."""
    action_dim = agent.network.params["modules_actor"]["mean_net"]["bias"].shape[-1]
    return {
        "observations": np.zeros((1, obs_dim), np.float32),
        "actions": np.zeros((1, action_dim), np.float32),
    }


def load_b_model(env_name: str, variant: str, seed: int) -> frozen.FrozenFBModel:
    """The frozen model with the pi_h of an approach B variant, for d-eval."""
    return load_retrained_model(env_name, approach_b_dir(env_name, variant, seed))


# ----- CLI -----


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("value", "policy"):
        p = sub.add_parser(name)
        p.add_argument("--env", required=True, choices=frozen.ENV_NAMES)
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--steps", type=int, default=500_000)
        p.add_argument("--log-every", type=int, default=5_000)
        p.add_argument("--run-dir", type=Path, help="default: runs/train_b/<maze>/<name>/seed<k>")
    value = sub.choices["value"]
    value.add_argument("--latents", choices=("single", "region"), default="single")
    value.add_argument("--expectile", type=float, default=ValueConfig.expectile)
    value.add_argument("--rho-max", type=float, default=OptionConfig.rho_max)
    value.add_argument(
        "--edge-reward", choices=("goal_zero", "exact", "paper"), default="goal_zero"
    )
    value.add_argument("--hop-cost", type=float, default=ValueConfig.hop_cost)
    policy = sub.choices["policy"]
    policy.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    policy.add_argument("--alpha", type=float, help="default: matched to upstream's spread")
    return parser.parse_args(argv)


def base_flags(env_name: str, seed: int, steps: int) -> dict[str, Any]:
    flags = json.loads((frozen.checkpoint_dir(env_name) / "flags.json").read_text())
    return flags | {"seed": seed, "train_steps": steps, "commit": git_commit()}


def train_value(args: argparse.Namespace, model: frozen.FrozenFBModel, run_dir: Path) -> None:
    agent = model.agent
    train_split, val_split = (Split.load(model, s, args.latents) for s in ("train", "val"))
    config = ValueConfig(
        option=OptionConfig(edge_reward=args.edge_reward, rho_max=args.rho_max),
        expectile=args.expectile,
        hop_cost=args.hop_cost,
    )
    spec = dataclasses.replace(
        HighBatchSpec.from_agent_config(agent.config), actor_p_randomgoal=VALUE_RANDOM_GOALS
    )
    gaussian = float(agent.config["critic_latent_mix_prob"])
    init_key = jax.random.PRNGKey(args.seed)
    obs_dim = train_split.data.observations.shape[-1]
    value = ValueState.create(init_key, obs_dim, model.latent_dim, config)
    value = run_logged(
        value,
        chunk=lambda v, k, n: value_chunk(
            v, agent, train_split.data, train_split.latents, k, n, spec, gaussian, config
        ),
        validate=lambda v, k: value_validation(
            v, agent, val_split.data, val_split.latents, k, spec, gaussian, config
        ),
        steps=args.steps,
        log_every=args.log_every,
        key=jax.random.fold_in(init_key, SAMPLER_STREAM),
        log_path=run_dir / "train.csv",
        progress_key="loss",
    )
    flags = base_flags(args.env, args.seed, args.steps) | {
        "latents": args.latents,
        "value_config": dataclasses.asdict(config),
        "value_spec": dataclasses.asdict(spec),
    }
    out_dir = approach_b_dir(args.env, f"value-{args.latents}", args.seed)
    save_value(value, flags, out_dir)
    print(f"saved {out_dir}")


def train_policy(args: argparse.Namespace, model: frozen.FrozenFBModel, run_dir: Path) -> None:
    latents, random_goals = VARIANTS[args.variant]
    train_split, val_split = (Split.load(model, s, latents) for s in ("train", "val"))
    obs_dim = train_split.data.observations.shape[-1]
    value_dir = approach_b_dir(args.env, f"value-{latents}", args.seed)
    value, config = load_value(value_dir, obs_dim, model.latent_dim)
    agent = with_fresh_high_actor(model.agent, args.seed, example_batch(model.agent, obs_dim))
    spec = dataclasses.replace(
        HighBatchSpec.from_agent_config(agent.config), actor_p_randomgoal=random_goals
    )
    gaussian = float(agent.config["critic_latent_mix_prob"])
    key = jax.random.fold_in(jax.random.PRNGKey(args.seed), SAMPLER_STREAM)
    calibration_key, key = jax.random.split(key)
    alpha, calibration = calibrate_alpha(
        agent, value, train_split, calibration_key, spec, gaussian, config
    )
    if args.alpha is not None:
        alpha = args.alpha
    print(json.dumps(calibration | {"alpha_used": alpha}), flush=True)
    alpha_array = jnp.asarray(alpha, dtype=jnp.float32)
    agent = run_logged(
        agent,
        chunk=lambda a, k, n: policy_chunk(
            a,
            value,
            train_split.data,
            train_split.latents,
            k,
            alpha_array,
            n,
            spec,
            gaussian,
            config,
        ),
        validate=lambda a, k: policy_validation(
            a, value, val_split.data, val_split.latents, k, alpha_array, spec, gaussian, config
        ),
        steps=args.steps,
        log_every=args.log_every,
        key=key,
        log_path=run_dir / "train.csv",
        progress_key="actor_loss",
    )
    flags = base_flags(args.env, args.seed, args.steps) | {
        "agent": dict(agent.config),
        "approach_b": {
            "variant": args.variant,
            "value_dir": str(value_dir),
            "value_config": dataclasses.asdict(config),
            "policy_spec": dataclasses.asdict(spec),
            "calibration": calibration,
            "alpha_used": alpha,
        },
    }
    out_dir = approach_b_dir(args.env, args.variant, args.seed)
    save_checkpoint(agent, flags, out_dir)
    print(f"saved {out_dir}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    name = f"value-{args.latents}" if args.stage == "value" else args.variant
    maze = args.env.split("-")[1]
    run_dir = args.run_dir or Path("runs/train_b") / maze / name / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model = frozen.load_model(args.env)
    (train_value if args.stage == "value" else train_policy)(args, model, run_dir)


if __name__ == "__main__":
    main()
