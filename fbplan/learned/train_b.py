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
    run_flags,
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
        """The `split` ("train" or "val") of the model's env with latents of the given kind."""
        raw = frozen.load_dataset(model.env_name, split)
        data = make_trajectory_data(raw["observations"], raw["terminals"])
        return cls(data, latent_table(model, raw["observations"], raw["terminals"], kind))

    @property
    def obs_dim(self) -> int:
        """Size of one observation."""
        return self.data.observations.shape[-1]

    def batch(self, key: jax.Array, stage: Stage) -> SwitchBatch:
        """One switch batch with the stage's goal distribution and share of Gaussian latents."""
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
    flags = run_flags(args.env, args.seed, args.steps) | {
        "latents": args.latents,
        "value_spec": dataclasses.asdict(stage.spec),
    }
    out_dir = args.out_dir or frozen.trained_dir(
        trained_key("approach_b", args.env, args.seed, args.name)
    )
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


@partial(jax.jit, static_argnames=("spec",))
def upstream_adv_std(
    agent: Agent, data: TrajectoryData, key: jax.Array, spec: HighBatchSpec
) -> jax.Array:
    """std of upstream's one-switch advantage on one batch, with its own draw of task latents.

    Upstream draws the latents from the agent's rng, so the rng is replaced per batch.
    """
    k_rng, k_batch = jax.random.split(key)
    batch = sample_high_batch(data, k_batch, spec)
    _, info = agent.replace(rng=k_rng).total_loss(batch, grad_params=None)
    return info["high_actor/adv_std"]


@partial(jax.jit, static_argnames=("stage",))
def switch_adv_std(
    agent: Agent, value: ValueState, split: Split, key: jax.Array, stage: Stage
) -> jax.Array:
    """std of the switch advantage over the valid edges of one batch."""
    adv, valid = switch_advantage(agent, value, split.batch(key, stage), stage.value)
    return masked_mean_std(adv, valid)[1]


def shared_alpha(
    agent: Agent,
    values: Sequence[ValueState],
    split: Split,
    key: jax.Array,
    stage: Stage,
    upstream_alpha: float,
    batches: int,
) -> dict[str, Any]:
    """One AWR temperature for all seeds of a policy family (design.md §8, AWR sharpness at 2x).

    alpha * spread(A) = upstream_alpha * spread(upstream advantage), where the spread is the
    median over `batches` batches of the per-batch std. Medians, because the per-batch std is
    heavy-tailed: the upstream advantage divides by F(w, z_w)^T B(w), which can be near 0. Each
    upstream batch has its own latent draw; the switch batches cycle over the seeds' critics.
    """
    k_up, k_sw = jax.random.split(key)
    upstream_spec = HighBatchSpec.from_agent_config(agent.config)
    up = [
        float(upstream_adv_std(agent, split.data, k, upstream_spec))
        for k in jax.random.split(k_up, batches)
    ]
    sw = [
        float(switch_adv_std(agent, values[i % len(values)], split, k, stage))
        for i, k in enumerate(jax.random.split(k_sw, batches))
    ]
    upstream_std, switch_std = float(np.median(up)), float(np.median(sw))
    return {
        "alpha": matched_alpha(upstream_std, switch_std, upstream_alpha),
        "upstream_alpha": upstream_alpha,
        "upstream_adv_std_median": upstream_std,
        "switch_adv_std_median": switch_std,
        "upstream_adv_std_q10_q90": [float(q) for q in np.percentile(up, [10, 90])],
        "switch_adv_std_q10_q90": [float(q) for q in np.percentile(sw, [10, 90])],
        "batches": batches,
    }


def calibrate(args: argparse.Namespace, model: frozen.FrozenFBModel) -> None:
    """Write one shared AWR temperature for a policy family to args.out (JSON)."""
    split = Split.load(model, "train", VARIANTS[args.variant][0])
    loaded = [
        load_value(args.env, args.value_name, seed, split.obs_dim, model.latent_dim)
        for seed in args.seeds
    ]
    stage = policy_stage(model.agent, args.variant, loaded[0][1])
    upstream_alpha = args.upstream_alpha or float(model.agent.config["high_alpha"])
    result = shared_alpha(
        model.agent,
        [v for v, _ in loaded],
        split,
        jax.random.PRNGKey(0),
        stage,
        upstream_alpha,
        args.batches,
    )
    result |= {
        "env": args.env,
        "variant": args.variant,
        "value_name": args.value_name,
        "value_seeds": args.seeds,
        "commit": git_commit(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


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
    config = dataclasses.replace(config, critic=args.critic)
    agent = with_fresh_high_actor(model.agent, args.seed)
    stage = policy_stage(agent, args.variant, config)

    key = jax.random.fold_in(jax.random.PRNGKey(args.seed), SAMPLER_STREAM)
    calibration_key, key = jax.random.split(key)  # also with --alpha-from: same batches
    if args.alpha_from is not None:
        calibration = json.loads(args.alpha_from.read_text())
        alpha = calibration["alpha"]
    else:
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
    flags = run_flags(args.env, args.seed, args.steps) | {
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
    out_dir = args.out_dir or frozen.trained_dir(
        trained_key("approach_b", args.env, args.seed, args.name)
    )
    frozen.save_trained_checkpoint(agent, flags, out_dir)
    print(f"saved {out_dir}")


# ----- CLI -----


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Command line of train_b; resolves the default folder names (see the comment below)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    stages = {
        "value": sub.add_parser("value", help="stage 1: fit V_H"),
        "policy": sub.add_parser("policy", help="stage 2: train pi_h with the switch advantage"),
    }
    for p in stages.values():
        p.add_argument("--env", required=True, choices=frozen.ENV_NAMES, help="OGBench env")
        p.add_argument("--seed", type=int, required=True, help="network init and batch sampling")
        p.add_argument("--steps", type=int, default=500_000, help="number of updates")
        p.add_argument("--log-every", type=int, default=5_000, help="updates per CSV row")
        p.add_argument("--run-dir", type=Path, help="default: runs/train_b/<maze>/<name>/seed<k>")
        p.add_argument("--name", help="output folder below approach_b/<maze>/; default below")
        p.add_argument("--out-dir", type=Path, help="default: the folder of --name in the cache")
    value = stages["value"]
    value.add_argument(
        "--latents", choices=get_args(LatentKind), default="single", help="B(w) or window mean"
    )
    value.add_argument("--expectile", type=float, default=ValueConfig.expectile, help="tau")
    value.add_argument("--rho-cap", type=float, default=OptionConfig.rho_cap, help="cap on rho")
    value.add_argument(
        "--edge-reward",
        choices=get_args(EdgeReward),
        default=OptionConfig.edge_reward,
        help="R(s, w): zero for goal latents, exact, or the paper's proxy",
    )
    value.add_argument("--hop-cost", type=float, default=ValueConfig.hop_cost, help="per switch")
    policy = stages["policy"]
    policy.add_argument(
        "--variant", choices=tuple(VARIANTS), required=True, help="see the module docstring"
    )
    policy.add_argument("--value-name", help="V_H folder to load; default value-<latents>")
    policy.add_argument(
        "--critic",
        choices=("v_h", "v_1"),
        default="v_h",
        help="v_1: use V_1 where the targets use V_H (control without a learned critic)",
    )
    temperature = policy.add_mutually_exclusive_group()
    temperature.add_argument("--alpha", type=float, help="AWR temperature; default: per seed")
    temperature.add_argument(
        "--alpha-from", type=Path, help="JSON of `calibrate`: one temperature for all seeds"
    )
    calib = sub.add_parser("calibrate", help="one AWR temperature shared by a policy family")
    calib.add_argument("--env", required=True, choices=frozen.ENV_NAMES, help="OGBench env")
    calib.add_argument("--variant", choices=tuple(VARIANTS), default="clean", help="batches")
    calib.add_argument("--value-name", default="value-single", help="V_H folder of the critics")
    calib.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="critics")
    calib.add_argument("--upstream-alpha", type=float, help="baseline temperature to match")
    calib.add_argument("--batches", type=int, default=200, help="batches per median")
    calib.add_argument("--out", type=Path, required=True, help="JSON file to write")
    args = parser.parse_args(argv)
    if args.stage == "calibrate":
        return args
    # Default folders: value-<latents> for V_H, the variant for pi_h. Give ablations their own
    # --name, so they do not replace the V_H that the policy stage loads.
    if args.name is None:
        args.name = f"value-{args.latents}" if args.stage == "value" else args.variant
    if args.stage == "policy" and args.value_name is None:
        args.value_name = f"value-{VARIANTS[args.variant][0]}"
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.stage == "calibrate":
        calibrate(args, frozen.load_model(args.env))
        return
    run_dir = (
        args.run_dir
        or Path("runs/train_b") / frozen.maze_name(args.env) / args.name / f"seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    model = frozen.load_model(args.env)
    (train_value if args.stage == "value" else train_policy)(args, model, run_dir)


if __name__ == "__main__":
    main()
