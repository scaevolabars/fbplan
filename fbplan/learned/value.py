"""Multi-switch value V_H, stage 1 of approach B (docs/plans/approach-b.md §2, §5).

    V_H(s, z) = V_1(s, z) + min_j gain_j(s, z),    gain_j = softplus(delta_j)

V_1 is the frozen single-intention value and delta_j (j = 1, 2) are the raw outputs of the
network, so V_H >= V_1: the option "never switch again" always exists, and the network learns
only the gain from switching. The gain is fit by expectile regression on the switch target over
subgoals w on the trajectory of s:

    y(s, w, z) = R(s, w; z) + rho(s, w) V_H(w, z) - hop_cost
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from fbplan.frozen import forward, save_numpy_pickle, upstream_module
from fbplan.learned.checkpoint import (
    Agent,
    load_trained_file,
    save_flags,
    trained_dir,
    trained_key,
)
from fbplan.learned.option_targets import OptionConfig, OptionTerms, direct_value, option_terms
from fbplan.learned.sampling import SwitchBatch


@dataclass(frozen=True)
class ValueConfig:
    """Settings of stage 1.

    expectile: tau of the in-sample soft max over w (1 would be a hard max).
    ema: rate of the target network, as upstream's tau for F and B.
    hop_cost: subtracted once per switch.
    """

    option: OptionConfig = field(default_factory=OptionConfig)
    hidden_dims: tuple[int, ...] = (512, 512, 512)
    lr: float = 3e-4
    expectile: float = 0.9
    ema: float = 0.005
    hop_cost: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ValueConfig:
        """Inverse of dataclasses.asdict, for the config stored in flags.json."""
        nested = {"option": OptionConfig(**d["option"]), "hidden_dims": tuple(d["hidden_dims"])}
        return cls(**(d | nested))


def delta_network(hidden_dims: tuple[int, ...]) -> nn.Module:
    """Upstream GCValue on (s, z) with 2 heads, the architecture of upstream value networks."""
    networks = upstream_module("utils.networks")
    return networks.GCValue(hidden_dims=hidden_dims, layer_norm=True, num_ensembles=2)


class ValueState(flax.struct.PyTreeNode):
    """The delta network, its EMA target copy and its optimizer state."""

    params: Any
    target_params: Any
    opt_state: Any
    step: jax.Array
    net: nn.Module = flax.struct.field(pytree_node=False)
    tx: optax.GradientTransformation = flax.struct.field(pytree_node=False)

    @classmethod
    def create(
        cls, key: jax.Array, obs_dim: int, latent_dim: int, config: ValueConfig
    ) -> ValueState:
        net = delta_network(config.hidden_dims)
        params = net.init(key, jnp.zeros((1, obs_dim)), jnp.zeros((1, latent_dim)))["params"]
        tx = optax.adam(config.lr)
        return cls(params, params, tx.init(params), jnp.zeros((), jnp.int32), net, tx)

    def gain(self, obs: jax.Array, z: jax.Array, params: Any = None) -> jax.Array:
        """gain_j(s, z) = softplus(delta_j(s, z)) per head: [2, N], >= 0."""
        params = self.params if params is None else params
        return jax.nn.softplus(self.net.apply({"params": params}, obs, z))

    def value(self, v1: jax.Array, obs: jax.Array, z: jax.Array, params: Any = None) -> jax.Array:
        """V_H(s, z) = V_1(s, z) + min_j gain_j(s, z), given V_1(s, z). [N]."""
        return v1 + self.gain(obs, z, params).min(axis=0)


def multi_switch_value(
    agent: Agent, value: ValueState, obs: jax.Array, z: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """(V_1(s, z), V_H(s, z)) for each sample. [N] each."""
    v1 = direct_value(forward(agent, obs, z), z)
    return v1, value.value(v1, obs, z)


def switch_target(
    reward: jax.Array, rho: jax.Array, value_at_w: jax.Array, hop_cost: float
) -> jax.Array:
    """y = R(s, w) + rho(s, w) V_H(w) - hop_cost: the value of switching to w now."""
    return reward + rho * value_at_w - hop_cost


def switch_targets(
    agent: Agent, value: ValueState, params: Any, batch: SwitchBatch, config: ValueConfig
) -> tuple[jax.Array, jax.Array, OptionTerms]:
    """(y, V_1(s, z), option terms) for each sample. `params` give the gain at w."""
    s, w, z_w, z = batch.observations, batch.subgoals, batch.subgoal_latents, batch.task_latents
    f_s_zw, f_w_zw = forward(agent, s, z_w), forward(agent, w, z_w)
    distinct = jnp.any(s != w, axis=-1)
    terms = option_terms(f_s_zw, f_w_zw, z_w, z, batch.is_goal, distinct, config.option)
    v_w = value.value(direct_value(forward(agent, w, z), z), w, z, params)
    y = switch_target(terms.reward, terms.rho, v_w, config.hop_cost)
    return y, direct_value(forward(agent, s, z), z), terms


def expectile_loss(diff: jax.Array, expectile: float) -> jax.Array:
    """Asymmetric squared error: weight `expectile` above the prediction, 1 - expectile below."""
    return jnp.where(diff >= 0, expectile, 1.0 - expectile) * diff**2


def value_loss(
    value: ValueState, params: Any, agent: Agent, batch: SwitchBatch, config: ValueConfig
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Expectile loss of both gain heads on the target-network switch targets, valid edges only."""
    y, v1_s, terms = switch_targets(agent, value, value.target_params, batch, config)
    y = jax.lax.stop_gradient(y)
    gain_s = value.gain(batch.observations, batch.task_latents, params)  # [2, N]
    mask = terms.valid.astype(jnp.float32)
    n_valid = jnp.maximum(mask.sum(), 1.0)
    loss = (expectile_loss(y - v1_s - gain_s, config.expectile) * mask).sum() / n_valid
    capped = terms.valid & (terms.rho_raw > config.option.rho_cap)
    return loss, {
        "loss": loss,
        "valid": mask.mean(),
        "rho_capped": capped.sum() / n_valid,
        "gain_s": (gain_s.min(axis=0) * mask).sum() / n_valid,
        "switch_better": ((y > v1_s) * mask).sum() / n_valid,
        "y": (y * mask).sum() / n_valid,
        "v1_s": v1_s.mean(),
    }


@partial(jax.jit, static_argnames=("config",))
def value_update(
    value: ValueState, agent: Agent, batch: SwitchBatch, config: ValueConfig
) -> tuple[ValueState, dict[str, jax.Array]]:
    """One Adam step on the delta network, then the EMA step of the target network."""
    grads, info = jax.grad(value_loss, argnums=1, has_aux=True)(
        value, value.params, agent, batch, config
    )
    updates, opt_state = value.tx.update(grads, value.opt_state, value.params)
    params = optax.apply_updates(value.params, updates)
    target = optax.incremental_update(params, value.target_params, config.ema)
    step = value.step + 1
    return value.replace(params=params, target_params=target, opt_state=opt_state, step=step), info


def save_value(
    value: ValueState, config: ValueConfig, flags: dict[str, Any], out_dir: Path
) -> None:
    """value.pkl (params and target params) and flags.json with the config under value_config."""
    state = {"params": value.params, "target_params": value.target_params, "step": value.step}
    save_numpy_pickle(state, out_dir / "value.pkl")
    save_flags(flags | {"value_config": dataclasses.asdict(config)}, out_dir)


def load_value(
    env_name: str, name: str, seed: int, obs_dim: int, latent_dim: int
) -> tuple[ValueState, ValueConfig]:
    """The V_H saved by save_value in approach_b/<maze>/<name>/seed<k>, and its config."""
    key = trained_key("approach_b", env_name, seed, name)
    state = load_trained_file(key, "value.pkl")
    folder = trained_dir("approach_b", env_name, seed, name)
    config = ValueConfig.from_dict(json.loads((folder / "flags.json").read_text())["value_config"])
    template = ValueState.create(jax.random.PRNGKey(0), obs_dim, latent_dim, config)
    value = template.replace(
        params=flax.serialization.from_state_dict(template.params, state["params"]),
        target_params=flax.serialization.from_state_dict(template.params, state["target_params"]),
        step=jnp.asarray(state["step"]),
    )
    return value, config
