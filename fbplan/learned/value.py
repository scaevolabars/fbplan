"""Multi-switch value V_H(s, z) = V_1(s, z) + softplus(gain_theta(s, z)), stage 1 of approach B.

V_1 is the frozen single-intention value, so V_H >= V_1: the option "never switch again" always
exists, and the network learns only the gain from switching. The gain is fit by expectile
regression on the switch target over subgoals w on the trajectory of s
(docs/plans/approach-b.md §2, §5):

    y(s, w, z) = R(s, w; z) + rho(s, w) [V_1(w, z) + min_j gain_j(w, z)] - hop_cost
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from fbplan.frozen import upstream_module
from fbplan.learned.option_targets import OptionConfig, OptionTerms, direct_value, option_terms
from fbplan.learned.sampling import SwitchBatch

Agent = Any  # upstream agents.fbpiswitch.FBpiSwitchAgent with the frozen F, B and pi_l


@dataclass(frozen=True)
class ValueConfig:
    """expectile: tau of the in-sample soft max over w (1 would be a hard max).
    ema: rate of the target network, as upstream's tau for F and B.
    """

    option: OptionConfig = field(default_factory=OptionConfig)
    hidden_dims: tuple[int, ...] = (512, 512, 512)
    lr: float = 3e-4
    expectile: float = 0.9
    ema: float = 0.005
    hop_cost: float = 0.0


def gain_network(hidden_dims: tuple[int, ...]) -> nn.Module:
    """Upstream GCValue on (s, z) with 2 heads, the architecture of upstream value networks."""
    networks = upstream_module("utils.networks")
    return networks.GCValue(hidden_dims=hidden_dims, layer_norm=True, num_ensembles=2)


class ValueState(flax.struct.PyTreeNode):
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
        net = gain_network(config.hidden_dims)
        params = net.init(key, jnp.zeros((1, obs_dim)), jnp.zeros((1, latent_dim)))["params"]
        tx = optax.adam(config.lr)
        return cls(params, params, tx.init(params), jnp.zeros((), jnp.int32), net, tx)

    def gain(self, obs: jax.Array, z: jax.Array, params: Any = None) -> jax.Array:
        """softplus(gain_j(s, z)) per head: [2, N], >= 0."""
        params = self.params if params is None else params
        return jax.nn.softplus(self.net.apply({"params": params}, obs, z))

    def value(self, agent: Agent, obs: jax.Array, z: jax.Array) -> jax.Array:
        """V_H(s, z) with the pessimistic (min) head of the gain. [N]."""
        return direct_value(forward(agent, obs, z), z) + self.gain(obs, z).min(axis=0)


def forward(agent: Agent, obs: jax.Array, z: jax.Array) -> jax.Array:
    """F(s, z) of both heads [2, N, d], as upstream successor_measure_extract calls it."""
    return agent.network.select("forward_repr")(obs, agent.normalize_z(z), goal_encoded=True)


def switch_targets(
    agent: Agent, value: ValueState, params: Any, batch: SwitchBatch, config: ValueConfig
) -> tuple[jax.Array, jax.Array, OptionTerms]:
    """(y, V_1(s, z), option terms) for each sample. `params` give the gain at w."""
    s, w, z_w, z = batch.observations, batch.subgoals, batch.subgoal_latents, batch.task_latents
    terms = option_terms(
        forward(agent, s, z_w), forward(agent, w, z_w), z_w, z, batch.is_goal, config.option
    )
    v_w = direct_value(forward(agent, w, z), z) + value.gain(w, z, params).min(axis=0)
    y = terms.reward + terms.rho * v_w - config.hop_cost
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
    capped = terms.valid & (terms.rho_raw > config.option.rho_max)
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
    """One Adam step on the gain heads, then the EMA step of the target network."""
    grads, info = jax.grad(value_loss, argnums=1, has_aux=True)(
        value, value.params, agent, batch, config
    )
    updates, opt_state = value.tx.update(grads, value.opt_state, value.params)
    params = optax.apply_updates(value.params, updates)
    target = optax.incremental_update(params, value.target_params, config.ema)
    step = value.step + 1
    return value.replace(params=params, target_params=target, opt_state=opt_state, step=step), info
