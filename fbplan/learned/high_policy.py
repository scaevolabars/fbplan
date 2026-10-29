"""Stage 2 of approach B: a new pi_h by AWR with the multi-switch advantage.

The network, its optimizer and the regression target are those of upstream `high_actor_loss`
(normalized subgoal latent, Gaussian policy with constant std, Adam on the high actor only).
Only the advantage differs (docs/plans/approach-b.md §4):

    A(s, w, z) = y(s, w, z) - V_H(s, z)        instead of upstream's one-switch proxy
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from fbplan.contracts import normalize_z
from fbplan.learned.checkpoint import Agent
from fbplan.learned.sampling import SwitchBatch
from fbplan.learned.value import ValueConfig, ValueState, switch_targets

MAX_LOG_WEIGHT = 5.0  # upstream clips alpha * A at 5 before exp


def switch_advantage(
    agent: Agent, value: ValueState, batch: SwitchBatch, config: ValueConfig
) -> tuple[jax.Array, jax.Array]:
    """(A, valid) per sample, with the trained V_H at s and at w."""
    y, v1_s, terms = switch_targets(agent, value, value.params, batch, config)
    return y - value.value(v1_s, batch.observations, batch.task_latents), terms.valid


def masked_mean_std(x: jax.Array, mask: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Mean and std of x over the samples where mask is True."""
    n = jnp.maximum(mask.sum(), 1)
    mean = (x * mask).sum() / n
    return mean, jnp.sqrt((((x - mean) * mask) ** 2).sum() / n)


def policy_loss(
    grad_params: Any,
    agent: Agent,
    value: ValueState,
    batch: SwitchBatch,
    alpha: jax.Array,
    config: ValueConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Upstream's AWR loss with the switch advantage; invalid edges get weight 0."""
    adv, valid = switch_advantage(agent, value, batch, config)
    weight = jnp.exp(jnp.minimum(alpha * adv, MAX_LOG_WEIGHT)) * valid
    target = normalize_z(batch.subgoal_latents, batch.subgoal_latents.shape[-1])
    dist = agent.network.select("high_actor")(
        batch.observations, batch.task_latents, params=grad_params
    )
    log_prob = dist.log_prob(target)
    loss = -(weight * log_prob).mean()
    adv_mean, adv_std = masked_mean_std(adv, valid)
    return loss, {
        "actor_loss": loss,
        "adv": adv_mean,
        "adv_std": adv_std,
        "valid": valid.mean(),
        "saturated": (valid & (alpha * adv >= MAX_LOG_WEIGHT)).sum() / jnp.maximum(valid.sum(), 1),
        "bc_log_prob": log_prob.mean(),
        "mse": jnp.mean((dist.mode() - target) ** 2),
    }


@partial(jax.jit, static_argnames=("config",))
def policy_update(
    agent: Agent, value: ValueState, batch: SwitchBatch, alpha: jax.Array, config: ValueConfig
) -> tuple[Agent, dict[str, jax.Array]]:
    """One upstream optimizer step; its mask keeps F, B and pi_l frozen."""

    def loss_fn(grad_params: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
        return policy_loss(grad_params, agent, value, batch, alpha, config)

    network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
    return agent.replace(network=network), info


def matched_alpha(upstream_adv_std: float, switch_adv_std: float, upstream_alpha: float) -> float:
    """AWR temperature that gives alpha * A the spread of the upstream run.

    Our advantage uses normalized task latents and V_H, so its scale differs from upstream's.
    Matching the spread keeps the weighting as sharp as in the baseline, so that only the critic
    changes.
    """
    return upstream_alpha * upstream_adv_std / switch_adv_std
