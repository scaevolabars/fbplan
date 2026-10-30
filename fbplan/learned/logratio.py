"""Log-ratio proxy advantage for the baseline pi_h (docs/design.md §8, wave 1, T2).

For goal latents (z = B(g), normalized to norm sqrt(d) in the dot products) the AWR advantage is

    A = log max(mean_h [F_h(s,z_w)^T z + rho~_h F_h(w,z)^T z], 1) - log max(mean_h F_h(s,z)^T z, 1)
    rho~_h = clip(F_h(s,z_w)^T B(w) / F_h(w,z_w)^T B(w), 0, 0.99)

with temperature goal_alpha (2.45: upstream's effective sample size at 0.1, so that only the shape
of the advantage changes). Gaussian latents keep upstream's proxy advantage and temperature
(high_alpha, 0.1). Everything else is upstream `high_actor_loss`: the same latents from the same
rng, the clip of alpha * A at 5, the mean over heads and samples, and upstream's optimizer step.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from fbplan.frozen import forward
from fbplan.learned.checkpoint import Agent

FLOOR = 1.0  # values below 1 count as 1, so the log stays finite (docs/hypotheses/h-learn.md H2)
RHO_CLIP = 0.99
MAX_LOG_WEIGHT = 5.0  # upstream clips alpha * A at 5 before exp
GOAL_ALPHA = 2.45


def log_ratio(
    f_s_zw_z: jax.Array, f_w_z_z: jax.Array, f_s_z_z: jax.Array, rho_h: jax.Array
) -> jax.Array:
    """A for goal latents from per-head terms [H, N]: the head mean is taken before the log. [N]."""
    rho = jnp.clip(rho_h, 0.0, RHO_CLIP)
    switch = (f_s_zw_z + rho * f_w_z_z).mean(axis=0)
    direct = f_s_z_z.mean(axis=0)
    return jnp.log(jnp.maximum(switch, FLOOR)) - jnp.log(jnp.maximum(direct, FLOOR))


def measure(agent: Agent, obs: jax.Array, z_goal: jax.Array, z_intent: jax.Array) -> jax.Array:
    """F_h(obs, z_intent)^T z_goal per head [H, N], with the ops of successor_measure_extract."""
    return jnp.sum(forward(agent, obs, z_intent) * z_goal[None], axis=-1)


def goal_latent_mask(rng: jax.Array, n: int, mix_prob: float) -> jax.Array:
    """True where upstream sample_latents(targets, rng, mix_prob) keeps B(g), not a Gaussian. [N]"""
    _, _, mix_rng = jax.random.split(rng, 3)  # the split of sample_latents
    return ~(jax.random.uniform(mix_rng, (n, 1)) < mix_prob)[:, 0]


def advantages(
    agent: Agent,
    obs: jax.Array,
    subgoals: jax.Array,
    z_norm: jax.Array,
    z_raw: jax.Array,
    latents: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Upstream's proxy advantage [H, N] (raw latents) and the log-ratio advantage [N] (normalized
    z). latents: the raw intentions B(w) of the subgoals."""
    rho_h = measure(agent, obs, latents, latents) / measure(agent, subgoals, latents, latents)
    proxy = (
        measure(agent, obs, z_raw, latents)
        + rho_h * measure(agent, subgoals, z_raw, z_raw)
        - measure(agent, obs, z_raw, z_raw)
    )
    adv_log = log_ratio(
        measure(agent, obs, z_norm, latents),
        measure(agent, subgoals, z_norm, z_raw),
        measure(agent, obs, z_norm, z_raw),
        rho_h,
    )
    return proxy, adv_log


def logratio_loss(
    agent: Agent, batch: dict[str, jax.Array], grad_params: Any, goal_alpha: float
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Upstream's high_actor_loss with the log-ratio advantage for goal latents."""
    obs, subgoals = batch["observations"], batch["high_actor_targets"]
    goals = batch["high_actor_goals"]
    new_rng, rng = jax.random.split(agent.rng)  # the latent draws of upstream high_actor_loss
    latents_norm, latents = jax.lax.stop_gradient(agent.sample_latents(subgoals, rng, 0.0))
    mix = agent.config["critic_latent_mix_prob"]
    z_norm, z_raw = jax.lax.stop_gradient(agent.sample_latents(goals, new_rng, mix))
    is_goal = goal_latent_mask(new_rng, goals.shape[0], mix)

    proxy, adv_log = advantages(agent, obs, subgoals, z_norm, z_raw, latents)
    log_weight = jnp.where(
        is_goal[None, :], goal_alpha * adv_log[None, :], proxy * agent.config["high_alpha"]
    )
    exp_a = jnp.exp(jnp.clip(log_weight, max=MAX_LOG_WEIGHT))  # [H, N]
    dist = agent.network.select("high_actor")(obs, z_norm, params=grad_params)
    log_prob = dist.log_prob(latents_norm)
    loss = -(exp_a * log_prob).mean()

    n_goal = jnp.maximum(is_goal.sum(), 1)
    w_goal = exp_a[0] * is_goal
    return loss, {
        "actor_loss": loss,
        "adv": proxy.mean(),
        "adv_log": (adv_log * is_goal).sum() / n_goal,
        "goal_share": is_goal.mean(),
        "saturated": (log_weight >= MAX_LOG_WEIGHT).mean(),
        "ess_goal": w_goal.sum() ** 2 / (n_goal * jnp.maximum((w_goal**2).sum(), 1e-12)),
        "bc_log_prob": log_prob.mean(),
        "mse": jnp.mean((dist.mode() - latents_norm) ** 2),
    }


@partial(jax.jit, static_argnames=("goal_alpha",))
def logratio_update(
    agent: Agent, batch: dict[str, jax.Array], goal_alpha: float
) -> tuple[Agent, dict[str, jax.Array]]:
    """One upstream optimizer step on the log-ratio loss; the rng advances as in upstream update."""
    new_rng, _ = jax.random.split(agent.rng)

    def loss_fn(grad_params: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
        loss, info = logratio_loss(agent, batch, grad_params, goal_alpha)
        return loss, {f"high_actor/{k}": v for k, v in info.items()}

    network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
    return agent.replace(network=network, rng=new_rng), info
