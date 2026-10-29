"""Option model of "follow pi_{z_w} until w is hit", one (s, w) pair per sample, in JAX.

Formulas and head convention are those of fbplan/option_model.py (approach A), which builds
(source, target) tables in numpy; a test checks that both agree. Here each sample carries one pair,
so the terms are computed inside a jitted training step (docs/plans/approach-b.md §2):

    rho_h(s, w) = F_h(s, z_w)^T z_w / F_h(w, z_w)^T z_w     min over heads, capped at rho_cap
    R_h(s, w)   = F_h(s, z_w)^T z - rho_h(s, w) F_h(w, z_w)^T z
    V_1(s, z)   = min_h F_h(s, z)^T z

z_w is the subgoal latent: B(w), or its trajectory-window mean in the B-region variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import flax
import jax
import jax.numpy as jnp

EdgeReward = Literal["goal_zero", "exact", "paper"]


@dataclass(frozen=True)
class OptionConfig:
    """edge_reward: "goal_zero" sets R = 0 for goal latents, as approach A does, because FB's exact
    R is a difference of two noisy values; Gaussian latents keep the exact R. "exact" and "paper"
    (drops -rho F(w, z_w)^T z, as the paper's pi_h advantage) are ablations.
    rho_cap: upper bound on rho in the targets. Below 1 it makes the hop-level Bellman operator a
    contraction with that factor. (The planner's rho_max is a different thing: a trust bound.)
    """

    edge_reward: EdgeReward = "goal_zero"
    rho_cap: float = 0.9

    def __post_init__(self) -> None:
        if self.edge_reward not in ("goal_zero", "exact", "paper"):
            raise ValueError(f"unknown edge_reward {self.edge_reward!r}")
        if not 0.0 < self.rho_cap <= 1.0:
            raise ValueError(f"rho_cap must be in (0, 1], got {self.rho_cap}")


@flax.struct.dataclass
class OptionTerms:
    rho: jax.Array  # [N] in [0, rho_cap]
    rho_raw: jax.Array  # [N] before the cap, for the clip statistics
    reward: jax.Array  # [N]
    valid: jax.Array  # [N] bool: a usable edge (see option_terms)


def head_dot(forward: jax.Array, latents: jax.Array) -> jax.Array:
    """F_h(x_i, .)^T latents_i for each head h and sample i: [H, N, d] x [N, d] -> [H, N]."""
    return jnp.einsum("hnd,nd->hn", forward, latents)


def direct_value(f_s_z: jax.Array, z: jax.Array) -> jax.Array:
    """V_1(s, z) = min_h F_h(s, z)^T z: the value of never switching again. [N]."""
    return head_dot(f_s_z, z).min(axis=0)


def option_terms(
    f_s_zw: jax.Array,
    f_w_zw: jax.Array,
    z_w: jax.Array,
    z: jax.Array,
    is_goal: jax.Array,
    distinct: jax.Array,
    config: OptionConfig,
) -> OptionTerms:
    """rho and R of the option to w, from F(s, z_w) and F(w, z_w) [H, N, d], z_w and z [N, d].

    An edge is invalid when s and w are the same state (`distinct` False), when
    F(w, z_w)^T z_w <= 0 in a head (w unreachable from itself), or when the raw rho >= 1:
    approach A treats such an edge as untrusted, and a self-loop with rho = 1 leaves the Bellman
    equation without a unique solution. `distinct` catches the self-loop even when rounding puts
    its raw rho below 1 (0.99999994 on a GPU).
    """
    self_measure = head_dot(f_w_zw, z_w)
    reachable = jnp.all(self_measure > 0, axis=0)
    rho_h = head_dot(f_s_zw, z_w) / jnp.where(self_measure > 0, self_measure, 1.0)
    rho_raw = rho_h.min(axis=0)
    return OptionTerms(
        rho=jnp.clip(rho_raw, 0.0, config.rho_cap),
        rho_raw=rho_raw,
        reward=edge_reward(f_s_zw, f_w_zw, rho_h, z, is_goal, config.edge_reward),
        valid=distinct & reachable & (rho_raw < 1.0),
    )


def edge_reward(
    f_s_zw: jax.Array,
    f_w_zw: jax.Array,
    rho_h: jax.Array,
    z: jax.Array,
    is_goal: jax.Array,
    kind: EdgeReward,
) -> jax.Array:
    """R(s, w; z) per sample, min over heads. [N]."""
    before = head_dot(f_s_zw, z)
    if kind == "paper":
        return before.min(axis=0)
    exact = (before - rho_h * head_dot(f_w_zw, z)).min(axis=0)
    return exact if kind == "exact" else jnp.where(is_goal, 0.0, exact)
