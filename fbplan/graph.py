"""Landmark graph: edge pruning, SMDP value iteration and route extraction (pure numpy).

Nodes are landmarks. Edge (u, w) is the option "follow pi_{B(w)} until w is hit", with
discount rho(u, w) and reward R(u, w) (docs/design.md §3). The graph Bellman equation is

    V_H(u) = max( direct(u),  max_{w trusted from u} R(u, w) + hop_discount * rho(u, w) * V_H(w) )

where direct(u) = F(u, z_r)^T z_r is the value of switching to the task policy now.
hop_discount < 1 is a cost per hop: in shortest-path form it adds -log(hop_discount) to
the edge cost -log rho, and it does not depend on the scale of z_r.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DIRECT = -1  # best_next value: switch to the task policy instead of another landmark
# Q values closer than this (relative) are ties. They differ only in rounding, which depends on
# the CPU and the BLAS build, so an argmax over them can differ between machines.
TIE_RTOL = 1e-9


@dataclass(frozen=True)
class GraphSolution:
    """Fixed point of the graph Bellman equation."""

    values: np.ndarray  # V_H [N]
    best_next: np.ndarray  # [N] greedy next landmark, or DIRECT
    iterations: int
    converged: bool


def trusted_edges(
    rho: np.ndarray,
    rho_min: float,
    top_k: int | None = None,
    exclude_diagonal: bool = False,
    rho_max: float = 1.0,
) -> np.ndarray:
    """Boolean mask of edges with rho_min <= rho < rho_max, then at most top_k per row.

    rho is the raw FB estimate, before clipping. rho >= 1 is an FB error, not a free hop: for
    distinct states E[gamma^H] <= gamma < 1. Trusted, such edges join far parts of the maze at
    almost no cost (wormholes). rho_max > 1 trusts them (an ablation). exclude_diagonal removes
    self-loops first, so that rho(u, u) = 1 does not use a top-k slot. top_k keeps the largest
    rho per row.
    """
    mask = (rho >= rho_min) & (rho < rho_max)
    if exclude_diagonal:
        mask &= ~np.eye(*rho.shape, dtype=bool)
    if top_k is not None and top_k < rho.shape[1]:
        ranked = np.argsort(-np.where(mask, rho, -np.inf), axis=1, kind="stable")[:, :top_k]
        keep = np.zeros_like(mask)
        np.put_along_axis(keep, ranked, True, axis=1)
        mask &= keep
    return mask


def q_values(
    rho: np.ndarray,
    reward: np.ndarray | None,
    values: np.ndarray,
    mask: np.ndarray,
    hop_discount: float = 1.0,
) -> np.ndarray:
    """Q(u, w) = R(u, w) + hop_discount * rho(u, w) * V(w) on trusted edges, -inf elsewhere.

    Sources u can be landmarks (rows of the edge table) or the current state (one row).
    """
    q = hop_discount * rho * values[None, :]
    if reward is not None:
        q = q + reward
    return np.where(mask, q, -np.inf)


def value_iteration(
    rho: np.ndarray,
    reward: np.ndarray | None,
    direct: np.ndarray,
    mask: np.ndarray,
    hop_discount: float = 1.0,
    max_iters: int = 1000,
    tol: float = 1e-8,
) -> GraphSolution:
    """Solve the graph Bellman equation by value iteration from V = direct.

    Self-loops are ignored. tol is relative to the largest |value|. On ties the direct
    option wins, so routes do not take hops that add no value.
    """
    n = direct.shape[0]
    shapes_ok = rho.shape == mask.shape == (n, n) and (reward is None or reward.shape == (n, n))
    if not shapes_ok:
        raise ValueError(f"expected [{n}, {n}] tables for {n} direct values")
    mask = mask & ~np.eye(n, dtype=bool)

    values, iterations, converged = direct.astype(np.float64), 0, False
    while not converged and iterations < max_iters:
        iterations += 1
        new = np.maximum(direct, q_values(rho, reward, values, mask, hop_discount).max(axis=1))
        converged = np.max(np.abs(new - values)) <= tol * max(1.0, np.max(np.abs(new)))
        values = new

    q = q_values(rho, reward, values, mask, hop_discount)
    return GraphSolution(values, best_options(q, values, direct), iterations, converged)


def best_options(q: np.ndarray, values: np.ndarray, direct: np.ndarray) -> np.ndarray:
    """Greedy choice in each row of q [R, N]: a landmark index, or DIRECT. [R]

    Ties (within TIE_RTOL) follow fixed rules instead of rounding: the direct option wins a tie
    with the landmarks (a hop that adds no value is not taken); among tied landmarks the one with
    the largest V_H (values [N]) wins, as it makes the most progress; then the lowest index.
    Example: with region latents in the exact tabular maze, a landmark ahead and one behind can
    tie exactly; choosing the one behind makes the agent turn back and forth.
    """
    best = q.max(axis=1)
    tol = TIE_RTOL * np.maximum(np.abs(best), np.abs(direct))
    tied_values = np.where(q >= (best - tol)[:, None], values[None, :], -np.inf)
    top = tied_values.max(axis=1)
    first = np.argmax(tied_values >= (top - TIE_RTOL * np.abs(top))[:, None], axis=1)
    return np.where(best > direct + tol, first, DIRECT)


def extract_route(solution: GraphSolution, start: int) -> list[int]:
    """Greedy route from landmark `start`: [start, next, ...] until the direct option.

    Stops before a repeated node, which can only happen on exact ties.
    """
    route = [start]
    nxt = int(solution.best_next[start])
    while nxt != DIRECT and nxt not in route:
        route.append(nxt)
        nxt = int(solution.best_next[nxt])
    return route
