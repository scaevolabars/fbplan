"""Landmark graph: SMDP value iteration, greedy choices and route extraction (pure numpy).

Nodes are landmarks. Edge (u, w) is the option "follow pi_{z_w} until w is hit", with
discount rho(u, w) and reward R(u, w) (docs/design.md §3). The graph Bellman equation is

    V_H(u) = max( direct(u),  max_{w trusted from u} R(u, w) + hop_discount * rho(u, w) * V_H(w) )

where direct(u) = F(u, z_r)^T z_r is the value of switching to the task policy now.
hop_discount < 1 is a cost per hop: in shortest-path form it adds -log(hop_discount) to
the edge cost -log rho, and it does not depend on the scale of z_r. Which edges are trusted
is the planner's rule (fbplan.controllers.planner.trusted_edges); value iteration takes a mask.

Data-certified edges (h-papers H1) need no FB value: an edge u -> w exists when a recorded
trajectory passes u and then w (landmark_visits, certify_edges); shortest_paths orders them.
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
    converged: bool


def trusted_edges(
    rho: np.ndarray, rho_min: float, exclude_diagonal: bool = False, rho_max: float = 1.0
) -> np.ndarray:
    """Boolean mask of edges with rho_min <= rho < rho_max: the rho part of the planner's
    trust rule (fbplan.controllers.planner.trusted_edges adds the xy cap).

    rho is the raw FB estimate, before clipping. rho >= 1 is an FB error, not a free hop: for
    distinct states E[gamma^H] <= gamma < 1. Trusted, such edges join far parts of the maze at
    almost no cost (wormholes). rho_max > 1 trusts them (an ablation). exclude_diagonal drops
    self-loops.
    """
    mask = (rho >= rho_min) & (rho < rho_max)
    if exclude_diagonal:
        mask &= ~np.eye(*rho.shape, dtype=bool)
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

    values, sweeps, converged = direct.astype(np.float64), 0, False
    while not converged and sweeps < max_iters:
        sweeps += 1
        new = np.maximum(direct, q_values(rho, reward, values, mask, hop_discount).max(axis=1))
        converged = np.max(np.abs(new - values)) <= tol * max(1.0, np.max(np.abs(new)))
        values = new

    q = q_values(rho, reward, values, mask, hop_discount)
    return GraphSolution(values, best_options(q, values, direct), converged)


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


# ----- data-certified edges (h-papers H1) -----


@dataclass(frozen=True)
class Visits:
    """Passes of recorded trajectories through landmark regions, in dataset-row order.

    A visit is a run of consecutive rows of one trajectory within the radius of one landmark;
    its row is the closest approach (the first one on a tie).
    """

    row: np.ndarray  # [V] dataset row
    label: np.ndarray  # [V] landmark id


@dataclass(frozen=True)
class CertifiedEdges:
    """Landmark edges that a recorded trajectory travels, each with one recorded walk: the pass
    at `source` on dataset row `start`, the pass at `target` on row `end`. [E] each"""

    source: np.ndarray
    target: np.ndarray
    start: np.ndarray
    end: np.ndarray


def trajectory_ends(terminals: np.ndarray) -> np.ndarray:
    """The last row of each row's trajectory, from the OGBench `terminals` flags (1 on the last
    row of a trajectory; the last row of the data always ends one). [P]"""
    ends = np.flatnonzero(terminals)
    if len(ends) == 0 or ends[-1] != len(terminals) - 1:
        ends = np.append(ends, len(terminals) - 1)
    return ends[np.searchsorted(ends, np.arange(len(terminals)))]


def landmark_visits(
    xy: np.ndarray, landmarks_xy: np.ndarray, radius: float, ends: np.ndarray, chunk: int = 8192
) -> Visits:
    """Visits of the recorded rows xy [P, 2] to the landmarks [N, 2]; a row belongs to its
    nearest landmark if that is within radius. ends: trajectory_ends of the rows."""
    labels = np.full(len(xy), -1)
    dist2 = np.full(len(xy), np.inf)
    for lo in range(0, len(xy), chunk):
        d2 = np.sum((xy[lo : lo + chunk, None, :] - landmarks_xy[None, :, :]) ** 2, axis=-1)
        nearest = np.argmin(d2, axis=1)
        best = d2[np.arange(len(nearest)), nearest]
        labels[lo : lo + chunk] = np.where(best <= radius**2, nearest, -1)
        dist2[lo : lo + chunk] = best
    rows = np.flatnonzero(labels >= 0)
    new_run = np.ones(len(rows), dtype=bool)
    new_run[1:] = (
        (labels[rows[1:]] != labels[rows[:-1]])
        | (rows[1:] != rows[:-1] + 1)
        | (ends[rows[1:]] != ends[rows[:-1]])
    )
    run = np.cumsum(new_run) - 1
    order = np.lexsort((rows, dist2[rows], run))  # by run, then distance, then row
    closest = order[np.r_[True, run[order][1:] != run[order][:-1]]] if len(rows) else order
    return Visits(rows[closest], labels[rows[closest]])


def walked_xy(xy: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Running xy path length of the rows [P]; steps across a trajectory end count 0, so the
    walk from row a to a later row b of one trajectory is arc[b] - arc[a]."""
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    steps[ends[:-1] == np.arange(len(xy) - 1)] = 0.0
    return np.concatenate([[0.0], np.cumsum(steps)])


def certify_edges(
    visits: Visits, ends: np.ndarray, max_steps: int, arc: np.ndarray | None = None
) -> CertifiedEdges:
    """Edges u -> w (u != w) where one trajectory passes u and then w at most max_steps rows
    later. Each keeps its shortest recorded walk (arc: walked_xy of the rows; None: all walks
    are equally long), then the fastest, then the earliest."""
    row, label = visits.row, visits.label
    first, second = [], []
    for offset in range(1, len(row)):
        a, b = row[:-offset], row[offset:]
        close = (b - a <= max_steps) & (ends[a] == ends[b])
        if not close.any():
            break  # rows increase, so larger offsets are farther apart
        keep = np.flatnonzero(close & (label[:-offset] != label[offset:]))
        first.append(keep)
        second.append(keep + offset)
    a = np.concatenate(first) if first else np.empty(0, dtype=int)
    b = np.concatenate(second) if second else np.empty(0, dtype=int)
    key = label[a] * (label.max(initial=0) + 1) + label[b]
    walked = np.zeros(len(a)) if arc is None else arc[row[b]] - arc[row[a]]
    order = np.lexsort((row[a], row[b] - row[a], walked, key))
    fastest = order[np.r_[True, key[order][1:] != key[order][:-1]]] if len(order) else order
    a, b = a[fastest], b[fastest]
    return CertifiedEdges(label[a], label[b], row[a], row[b])


def reachable_visits(
    visits: Visits, rows: np.ndarray, ends: np.ndarray, max_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """The visits that the trajectory of each dataset row in rows makes from that row on, at
    most max_steps rows later: (visit index, index into rows), both [M]."""
    lo = np.searchsorted(visits.row, rows, side="left")
    hi = np.searchsorted(visits.row, np.minimum(rows + max_steps, ends[rows]), side="right")
    counts = np.maximum(hi - lo, 0)
    owner = np.repeat(np.arange(len(rows)), counts)
    offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    return np.repeat(lo, counts) + offsets, owner


def shortest_paths(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All-pairs shortest paths (Floyd-Warshall) over edge weights [N, N] (inf: no edge).

    Returns the distances [N, N] and the next hop [N, N] from u towards w (-1: no path).
    A path changes only for a strictly shorter one, so ties keep the earlier path.
    """
    n = len(weights)
    dist = np.array(weights, dtype=np.float64)
    np.fill_diagonal(dist, 0.0)
    next_hop = np.where(np.isfinite(dist), np.arange(n)[None, :], -1)
    for k in range(n):
        via = dist[:, k, None] + dist[None, k, :]
        shorter = via < dist
        dist = np.where(shorter, via, dist)
        next_hop = np.where(shorter, next_hop[:, k, None], next_hop)
    return dist, next_hop
