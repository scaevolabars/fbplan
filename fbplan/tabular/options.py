"""Exact option models for "follow pi_w until w is hit", and planning over chains of them.

Notation follows docs/design.md §3 and docs/theory.md. For an option with Markov policy pi_w and
subgoal state w, with H the hitting time of w (H = 0 if s = w):

    rho(s, w) = E[gamma^H]                          = M_w(s, w) / M_w(w, w)
    R(s, w)   = E[sum_{t < H} gamma^t r(s_t)]       = V^{pi_w}(s) - rho(s, w) V^{pi_w}(w)

Tables are [n_states, L]: column j is the option to `subgoals[j]`, evaluated from every state.
The functions `hitting_discount`, `pre_hit_reward` and `switching_successor_measure` compute the
same objects from their definitions (first-step analysis, an augmented Markov chain), so tests can
check the formulas against them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from fbplan.tabular.maze import GridMaze
from fbplan.tabular.mdp import policy_matrix, successor_measure


def hitting_discount(maze: GridMaze, policy: np.ndarray, target: int, gamma: float) -> np.ndarray:
    """E[gamma^H] by first-step analysis: h(w) = 1, h(s) = gamma E[h(s')] otherwise. [n]."""
    a = np.eye(maze.n_states) - gamma * policy_matrix(maze, policy)
    a[target] = np.eye(maze.n_states)[target]
    return np.linalg.solve(a, np.eye(maze.n_states)[target])


def pre_hit_reward(
    maze: GridMaze, policy: np.ndarray, target: int, reward: np.ndarray, gamma: float
) -> np.ndarray:
    """E[sum_{t<H} gamma^t r(s_t)] by first-step analysis: g(w) = 0. [n]."""
    a = np.eye(maze.n_states) - gamma * policy_matrix(maze, policy)
    a[target] = np.eye(maze.n_states)[target]
    b = reward.astype(np.float64).copy()
    b[target] = 0.0
    return np.linalg.solve(a, b)


def option_model(m_w: np.ndarray, target: int, reward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(rho, R) of one option from the successor measure M_w of its policy. Each [n]."""
    rho = m_w[:, target] / m_w[target, target]
    v_w = m_w @ reward
    return rho, v_w - rho * v_w[target]


def option_tables(
    maze: GridMaze,
    option_policies: Sequence[np.ndarray],
    subgoals: np.ndarray,
    reward: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(rho, R) tables [n, L] for the options "follow option_policies[j] until subgoals[j]"."""
    columns = [
        option_model(successor_measure(maze, pi_w, gamma), int(w), reward)
        for pi_w, w in zip(option_policies, subgoals, strict=True)
    ]
    return np.stack([c[0] for c in columns], 1), np.stack([c[1] for c in columns], 1)


def chained_value(
    rho: np.ndarray,
    r_before: np.ndarray,
    v_final: np.ndarray,
    subgoals: np.ndarray,
    start: int,
    chain: Sequence[int],
) -> float:
    """Value of the chain formula: R(s,w1) + rho(s,w1)[R(w1,w2) + ... + rho(.,wk) V_final(wk)].

    `chain` lists column indices into the tables; `v_final` [n] is the value of the last policy.
    """
    value, discount, state = 0.0, 1.0, start
    for j in chain:
        value += discount * r_before[state, j]
        discount *= rho[state, j]
        state = int(subgoals[j])
    return value + discount * v_final[state]


def switching_successor_measure(
    maze: GridMaze,
    option_policies: Sequence[np.ndarray],
    subgoals: Sequence[int],
    final_policy: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Successor measure [n, n] of "pi_1 until w_1, then pi_2 until w_2, ..., then pi_final".

    Built from the definition on the augmented chain (state, phase), where phase i < k means
    "follow pi_{i+1}" and advances when the state is w_{i+1} (at once, also at t = 0).
    """
    n, k = maze.n_states, len(subgoals)
    policies = [*option_policies, final_policy]

    def phase_at(state: int, phase: int) -> int:
        while phase < k and state == subgoals[phase]:
            phase += 1
        return phase

    p = np.zeros((n * (k + 1), n * (k + 1)))
    for phase in range(k + 1):
        for s in range(n):
            active = phase_at(s, phase)
            a = policies[active][s]
            for nxt, prob in zip(maze.next_states[s, a], maze.probs[s, a], strict=True):
                p[phase * n + s, phase_at(int(nxt), active) * n + nxt] += prob
    m_aug = np.linalg.inv(np.eye(n * (k + 1)) - gamma * p)
    return m_aug[:n].reshape(n, k + 1, n).sum(axis=1)


def graph_value_iteration(
    rho: np.ndarray,
    r_before: np.ndarray,
    v_direct: np.ndarray,
    hop_cost: float = 0.0,
    n_iters: int | None = None,
    init: np.ndarray | None = None,
    tol: float = 1e-12,
) -> np.ndarray:
    """Solve V(u) = max(v_direct(u), max_{w != u} R(u,w) - hop_cost + rho(u,w) V(w)) on landmarks.

    `rho`, `r_before` are [L, L] landmark-to-landmark tables. Self-loops are excluded: with them
    (rho = 1, R = 0) every V(u) above the other terms solves the equation, so the fixed point is not
    unique. Without them the operator is a contraction with modulus max rho <= gamma.
    `n_iters` runs exactly that many sweeps from `init` (default v_direct) instead of converging.
    """
    values = v_direct.copy() if init is None else init.astype(np.float64).copy()
    off_diagonal = ~np.eye(len(v_direct), dtype=bool)
    sweeps = range(n_iters) if n_iters is not None else iter(int, 1)  # iter(int, 1): forever
    for _ in sweeps:
        q = np.where(off_diagonal, r_before - hop_cost + rho * values[None, :], -np.inf)
        new = np.maximum(v_direct, q.max(axis=1))
        converged = np.max(np.abs(new - values)) <= tol * (1.0 + np.max(np.abs(new)))
        values = new
        if n_iters is None and converged:
            break
    return values


def option_values(
    rho: np.ndarray, r_before: np.ndarray, v_graph: np.ndarray, subgoals: np.ndarray
) -> np.ndarray:
    """Q(s, j) = R(s, w_j) + rho(s, w_j) V_graph(w_j) for all states, -inf where s = w_j: [n, L]."""
    q = r_before + rho * v_graph[None, :]
    q[subgoals, np.arange(len(subgoals))] = -np.inf
    return q


def replanning_policy(
    q: np.ndarray,
    v_final: np.ndarray,
    subgoals: np.ndarray,
    option_policies: Sequence[np.ndarray],
    final_policy: np.ndarray,
) -> np.ndarray:
    """Markov policy that re-picks the best option at every step (option interruption). [n]."""
    n = len(v_final)
    best = q.argmax(axis=1)
    use_option = q[np.arange(n), best] > v_final
    option_actions = np.stack(option_policies)[best, np.arange(n)]
    return np.where(use_option, option_actions, final_policy)
