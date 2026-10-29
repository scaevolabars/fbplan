"""Exact option models for "follow pi_w until w is hit", and planning over chains of them.

Notation follows docs/design.md §3 and docs/theory.md. For an option with Markov policy pi_w and
subgoal state w, with H the hitting time of w (H = 0 if s = w):

    rho(s, w) = E[gamma^H]                          = M_w(s, w) / M_w(w, w)
    R(s, w)   = E[sum_{t < H} gamma^t r(s_t)]       = V^{pi_w}(s) - rho(s, w) V^{pi_w}(w)

Tables are [n_states, L]: column j is the option to `subgoals[j]`, evaluated from every state.
The functions `hitting_discount`, `pre_hit_reward` and `switching_successor_measure` compute the
same objects from their definitions (first-step analysis, an augmented Markov chain), so tests can
check the formulas against them.

Everything here uses the true model. The planner computes the same option model from F and B
(`fbplan.option_model`); graph value iteration and Q-values come from `fbplan.graph` in both cases.
Names: `r_before` = R, `v_direct` = V_d (value of the last or task policy), `v_graph` = V_H.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from fbplan.graph import value_iteration
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
    v_direct: np.ndarray,
    subgoals: np.ndarray,
    start: int,
    chain: Sequence[int],
) -> float:
    """Value of the chain formula: R(s,w1) + rho(s,w1)[R(w1,w2) + ... + rho(.,wk) V_d(wk)].

    `chain` lists column indices into the tables; `v_direct` [n] is the value of the last
    policy (in the graph Bellman equation, the task policy: V_d in docs/theory.md).
    """
    value, discount, state = 0.0, 1.0, start
    for j in chain:
        value += discount * r_before[state, j]
        discount *= rho[state, j]
        state = int(subgoals[j])
    return value + discount * v_direct[state]


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
    rho: np.ndarray, r_before: np.ndarray, v_direct: np.ndarray, max_iters: int = 10_000
) -> np.ndarray:
    """V_H on exact landmark tables [L, L]: `fbplan.graph.value_iteration` with every edge
    between distinct landmarks trusted, solved to float precision.

    The planner's code runs the same equation on FB estimates with trust rules. Here the tables
    are exact, so no edge needs a guard. Self-loops are excluded: with them (rho = 1, R = 0)
    every V(u) above the other terms solves the equation, so the fixed point is not unique.
    Raises if the iteration does not converge (for instance with a rho >= 1 off the diagonal).
    """
    trusted = np.ones(rho.shape, dtype=bool)
    solution = value_iteration(rho, r_before, v_direct, trusted, max_iters=max_iters, tol=1e-13)
    if not solution.converged:
        raise RuntimeError(f"graph value iteration did not converge in {max_iters} sweeps")
    return solution.values


def replanning_policy(
    q: np.ndarray,
    v_direct: np.ndarray,
    option_policies: Sequence[np.ndarray],
    final_policy: np.ndarray,
) -> np.ndarray:
    """Markov policy that re-picks the best option at every step (option interruption). [n].

    q: [n, L] option values (-inf where an option is not available), v_direct: [n] value of
    the final policy, which acts where no option beats it.
    """
    n = len(v_direct)
    best = q.argmax(axis=1)
    use_option = q[np.arange(n), best] > v_direct
    option_actions = np.stack(option_policies)[best, np.arange(n)]
    return np.where(use_option, option_actions, final_policy)
