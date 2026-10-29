"""Exact dynamic programming on a GridMaze.

Rewards are state rewards r(s_t), collected from t = 0, as in the paper:
V^pi(s; r) = sum_t gamma^t E[r(s_t)] = (M^pi r)(s) with M^pi = (I - gamma P_pi)^{-1}.
Policies are deterministic: one action id per state.
"""

from __future__ import annotations

import numpy as np

from fbplan.tabular.maze import GridMaze

# Greedy ties: actions within this tolerance of the best are equal. The absolute part covers float
# error of linear solves (relative to the largest value); the relative part keeps far states,
# whose values are tiny, distinguishable.
_ATOL_REL_TO_MAX = 1e-12
_RTOL_PER_STATE = 1e-9


def action_values(maze: GridMaze, values: np.ndarray) -> np.ndarray:
    """E[values(s') | s, a] for every state and action: [n, 5]."""
    return np.sum(maze.probs * values[maze.next_states], axis=-1)


def greedy_policy(
    maze: GridMaze, values: np.ndarray, prefer: np.ndarray | None = None
) -> np.ndarray:
    """Actions that maximise E[values(s')]. Ties go to `prefer`, else to the lowest action id.

    The tie rule makes the policy a deterministic function of the values, so the same value
    function always gives the same policy (exact mode of the fake FB relies on this).
    """
    q = action_values(maze, values)
    best = q.max(axis=1, keepdims=True)
    tol = _ATOL_REL_TO_MAX * np.max(np.abs(values)) + _RTOL_PER_STATE * np.abs(best)
    near_best = q >= best - tol
    policy = np.argmax(near_best, axis=1)  # first True = lowest action id
    if prefer is not None:
        keep = near_best[np.arange(maze.n_states), prefer]
        policy = np.where(keep, prefer, policy)
    return policy


def policy_matrix(maze: GridMaze, policy: np.ndarray) -> np.ndarray:
    """State transition matrix P_pi [n, n] of a deterministic policy."""
    n = maze.n_states
    rows = np.arange(n)
    p = np.zeros((n, n))
    np.add.at(
        p,
        (np.repeat(rows, maze.next_states.shape[-1]), maze.next_states[rows, policy].ravel()),
        maze.probs[rows, policy].ravel(),
    )
    return p


def successor_measure(maze: GridMaze, policy: np.ndarray, gamma: float) -> np.ndarray:
    """M^pi = (I - gamma P_pi)^{-1}: M[s, s'] = sum_t gamma^t P(s_t = s' | s_0 = s)."""
    return np.linalg.inv(np.eye(maze.n_states) - gamma * policy_matrix(maze, policy))


def policy_value(
    maze: GridMaze, policy: np.ndarray, reward: np.ndarray, gamma: float
) -> np.ndarray:
    """V^pi(s; r) by one linear solve."""
    return np.linalg.solve(np.eye(maze.n_states) - gamma * policy_matrix(maze, policy), reward)


def optimal_policy(
    maze: GridMaze, reward: np.ndarray, gamma: float, max_vi_iters: int = 100_000
) -> tuple[np.ndarray, np.ndarray]:
    """Exact optimal policy and value V* for a state reward.

    Value iteration gives a warm start; policy iteration then makes the values exact. The
    returned policy is greedy with respect to V* with the lowest-id tie rule, which is optimal
    for any tie rule because gamma < 1.
    """
    values = np.zeros(maze.n_states)
    for _ in range(max_vi_iters):
        new = reward + gamma * action_values(maze, values).max(axis=1)
        done = np.max(np.abs(new - values)) <= 1e-12 * (1.0 + np.max(np.abs(new)))
        values = new
        if done:
            break
    policy = greedy_policy(maze, values)
    while True:
        values = policy_value(maze, policy, reward, gamma)
        improved = greedy_policy(maze, values, prefer=policy)
        if np.array_equal(improved, policy):
            break
        policy = improved
    return greedy_policy(maze, values), values
