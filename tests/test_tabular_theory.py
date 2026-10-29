"""Numeric checks of the switching theory (docs/theory.md) against independent ground truth."""

import itertools

import numpy as np
import pytest

from fbplan.tabular.maze import GridMaze
from fbplan.tabular.mdp import optimal_policy, policy_value, successor_measure
from fbplan.tabular.options import (
    chained_value,
    graph_value_iteration,
    hitting_discount,
    option_model,
    option_tables,
    option_values,
    pre_hit_reward,
    replanning_policy,
    switching_successor_measure,
)

GAMMA = 0.9
LANDMARKS = np.array([0, 6, 13, 19, 24])


@pytest.fixture(scope="module", params=["medium", "teleport"])
def maze(request) -> GridMaze:
    return GridMaze(request.param, resolution=1)


def _random_policy(maze: GridMaze, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 5, maze.n_states)


def _detour_policy(maze: GridMaze, target: int, seed: int) -> np.ndarray:
    """Reaches `target` from every state, with detours; a random policy almost never hits it."""
    reward = 0.05 * np.random.default_rng(seed).normal(size=maze.n_states)
    reward[target] += 1.0
    return optimal_policy(maze, reward, GAMMA)[0]


def _random_reward(maze: GridMaze, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=maze.n_states)


def _goal_policy(maze: GridMaze, target: int) -> np.ndarray:
    reward = np.zeros(maze.n_states)
    reward[target] = 1.0
    return optimal_policy(maze, reward, GAMMA)[0]


def _goal_policies(maze: GridMaze) -> list[np.ndarray]:
    return [_goal_policy(maze, w) for w in LANDMARKS]


def _graph_inputs(maze, reward, final):
    """Landmark-to-landmark tables, direct values at landmarks, and the full tables."""
    rho, r_before = option_tables(maze, _goal_policies(maze), LANDMARKS, reward, GAMMA)
    v_final = policy_value(maze, final, reward, GAMMA)
    return rho, r_before, v_final


def test_lemma_b1_rho_is_the_expected_discount_at_the_hitting_time(maze):
    w = 7
    policy = _detour_policy(maze, w, 0)
    m = successor_measure(maze, policy, GAMMA)
    expected = hitting_discount(maze, policy, w, GAMMA)
    assert np.sum(expected > 0) > maze.n_states // 2
    np.testing.assert_allclose(m[:, w] / m[w, w], expected, atol=1e-10)


def test_rho_is_at_most_gamma_away_from_the_subgoal(maze):
    w = 11
    m = successor_measure(maze, _goal_policy(maze, w), GAMMA)
    rho = m[:, w] / m[w, w]
    assert rho[w] == pytest.approx(1.0)
    assert np.all(np.delete(rho, w) <= GAMMA + 1e-12)


def test_option_reward_is_the_reward_collected_before_the_hit(maze):
    w, reward = 5, _random_reward(maze, 2)
    policy = _detour_policy(maze, w, 1)
    _, r_before = option_model(successor_measure(maze, policy, GAMMA), w, reward)
    expected = pre_hit_reward(maze, policy, w, reward, GAMMA)
    np.testing.assert_allclose(r_before, expected, atol=1e-9)


def test_theorem_1_holds_for_the_switching_successor_measure(maze):
    w = 9
    pi_w, pi = _detour_policy(maze, w, 3), _random_policy(maze, 4)
    m_w, m = successor_measure(maze, pi_w, GAMMA), successor_measure(maze, pi, GAMMA)
    rho = m_w[:, w] / m_w[w, w]
    theorem = m_w + rho[:, None] * (m[w] - m_w[w])[None, :]
    switching = switching_successor_measure(maze, [pi_w], [w], pi, GAMMA)
    np.testing.assert_allclose(switching, theorem, atol=1e-8)


def test_chained_formula_equals_the_value_of_the_switching_policy(maze):
    reward = _random_reward(maze, 5)
    subgoals = np.array([3, 17, 8, 21])
    policies = [_detour_policy(maze, w, 10 + i) for i, w in enumerate(subgoals)]
    final = _random_policy(maze, 99)
    rho, r_before = option_tables(maze, policies, subgoals, reward, GAMMA)
    v_final = policy_value(maze, final, reward, GAMMA)
    truth = switching_successor_measure(maze, policies, list(subgoals), final, GAMMA) @ reward
    for start in (0, 12, 17):
        value = chained_value(rho, r_before, v_final, subgoals, start, [0, 1, 2, 3])
        assert value == pytest.approx(truth[start], abs=1e-8)


def test_k_steps_of_graph_value_iteration_equal_the_best_chain_of_at_most_k_switches(maze):
    reward = _random_reward(maze, 6)
    rho, r_before, v_final = _graph_inputs(maze, reward, _random_policy(maze, 7))
    columns = range(len(LANDMARKS))
    for k in range(4):
        v_graph = graph_value_iteration(
            rho[LANDMARKS], r_before[LANDMARKS], v_final[LANDMARKS], n_iters=k
        )
        for u, start in enumerate(LANDMARKS):
            chains = (c for n in range(k + 1) for c in itertools.product(columns, repeat=n))
            best = max(chained_value(rho, r_before, v_final, LANDMARKS, start, c) for c in chains)
            assert v_graph[u] == pytest.approx(best, abs=1e-9)


def test_graph_value_iteration_has_a_unique_fixed_point_without_self_loops(maze):
    reward = _random_reward(maze, 8)
    rho, r_before, v_final = _graph_inputs(maze, reward, _random_policy(maze, 9))
    args = (rho[LANDMARKS], r_before[LANDMARKS], v_final[LANDMARKS])
    low = graph_value_iteration(*args)
    high = graph_value_iteration(*args, init=v_final[LANDMARKS] + 1e3)
    np.testing.assert_allclose(low, high, atol=1e-8)


def test_exact_planning_cannot_beat_the_optimal_direct_value(maze):
    reward = _random_reward(maze, 10)
    optimal, v_star = optimal_policy(maze, reward, GAMMA)
    rho, r_before, v_final = _graph_inputs(maze, reward, optimal)
    v_graph = graph_value_iteration(rho[LANDMARKS], r_before[LANDMARKS], v_final[LANDMARKS])
    np.testing.assert_allclose(v_graph, v_star[LANDMARKS], atol=1e-8)


def test_option_at_its_own_subgoal_is_not_available():
    rho, r_before = np.ones((3, 2)), np.zeros((3, 2))
    q = option_values(rho, r_before, np.array([5.0, 7.0]), np.array([0, 2]))
    assert q[0, 0] == -np.inf and q[2, 1] == -np.inf
    assert q[1, 0] == 5.0 and q[0, 1] == 7.0


def test_replanning_every_step_is_at_least_as_good_as_the_option_plan(maze):
    reward = _random_reward(maze, 11)
    final = _random_policy(maze, 12)
    rho, r_before, v_final = _graph_inputs(maze, reward, final)
    v_graph = graph_value_iteration(rho[LANDMARKS], r_before[LANDMARKS], v_final[LANDMARKS])
    q = option_values(rho, r_before, v_graph, LANDMARKS)
    plan_value = np.maximum(v_final, q.max(1))
    policy = replanning_policy(q, v_final, LANDMARKS, _goal_policies(maze), final)
    assert np.all(policy_value(maze, policy, reward, GAMMA) >= plan_value - 1e-9)
