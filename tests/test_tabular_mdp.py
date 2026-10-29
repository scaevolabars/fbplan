import numpy as np

from fbplan.tabular.maze import GridMaze
from fbplan.tabular.mdp import (
    action_values,
    greedy_policy,
    optimal_policy,
    policy_matrix,
    successor_measure,
)

GAMMA = 0.9


def test_policy_matrix_is_row_stochastic():
    maze = GridMaze("teleport", resolution=1)
    policy = np.random.default_rng(0).integers(0, 5, maze.n_states)
    np.testing.assert_allclose(policy_matrix(maze, policy).sum(1), 1.0)


def test_successor_measure_solves_its_bellman_equation():
    maze = GridMaze("teleport", resolution=1)
    policy = np.random.default_rng(1).integers(0, 5, maze.n_states)
    m = successor_measure(maze, policy, GAMMA)
    p = policy_matrix(maze, policy)
    np.testing.assert_allclose(m, np.eye(maze.n_states) + GAMMA * p @ m, atol=1e-10)
    np.testing.assert_allclose(m.sum(1), 1 / (1 - GAMMA))


def test_optimal_value_for_goal_indicator_is_discounted_distance():
    maze = GridMaze("medium", resolution=2)
    goal = maze.centre_state((6, 6))
    reward = np.zeros(maze.n_states)
    reward[goal] = 1.0
    policy, values = optimal_policy(maze, reward, GAMMA)
    dist = maze.geodesic_distances()[:, goal]
    np.testing.assert_allclose(values, GAMMA**dist / (1 - GAMMA), rtol=1e-10)
    assert policy[goal] == 0  # stay at the goal


def test_optimal_values_satisfy_bellman_optimality_in_stochastic_maze():
    maze = GridMaze("teleport", resolution=1)
    reward = np.random.default_rng(2).normal(size=maze.n_states)
    policy, values = optimal_policy(maze, reward, GAMMA)
    np.testing.assert_allclose(
        values, reward + GAMMA * action_values(maze, values).max(1), atol=1e-9
    )
    m = successor_measure(maze, policy, GAMMA)
    np.testing.assert_allclose(m @ reward, values, atol=1e-9)


def test_greedy_policy_breaks_ties_towards_the_lowest_action_id():
    maze = GridMaze("medium", resolution=1)
    np.testing.assert_array_equal(greedy_policy(maze, np.ones(maze.n_states)), 0)


def test_greedy_policy_ignores_float_noise_below_tolerance():
    maze = GridMaze("medium", resolution=1)
    values = 1.0 + 1e-14 * np.random.default_rng(3).normal(size=maze.n_states)
    np.testing.assert_array_equal(greedy_policy(maze, values), 0)
