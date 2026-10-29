"""Numeric checks of the switching theory (docs/theory.md) against independent ground truth."""

import itertools
from collections import deque

import numpy as np
import pytest

from fbplan.graph import q_values, value_iteration
from fbplan.tabular.maze import GridMaze
from fbplan.tabular.mdp import optimal_policy, policy_value, successor_measure
from fbplan.tabular.options import (
    chained_value,
    graph_value_iteration,
    hitting_discount,
    option_model,
    option_tables,
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


def _graph_inputs(
    maze: GridMaze, reward: np.ndarray, final: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rho [n, L], r_before [n, L], v_direct [n]): exact tables of the goal options to
    LANDMARKS from every state, and the value of the final policy. Rows LANDMARKS give the
    landmark-to-landmark tables."""
    rho, r_before = option_tables(maze, _goal_policies(maze), LANDMARKS, reward, GAMMA)
    v_direct = policy_value(maze, final, reward, GAMMA)
    return rho, r_before, v_direct


def _least_steps(maze: GridMaze, target: int) -> np.ndarray:
    """Fewest steps from every state to `target` over transitions with positive probability
    (teleport jumps included). [n]."""
    predecessors: list[set[int]] = [set() for _ in range(maze.n_states)]
    for s, a, k in np.argwhere(maze.probs > 0):
        predecessors[maze.next_states[s, a, k]].add(int(s))
    steps = np.full(maze.n_states, np.inf)
    steps[target] = 0
    queue = deque([target])
    while queue:
        u = queue.popleft()
        for v in predecessors[u]:
            if steps[v] == np.inf:
                steps[v] = steps[u] + 1
                queue.append(v)
    return steps


def _own_subgoal(maze: GridMaze) -> np.ndarray:
    """[n, L] True where the state is the option's own subgoal (that option is unavailable)."""
    return np.arange(maze.n_states)[:, None] == LANDMARKS[None, :]


def test_lemma_b1_rho_is_the_expected_discount_at_the_hitting_time(maze):
    w = 7
    policy = _detour_policy(maze, w, 0)
    m = successor_measure(maze, policy, GAMMA)
    expected = hitting_discount(maze, policy, w, GAMMA)
    assert np.sum(expected > 0) > maze.n_states // 2
    np.testing.assert_allclose(m[:, w] / m[w, w], expected, atol=1e-10)


def test_rho_is_at_most_gamma_to_the_least_number_of_steps(maze):
    for w in (11, 20):
        for policy in (_goal_policy(maze, w), _detour_policy(maze, w, 5)):
            m = successor_measure(maze, policy, GAMMA)
            rho = m[:, w] / m[w, w]
            assert rho[w] == pytest.approx(1.0)
            assert np.all(np.delete(rho, w) <= GAMMA + 1e-12)
            assert np.all(rho <= GAMMA ** _least_steps(maze, w) + 1e-12)


def test_shortest_path_option_in_a_deterministic_maze_has_rho_gamma_to_the_distance():
    maze, w = GridMaze("medium", resolution=1), 11
    m = successor_measure(maze, _goal_policy(maze, w), GAMMA)
    np.testing.assert_allclose(m[:, w] / m[w, w], GAMMA ** maze.geodesic_distances()[:, w])


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
    v_direct = policy_value(maze, final, reward, GAMMA)
    truth = switching_successor_measure(maze, policies, list(subgoals), final, GAMMA) @ reward
    # The upstream proxy R~(s, w) = V^{pi_w}(s) keeps the term rho V^{pi_w}(w) (theory.md §7).
    v_options = np.stack([policy_value(maze, pi, reward, GAMMA) for pi in policies], axis=1)
    gaps = []
    for start in (0, 12, 17):
        value = chained_value(rho, r_before, v_direct, subgoals, start, [0, 1, 2, 3])
        assert value == pytest.approx(truth[start], abs=1e-8)
        proxy = chained_value(rho, v_options, v_direct, subgoals, start, [0, 1, 2, 3])
        gaps.append(abs(proxy - truth[start]))
    # The check is not trivial: the proxy misses the chain value (theory.md §4 quotes these).
    assert max(gaps) == pytest.approx({"medium": 0.561, "teleport": 0.355}[maze.name], abs=1e-3)


def test_k_steps_of_graph_value_iteration_equal_the_best_chain_of_at_most_k_switches(maze):
    reward = _random_reward(maze, 6)
    rho, r_before, v_direct = _graph_inputs(maze, reward, _random_policy(maze, 7))
    columns = range(len(LANDMARKS))
    all_edges = np.ones((len(LANDMARKS), len(LANDMARKS)), dtype=bool)
    for k in range(1, 4):
        v_graph = value_iteration(  # tol 0: exactly k sweeps
            rho[LANDMARKS],
            r_before[LANDMARKS],
            v_direct[LANDMARKS],
            all_edges,
            max_iters=k,
            tol=0.0,
        ).values
        for u, start in enumerate(LANDMARKS):
            chains = (c for n in range(k + 1) for c in itertools.product(columns, repeat=n))
            best = max(chained_value(rho, r_before, v_direct, LANDMARKS, start, c) for c in chains)
            assert v_graph[u] == pytest.approx(best, abs=1e-9)


def _bellman(rho, r_before, v_direct, trusted, values):
    return np.maximum(v_direct, q_values(rho, r_before, values, trusted).max(axis=1))


def test_graph_bellman_operator_is_a_contraction_without_self_loops_only(maze):
    reward = _random_reward(maze, 8)
    rho, r_before, v_direct = _graph_inputs(maze, reward, _random_policy(maze, 9))
    rho, r_before, v_direct = rho[LANDMARKS], r_before[LANDMARKS], v_direct[LANDMARKS]
    off_diagonal = ~np.eye(len(LANDMARKS), dtype=bool)
    rng = np.random.default_rng(0)
    for _ in range(20):
        v1, v2 = rng.normal(scale=10, size=(2, len(LANDMARKS)))
        t1 = _bellman(rho, r_before, v_direct, off_diagonal, v1)
        t2 = _bellman(rho, r_before, v_direct, off_diagonal, v2)
        assert np.abs(t1 - t2).max() <= GAMMA * np.abs(v1 - v2).max() + 1e-12
    v_graph = graph_value_iteration(rho, r_before, v_direct)
    np.testing.assert_allclose(_bellman(rho, r_before, v_direct, off_diagonal, v_graph), v_graph)
    # With self-loops (rho(u, u) = 1, R(u, u) = 0) every larger vector is a fixed point too.
    higher = v_graph + 1e3
    with_loops = _bellman(rho, r_before, v_direct, np.ones_like(off_diagonal), higher)
    np.testing.assert_allclose(with_loops, higher)


def test_exact_planning_cannot_beat_the_optimal_direct_value(maze):
    reward = _random_reward(maze, 10)
    optimal, v_star = optimal_policy(maze, reward, GAMMA)
    rho, r_before, v_direct = _graph_inputs(maze, reward, optimal)
    v_graph = graph_value_iteration(rho[LANDMARKS], r_before[LANDMARKS], v_direct[LANDMARKS])
    np.testing.assert_allclose(v_graph, v_star[LANDMARKS], atol=1e-8)


def test_replanning_every_step_is_at_least_as_good_as_the_option_plan(maze):
    reward = _random_reward(maze, 11)
    final = _random_policy(maze, 12)
    rho, r_before, v_direct = _graph_inputs(maze, reward, final)
    v_graph = graph_value_iteration(rho[LANDMARKS], r_before[LANDMARKS], v_direct[LANDMARKS])
    q = q_values(rho, r_before, v_graph, ~_own_subgoal(maze))
    plan_value = np.maximum(v_direct, q.max(1))
    policy = replanning_policy(q, v_direct, _goal_policies(maze), final)
    value = policy_value(maze, policy, reward, GAMMA)
    assert np.all(value >= plan_value - 1e-9)
    # theory.md §6 quotes how often re-planning is strictly better.
    strict = int(np.sum(value > plan_value + 1e-9))
    assert (strict, maze.n_states) == {"medium": (7, 26), "teleport": (20, 45)}[maze.name]
