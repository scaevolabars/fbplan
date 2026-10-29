import numpy as np
import pytest

from fbplan.graph import GraphSolution, extract_route, q_values, trusted_edges, value_iteration

NEG = -np.inf


def chain(rho_edge: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """0 -> 1 -> 2, the task value is 10 at node 2 and 0 elsewhere."""
    rho = np.zeros((3, 3))
    rho[0, 1] = rho[1, 2] = rho_edge
    return rho, rho > 0, np.array([0.0, 0.0, 10.0])


def test_trusted_edges_keeps_rho_at_or_above_threshold():
    rho = np.array([[0.2, 0.5, 0.9]])
    np.testing.assert_array_equal(trusted_edges(rho, rho_min=0.5), [[False, True, True]])


def test_trusted_edges_top_k_keeps_the_k_largest_per_row():
    rho = np.array([[0.6, 0.9, 0.7, 0.8], [0.95, 0.1, 0.2, 0.3]])
    mask = trusted_edges(rho, rho_min=0.5, top_k=2)
    np.testing.assert_array_equal(mask, [[False, True, False, True], [True, False, False, False]])


def test_trusted_edges_can_drop_self_loops_before_top_k():
    rho = np.array([[1.0, 0.8, 0.6], [0.7, 1.0, 0.9], [0.5, 0.4, 1.0]])
    mask = trusted_edges(rho, rho_min=0.45, top_k=1, exclude_diagonal=True)
    np.testing.assert_array_equal(
        mask, [[False, True, False], [False, False, True], [True, False, False]]
    )


def test_value_iteration_on_a_chain_discounts_by_rho_per_hop():
    rho, mask, direct = chain()
    sol = value_iteration(rho, None, direct, mask)
    np.testing.assert_allclose(sol.values, [2.5, 5.0, 10.0])
    np.testing.assert_array_equal(sol.best_next, [1, 2, -1])
    assert sol.converged


def test_value_iteration_adds_reward_collected_before_the_hit():
    rho, mask, direct = chain()
    reward = np.zeros((3, 3))
    reward[0, 1] = 1.0
    sol = value_iteration(rho, reward, direct, mask)
    np.testing.assert_allclose(sol.values[0], 1.0 + 0.5 * 5.0)


def test_value_iteration_hop_discount_is_a_cost_per_hop():
    rho, mask, direct = chain()
    sol = value_iteration(rho, None, direct, mask, hop_discount=0.5)
    np.testing.assert_allclose(sol.values, [0.625, 2.5, 10.0])


def test_value_iteration_keeps_the_direct_option_when_it_is_better():
    rho, mask, direct = chain()
    direct = np.array([3.0, 0.0, 10.0])
    sol = value_iteration(rho, None, direct, mask)
    assert sol.values[0] == 3.0
    assert sol.best_next[0] == -1


def test_value_iteration_ignores_untrusted_edges():
    rho, mask, direct = chain()
    rho[0, 2] = 0.99  # a wormhole that pruning removed
    sol = value_iteration(rho, None, direct, mask)
    assert sol.best_next[0] == 1


def test_value_iteration_ignores_self_loops():
    rho, mask, direct = chain()
    np.fill_diagonal(rho, 1.0)
    np.fill_diagonal(mask, True)
    reward = np.eye(3)  # a positive self-loop would make the values diverge
    sol = value_iteration(rho, reward, direct, mask)
    np.testing.assert_allclose(sol.values, [2.5, 5.0, 10.0])


def test_value_iteration_reports_no_convergence_when_iterations_run_out():
    rho, mask, direct = chain()
    assert not value_iteration(rho, None, direct, mask, max_iters=1).converged


def test_value_iteration_with_zero_reward_finds_the_max_product_route():
    # 0 -> 1 -> 3 has product 0.81; 0 -> 2 -> 3 has product 0.495.
    rho = np.zeros((4, 4))
    rho[0, 1], rho[1, 3], rho[0, 2], rho[2, 3] = 0.9, 0.9, 0.5, 0.99
    direct = np.array([0.0, 0.0, 0.0, 1.0])
    sol = value_iteration(rho, None, direct, rho > 0)
    assert extract_route(sol, 0) == [0, 1, 3]
    np.testing.assert_allclose(sol.values[0], 0.81)


def test_q_values_score_edges_from_any_source_against_landmark_values():
    rho = np.array([[0.5, 0.0, 0.8]])
    values = np.array([4.0, 1.0, 2.0])
    q = q_values(rho, np.ones((1, 3)), values, rho > 0, hop_discount=0.5)
    np.testing.assert_allclose(q, [[1.0 + 0.5 * 0.5 * 4.0, NEG, 1.0 + 0.5 * 0.8 * 2.0]])


def test_extract_route_starts_at_the_source_and_ends_at_the_direct_option():
    rho, mask, direct = chain()
    sol = value_iteration(rho, None, direct, mask)
    assert extract_route(sol, 0) == [0, 1, 2]
    assert extract_route(sol, 2) == [2]


def test_extract_route_stops_when_a_node_repeats():
    sol = GraphSolution(
        values=np.zeros(3), best_next=np.array([1, 2, 1]), iterations=1, converged=True
    )
    assert extract_route(sol, 0) == [0, 1, 2]


def test_value_iteration_rejects_mismatched_shapes():
    rho, mask, direct = chain()
    with pytest.raises(ValueError):
        value_iteration(rho, None, direct[:2], mask)
