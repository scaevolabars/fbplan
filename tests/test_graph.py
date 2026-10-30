import numpy as np
import pytest

from fbplan.graph import (
    DIRECT,
    GraphSolution,
    best_options,
    certify_edges,
    extract_route,
    landmark_visits,
    q_values,
    reachable_visits,
    shortest_paths,
    trajectory_ends,
    trusted_edges,
    value_iteration,
    walked_xy,
)

NEG = -np.inf


def chain(rho_edge: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """0 -> 1 -> 2, the task value is 10 at node 2 and 0 elsewhere."""
    rho = np.zeros((3, 3))
    rho[0, 1] = rho[1, 2] = rho_edge
    return rho, rho > 0, np.array([0.0, 0.0, 10.0])


def test_trusted_edges_keep_rho_from_rho_min_up_to_rho_max():
    # For distinct states E[gamma^H] <= gamma < 1, so rho >= 1 is an FB error, not a free hop.
    rho = np.array([[0.2, 0.5, 0.99, 1.0]])
    np.testing.assert_array_equal(trusted_edges(rho, rho_min=0.5), [[0, 1, 1, 0]])
    np.testing.assert_array_equal(trusted_edges(rho, 0.5, rho_max=np.inf), [[0, 1, 1, 1]])


def test_trusted_edges_can_drop_self_loops():
    rho = np.full((2, 2), 0.8)
    np.testing.assert_array_equal(trusted_edges(rho, 0.5, exclude_diagonal=True), [[0, 1], [1, 0]])


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
        values=np.zeros(3), best_next=np.array([1, 2, 1]), converged=True
    )
    assert extract_route(sol, 0) == [0, 1, 2]


def test_best_options_treat_q_values_within_rounding_as_ties_and_prefer_the_larger_value():
    # 1 + 1e-13 and 1 differ only by rounding: the tied landmark with the larger V_H wins.
    q = np.array([[1.0 + 1e-13, 1.0, 0.5]])
    assert best_options(q, np.array([2.0, 3.0, 9.0]), np.array([0.0])).tolist() == [1]


def test_best_options_break_value_ties_by_the_lowest_index():
    q = np.array([[1.0, 1.0 + 1e-13]])
    assert best_options(q, np.array([3.0, 3.0 - 1e-12]), np.array([0.0])).tolist() == [0]


def test_best_options_give_ties_with_the_direct_value_to_the_task_policy():
    q = np.array([[1.0 + 1e-13], [1.1]])
    assert best_options(q, np.array([5.0]), np.array([1.0, 1.0])).tolist() == [DIRECT, 0]


def test_best_options_without_a_trusted_edge_choose_the_task_policy():
    q = np.full((1, 3), -np.inf)
    assert best_options(q, np.ones(3), np.array([-2.0])).tolist() == [DIRECT]


def test_value_iteration_breaks_rounding_ties_toward_the_larger_value():
    # From node 0, hops to 1 and to 2 are worth 1.35 each up to rounding; 2 is nearer the goal.
    direct = np.array([0.0, 2.7, 3.0])
    rho = np.zeros((3, 3))
    rho[0, 1], rho[0, 2] = (1.35 + 1e-13) / 2.7, 0.45
    mask = rho > 0
    assert value_iteration(rho, None, direct, mask).best_next[0] == 2


def test_value_iteration_rejects_mismatched_shapes():
    rho, mask, direct = chain()
    with pytest.raises(ValueError):
        value_iteration(rho, None, direct[:2], mask)


# ----- data-certified edges (h-papers H1) -----

# Two landmarks on a line, 4 xy apart. Each row is one recorded state; xy along x.
MARKS = np.array([[0.0, 0.0], [4.0, 0.0]])


def walk(*x: float) -> np.ndarray:
    return np.stack([np.asarray(x, dtype=float), np.zeros(len(x))], axis=1)


def test_trajectory_ends_give_the_last_row_of_each_rows_trajectory():
    terminals = np.array([0, 0, 1, 0, 1, 0])  # the last trajectory has no terminal flag
    np.testing.assert_array_equal(trajectory_ends(terminals), [2, 2, 2, 4, 4, 5])


def test_a_visit_is_a_run_near_one_landmark_and_keeps_its_closest_row():
    xy = walk(-0.4, -0.1, 0.3, 2.0, 3.8, 4.2, 9.0)
    visits = landmark_visits(xy, MARKS, radius=0.5, ends=np.full(7, 6))
    np.testing.assert_array_equal(visits.row, [1, 4])
    np.testing.assert_array_equal(visits.label, [0, 1])


def test_visits_split_where_a_trajectory_ends_or_the_walk_leaves_the_radius():
    xy = walk(0.1, 0.2, 0.1, 1.0, 0.0)
    ends = trajectory_ends(np.array([0, 1, 0, 0, 1]))  # trajectories: rows 0-1 and 2-4
    visits = landmark_visits(xy, MARKS, radius=0.5, ends=ends)
    np.testing.assert_array_equal(visits.row, [0, 2, 4])
    np.testing.assert_array_equal(visits.label, [0, 0, 0])


def test_an_edge_is_certified_in_the_recorded_direction_within_max_steps():
    xy = walk(0.0, 1.0, 2.0, 3.0, 4.0)
    visits = landmark_visits(xy, MARKS, 0.5, np.full(5, 4))
    edges = certify_edges(visits, np.full(5, 4), max_steps=4)
    assert (edges.source.tolist(), edges.target.tolist()) == ([0], [1])
    assert (edges.start.tolist(), edges.end.tolist()) == ([0], [4])
    assert len(certify_edges(visits, np.full(5, 4), max_steps=3).source) == 0


def test_an_edge_is_not_certified_across_two_trajectories():
    xy = walk(0.0, 1.0, 3.0, 4.0)
    ends = trajectory_ends(np.array([0, 1, 0, 1]))
    edges = certify_edges(landmark_visits(xy, MARKS, 0.5, ends), ends, max_steps=10)
    assert len(edges.source) == 0


def certified_walks(xy: np.ndarray, terminals: list[int], arc: bool) -> list[tuple[int, int]]:
    """(start, end) rows of each certified edge between MARKS, with max_steps 10."""
    ends = trajectory_ends(np.array(terminals))
    lengths = walked_xy(xy, ends) if arc else None
    edges = certify_edges(landmark_visits(xy, MARKS, 0.5, ends), ends, 10, lengths)
    return list(zip(edges.start.tolist(), edges.end.tolist(), strict=True))


def test_an_edge_keeps_its_shortest_recorded_walk_then_its_fastest():
    # 0 -> 1 three times: rows 0-3 walk 8 xy in 3 steps, rows 4-8 4 xy in 4, rows 9-11 4 xy in 2.
    xy = walk(0.0, 3.0, 1.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 2.0, 4.0)
    terminals = [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1]
    assert certified_walks(xy, terminals, arc=True) == [(9, 11)]
    assert certified_walks(xy[:9], terminals[:9], arc=True) == [(4, 8)]
    assert certified_walks(xy[:9], terminals[:9], arc=False) == [(0, 3)]  # the fastest


def test_walked_xy_sums_the_steps_within_each_trajectory():
    xy = walk(0.0, 1.0, 3.0, 10.0, 9.0)
    ends = trajectory_ends(np.array([0, 0, 1, 0, 1]))
    np.testing.assert_allclose(walked_xy(xy, ends), [0.0, 1.0, 3.0, 3.0, 4.0])


def test_reachable_visits_lie_in_the_rows_own_trajectory_within_max_steps():
    xy = walk(0.0, 2.0, 4.0, 2.0, 0.0, 4.0)
    ends = trajectory_ends(np.array([0, 0, 0, 0, 1, 1]))  # rows 0-4 and row 5
    visits = landmark_visits(xy, MARKS, 0.5, ends)  # rows 0, 2, 4, 5
    visit, owner = reachable_visits(visits, np.array([1, 3, 4]), ends, max_steps=2)
    found = sorted(zip(owner.tolist(), visits.row[visit].tolist(), strict=True))
    assert found == [(0, 2), (1, 4), (2, 4)]


def test_shortest_paths_follow_the_cheapest_edges_and_give_the_next_hop():
    inf = np.inf
    weights = np.array([[inf, 1.0, 5.0, inf], [inf, inf, 1.0, inf], [inf, inf, inf, inf],
                        [inf, inf, inf, inf]])
    dist, next_hop = shortest_paths(weights)
    assert dist[0, 2] == 2.0 and next_hop[0, 2] == 1 and next_hop[1, 2] == 2
    assert dist[2, 0] == inf and next_hop[2, 0] == -1
    assert dist[3, 3] == 0.0 and next_hop[3, 3] == 3
