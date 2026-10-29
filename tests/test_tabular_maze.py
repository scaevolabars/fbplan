import numpy as np
import pytest

from fbplan.tabular.maze import ACTION_VECTORS, GridMaze


def test_medium_maze_with_resolution_2_has_104_states_like_the_paper():
    assert GridMaze("medium", resolution=2).n_states == 104


def test_xy_matches_ogbench_cell_centres_at_resolution_1():
    maze = GridMaze("medium", resolution=1)
    s = maze.cell_states((1, 1))[0]
    np.testing.assert_allclose(maze.xy[s], [0.0, 0.0])  # OGBench: x = 4j - 4, y = 4i - 4
    s = maze.cell_states((6, 6))[0]
    np.testing.assert_allclose(maze.xy[s], [20.0, 20.0])


def test_state_index_inverts_xy_for_all_states():
    maze = GridMaze("large", resolution=2)
    np.testing.assert_array_equal(maze.state_index(maze.xy), np.arange(maze.n_states))


def test_state_index_rejects_wall_positions():
    maze = GridMaze("medium", resolution=1)
    with pytest.raises(ValueError, match="wall"):
        maze.state_index(np.array([[-4.0, -4.0]]))  # cell (0, 0) is the outer wall


def test_transition_probabilities_sum_to_one():
    for name in ("medium", "teleport"):
        maze = GridMaze(name, resolution=2)
        np.testing.assert_allclose(maze.probs.sum(-1), 1.0)


def test_move_into_wall_keeps_the_agent_in_place():
    maze = GridMaze("medium", resolution=1)
    s = maze.cell_states((1, 1))[0]  # wall at (0, 1) below and (1, 0) to the left
    minus_y = 4
    assert maze.next_states[s, minus_y, 0] == s


def test_move_into_free_cell_changes_state():
    maze = GridMaze("medium", resolution=1)
    s = maze.cell_states((1, 1))[0]
    plus_x = 1
    assert maze.next_states[s, plus_x, 0] == maze.cell_states((1, 2))[0]


def test_teleport_sends_the_agent_to_one_of_three_exits():
    maze = GridMaze("teleport", resolution=1)
    entry_neighbour = maze.cell_states((3, 6))[0]  # (4, 6) is a teleport entry
    plus_y = 3
    exits = {maze.cell_states(ij)[0] for ij in [(1, 7), (6, 1), (6, 10)]}
    assert set(maze.next_states[entry_neighbour, plus_y].tolist()) == exits
    np.testing.assert_allclose(maze.probs[entry_neighbour, plus_y], 1 / 3)


def test_geodesic_distance_counts_moves_on_the_fine_grid():
    maze = GridMaze("medium", resolution=2)
    a, b = maze.cell_states((1, 1))[0], maze.cell_states((1, 2))[0]
    dist = maze.geodesic_distances()
    assert dist[a, a] == 0
    assert dist[a, b] == dist[b, a] == 2  # two fine steps per maze cell at resolution 2


def test_action_from_vector_inverts_action_vectors():
    maze = GridMaze("medium", resolution=1)
    np.testing.assert_array_equal(maze.action_from_vector(ACTION_VECTORS), np.arange(5))


def test_small_action_vector_means_stay():
    maze = GridMaze("medium", resolution=1)
    assert maze.action_from_vector(np.array([[0.1, -0.2]]))[0] == 0


def test_tasks_follow_ogbench_cells():
    maze = GridMaze("medium", resolution=2)
    init_ij, goal_ij = maze.tasks[0]
    assert (init_ij, goal_ij) == ((1, 1), (6, 6))
    np.testing.assert_allclose(maze.cell_xy(goal_ij), [20.0, 20.0])
