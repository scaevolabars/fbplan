"""Success within a time limit vs. the discounted value on the cell-level mazes."""

import numpy as np
import pytest

from fbplan.controllers.oracle_teleport import (
    Teleports,
    discounted_plan,
    discounted_values,
    reach_plan,
)
from fbplan.maze import MazeGeometry
from fbplan.tabular import GridMaze
from fbplan.tabular.maze import ACTION_VECTORS
from fbplan.tabular.mdp import optimal_policy, policy_value
from fbplan.tabular.time_limit import (
    best_success,
    discounted_policy,
    enters_teleporter,
    hitting_cdf,
    main,
    walk_policy,
    walk_threshold,
)

SAFE_WALK_CELLS = [23, 15, 14, 25, 27]  # teleport tasks, counted by hand
TELEPORTS = Teleports(entries=((4, 6), (5, 1)), exits=((1, 7), (6, 1), (6, 10)), trigger_xy=1.5)


@pytest.fixture(scope="module")
def teleport() -> GridMaze:
    return GridMaze("teleport", resolution=1)


def _geometry(maze: GridMaze) -> MazeGeometry:
    return MazeGeometry(maze.maze_map.astype(int), 4.0, (4.0, 4.0))


def _plan_policy(maze: GridMaze, plan: dict) -> np.ndarray:
    """GridMaze actions (resolution 1) that move each cell to its next cell in an oracle plan."""
    policy = np.zeros(maze.n_states, dtype=np.int64)
    for state, (i, j) in enumerate(maze.fine_rc):
        nxt = plan.get((int(i), int(j)), (int(i), int(j)))
        move = [nxt[1] - j, nxt[0] - i]  # (dx, dy): columns go along +x, rows along +y
        policy[state] = int(np.flatnonzero((ACTION_VECTORS == move).all(axis=1))[0])
    return policy


def _ends(maze, task_id):
    start_ij, goal_ij = maze.tasks[task_id - 1]
    return maze.centre_state(start_ij), maze.centre_state(goal_ij)


@pytest.mark.parametrize("task_id", [1, 2, 3, 4, 5])
def test_walk_reaches_the_goal_in_the_shortest_safe_walk(teleport, task_id):
    start, goal = _ends(teleport, task_id)
    policy = walk_policy(teleport, goal)
    length = SAFE_WALK_CELLS[task_id - 1]
    cdf = hitting_cdf(teleport, policy, start, goal, length)
    assert cdf[length] == 1.0
    assert cdf[length - 1] == 0.0
    assert not enters_teleporter(teleport, policy, start, goal)


def test_gamble_succeeds_with_probability_one_half(teleport):
    # Task 3, (5, 6) -> (7, 10): one move into entry (4, 6); exit (6, 10) is next to the goal,
    # (1, 7) is the trap, and from (6, 1) one move into entry (5, 1) repeats the jump.
    # So P(H = k) = 3^(1-k) for k >= 2 and P(H <= k) = (1/2)(1 - 3^(1-k)).
    start, goal = _ends(teleport, 3)
    policy = discounted_policy(teleport, goal, 0.67)
    assert enters_teleporter(teleport, policy, start, goal)
    cdf = hitting_cdf(teleport, policy, start, goal, 41)
    k = np.arange(1, 42)
    np.testing.assert_allclose(cdf[1:], 0.5 * (1 - 3.0 ** (1 - k)))


def test_discounted_values_equal_those_of_the_oracle_cell_mdp(teleport):
    # Two implementations of the same cell MDP: mdp.optimal_policy (with a stay action, reward
    # 1 at the goal, so V = gamma^H / (1 - gamma)) and the oracle's value iteration.
    gamma = 0.8
    geometry, teleports = _geometry(teleport), TELEPORTS
    for task_id in (1, 3):
        start_ij, goal_ij = teleport.tasks[task_id - 1]
        reward = np.zeros(teleport.n_states)
        reward[teleport.centre_state(goal_ij)] = 1.0
        v_tabular = optimal_policy(teleport, reward, gamma)[1] * (1 - gamma)
        v_oracle = discounted_values(geometry, teleports, goal_ij, gamma)
        for state, (i, j) in enumerate(teleport.fine_rc):
            if (i, j) not in teleports.entries:
                np.testing.assert_allclose(v_tabular[state], v_oracle[i, j], atol=1e-9)


def test_best_success_walks_when_the_walk_fits_and_gambles_when_it_does_not(teleport):
    start, goal = _ends(teleport, 5)  # walk 27 cells
    walk = hitting_cdf(teleport, walk_policy(teleport, goal), start, goal, 27)
    assert best_success(teleport, start, goal, 27) == 1.0
    assert walk[26] == 0.0
    # One jump: exit (6, 1) is next to the goal, and from exit (6, 10) the walk takes 24 moves,
    # which still fit; only the trap fails.
    np.testing.assert_allclose(best_success(teleport, start, goal, 26), 2 / 3)


@pytest.mark.parametrize("task_id", [1, 2, 3, 4, 5])
def test_best_success_dominates_walk_and_risk_neutral_policies(teleport, task_id):
    start, goal = _ends(teleport, task_id)
    for n_moves in (10, 20, 30):
        best = best_success(teleport, start, goal, n_moves)
        for policy in (walk_policy(teleport, goal), discounted_policy(teleport, goal, 0.67)):
            assert best >= hitting_cdf(teleport, policy, start, goal, n_moves)[-1] - 1e-12


def test_deterministic_maze_has_no_gamble():
    giant = GridMaze("giant", resolution=1)
    start, goal = _ends(giant, 1)
    policy = discounted_policy(giant, goal, 0.8)
    assert not enters_teleporter(giant, policy, start, goal)
    np.testing.assert_array_equal(policy, walk_policy(giant, goal))


def test_walk_threshold_solves_the_gamble_closed_form(teleport):
    # Task 3 at the threshold: gamble g^2 / (3 - g) = walk g^14 with g = gamma^steps_per_cell.
    gamma = walk_threshold(teleport, task_id=3, steps_per_cell=40)
    g = gamma**40
    np.testing.assert_allclose(g**2 / (3 - g), g**14, rtol=1e-6)
    start, goal = _ends(teleport, 3)
    assert enters_teleporter(
        teleport, discounted_policy(teleport, goal, (gamma - 1e-5) ** 40), start, goal
    )
    assert not enters_teleporter(
        teleport, discounted_policy(teleport, goal, gamma**40), start, goal
    )


def test_main_writes_its_tables_and_figure_into_a_new_folder(tmp_path, monkeypatch):
    out = tmp_path / "new" / "folder"
    monkeypatch.setattr("sys.argv", ["time_limit", "--out-dir", str(out)])
    main()
    names = {"time_limit.csv", "time_limit.png", "time_limit_thresholds.csv"}
    assert {p.name for p in out.iterdir()} == names


@pytest.mark.parametrize("task_id", [1, 2, 3, 4, 5])
def test_oracle_reach_plan_walks_as_far_as_the_walk_policy_from_every_cell(teleport, task_id):
    # The teleport oracle and this module implement the walk separately; pin them together.
    goal_ij = teleport.tasks[task_id - 1][1]
    goal = teleport.centre_state(goal_ij)
    walk = walk_policy(teleport, goal)
    oracle = _plan_policy(teleport, reach_plan(_geometry(teleport), TELEPORTS, goal_ij))
    entries = set(teleport.entry_states.tolist())
    for state in range(teleport.n_states):
        if state in entries:  # the walk policy has no plan there (the agent never stays)
            continue
        walk_cdf = hitting_cdf(teleport, walk, state, goal, teleport.n_states)
        oracle_cdf = hitting_cdf(teleport, oracle, state, goal, teleport.n_states)
        np.testing.assert_array_equal(oracle_cdf, walk_cdf)  # same length, never a jump
        if walk_cdf[-1] > 0:
            assert not enters_teleporter(teleport, oracle, state, goal)


@pytest.mark.parametrize("gamma", [0.67, 0.99])
def test_oracle_discounted_plan_is_optimal_for_the_tabular_cell_mdp(teleport, gamma):
    for task_id in (1, 3, 5):
        goal_ij = teleport.tasks[task_id - 1][1]
        goal = teleport.centre_state(goal_ij)
        reward = np.eye(teleport.n_states)[goal]
        plan = discounted_plan(_geometry(teleport), TELEPORTS, goal_ij, gamma)
        v_plan = policy_value(teleport, _plan_policy(teleport, plan), reward, gamma)
        _, v_star = optimal_policy(teleport, reward, gamma)
        other = np.setdiff1d(np.arange(teleport.n_states), teleport.entry_states)
        np.testing.assert_allclose(v_plan[other], v_star[other], rtol=1e-9, atol=1e-12)
