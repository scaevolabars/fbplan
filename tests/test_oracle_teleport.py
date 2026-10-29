"""Tests of the teleport-aware oracle on the OGBench teleport map (no MuJoCo needed)."""

import numpy as np
import pytest

from fbplan.contracts import TaskSpec
from fbplan.controllers.oracle_teleport import (
    TeleportOracleConfig,
    TeleportOracleController,
    Teleports,
    discounted_plan,
    discounted_values,
    reach_plan,
)
from fbplan.maze import MazeGeometry
from fbplan.tabular import GridMaze

TELEPORT = GridMaze("teleport", resolution=1)  # the OGBench map (tests/test_tabular_maze.py)
MAZE = MazeGeometry(maze_map=TELEPORT.maze_map.astype(int), unit=4.0, offset_xy=(4.0, 4.0))
TELEPORTS = Teleports(entries=((4, 6), (5, 1)), exits=((1, 7), (6, 1), (6, 10)), trigger_xy=1.5)
TRAP = (1, 7)  # an exit with walls on all four sides
TASKS = [(tuple(a), tuple(b)) for a, b in TELEPORT.tasks]


def _follow(plan, start, goal, max_moves=100):
    path = [start]
    while path[-1] != goal and len(path) <= max_moves:
        path.append(plan[path[-1]])
    return path


def test_plain_bfs_path_enters_a_teleporter_on_four_tasks():
    entered = [
        any(cell in TELEPORTS.entries for cell in MAZE.path(start, goal)) for start, goal in TASKS
    ]
    assert entered == [True, False, True, True, True]


@pytest.mark.parametrize("task", TASKS)
def test_reach_plan_walks_to_the_goal_in_unit_moves_without_teleporters(task):
    # Its length equals the tabular walk policy's on every cell: tests/test_tabular_time_limit.py.
    start, goal = task
    path = _follow(reach_plan(MAZE, TELEPORTS, goal), start, goal)
    assert path[-1] == goal
    assert not set(path) & set(TELEPORTS.entries)
    moves = zip(path[:-1], path[1:], strict=True)
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in moves)


@pytest.mark.parametrize("goal", [(7, 1), (7, 10)])
def test_reach_plan_leads_every_cell_but_the_trap_to_the_goal(goal):
    plan = reach_plan(MAZE, TELEPORTS, goal)
    assert TRAP not in plan
    for cell in MAZE.free_cells():
        if cell == TRAP:
            continue
        path = _follow(plan, cell, goal)
        assert path[-1] == goal
        assert not set(path[1:]) & set(TELEPORTS.entries)  # an entry cell is left at once


def test_discounted_value_of_the_gamble_has_the_closed_form():
    # From (5, 6) with goal (7, 10): step into entry (4, 6); exits (6, 10) -> goal in 1 move,
    # (1, 7) trap, (6, 1) -> entry (5, 1) and gamble again. V = g (g + 0 + V) / 3 at the cells
    # next to an entry, so V = g^2 / (3 - g), while the walk is worth g^14.
    g = 0.67
    values = discounted_values(MAZE, TELEPORTS, (7, 10), g)
    np.testing.assert_allclose(values[5, 6], g**2 / (3 - g), rtol=1e-10)
    np.testing.assert_allclose(values[6, 1], g**2 / (3 - g), rtol=1e-10)
    np.testing.assert_allclose(values[6, 10], g, rtol=1e-10)
    assert values[TRAP] == 0.0


def test_discounted_plan_gambles_at_a_low_cell_discount_and_walks_at_a_high_one():
    assert discounted_plan(MAZE, TELEPORTS, (7, 10), 0.67)[(5, 6)] == (4, 6)
    assert discounted_plan(MAZE, TELEPORTS, (7, 10), 0.99)[(5, 6)] == (6, 6)


class _FakeFB:
    """B(s) keeps the xy of s, so the latent given to pi_l shows the waypoint region."""

    env_name = "antmaze-teleport-navigate-v0"
    latent_dim = 3
    discount = 0.99

    def __init__(self) -> None:
        self.low_latents: list[np.ndarray] = []

    def backward(self, obs):
        return np.concatenate([obs[:, :2], np.ones((len(obs), 1))], axis=1)

    def high_latent(self, obs, z_task):
        return z_task + 100.0  # marks that pi_h picked the intention

    def low_action(self, obs, z):
        self.low_latents.append(np.asarray(z)[0].copy())
        return np.tanh(z[:, :2] - obs[:, :2])


def _dataset() -> np.ndarray:
    """States at each free cell centre, plus states 1.2 from each entry centre (inside the
    trigger disk but not within 0.5 of the centre, as in the real data)."""
    centres = [MAZE.centre(c) for c in MAZE.free_cells() if c not in TELEPORTS.entries]
    near_entries = [MAZE.centre(e) + [1.2, 0.0] for e in TELEPORTS.entries]
    xy = np.array(centres + near_entries)
    return np.concatenate([xy, np.zeros((len(xy), 3))], axis=1)


def _controller(**options) -> tuple[TeleportOracleController, _FakeFB]:
    model = _FakeFB()
    config = TeleportOracleConfig(**options)
    return TeleportOracleController(model, MAZE, TELEPORTS, _dataset(), config), model


def _act_at(controller, cell):
    obs = np.concatenate([MAZE.centre(cell), np.zeros(3)])
    controller.act(obs)
    info = controller.step_info()
    return info["target_i"], info["target_j"]


def _task(goal) -> TaskSpec:
    return TaskSpec("antmaze-teleport-navigate-v0", 3, np.zeros(3), MAZE.centre(goal))


def test_controller_targets_the_cell_lookahead_moves_ahead_on_the_walk():
    controller, model = _controller(lookahead=2)
    controller.reset(_task((7, 10)), seed=0)
    assert _act_at(controller, (5, 6)) == (7, 6)  # down and around, not up into (4, 6)
    np.testing.assert_allclose(model.low_latents[-1][:2], MAZE.centre((7, 6)))


def test_discounted_controller_stops_the_waypoint_at_the_entry():
    controller, model = _controller(objective="discounted", steps_per_cell=40, lookahead=2)
    controller.reset(_task((7, 10)), seed=0)
    assert _act_at(controller, (5, 6)) == (4, 6)
    np.testing.assert_allclose(model.low_latents[-1][:2], MAZE.centre((4, 6)) + [1.2, 0.0])


def test_controller_can_give_the_waypoint_to_pi_h():
    controller, model = _controller(follow="pi_h")
    controller.reset(_task((7, 10)), seed=0)
    _act_at(controller, (5, 6))
    np.testing.assert_allclose(model.low_latents[-1][:2], MAZE.centre((7, 6)) + 100.0)


def test_controller_acts_in_the_trap():
    controller, _ = _controller()
    controller.reset(_task((7, 10)), seed=0)
    assert _act_at(controller, TRAP) == TRAP


def test_teleports_must_reach_the_dataset():
    with pytest.raises(ValueError, match="no dataset state"):
        TeleportOracleController(_FakeFB(), MAZE, TELEPORTS, _dataset()[:-2])


def test_unknown_objective_is_rejected():
    with pytest.raises(ValueError, match="objective"):
        TeleportOracleConfig(objective="fast")


def test_without_teleporters_the_reach_plan_is_the_plain_oracle_plan():
    medium = MazeGeometry(GridMaze("medium", 1).maze_map.astype(int), 4.0, (4.0, 4.0))
    none = Teleports(entries=(), exits=(), trigger_xy=1.5)
    for goal in medium.free_cells():
        steps = medium.bfs_steps(goal)
        plan = reach_plan(medium, none, goal)
        assert plan == {cell: medium.next_cell(cell, steps) for cell in medium.free_cells()}


def test_value_iteration_raises_when_it_does_not_converge():
    with pytest.raises(RuntimeError, match="converge"):
        discounted_values(MAZE, TELEPORTS, (7, 10), cell_discount=0.9, max_iters=2)
