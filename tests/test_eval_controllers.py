import numpy as np
import ogbench
import pytest

from fbplan.contracts import TaskSpec
from fbplan.controllers.baseline import BaselineController, NonHierController
from fbplan.controllers.oracle import (
    MazeGeometry,
    OracleConfig,
    OracleWaypointController,
    nearest_observations,
)

MEDIUM_MAP = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
])
MEDIUM = MazeGeometry(maze_map=MEDIUM_MAP, unit=4.0, offset_xy=(4.0, 4.0))


class _FakeFB:
    """Tiny FBModel: B(s) keeps the xy of s, so a latent shows which waypoint it came from."""

    env_name = "antmaze-medium-navigate-v0"
    latent_dim = 4
    discount = 0.99

    def __init__(self) -> None:
        self.low_latents: list[np.ndarray] = []

    def backward(self, obs):
        return np.concatenate([obs[:, :2], np.ones((len(obs), 2))], axis=1)

    def high_latent(self, obs, z_task, temperature=0.0, seed=None):
        return z_task + obs[:, :4]

    def low_action(self, obs, z, temperature=0.0, seed=None):
        self.low_latents.append(np.asarray(z)[0].copy())
        return np.tanh(z[:, :2] - obs[:, :2])


def _task(goal_xy, z_r=(0.0, 0.0, 0.0, 0.0)) -> TaskSpec:
    return TaskSpec("antmaze-medium-navigate-v0", 1, np.asarray(z_r, float), np.asarray(goal_xy))


def _obs_at(xy) -> np.ndarray:
    return np.concatenate([np.asarray(xy, float), [0.3, -0.2, 0.0, 0.0]])


def test_baseline_feeds_the_intention_of_pi_h_at_the_current_state_to_pi_l():
    model, z_r = _FakeFB(), np.array([1.0, 2.0, 3.0, 4.0])
    controller = BaselineController(model)
    controller.reset(_task([0, 0], z_r), seed=0)
    for xy in ([0.0, 0.0], [5.0, 1.0]):
        obs = _obs_at(xy)
        action = controller.act(obs)
        z = z_r + obs[:4]
        np.testing.assert_allclose(model.low_latents[-1], z)
        np.testing.assert_allclose(action, np.tanh(z[:2] - obs[:2]))


def test_nonhier_feeds_the_task_latent_to_pi_l():
    model, z_r = _FakeFB(), np.array([1.0, 2.0, 3.0, 4.0])
    controller = NonHierController(model)
    controller.reset(_task([0, 0], z_r), seed=0)
    controller.act(_obs_at([5.0, 1.0]))
    np.testing.assert_allclose(model.low_latents[-1], z_r)


def test_maze_geometry_matches_ogbench_cell_conversions():
    env = ogbench.make_env_and_datasets("antmaze-medium-navigate-v0", env_only=True)
    maze = MazeGeometry.from_env(env)
    np.testing.assert_array_equal(maze.maze_map, MEDIUM_MAP)
    xys = np.random.default_rng(0).uniform(-6, 26, size=(200, 2))
    assert [maze.cell_of(xy) for xy in xys] == [env.unwrapped.xy_to_ij(xy) for xy in xys]
    for cell in maze.free_cells():
        np.testing.assert_allclose(maze.centre(cell), env.unwrapped.ij_to_xy(cell))


def test_bfs_steps_give_shortest_cell_distance_and_mark_walls():
    steps = MEDIUM.bfs_steps((6, 6))
    assert steps[6, 6] == 0
    assert steps[1, 1] == 10  # (1,1) (2,1) (2,2) (3,2) (3,3) (3,4) (4,4) (4,5) (4,6) (5,6) (6,6)
    assert steps[0, 0] == -1


def test_path_follows_free_adjacent_cells_to_the_goal():
    path = MEDIUM.path((1, 1), (6, 6))
    assert path[0] == (1, 1) and path[-1] == (6, 6) and len(path) == 11
    for (i, j), (k, m) in zip(path, path[1:], strict=False):
        assert abs(i - k) + abs(j - m) == 1 and MEDIUM_MAP[k, m] == 0


def test_nearest_observations_use_xy_only():
    obs = np.array([[0.0, 0.0, 100.0], [1.0, 1.0, 0.0], [5.0, 5.0, 0.0]])
    picked = nearest_observations(obs, np.array([[0.2, 0.1], [4.0, 4.0]]))
    np.testing.assert_array_equal(picked, obs[[0, 2]])


def _dataset_at_cell_centres() -> np.ndarray:
    """Observations 0.1 off each free cell centre, so a waypoint shows its cell."""
    return np.stack([_obs_at(MEDIUM.centre(c) + 0.1) for c in MEDIUM.free_cells()])


@pytest.mark.parametrize("lookahead", [1, 3])
def test_oracle_targets_the_waypoint_lookahead_cells_ahead_on_the_shortest_path(lookahead):
    model = _FakeFB()
    oracle = OracleWaypointController(
        model, MEDIUM, _dataset_at_cell_centres(), OracleConfig(lookahead=lookahead)
    )
    oracle.reset(_task(MEDIUM.centre((6, 6))), seed=0)
    oracle.act(_obs_at(MEDIUM.centre((1, 1)) + [0.5, -0.5]))
    target = MEDIUM.path((1, 1), (6, 6))[lookahead]
    np.testing.assert_allclose(model.low_latents[-1][:2], MEDIUM.centre(target) + 0.1)
    assert oracle.step_info() == {"target_i": target[0], "target_j": target[1]}


def test_oracle_lookahead_stops_at_the_goal_cell():
    model = _FakeFB()
    oracle = OracleWaypointController(
        model, MEDIUM, _dataset_at_cell_centres(), OracleConfig(lookahead=5)
    )
    oracle.reset(_task(MEDIUM.centre((6, 6))), seed=0)
    oracle.act(_obs_at(MEDIUM.centre((5, 6))))
    np.testing.assert_allclose(model.low_latents[-1][:2], MEDIUM.centre((6, 6)) + 0.1)
