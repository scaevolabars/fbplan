import numpy as np
import pytest

from fbplan.contracts import TaskSpec
from fbplan.controllers import resolve_controller
from fbplan.controllers.baseline import (
    BaselineController,
    EnsembleBaselineController,
    EnsembleConfig,
    NonHierController,
    RawBaselineController,
)
from fbplan.controllers.oracle import (
    OracleConfig,
    OracleWaypointController,
    make_oracle,
    nearest_observations,
    region_latents,
)
from fbplan.frozen import make_env
from fbplan.maze import MazeGeometry

MEDIUM_ENV = make_env("antmaze-medium-navigate-v0")
MEDIUM = MazeGeometry.from_env(MEDIUM_ENV)


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


class _FakeUpstreamAgent:
    """Records the calls of upstream FBpiSwitchAgent.sample_actions."""

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray, float]] = []

    def sample_actions(self, observations, latents=None, seed=None, temperature=1.0):
        self.calls.append((np.array(observations), np.array(latents), temperature))
        return np.tanh(latents[:2] - observations[:2])


def test_raw_baseline_passes_the_task_latent_unnormalized_to_upstream_sample_actions():
    model, z_r = _FakeFB(), np.array([0.1, 0.2, 0.3, 0.4])  # norm far from sqrt(d)
    model.agent = _FakeUpstreamAgent()
    controller = RawBaselineController(model)
    controller.reset(_task([0, 0], z_r), seed=0)
    obs = _obs_at([5.0, 1.0])
    action = controller.act(obs)
    ((obs_in, z_in, temperature),) = model.agent.calls
    np.testing.assert_array_equal(obs_in, obs)
    np.testing.assert_array_equal(z_in, z_r)
    assert temperature == 0.0
    np.testing.assert_allclose(action, np.tanh(z_r[:2] - obs[:2]))


class _FixedIntention(_FakeFB):
    """A pi_h member that always proposes the same normalized intention."""

    def __init__(self, intention) -> None:
        super().__init__()
        self.intention = np.asarray(intention, float)
        self.tasks: list[np.ndarray] = []

    def high_latent(self, obs, z_task, temperature=0.0, seed=None):
        self.tasks.append(np.asarray(z_task)[0].copy())
        return self.intention[None]


def test_ensemble_acts_on_the_normalized_mean_of_the_members_intentions():
    model, z_r = _FakeFB(), np.array([1.0, 2.0, 3.0, 4.0])
    members = [_FixedIntention([2.0, 0, 0, 0]), _FixedIntention([0, 2.0, 0, 0])]  # norm sqrt(d)
    controller = EnsembleBaselineController(model, members, EnsembleConfig("retrained", 2))
    controller.reset(_task([0, 0], z_r), seed=0)
    controller.act(_obs_at([5.0, 1.0]))
    np.testing.assert_allclose(model.low_latents[-1], [np.sqrt(2), np.sqrt(2), 0, 0])
    for member in members:
        np.testing.assert_array_equal(member.tasks[-1], z_r)
        assert member.low_latents == []  # only the shared model's pi_l acts


def test_maze_geometry_matches_ogbench_cell_conversions():
    xys = np.random.default_rng(0).uniform(-6, 26, size=(200, 2))
    assert [MEDIUM.cell_of(xy) for xy in xys] == [MEDIUM_ENV.unwrapped.xy_to_ij(xy) for xy in xys]
    for cell in MEDIUM.free_cells():
        np.testing.assert_allclose(MEDIUM.centre(cell), MEDIUM_ENV.unwrapped.ij_to_xy(cell))


def test_bfs_steps_give_shortest_cell_distance_and_mark_walls():
    steps = MEDIUM.bfs_steps((6, 6))
    assert steps[6, 6] == 0
    assert steps[1, 1] == 10  # (1,1) (2,1) (2,2) (3,2) (3,3) (3,4) (4,4) (4,5) (4,6) (5,6) (6,6)
    assert steps[0, 0] == -1


def test_path_follows_free_adjacent_cells_to_the_goal():
    path = MEDIUM.path((1, 1), (6, 6))
    assert path[0] == (1, 1) and path[-1] == (6, 6) and len(path) == 11
    for (i, j), (k, m) in zip(path, path[1:], strict=False):
        assert abs(i - k) + abs(j - m) == 1 and MEDIUM.maze_map[k, m] == 0


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
        model, MEDIUM, _dataset_at_cell_centres(),
        OracleConfig(lookahead=lookahead, waypoint="state"),
    )
    oracle.reset(_task(MEDIUM.centre((6, 6))), seed=0)
    oracle.act(_obs_at(MEDIUM.centre((1, 1)) + [0.5, -0.5]))
    target = MEDIUM.path((1, 1), (6, 6))[lookahead]
    np.testing.assert_allclose(model.low_latents[-1][:2], MEDIUM.centre(target) + 0.1)
    assert oracle.step_info() == {"target_i": target[0], "target_j": target[1]}


def test_oracle_lookahead_stops_at_the_goal_cell():
    model = _FakeFB()
    oracle = OracleWaypointController(
        model, MEDIUM, _dataset_at_cell_centres(), OracleConfig(lookahead=5, waypoint="state")
    )
    oracle.reset(_task(MEDIUM.centre((6, 6))), seed=0)
    oracle.act(_obs_at(MEDIUM.centre((5, 6))))
    np.testing.assert_allclose(model.low_latents[-1][:2], MEDIUM.centre((6, 6)) + 0.1)


def test_resolve_controller_accepts_aliases_and_module_paths():
    assert resolve_controller("baseline") is BaselineController
    assert resolve_controller("oracle") is make_oracle
    assert resolve_controller("fbplan.controllers.baseline:NonHierController") is NonHierController


def test_resolve_controller_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown controller"):
        resolve_controller("planner-that-does-not-exist")


def test_geodesic_distance_is_euclidean_inside_an_open_area():
    a, b = MEDIUM.centre((1, 1)), MEDIUM.centre((2, 2))  # the 2x2 open block at the top left
    np.testing.assert_allclose(MEDIUM.geodesic_distances(a[None], b[None]), [[np.hypot(4, 4)]])


def test_geodesic_distance_bends_at_wall_corners():
    # From (4, 0) around wall cell (2, 3) [x 6..10, y 2..6] to (12, 4): via corners (6, 6), (10, 6).
    a, b = MEDIUM.centre((1, 2)), MEDIUM.centre((2, 4))
    expected = np.hypot(2, 6) + 4 + np.hypot(2, 2)
    np.testing.assert_allclose(MEDIUM.geodesic_distances(a[None], b[None]), [[expected]])


def test_geodesic_distances_are_symmetric_and_bounded_by_cell_paths():
    centres = np.stack([MEDIUM.centre(c) for c in MEDIUM.free_cells()])
    dist = MEDIUM.geodesic_distances(centres, centres)
    np.testing.assert_allclose(dist, dist.T, atol=1e-9)
    euclid = np.linalg.norm(centres[:, None] - centres[None], axis=-1)
    cell_steps = np.stack([MEDIUM.bfs_steps(c)[tuple(zip(*MEDIUM.free_cells(), strict=True))]
                           for c in MEDIUM.free_cells()])
    assert np.all(dist >= euclid - 1e-9)
    assert np.all(dist <= MEDIUM.unit * cell_steps + 1e-9)


# Wall cells (2, 2) and (3, 3) touch only at a corner: a pinch the ant cannot pass.
PINCH = MazeGeometry(
    maze_map=np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 1, 1],
        [1, 1, 1, 1, 1],
    ]),
    unit=4.0,
    offset_xy=(4.0, 4.0),
)


def test_geodesic_distance_does_not_pass_between_walls_that_touch_at_a_corner():
    a, b = PINCH.centre((2, 3)), PINCH.centre((3, 2))  # 4 sqrt(2) apart through the pinch
    # around wall (2, 2): over its top edge and down its left edge
    expected = 8 + 4 * np.sqrt(2)
    np.testing.assert_allclose(PINCH.geodesic_distances(a[None], b[None]), [[expected]])


def test_geodesic_distance_is_infinite_when_only_a_pinch_joins_two_cells():
    maze = MazeGeometry(np.array([[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]]),
                        4.0, (4.0, 4.0))
    a, b = maze.centre((1, 1)), maze.centre((2, 2))
    assert np.isinf(maze.geodesic_distances(a[None], b[None])).all()


def test_region_latents_average_b_over_states_within_the_radius():
    obs = np.stack([_obs_at(xy) for xy in ([0.2, 0.0], [-0.2, 0.4], [3.0, 3.0], [8.0, 0.1])])
    latents = region_latents(_FakeFB(), obs, np.array([[0.0, 0.0], [8.0, 0.0]]), radius=1.0)
    np.testing.assert_allclose(latents[:, :2], [[0.0, 0.2], [8.0, 0.1]])


def test_region_latents_reject_a_region_without_states():
    with pytest.raises(ValueError, match="no dataset state"):
        region_latents(_FakeFB(), _obs_at([0.0, 0.0])[None], np.array([[5.0, 5.0]]), radius=1.0)


def test_region_oracle_uses_the_mean_latent_of_the_target_cell():
    centre = MEDIUM.centre((2, 1))
    extra = np.stack([_obs_at(centre + d) for d in ([0.6, 0.0], [-0.2, 0.0])])
    model = _FakeFB()
    oracle = OracleWaypointController(
        model, MEDIUM, np.concatenate([_dataset_at_cell_centres(), extra]),
        OracleConfig(lookahead=1, waypoint="region", radius=1.0),
    )
    oracle.reset(_task(MEDIUM.centre((6, 6))), seed=0)
    oracle.act(_obs_at(MEDIUM.centre((1, 1))))
    expected = np.mean([centre + 0.1, centre + [0.6, 0.0], centre + [-0.2, 0.0]], axis=0)
    np.testing.assert_allclose(model.low_latents[-1][:2], expected)
