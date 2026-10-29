from dataclasses import replace

import numpy as np
import pytest
from planner_fakes import GridFB

from fbplan.contracts import TaskSpec
from fbplan.controllers.planner import (
    GraphPlannerController,
    GraphPlusHighController,
    PlannerConfig,
    config_from_options,
)
from fbplan.landmarks import LandmarkConfig
from fbplan.option_model import OptionModelConfig

START, GOAL = (1, 4), (5, 4)  # the two bottom ends of the U maze, 10 steps apart
# With horizon 4, edges are trusted up to 4 steps (gamma^4 = 0.656); a waypoint counts as
# reached only on its own cell (adjacent cells have rho = 0.9). A small cost per hop makes
# the route take as few hops as it can, so it is unique.
HORIZON_CFG = PlannerConfig(rho_min=0.65, rho_reached=0.95, hop_discount=0.99, commit_timeout=20)


def task(fb: GridFB, task_id: int = 1) -> TaskSpec:
    # goal_xy is NaN: the planner must not read privileged information.
    return TaskSpec(fb.env_name, task_id, fb.infer_task_latent(task_id), np.full(2, np.nan))


def all_cells(fb: GridFB) -> np.ndarray:
    return np.stack([fb.obs(c) for c in fb.cells])


def planner(fb, config=HORIZON_CFG, cls=GraphPlannerController):
    ctrl = cls(fb, all_cells(fb), config)
    ctrl.reset(task(fb), seed=0)
    return ctrl


def rollout(fb, ctrl, steps: int, start=START) -> tuple[list[tuple[int, int]], list[int]]:
    """Cells visited before each step and the waypoint that each step used."""
    obs, cells, waypoints = fb.obs(start), [], []
    for _ in range(steps):
        cells.append((int(obs[0]), int(obs[1])))
        obs = fb.step(obs, ctrl.act(obs))
        waypoints.append(ctrl.step_info()["waypoint"])
    cells.append((int(obs[0]), int(obs[1])))
    return cells, waypoints


class StandStill:
    """Wraps an FBModel whose low-level policy never moves."""

    def __init__(self, fb: GridFB) -> None:
        self.fb = fb

    def __getattr__(self, name: str):
        return getattr(self.fb, name)

    def low_action(self, obs, z, temperature=0.0, seed=None):
        return np.zeros((len(obs), 2), dtype=np.float32)


class CountingForward:
    """Wraps an FBModel and counts the rows that go through F."""

    def __init__(self, fb: GridFB) -> None:
        self.fb, self.rows = fb, 0

    def __getattr__(self, name: str):
        return getattr(self.fb, name)

    def forward(self, obs, z):
        self.rows += len(obs)
        return self.fb.forward(obs, z)


def test_reset_solves_the_graph_for_the_task():
    fb = GridFB()  # exact: no route beats the exact task policy, so V_H = direct value
    ctrl = planner(fb, PlannerConfig(rho_min=0.0))
    expected = 3.0 * fb.discount ** fb.dist[:, fb.index[GOAL]] / (1 - fb.discount)
    np.testing.assert_allclose(ctrl.plan.solution.values, expected, rtol=1e-9)
    assert ctrl.plan.solution.converged


def test_task_policy_alone_cannot_reach_a_goal_beyond_the_horizon():
    fb = GridFB(horizon=4)
    action = fb.low_action(fb.obs(START)[None], fb.infer_task_latent(1)[None])
    assert not action.any()


def test_planner_reaches_a_goal_beyond_the_horizon_through_waypoints():
    fb = GridFB(horizon=4)
    cells, waypoints = rollout(fb, planner(fb), steps=12)
    assert cells.index(GOAL) == 10  # the shortest path
    # 10 steps with edges of at most 4 steps need 2 waypoints; the hop cost forbids more.
    assert len({w for w in waypoints if w >= 0}) == 2


def test_waypoint_is_kept_until_it_is_reached():
    fb = GridFB(horizon=4)
    cells, waypoints = rollout(fb, planner(fb), steps=10)
    for t in range(1, len(waypoints)):
        if waypoints[t] != waypoints[t - 1]:
            assert cells[t] == fb.cells[waypoints[t - 1]]


def test_waypoint_is_dropped_after_the_timeout():
    fb = StandStill(GridFB(horizon=4))
    ctrl = planner(fb, PlannerConfig(rho_min=0.65, rho_reached=0.95, commit_timeout=3))
    rollout(fb, ctrl, steps=7)
    assert ctrl.step_info()["replans"] == 3  # at steps 0, 3 and 6


def test_a_reached_landmark_is_not_chosen_again():
    # Without a hop cost, the landmark under the agent has Q = 1 * V_H = the best Q, so only
    # the exclusion stops the planner from choosing the landmark it already stands on.
    fb = GridFB(horizon=4)
    ctrl = planner(fb, replace(HORIZON_CFG, hop_discount=1.0))
    ctrl.act(fb.obs((2, 1)))
    assert fb.cells[ctrl.step_info()["waypoint"]] != (2, 1)


def test_without_a_trusted_edge_the_task_policy_acts():
    fb = GridFB()
    ctrl = planner(fb, PlannerConfig(rho_min=1.0))
    obs = fb.obs(START)
    action = ctrl.act(obs)
    assert ctrl.step_info()["waypoint"] == -1
    np.testing.assert_array_equal(
        action, fb.low_action(obs[None], fb.infer_task_latent(1)[None])[0]
    )


def test_graph_plus_high_routes_the_waypoint_through_pi_h():
    fb = GridFB(horizon=4)
    plain, _ = rollout(fb, planner(fb), steps=10)
    via_high, _ = rollout(fb, planner(fb, cls=GraphPlusHighController), steps=10)
    assert via_high == plain  # pi_h is the identity in the fake
    assert fb.high_calls == 10


def test_edge_table_is_computed_once_per_env_and_plans_once_per_task():
    fb = CountingForward(GridFB(goals={1: (5, 4), 2: (3, 1)}))
    n = fb.latent_dim
    cfg = PlannerConfig(options=OptionModelConfig(edge_reward="zero"))
    ctrl = GraphPlannerController(fb, all_cells(fb), cfg)
    ctrl.reset(task(fb, 1), seed=0)
    rows = fb.rows
    ctrl.reset(task(fb, 2), seed=0)
    assert 0 < fb.rows - rows < n * n  # only the direct values of the new task
    rows = fb.rows
    ctrl.reset(task(fb, 1), seed=1)
    assert fb.rows == rows


def test_from_dataset_selects_landmarks_with_the_config():
    fb = GridFB()
    cfg = PlannerConfig(landmarks=LandmarkConfig(count=5))
    ctrl = GraphPlannerController.from_dataset(fb, np.tile(all_cells(fb), (3, 1)), cfg)
    assert len(ctrl.landmarks) == 5


def test_route_lists_the_waypoints_that_the_rollout_visits():
    fb = GridFB(horizon=4)
    ctrl = planner(fb)
    route = ctrl.route(fb.obs(START))
    _, waypoints = rollout(fb, ctrl, steps=10)
    assert route == [w for w in dict.fromkeys(waypoints) if w >= 0]


def test_route_is_empty_when_the_task_policy_is_best():
    fb = GridFB(horizon=4)
    assert planner(fb).route(fb.obs((5, 3))) == []


def test_config_from_options_sets_top_level_and_nested_fields():
    cfg = config_from_options({"rho_min": 0.5, "landmarks.count": 40, "options.head_agg": "mean"})
    assert (cfg.rho_min, cfg.landmarks.count, cfg.options.head_agg) == (0.5, 40, "mean")
    assert cfg.landmarks.space == PlannerConfig().landmarks.space


def test_config_from_options_rejects_unknown_keys():
    for bad in ({"rho_mn": 0.5}, {"landmark.count": 3}, {"landmarks.cnt": 3}):
        with pytest.raises(TypeError):
            config_from_options(bad)


def test_rho_at_one_is_not_a_trusted_edge():
    fb = GridFB(wormholes={(h, START, GOAL): 1.5 for h in (0, 1)})  # clipped to rho = 1
    i, j = fb.index[START], fb.index[GOAL]
    assert not planner(fb, PlannerConfig(rho_min=0.3)).trusted[i, j]
    assert planner(fb, PlannerConfig(rho_min=0.3, rho_max=np.inf)).trusted[i, j]


def test_xy_cap_removes_long_landmark_edges():
    fb = GridFB()
    ctrl = planner(fb, PlannerConfig(rho_min=0.0, max_edge_xy=1.5))
    xy = all_cells(fb)
    lengths = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    assert ctrl.trusted.any()
    assert (lengths[ctrl.trusted] <= 1.5).all()


def test_xy_cap_applies_to_the_first_hop():
    fb = GridFB(horizon=4)
    ctrl = planner(fb, replace(HORIZON_CFG, max_edge_xy=1.0))
    ctrl.act(fb.obs(START))
    waypoint = fb.obs(fb.cells[ctrl.step_info()["waypoint"]])
    assert np.linalg.norm(waypoint - fb.obs(START)) <= 1.0


# FB says the cell next to the goal is close to START (rho 0.85, below rho_reached = 0.9);
# the way back is gamma^9 = 0.39. A wormhole to GOAL itself would also raise the direct
# value, since z_r targets GOAL.
NEAR_GOAL = (5, 3)
ONE_WAY = {(h, START, NEAR_GOAL): 0.85 for h in (0, 1)}


def test_two_way_consistency_removes_a_one_way_wormhole():
    fb = GridFB(wormholes=ONE_WAY)
    i, j = fb.index[START], fb.index[NEAR_GOAL]
    cfg = PlannerConfig(rho_min=0.5, max_edge_xy=None)
    assert planner(fb, cfg).trusted[i, j]
    assert not planner(fb, replace(cfg, two_way=True)).trusted[i, j]


def test_two_way_consistency_applies_to_the_first_hop():
    fb = GridFB(wormholes=ONE_WAY)
    cfg = PlannerConfig(rho_min=0.5, max_edge_xy=None)
    one_way, two_way = planner(fb, cfg), planner(fb, replace(cfg, two_way=True))
    one_way.act(fb.obs(START))
    two_way.act(fb.obs(START))
    assert one_way.step_info()["waypoint"] == fb.index[NEAR_GOAL]
    assert two_way.step_info()["waypoint"] != fb.index[NEAR_GOAL]


def test_reached_landmarks_do_not_use_top_k_slots():
    # top_k = 2 keeps the two corridor neighbours of every cell. With rho_reached = 0.85 the
    # neighbours of (2, 1) (rho 0.9) count as reached. If they took the two slots before their
    # exclusion, no edge would be left; the next cells towards the goal must get the slots.
    fb = GridFB(horizon=4)
    ctrl = planner(fb, replace(HORIZON_CFG, top_k=2, rho_reached=0.85))
    ctrl.act(fb.obs((2, 1)))
    assert fb.cells[ctrl.step_info()["waypoint"]] == (4, 1)
