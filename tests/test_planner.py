from dataclasses import replace

import numpy as np
import pytest
from planner_fakes import GOAL, START, GridFB, navigate_dataset

from fbplan.contracts import TaskSpec
from fbplan.controllers.baseline import NonHierController
from fbplan.controllers.planner import (
    GraphPlannerController,
    GraphPlusHighController,
    PlannerConfig,
    RecordedRouteConfig,
    RecordedRouteController,
    config_from_options,
    trusted_edges,
)
from fbplan.graph import DIRECT, GraphSolution, value_iteration
from fbplan.landmarks import LandmarkConfig, RegionField, region_latents
from fbplan.option_model import OptionModelConfig
from fbplan.tabular.env import rollout as rollout_tabular
from fbplan.tabular.fake_fb import FakeFBConfig, TabularFB

# With horizon 4, edges are trusted up to 4 steps (gamma^4 = 0.656); a waypoint counts as
# reached only on its own cell (cells are 1 xy unit apart). A small cost per hop makes the
# route take as few hops as it can, so it is unique. These tests check the plain commitment
# rule, so look-ahead is off; the look-ahead tests turn it on.
HORIZON_CFG = PlannerConfig(
    rho_min=0.65, reached_xy=0.5, hop_discount=0.99, commit_timeout=20, lookahead=False
)
# The same with the rho rule: only the waypoint's own cell has rho >= 0.95 (neighbours: 0.9).
RHO_RULE_CFG = replace(HORIZON_CFG, reached_xy=None, rho_reached=0.95)


def task(fb: GridFB, task_id: int = 1) -> TaskSpec:
    # goal_xy is NaN: the planner must not read privileged information.
    return TaskSpec(fb.env_name, task_id, fb.infer_task_latent(task_id), np.full(2, np.nan))


def planner(fb, config=HORIZON_CFG, cls=GraphPlannerController):
    ctrl = cls(fb, fb.all_obs(), config)
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


def cell_of(fb: GridFB, waypoint: int) -> tuple[int, int]:
    """The cell of a landmark waypoint (every cell is a landmark). DIRECT (-1) would index the
    last cell, so a wrong fallback to the task policy could pass: it fails here instead."""
    assert waypoint != DIRECT
    return fb.cells[waypoint]


class StandStill:
    """Wraps an FBModel whose low-level policy never moves."""

    def __init__(self, fb: GridFB) -> None:
        self.fb = fb

    def __getattr__(self, name: str):
        return getattr(self.fb, name)

    def low_action(self, obs, z):
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


def test_a_task_whose_value_iteration_does_not_converge_warns_and_plans_with_the_last_sweep():
    fb = GridFB(horizon=4)  # waypoints add value beyond the horizon: one sweep is no fixed point
    cfg = replace(HORIZON_CFG, vi_max_iters=1)
    ctrl = GraphPlannerController(fb, fb.all_obs(), cfg)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        ctrl.reset(task(fb), seed=0)
    direct = ctrl.options.direct_value(ctrl.landmarks, task(fb).z_r)
    one_sweep = value_iteration(
        ctrl.edge_rho, None, direct, ctrl.trusted, cfg.hop_discount, 1, cfg.vi_tol
    )
    assert not ctrl.plan.solution.converged
    np.testing.assert_array_equal(ctrl.plan.solution.values, one_sweep.values)
    assert ctrl.act(fb.all_obs()[0]).shape == (2,)


@pytest.mark.parametrize("cfg", [HORIZON_CFG, RHO_RULE_CFG], ids=["xy_rule", "rho_rule"])
def test_waypoint_is_kept_until_it_is_reached(cfg):
    fb = GridFB(horizon=4)
    cells, waypoints = rollout(fb, planner(fb, cfg), steps=10)
    assert GOAL in cells
    for t in range(1, len(waypoints)):
        if waypoints[t] != waypoints[t - 1]:
            assert cells[t] == cell_of(fb, waypoints[t - 1])


def test_xy_rule_ends_a_waypoint_within_the_radius():
    fb = GridFB(horizon=4)
    cells, waypoints = rollout(fb, planner(fb, replace(HORIZON_CFG, reached_xy=1.0)), steps=14)
    assert GOAL in cells
    switches = [t for t in range(1, len(waypoints)) if waypoints[t] != waypoints[t - 1]]
    assert switches
    for t in switches:
        gap = np.subtract(cells[t], cell_of(fb, waypoints[t - 1]))
        assert 0 < np.linalg.norm(gap) <= 1.0  # one cell before the waypoint


def test_waypoint_is_dropped_after_the_timeout():
    fb = StandStill(GridFB(horizon=4))
    ctrl = planner(fb, PlannerConfig(rho_min=0.65, commit_timeout=3))
    rollout(fb, ctrl, steps=7)
    assert ctrl.step_info()["replans"] == 3  # at steps 0, 3 and 6


@pytest.mark.parametrize(
    "rule", [{"reached_xy": 1.0}, {"reached_xy": None, "rho_reached": 0.85}], ids=["xy", "rho"]
)
def test_reached_landmarks_are_not_chosen_again(rule):
    # Both rules make the neighbours of (2, 1) count as reached (1 xy unit away, rho = 0.9).
    # Without a hop cost every landmark on the shortest path ties, (3, 1) has the lowest index
    # among them, and only the exclusion stops the planner from choosing a reached landmark,
    # which would stall the agent. (2, 1) itself has rho = 1 and is not a trusted edge anyway.
    fb = GridFB(horizon=4)
    ctrl = planner(fb, replace(HORIZON_CFG, hop_discount=1.0, **rule))
    ctrl.act(fb.obs((2, 1)))
    assert cell_of(fb, ctrl.step_info()["waypoint"]) not in {(1, 1), (2, 1), (3, 1)}


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
    ctrl = GraphPlannerController(fb, fb.all_obs(), cfg)
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
    ctrl = GraphPlannerController.from_dataset(fb, np.tile(fb.all_obs(), (3, 1)), cfg)
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
    assert cfg.landmarks.pool_size == PlannerConfig().landmarks.pool_size


def test_config_from_options_rejects_unknown_keys():
    for bad in ({"rho_mn": 0.5}, {"landmark.count": 3}, {"landmarks.cnt": 3}):
        with pytest.raises(TypeError):
            config_from_options(bad)


def test_trusted_edges_apply_the_rho_bounds_of_the_config():
    rho, xy = np.array([[0.2, 0.5, 0.99, 1.0]]), np.zeros((4, 2))
    cfg = PlannerConfig(rho_min=0.5)
    np.testing.assert_array_equal(trusted_edges(rho, xy[:1], xy, cfg), [[0, 1, 1, 0]])
    loose = replace(cfg, rho_max=np.inf)
    np.testing.assert_array_equal(trusted_edges(rho, xy[:1], xy, loose), [[0, 1, 1, 1]])


def test_trusted_edges_cap_the_xy_length_and_can_drop_self_loops():
    xy = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])  # neighbours 5 xy apart, ends 10
    mask = trusted_edges(np.full((3, 3), 0.8), xy, xy, PlannerConfig(), exclude_diagonal=True)
    np.testing.assert_array_equal(mask, [[0, 1, 0], [1, 0, 1], [0, 1, 0]])


def test_trusted_edges_use_the_planner_defaults_without_a_config():
    xy = np.array([[0.0, 0.0], [0.0, 0.0], [7.0, 0.0]])  # the third is beyond the 6 xy cap
    mask = trusted_edges(np.array([[0.29, 0.31, 0.31]]), xy[:1], xy)
    np.testing.assert_array_equal(mask, [[0, 1, 0]])


def test_a_raw_rho_at_or_above_one_is_not_a_trusted_edge():
    fb = GridFB(wormholes={(h, START, GOAL): 1.5 for h in (0, 1)})  # raw 1.5, valued at gamma
    i, j = fb.index[START], fb.index[GOAL]
    assert not planner(fb, PlannerConfig(rho_min=0.3)).trusted[i, j]
    assert planner(fb, PlannerConfig(rho_min=0.3, rho_max=np.inf)).trusted[i, j]


def test_first_hop_does_not_trust_a_raw_rho_at_or_above_one():
    # Clipped to gamma, this edge would win the first hop (0.9 * 27 against 10.5 direct).
    # Untrusted, it leaves the exact task policy as the best choice.
    fb = GridFB(wormholes={(h, START, (5, 3)): 1.5 for h in (0, 1)})
    ctrl = planner(fb, PlannerConfig(rho_min=0.3, max_edge_xy=None))
    ctrl.act(fb.obs(START))
    assert ctrl.step_info()["waypoint"] == DIRECT


def test_value_iteration_uses_rho_clipped_to_gamma():
    # Raw rho 0.95 from START to the cell next to the goal: trusted (raw < 1), valued at gamma.
    fb = GridFB(wormholes={(h, START, (5, 3)): 0.95 for h in (0, 1)})
    ctrl = planner(fb, PlannerConfig(rho_min=0.3, max_edge_xy=None))
    v_next_to_goal = 3.0 * fb.discount / (1 - fb.discount)
    expected = fb.discount * v_next_to_goal
    assert ctrl.plan.solution.values[fb.index[START]] == pytest.approx(expected)


def test_xy_cap_removes_long_landmark_edges():
    fb = GridFB()
    ctrl = planner(fb, PlannerConfig(rho_min=0.0, max_edge_xy=1.5))
    xy = fb.all_obs()
    lengths = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    assert ctrl.trusted.any()
    assert (lengths[ctrl.trusted] <= 1.5).all()


def test_xy_cap_applies_to_the_first_hop():
    fb = GridFB(horizon=4)
    ctrl = planner(fb, replace(HORIZON_CFG, max_edge_xy=1.0))
    ctrl.act(fb.obs(START))
    waypoint = fb.obs(cell_of(fb, ctrl.step_info()["waypoint"]))
    assert np.linalg.norm(waypoint - fb.obs(START)) <= 1.0


# FB says the cell next to the goal is close to START (rho 0.85); the way back is
# gamma^9 = 0.39. A wormhole to GOAL itself would also raise the direct value, since z_r
# targets GOAL.
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


# Landmarks on the U path with hops shorter than the 4-step horizon: (1, 1) is 3 steps from
# START, (3, 1) 2 steps further, (5, 2) 3 steps further, and the goal 2 steps after that.
SHORT_HOPS = [(1, 1), (3, 1), (5, 2)]


def short_hop_planner(fb: GridFB, lookahead: bool, **overrides) -> GraphPlannerController:
    landmarks = np.stack([fb.obs(c) for c in SHORT_HOPS])
    cfg = replace(HORIZON_CFG, lookahead=lookahead, **overrides)
    ctrl = GraphPlannerController(fb, landmarks, cfg)
    ctrl.reset(task(fb), seed=0)
    return ctrl


def test_lookahead_switches_to_the_next_route_stop_before_the_waypoint_is_reached():
    fb = GridFB(horizon=4)
    cells, waypoints = rollout(fb, short_hop_planner(fb, lookahead=True), steps=12)
    assert cells.index(GOAL) == 10
    assert SHORT_HOPS[waypoints[0]] == (1, 1)  # (3, 1) is 5 steps away: not trusted from START
    # From (1, 3), (3, 1) is 4 steps away and trusted: going there directly is worth more.
    assert SHORT_HOPS[waypoints[1]] == (3, 1)
    # The task policy takes over once the goal is within the horizon, before (5, 2).
    assert waypoints[cells.index((4, 1))] == -1


def test_without_lookahead_waypoints_end_only_when_reached():
    fb = GridFB(horizon=4)
    cells, waypoints = rollout(fb, short_hop_planner(fb, lookahead=False), steps=12)
    assert cells.index(GOAL) == 10
    for t in range(1, len(waypoints)):
        if waypoints[t] != waypoints[t - 1]:
            assert cells[t] == SHORT_HOPS[waypoints[t - 1]]


def test_lookahead_counts_advances():
    fb = GridFB(horizon=4)
    ctrl = short_hop_planner(fb, lookahead=True)
    rollout(fb, ctrl, steps=12)
    assert ctrl.step_info()["advances"] >= 2
    assert ctrl.step_info()["replans"] == 1


def test_lookahead_moves_on_only_along_trusted_edges():
    # rho_min = 0.72 trusts edges of at most 3 steps (gamma^3 = 0.729). From (1, 3), (3, 1) is
    # 4 steps away: FB still sees it (rho 0.66) and values it more, but the edge is not trusted.
    fb = GridFB(horizon=4)
    cells, waypoints = rollout(fb, short_hop_planner(fb, lookahead=True, rho_min=0.72), steps=4)
    assert cells[1:3] == [(1, 3), (1, 2)]
    assert [SHORT_HOPS[w] for w in waypoints[:3]] == [(1, 1), (1, 1), (3, 1)]


# Route following at a reach (switch_steps). Edges of at most 3 steps are trusted
# (rho_min 0.72 > gamma^4) and a waypoint counts as reached one cell before it.
FOLLOW_CFG = replace(HORIZON_CFG, reached_xy=1.0, rho_min=0.72)
# From (1, 1) the waypoint (2, 1) counts as reached, but its next stop (5, 1) is 4 steps away
# and not trusted from there.
STALL_LANDMARKS = [(1, 3), (2, 1), (5, 1)]
# The plan at (5, 1) hands over to the task policy. FB underestimates the task value at (4, 1)
# on one head (rho 0.3 instead of gamma^4): 9 instead of 19.7. Going back to (2, 1) is worth
# Q = 12.7 from there, 12.7 / 9 = gamma^-3.3.
DIP = {(0, (4, 1), GOAL): 0.3}
NEAR_GOAL_LANDMARKS = [(2, 1), (5, 1)]


def landmark_planner(fb: GridFB, cells: list[tuple[int, int]], cfg: PlannerConfig):
    ctrl = GraphPlannerController(fb, np.stack([fb.obs(c) for c in cells]), cfg)
    ctrl.reset(task(fb), seed=0)
    return ctrl


def test_full_replans_at_a_reach_can_stall_between_two_cells():
    fb = GridFB(horizon=4)
    ctrl = landmark_planner(fb, STALL_LANDMARKS, FOLLOW_CFG)
    cells, _ = rollout(fb, ctrl, steps=30, start=(1, 2))
    assert set(cells) == {(1, 1), (1, 2)}  # back to (1, 3), forward to (2, 1), back...


@pytest.mark.parametrize("switch_steps", [0.0, np.inf])
def test_route_following_keeps_a_reached_waypoint_until_its_next_stop_is_trusted(switch_steps):
    fb = GridFB(horizon=4)
    cfg = replace(FOLLOW_CFG, switch_steps=switch_steps)
    ctrl = landmark_planner(fb, STALL_LANDMARKS, cfg)
    cells, waypoints = rollout(fb, ctrl, steps=10, start=(1, 2))
    assert cells.index(GOAL) == 8
    assert STALL_LANDMARKS[waypoints[1]] == (2, 1)  # kept at (1, 1), where it counts as reached


def test_full_replans_near_the_goal_circle_between_landmarks():
    fb = GridFB(horizon=4, wormholes=DIP)
    ctrl = landmark_planner(fb, NEAR_GOAL_LANDMARKS, FOLLOW_CFG)
    cells, _ = rollout(fb, ctrl, steps=30, start=(3, 1))
    assert set(cells) == {(3, 1), (4, 1)}


def test_route_following_hands_over_to_the_task_policy_where_the_plan_does():
    fb = GridFB(horizon=4, wormholes=DIP)
    cfg = replace(FOLLOW_CFG, switch_steps=4.0)
    ctrl = landmark_planner(fb, NEAR_GOAL_LANDMARKS, cfg)
    cells, waypoints = rollout(fb, ctrl, steps=5, start=(3, 1))
    assert cells.index(GOAL) == 5
    assert waypoints[1:] == [-1] * 4


@pytest.mark.parametrize(("switch_steps", "goes_back"), [(3.0, True), (4.0, False)])
def test_a_switch_away_from_the_plan_must_save_more_than_switch_steps(switch_steps, goes_back):
    fb = GridFB(horizon=4, wormholes=DIP)
    cfg = replace(FOLLOW_CFG, switch_steps=switch_steps)
    ctrl = landmark_planner(fb, NEAR_GOAL_LANDMARKS, cfg)
    _, waypoints = rollout(fb, ctrl, steps=2, start=(3, 1))
    assert (waypoints[1] == NEAR_GOAL_LANDMARKS.index((2, 1))) == goes_back


def test_planner_config_rejects_a_negative_switch_margin():
    with pytest.raises(ValueError, match="switch_steps"):
        PlannerConfig(switch_steps=-1.0)


def test_lookahead_ends_on_a_cycle_of_tied_route_stops():
    # A plan whose route loops (0 -> 1 -> 0) with equal values: from (3, 1) each stop is worth
    # as much as the other (rho 0.9 to both), so the look-ahead could move on forever.
    fb = GridFB()
    landmarks = np.stack([fb.obs((2, 1)), fb.obs((4, 1))])
    ctrl = GraphPlannerController(fb, landmarks, replace(HORIZON_CFG, lookahead=True))
    ctrl.reset(task(fb), seed=0)
    loop = GraphSolution(np.full(2, 100.0), np.array([1, 0]), converged=True)
    ctrl.plan = replace(ctrl.plan, solution=loop)
    ctrl.act(fb.obs((3, 1)))  # chooses a waypoint (Q 89 > task value 18)
    ctrl.act(fb.obs((3, 1)))  # looks ahead along the loop and must return
    assert ctrl.step_info()["advances"] == len(landmarks)


class LatentSpy:
    """Wraps an FBModel and keeps the last latent given to pi_l and to pi_h."""

    def __init__(self, fb: GridFB) -> None:
        self.fb, self.low_z, self.high_z = fb, None, None

    def __getattr__(self, name: str):
        return getattr(self.fb, name)

    def low_action(self, obs, z):
        self.low_z = np.asarray(z)[0]
        return self.fb.low_action(obs, z)

    def high_latent(self, obs, z_task):
        self.high_z = np.asarray(z_task)[0]
        return self.fb.high_latent(obs, z_task)


def act_ahead_planner(fb, ahead: float, cls=GraphPlannerController) -> GraphPlannerController:
    """SHORT_HOPS landmarks; the plan from START is (1, 1) -> (3, 1) -> (5, 2) -> task policy."""
    landmarks = np.stack([fb.obs(c) for c in SHORT_HOPS])
    field = RegionField(fb.all_obs()[:, :2], fb.backward(fb.all_obs()), radius=0.5)
    cfg = replace(HORIZON_CFG, act_ahead_xy=ahead)
    ctrl = cls(fb, landmarks, cfg, region_field=field)
    ctrl.reset(task(fb), seed=0)
    return ctrl


@pytest.mark.parametrize(("ahead", "point"), [(2.0, (1.0, 2.0)), (4.0, (2.0, 1.0))])
def test_act_ahead_gives_pi_l_the_region_latent_of_the_route_point_ahead(ahead, point):
    # From START (1, 4) the route runs to (1, 1) (3 xy), then to (3, 1).
    fb = LatentSpy(GridFB(horizon=4))
    ctrl = act_ahead_planner(fb, ahead)
    ctrl.act(fb.obs(START))
    assert SHORT_HOPS[ctrl.step_info()["waypoint"]] == (1, 1)
    np.testing.assert_allclose(fb.low_z, ctrl.region_field.at(np.array(point)))


def test_act_ahead_acts_with_the_task_latent_when_the_route_is_shorter():
    fb = LatentSpy(GridFB(horizon=4))
    ctrl = act_ahead_planner(fb, ahead=100.0)
    ctrl.act(fb.obs(START))
    np.testing.assert_allclose(fb.low_z, fb.infer_task_latent(1))


def test_act_ahead_gives_pi_h_the_region_latent_as_its_task():
    fb = LatentSpy(GridFB(horizon=4))
    ctrl = act_ahead_planner(fb, 2.0, cls=GraphPlusHighController)
    ctrl.act(fb.obs(START))
    np.testing.assert_allclose(fb.high_z, ctrl.region_field.at(np.array([1.0, 2.0])))


def test_from_dataset_builds_the_region_field_only_for_act_ahead():
    fb = GridFB()
    data = np.tile(fb.all_obs(), (3, 1))
    base = PlannerConfig(landmarks=LandmarkConfig(count=5))
    plain = GraphPlannerController.from_dataset(fb, data, base)
    cfg = replace(base, act_ahead_xy=4.0, act_region_xy=0.7)
    ahead = GraphPlannerController.from_dataset(fb, data, cfg)
    assert plain.region_field is None
    assert ahead.region_field.radius == 0.7 and len(ahead.region_field.xy) == len(data)


def test_replan_every_step_rescores_all_landmarks_at_each_step():
    fb = GridFB(horizon=4)
    ctrl = planner(fb, replace(HORIZON_CFG, replan_every_step=True))
    cells, _ = rollout(fb, ctrl, steps=12)
    assert cells.index(GOAL) == 10
    assert ctrl.step_info()["replans"] == 12


def successes(ctrl, fb: TabularFB, episodes: int = 3) -> int:
    """Successes over the 5 OGBench tasks of fb's maze, `episodes` start states each."""
    return sum(
        rollout_tabular(ctrl, fb, task_id, seed=ep, max_steps=200).success
        for task_id in range(1, 6)
        for ep in range(episodes)
    )


def test_planner_solves_every_task_of_the_exact_tabular_medium_maze():
    fb = TabularFB()
    assert successes(GraphPlannerController(fb, fb.maze.xy), fb) == 15


def test_planner_beats_the_task_policy_under_fb_noise_on_the_tabular_maze():
    # the tabular study (docs/theory.md §9.1): with a greedy controller on noisy F, the task
    # policy fails at long range and chaining short trusted edges recovers part of it.
    planner_wins, direct_wins = 0, 0
    for seed in range(3):
        fb = TabularFB(FakeFBConfig(rho_noise=0.02, seed=seed))
        planner_wins += successes(GraphPlannerController(fb, fb.maze.xy), fb)
        direct_wins += successes(NonHierController(fb), fb)
    assert planner_wins >= direct_wins + 10


def test_planner_acting_ahead_on_region_latents_solves_every_task_of_the_exact_tabular_maze():
    fb = TabularFB()
    # Fine cells are 2 xy apart: act 4 xy (two cells) ahead, on a region of a cell and its
    # neighbours.
    cfg = PlannerConfig(
        landmarks=LandmarkConfig(count=60), act_ahead_xy=4.0, act_region_xy=2.1
    )
    ctrl = GraphPlannerController.from_dataset(fb, fb.maze.xy, cfg)
    assert successes(ctrl, fb) == 15


class RoundedFB(TabularFB):
    """The exact tabular model with F perturbed by about 1e-12 relative: other rounding, as on
    another CPU or BLAS build (Linux x86 against macOS arm64)."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        noise = np.random.default_rng(seed).standard_normal((self.maze.n_states, self.latent_dim))
        self.factor = 1.0 + 1e-12 * noise

    def forward(self, obs, z):
        return super().forward(obs, z) * self.factor[self.maze.state_index(np.asarray(obs))]


# Region latents make exact ties: rho to a region counts the hit one step before w, so a hop
# back towards a region can be worth exactly as much as a hop ahead. Rounding must not decide.
@pytest.mark.parametrize("rounding", [None, 1, 2, 3, 4])
def test_planner_with_region_latents_solves_every_task_of_the_exact_tabular_maze(rounding):
    fb = TabularFB() if rounding is None else RoundedFB(rounding)
    # Fine cells are 2 xy units apart, so a radius of 2.1 takes each state and its neighbours.
    cfg = PlannerConfig(landmarks=LandmarkConfig(count=60, latent="region", region_xy=2.1))
    ctrl = GraphPlannerController.from_dataset(fb, fb.maze.xy, cfg)
    expected = region_latents(fb, fb.maze.xy, ctrl.landmarks[:, :2], radius=2.1)
    np.testing.assert_allclose(ctrl.targets.z, expected, atol=1e-9)
    assert successes(ctrl, fb) == 15


# ----- recorded routes (docs/hypotheses/h-papers.md H1) -----

# The U corridor walked from START up, across and down to GOAL: 10 steps of 1 xy. START and
# GOAL are 4 xy apart in a straight line, through the wall.
U_WALK = [(1, 4), (1, 3), (1, 2), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (5, 2), (5, 3), (5, 4)]


def one_hot(fb: GridFB, cell: tuple[int, int]) -> np.ndarray:
    return np.eye(fb.latent_dim)[fb.index[cell]]


def recorded_route(fb, walks, at=START, max_steps=20, **overrides) -> RecordedRouteController:
    """Landmarks at START and GOAL, the recorded walks as data, regions of radius 0.5 (one cell).
    The controller has acted once, from the cell `at`."""
    xy = np.concatenate([np.asarray(w, dtype=float) for w in walks])
    terminals = np.concatenate([np.r_[np.zeros(len(w) - 1), 1.0] for w in walks])
    field = RegionField(fb.all_obs()[:, :2], fb.backward(fb.all_obs()), radius=0.5)
    landmarks = np.stack([fb.obs(START), fb.obs(GOAL)])
    cfg = RecordedRouteConfig(max_steps=max_steps, **overrides)
    ctrl = RecordedRouteController(fb, landmarks, xy, terminals, field, cfg)
    ctrl.reset(task(fb), seed=0)
    ctrl.act(fb.obs(at))
    return ctrl


@pytest.mark.parametrize(("ahead", "point"), [(2.0, (1, 2)), (5.0, (3, 1))])
def test_recorded_route_acts_on_the_point_ahead_along_the_recorded_walk(ahead, point):
    # A straight line from START to GOAL would put the point in the wall at (3, 4).
    spy = LatentSpy(GridFB())
    ctrl = recorded_route(spy, [U_WALK], ahead_xy=ahead)
    np.testing.assert_array_equal(spy.low_z, one_hot(spy.fb, point))
    assert ctrl.step_info()["goal_landmark"] == 1  # GOAL has the largest F(w, z_r)^T z_r


def test_recorded_route_acts_with_the_task_latent_past_the_goal_landmark():
    spy = LatentSpy(GridFB())
    recorded_route(spy, [U_WALK], ahead_xy=11.0)
    np.testing.assert_array_equal(spy.low_z, spy.fb.infer_task_latent(1))


@pytest.mark.parametrize(
    ("window", "cells"), [((1.0, 2.0), [(1, 3), (1, 2)]), ((2.0, 20.0), [(1, 2), GOAL])]
)
def test_recorded_route_window_averages_the_unit_latents_of_the_points_ahead(window, cells):
    # Past the goal landmark a window point takes z_r / |z_r| (here the one-hot of GOAL).
    spy = LatentSpy(GridFB())
    recorded_route(spy, [U_WALK], executor="window", window_xy=window)
    expected = np.mean([one_hot(spy.fb, c) for c in cells], axis=0)
    np.testing.assert_allclose(spy.low_z, expected)


def test_recorded_route_gives_pi_h_the_region_latent_of_the_point_as_its_task():
    spy = LatentSpy(GridFB())
    recorded_route(spy, [U_WALK], executor="high", ahead_xy=2.0)
    np.testing.assert_array_equal(spy.high_z, one_hot(spy.fb, (1, 2)))


def test_recorded_route_uses_edges_only_in_their_recorded_direction():
    # Only the walk from GOAL to START is recorded, so no certified route leads to GOAL.
    spy = LatentSpy(GridFB())
    ctrl = recorded_route(spy, [U_WALK[::-1]], ahead_xy=2.0)
    np.testing.assert_array_equal(spy.low_z, spy.fb.infer_task_latent(1))
    assert ctrl.step_info()["hops"] == 0


@pytest.mark.parametrize(("max_steps", "routed"), [(5, True), (4, False)])
def test_recorded_route_first_hop_is_a_recorded_walk_of_at_most_max_steps(max_steps, routed):
    # From (3, 1) the walk reaches GOAL 5 steps later.
    spy = LatentSpy(GridFB())
    ctrl = recorded_route(spy, [U_WALK], at=(3, 1), max_steps=max_steps, ahead_xy=2.0)
    expected = one_hot(spy.fb, (5, 1)) if routed else spy.fb.infer_task_latent(1)
    np.testing.assert_array_equal(spy.low_z, expected)
    assert ctrl.step_info()["hops"] == int(routed)


def test_recorded_route_first_hop_follows_the_shortest_recorded_walk():
    # From (3, 1) two walks reach GOAL: one in 6 steps that first turns back (7 xy), one in
    # 7 steps that stands still twice (5 xy). The point 2 xy ahead along the first is (3, 1).
    back = [(3, 1), (2, 1), (4, 1), (5, 1), (5, 2), (5, 3), (5, 4)]
    still = [(3, 1), (4, 1), (4, 1), (4, 1), (5, 1), (5, 2), (5, 3), (5, 4)]
    spy = LatentSpy(GridFB())
    recorded_route(spy, [back, still], at=(3, 1), ahead_xy=2.0)
    np.testing.assert_array_equal(spy.low_z, one_hot(spy.fb, (5, 1)))


def test_recorded_route_config_rejects_unknown_executors_and_bad_bounds():
    with pytest.raises(ValueError):
        RecordedRouteConfig(executor="cells")
    with pytest.raises(ValueError):
        RecordedRouteConfig(max_steps=0)
    with pytest.raises(ValueError):
        RecordedRouteConfig(window_xy=())


def test_config_from_options_builds_a_recorded_route_config():
    cfg = config_from_options(
        {"executor": "window", "max_steps": 150, "landmarks.count": 5}, RecordedRouteConfig()
    )
    assert (cfg.executor, cfg.max_steps, cfg.landmarks.count) == ("window", 150, 5)


EXECUTORS = ["point", "window", "high"]


def recorded_route_on(fb: TabularFB, executor: str, count: int, max_steps: int):
    observations, terminals = navigate_dataset(fb.maze, trajectories=100, length=100)
    cfg = RecordedRouteConfig(
        landmarks=LandmarkConfig(count=count), executor=executor, max_steps=max_steps
    )
    return RecordedRouteController.from_dataset(fb, observations, terminals, cfg)


@pytest.mark.parametrize("executor", EXECUTORS)
def test_recorded_route_solves_every_task_of_the_exact_tabular_maze(executor):
    # One state per maze cell. With 2 x 2 states per cell, two equally short recorded walks
    # down one corridor can take turns as the route, and the exact greedy pi_l has tied moves
    # towards both, so the agent steps from side to side.
    fb = TabularFB(FakeFBConfig(resolution=1))
    assert successes(recorded_route_on(fb, executor, count=12, max_steps=5), fb) == 15


@pytest.mark.parametrize("executor", EXECUTORS)
def test_recorded_route_beats_the_task_policy_under_fb_noise_on_the_tabular_maze(executor):
    # Under F noise pi_l(z_r) fails at long range (docs/theory.md §9.1); the route
    # gives pi_l targets a few cells ahead. 10 steps of 2 xy: about K = 150 on AntMaze.
    route_wins, direct_wins = 0, 0
    for seed in range(3):
        fb = TabularFB(FakeFBConfig(rho_noise=0.02, seed=seed))
        route_wins += successes(recorded_route_on(fb, executor, count=48, max_steps=10), fb)
        direct_wins += successes(NonHierController(fb), fb)
    assert route_wins >= direct_wins + 20
