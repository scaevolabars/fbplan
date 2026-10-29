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
from fbplan.landmarks import LandmarkConfig, region_latents
from fbplan.option_model import OptionModelConfig
from fbplan.tabular.env import rollout as rollout_tabular
from fbplan.tabular.fake_fb import FakeFBConfig, TabularFB

START, GOAL = (1, 4), (5, 4)  # the two bottom ends of the U maze, 10 steps apart
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


@pytest.mark.parametrize("cfg", [HORIZON_CFG, RHO_RULE_CFG], ids=["xy_rule", "rho_rule"])
def test_waypoint_is_kept_until_it_is_reached(cfg):
    fb = GridFB(horizon=4)
    cells, waypoints = rollout(fb, planner(fb, cfg), steps=10)
    assert GOAL in cells
    for t in range(1, len(waypoints)):
        if waypoints[t] != waypoints[t - 1]:
            assert cells[t] == fb.cells[waypoints[t - 1]]


def test_xy_rule_ends_a_waypoint_within_the_radius():
    fb = GridFB(horizon=4)
    cells, waypoints = rollout(fb, planner(fb, replace(HORIZON_CFG, reached_xy=1.0)), steps=14)
    assert GOAL in cells
    switches = [t for t in range(1, len(waypoints)) if waypoints[t] != waypoints[t - 1]]
    assert switches
    for t in switches:
        gap = np.subtract(cells[t], fb.cells[waypoints[t - 1]])
        assert 0 < np.linalg.norm(gap) <= 1.0  # one cell before the waypoint


def test_waypoint_is_dropped_after_the_timeout():
    fb = StandStill(GridFB(horizon=4))
    ctrl = planner(fb, PlannerConfig(rho_min=0.65, rho_reached=0.95, commit_timeout=3))
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
    assert fb.cells[ctrl.step_info()["waypoint"]] not in {(1, 1), (2, 1), (3, 1)}


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


def test_first_hop_does_not_trust_a_raw_rho_at_or_above_one():
    # Clipped to gamma, this edge would win the first hop (0.9 * 27 against 10.5 direct).
    fb = GridFB(wormholes={(h, START, (5, 3)): 1.5 for h in (0, 1)})
    ctrl = planner(fb, PlannerConfig(rho_min=0.3, max_edge_xy=None))
    ctrl.act(fb.obs(START))
    assert fb.cells[ctrl.step_info()["waypoint"]] != (5, 3)


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


def test_reached_landmarks_do_not_use_top_k_slots():
    # top_k = 2 keeps the two corridor neighbours of every cell. With reached_xy = 1 the
    # neighbours of (2, 1) (rho 0.9, the largest) count as reached. If they took the two slots
    # before their exclusion, no edge would be left; the next cells must get the slots.
    fb = GridFB(horizon=4)
    ctrl = planner(fb, replace(HORIZON_CFG, top_k=2, reached_xy=1.0))
    ctrl.act(fb.obs((2, 1)))
    assert fb.cells[ctrl.step_info()["waypoint"]] == (4, 1)


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


def test_replan_every_step_rescores_all_landmarks_at_each_step():
    fb = GridFB(horizon=4)
    ctrl = planner(fb, replace(HORIZON_CFG, replan_every_step=True))
    cells, _ = rollout(fb, ctrl, steps=12)
    assert cells.index(GOAL) == 10
    assert ctrl.step_info()["replans"] == 12


class TaskPolicy:
    """pi_l(a | s, z_r): the task policy alone, for comparison."""

    name = "task_policy"

    def __init__(self, fb: TabularFB) -> None:
        self.fb = fb

    def reset(self, task: TaskSpec, seed: int) -> None:
        self.z_r = task.z_r

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self.fb.low_action(obs[None], self.z_r[None]))[0]

    def step_info(self) -> dict[str, float | int | str]:
        return {}


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
    # b-theory's study (docs/theory.md §9.1): with a greedy controller on noisy F, the task
    # policy fails at long range and chaining short trusted edges recovers part of it.
    planner_wins, direct_wins = 0, 0
    for seed in range(3):
        fb = TabularFB(FakeFBConfig(rho_noise=0.02, seed=seed))
        planner_wins += successes(GraphPlannerController(fb, fb.maze.xy), fb)
        direct_wins += successes(TaskPolicy(fb), fb)
    assert planner_wins >= direct_wins + 10


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
