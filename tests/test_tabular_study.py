from itertools import pairwise

import numpy as np
import pytest

from fbplan.tabular.fake_fb import FakeFBConfig, TabularFB
from fbplan.tabular.mdp import optimal_policy, policy_value, successor_measure
from fbplan.tabular.options import option_tables
from fbplan.tabular.study import (
    LandmarkModel,
    StudyPlannerConfig,
    evaluate_goal_reaching,
    goal_latent,
    make_plan,
    plan_actions,
    plan_route,
    rho_bound_wormholes,
)

GAMMA = 0.9
CHAINED = StudyPlannerConfig("chained")
PLANNERS = [
    StudyPlannerConfig("direct", kind="direct"),
    StudyPlannerConfig("baseline", kind="baseline"),
    StudyPlannerConfig("single", kind="single"),
    CHAINED,
    StudyPlannerConfig("chained-commit", commit=True),
]


@pytest.fixture(scope="module")
def exact() -> LandmarkModel:
    model = TabularFB(FakeFBConfig(discount=GAMMA))
    return LandmarkModel(model, np.arange(model.maze.n_states))


def test_option_model_on_the_exact_fake_equals_the_exact_option_tables(exact):
    # The study's FB estimates come from c-planner's OptionModel; the truth from the maze.
    maze, model, goal = exact.model.maze, exact.model, 77
    reward = np.eye(maze.n_states)[goal]  # r of the goal latent in full rank
    z_r = goal_latent(model, goal)
    policies = [model.policy(b) for b in model.basis]
    rho, r_before = option_tables(maze, policies, exact.landmarks, reward, GAMMA)
    table = exact.table("mean", "exact", z_r)
    np.testing.assert_allclose(table.rho, rho, atol=1e-9)
    np.testing.assert_allclose(table.reward, r_before, atol=1e-9)
    # The upstream proxy ("paper") is the value of following pi_w forever, w included.
    v_options = np.stack([policy_value(maze, pi, reward, GAMMA) for pi in policies], axis=1)
    np.testing.assert_allclose(exact.table("min", "paper", z_r).reward, v_options, atol=1e-9)
    m_goal = successor_measure(maze, model.policy(model.basis[goal]), GAMMA)
    np.testing.assert_allclose(exact.v_direct(z_r, "mean"), m_goal[:, goal], atol=1e-9)


@pytest.mark.parametrize("planner", PLANNERS, ids=lambda p: p.name)
def test_every_planner_reaches_every_goal_with_the_exact_model(exact, planner):
    goals = np.array([0, 40, 103])
    success, _ = evaluate_goal_reaching(exact, planner, goals, horizon=100)
    assert success.all()


def test_exact_pi_h_with_a_subgoal_radius_reaches_every_goal():
    model = TabularFB(FakeFBConfig(discount=GAMMA, high_radius=6))
    lm = LandmarkModel(model, np.arange(model.maze.n_states))
    goals = np.array([0, 40, 103])
    planner = StudyPlannerConfig("pi_h", kind="baseline")
    success, _ = evaluate_goal_reaching(lm, planner, goals, 100)
    assert success.all()


def test_exact_model_has_no_wormholes(exact):
    dist = exact.model.maze.geodesic_distances()
    assert not rho_bound_wormholes(exact.table("mean").rho, dist, GAMMA).any()


def test_wall_blind_low_rank_basis_creates_wormholes():
    model = TabularFB(FakeFBConfig(discount=GAMMA, rank=24, basis="euclidean"))
    lm = LandmarkModel(model, np.arange(model.maze.n_states))
    dist = model.maze.geodesic_distances()
    assert rho_bound_wormholes(lm.table("mean").rho, dist, GAMMA).sum() > 0


def test_exact_planner_on_exact_model_is_optimal_for_a_general_reward(exact):
    maze = exact.model.maze
    reward = np.random.default_rng(0).normal(size=maze.n_states)
    z_r = exact.model.basis.T @ reward / np.sqrt(maze.n_states)
    actions = plan_actions(exact, make_plan(exact, CHAINED, z_r), z_r)
    _, v_star = optimal_policy(maze, reward, GAMMA)
    np.testing.assert_allclose(policy_value(maze, actions, reward, GAMMA), v_star, atol=1e-8)


def test_route_uses_only_trusted_hops_and_never_repeats_a_landmark():
    model = TabularFB(FakeFBConfig(discount=GAMMA, rho_noise=0.02, low_policy="optimal"))
    lm = LandmarkModel(model, np.arange(model.maze.n_states))
    goal, planner = 90, StudyPlannerConfig("chained", rho_min=0.3)
    z_r = goal_latent(model, goal)
    plan = make_plan(lm, planner, z_r)
    rho_raw = lm.table("mean").rho_raw
    hops = 0
    for start in range(0, model.maze.n_states, 9):
        route = plan_route(lm, plan, start)
        assert len(set(route)) == len(route)
        for u, w in pairwise([start, *route]):
            assert planner.rho_min <= rho_raw[u, w] < 1.0
            hops += 1
    assert hops > 0


def test_hop_discount_shortens_routes_on_noisy_values():
    model = TabularFB(FakeFBConfig(discount=GAMMA, rho_noise=0.02, low_policy="optimal"))
    lm = LandmarkModel(model, np.arange(model.maze.n_states))
    z_r = goal_latent(model, 90)
    lengths = []
    for hop_discount in (1.0, 0.8):
        plan = make_plan(lm, StudyPlannerConfig("c", hop_discount=hop_discount), z_r)
        lengths.append(sum(len(plan_route(lm, plan, s)) for s in range(model.maze.n_states)))
    assert lengths[1] < lengths[0]


@pytest.mark.parametrize(
    ("field", "value"), [("kind", "double"), ("head_agg", "median"), ("edge_reward", "full")]
)
def test_unknown_planner_settings_are_rejected(field, value):
    with pytest.raises(ValueError, match=value):
        StudyPlannerConfig("bad", **{field: value})
