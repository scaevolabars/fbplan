import numpy as np
import pytest

from fbplan.tabular.fake_fb import FakeFBConfig, TabularFB
from fbplan.tabular.mdp import optimal_policy, policy_value, successor_measure
from fbplan.tabular.options import option_tables
from fbplan.tabular.study import (
    LandmarkModel,
    PlannerConfig,
    evaluate_goal_reaching,
    goal_latent,
    plan_policy,
    plan_route,
    wormhole_mask,
)

GAMMA = 0.9
CHAINED = PlannerConfig("chained")
PLANNERS = [
    PlannerConfig("direct", kind="direct"),
    PlannerConfig("baseline", kind="baseline"),
    PlannerConfig("single", kind="single"),
    CHAINED,
    PlannerConfig("chained-commit", commit=True),
]


@pytest.fixture(scope="module")
def exact() -> LandmarkModel:
    model = TabularFB(FakeFBConfig(discount=GAMMA))
    return LandmarkModel(model, np.arange(model.maze.n_states))


def test_exact_estimates_equal_the_exact_option_model(exact):
    maze, goal = exact.model.maze, 77
    reward = np.eye(maze.n_states)[goal]
    rho_hat, r_hat, v_hat = exact.estimates(goal_latent(exact.model, goal), aggregate="mean")
    policies = [exact.model.policy(b) for b in exact.model.basis]
    rho, r_before = option_tables(maze, policies, exact.landmarks, reward, GAMMA)
    np.testing.assert_allclose(rho_hat, rho, atol=1e-9)
    np.testing.assert_allclose(r_hat, r_before, atol=1e-9)
    m_goal = successor_measure(maze, exact.model.policy(exact.model.basis[goal]), GAMMA)
    np.testing.assert_allclose(v_hat, m_goal[:, goal], atol=1e-9)


@pytest.mark.parametrize("planner", PLANNERS, ids=lambda p: p.name)
def test_every_planner_reaches_every_goal_with_the_exact_model(exact, planner):
    goals = np.array([0, 40, 103])
    success, _ = evaluate_goal_reaching(exact, planner, goals, horizon=100)
    assert success.all()


def test_exact_model_has_no_wormholes(exact):
    rho_hat, _, _ = exact.estimates(goal_latent(exact.model, 5), aggregate="mean")
    dist = exact.model.maze.geodesic_distances()
    assert not wormhole_mask(rho_hat, dist, GAMMA).any()


def test_wall_blind_low_rank_basis_creates_wormholes():
    model = TabularFB(FakeFBConfig(discount=GAMMA, rank=24, basis="euclidean"))
    lm = LandmarkModel(model, np.arange(model.maze.n_states))
    rho_hat, _, _ = lm.estimates(goal_latent(model, 5), aggregate="mean")
    assert wormhole_mask(rho_hat, model.maze.geodesic_distances(), GAMMA).sum() > 0


def test_full_planner_on_exact_model_is_optimal_for_a_general_reward(exact):
    maze = exact.model.maze
    reward = np.random.default_rng(0).normal(size=maze.n_states)
    z_r = exact.model.basis.T @ reward / np.sqrt(maze.n_states)
    actions = plan_policy(exact, CHAINED, z_r)
    _, v_star = optimal_policy(maze, reward, GAMMA)
    np.testing.assert_allclose(policy_value(maze, actions, reward, GAMMA), v_star, atol=1e-8)


def test_route_uses_only_trusted_hops_and_never_repeats_a_landmark():
    model = TabularFB(FakeFBConfig(discount=GAMMA, rho_noise=0.02, low_policy="optimal"))
    lm = LandmarkModel(model, np.arange(model.maze.n_states))
    goal, planner = 90, PlannerConfig("chained", rho_min=0.3)
    z_r = goal_latent(model, goal)
    rho_hat, _, _ = lm.estimates(z_r, aggregate="mean")
    for start in range(0, model.maze.n_states, 9):
        route = plan_route(lm, planner, z_r, start)
        assert len(set(route)) == len(route)
        for u, w in zip([start, *route], route, strict=False):  # consecutive hops
            assert rho_hat[u, w] >= planner.rho_min


def test_reward_variants_follow_their_definitions(exact):
    maze, goal = exact.model.maze, 30
    z_r = goal_latent(exact.model, goal)
    rho, full, _ = exact.estimates(z_r, "mean", reward="full")
    _, proxy, _ = exact.estimates(z_r, "mean", reward="proxy")
    _, zero, _ = exact.estimates(z_r, "mean", reward="zero")
    stay_value = proxy[exact.landmarks, np.arange(maze.n_states)]  # F(w, z_w)^T z_r
    np.testing.assert_allclose(proxy - full, rho * stay_value[None, :], atol=1e-9)
    assert not zero.any()
