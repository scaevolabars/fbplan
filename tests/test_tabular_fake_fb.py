import numpy as np
import pytest

from fbplan.contracts import FBModel, normalize_z
from fbplan.tabular.fake_fb import FakeFBConfig, TabularFB
from fbplan.tabular.mdp import optimal_policy, successor_measure

GAMMA = 0.9


@pytest.fixture(scope="module")
def exact() -> TabularFB:
    return TabularFB(FakeFBConfig(discount=GAMMA))


@pytest.fixture(scope="module")
def noisy() -> TabularFB:
    return TabularFB(FakeFBConfig(discount=GAMMA, rho_noise=0.05, seed=3))


def _subgoal(model: TabularFB, w: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Observations of all states and the latent B(w) repeated n times."""
    obs = model.maze.xy
    return obs, np.repeat(model.backward(obs[[w]]), n, axis=0)


def test_fake_satisfies_the_fbmodel_protocol_shapes(exact):
    model: FBModel = exact
    obs = exact.maze.xy[:7]
    z = np.random.default_rng(0).normal(size=(7, exact.latent_dim))
    assert model.forward(obs, z).shape == (2, 7, exact.latent_dim)
    assert model.backward(obs).shape == (7, exact.latent_dim)
    actions = model.low_action(obs, z)
    assert actions.shape == (7, 2) and np.all(np.abs(actions) <= 1)
    high = model.high_latent(obs, z)
    np.testing.assert_allclose(np.linalg.norm(high, axis=-1), np.sqrt(exact.latent_dim))
    z_r = model.infer_task_latent(1)
    np.testing.assert_allclose(np.linalg.norm(z_r), np.sqrt(exact.latent_dim))
    assert model.discount == GAMMA


def test_exact_mode_has_one_latent_dimension_per_state(exact):
    assert exact.latent_dim == exact.maze.n_states == 104


def test_backward_is_orthonormal_up_to_sqrt_n_like_upstream_regulariser():
    for basis in ("laplacian", "euclidean"):
        model = TabularFB(FakeFBConfig(rank=20, basis=basis))
        b = model.backward(model.maze.xy)
        np.testing.assert_allclose(b.T @ b, model.maze.n_states * np.eye(20), atol=1e-9)


def test_exact_forward_backward_is_the_successor_measure_of_the_executed_policy(exact):
    n = exact.maze.n_states
    for w in (0, 37, 101):
        obs, z_w = _subgoal(exact, w, n)
        m_hat = exact.forward(obs, z_w)[0] @ exact.backward(obs).T
        np.testing.assert_allclose(m_hat, exact.true_successor_measure(z_w[0]), atol=1e-9)


def test_exact_rho_is_discount_to_the_power_of_geodesic_distance(exact):
    n = exact.maze.n_states
    dist = exact.maze.geodesic_distances()
    for w in (5, 60):
        obs, z_w = _subgoal(exact, w, n)
        m_sw = np.sum(exact.forward(obs, z_w)[0] * z_w, axis=-1)
        np.testing.assert_allclose(m_sw / m_sw[w], GAMMA ** dist[:, w], rtol=1e-9)


def test_exact_task_value_is_a_positive_multiple_of_the_optimal_value(exact):
    n = exact.maze.n_states
    z_r = exact.infer_task_latent(1)
    v_hat = np.sum(exact.forward(exact.maze.xy, np.tile(z_r, (n, 1)))[0] * z_r, axis=-1)
    _, v_star = optimal_policy(exact.maze, exact.task_reward(1), GAMMA)
    ratio = v_hat / v_star
    assert ratio[0] > 0
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-9)


def test_forward_ignores_the_scale_of_the_latent(noisy):
    obs = noisy.maze.xy[:9]
    z = np.random.default_rng(4).normal(size=(9, noisy.latent_dim))
    np.testing.assert_allclose(noisy.forward(obs, 7.3 * z), noisy.forward(obs, z), atol=1e-12)


def test_low_action_follows_a_shortest_path_to_the_subgoal_in_exact_mode(exact):
    maze = exact.maze
    dist = maze.geodesic_distances()
    start, goal = maze.centre_state((1, 1)), maze.centre_state((6, 6))
    z_w = exact.backward(maze.xy[[goal]])
    state = start
    for _ in range(int(dist[start, goal])):
        action = maze.action_from_vector(exact.low_action(maze.xy[[state]], z_w))[0]
        nxt = maze.next_states[state, action, 0]
        assert dist[nxt, goal] == dist[state, goal] - 1
        state = nxt
    assert state == goal


def test_high_latent_returns_the_backward_embedding_of_a_state(exact):
    obs = exact.maze.xy[[3]]
    z = exact.high_latent(obs, exact.infer_task_latent(2)[None])
    candidates = normalize_z(exact.backward(exact.maze.xy), exact.latent_dim)
    assert np.min(np.linalg.norm(candidates - z, axis=-1)) < 1e-9


def test_noisy_heads_differ_and_calls_are_deterministic(noisy):
    obs, z_w = _subgoal(noisy, 10, noisy.maze.n_states)
    f = noisy.forward(obs, z_w)
    assert not np.allclose(f[0], f[1])
    np.testing.assert_array_equal(f, noisy.forward(obs, z_w))
    fresh = TabularFB(noisy.config)
    np.testing.assert_array_equal(f, fresh.forward(obs, z_w))


def _optimal_successor_measure(model: TabularFB, z: np.ndarray) -> np.ndarray:
    policy, _ = optimal_policy(model.maze, model.reward_of_latent(z), model.discount)
    return successor_measure(model.maze, policy, model.discount)


def test_noise_on_rho_has_the_configured_scale(noisy):
    n = noisy.maze.n_states
    errors = []
    for w in range(0, n, 7):
        obs, z_w = _subgoal(noisy, w, n)
        m_sw = np.sum(noisy.forward(obs, z_w) * z_w, axis=-1)  # [2, n]
        errors.append((m_sw - _optimal_successor_measure(noisy, z_w[0])[:, w]) * (1 - GAMMA))
    assert np.std(errors) == pytest.approx(0.05, rel=0.1)


def test_greedy_low_policy_deviates_from_the_optimal_policy_under_noise(noisy):
    z_w = noisy.backward(noisy.maze.xy[[10]])[0]
    optimal, _ = optimal_policy(noisy.maze, noisy.reward_of_latent(z_w), GAMMA)
    assert np.any(noisy.policy(z_w) != optimal)


def test_optimal_low_policy_option_executes_the_optimal_policy():
    model = TabularFB(FakeFBConfig(discount=GAMMA, rho_noise=0.05, low_policy="optimal"))
    z_w = model.backward(model.maze.xy[[10]])[0]
    np.testing.assert_allclose(
        model.true_successor_measure(z_w), _optimal_successor_measure(model, z_w), atol=1e-9
    )


def test_low_rank_value_is_the_optimal_value_of_the_projected_reward():
    model = TabularFB(FakeFBConfig(rank=16, discount=GAMMA))
    n = model.maze.n_states
    z = normalize_z(model.backward(model.maze.xy[[40]])[0], model.latent_dim)
    v_hat = np.sum(model.forward(model.maze.xy, np.tile(z, (n, 1)))[0] * z, axis=-1)
    _, v_star = optimal_policy(model.maze, model.reward_of_latent(z), GAMMA)
    np.testing.assert_allclose(v_hat, v_star, atol=1e-9)


def test_task_spec_uses_the_ogbench_goal_cell(exact):
    task = exact.task_spec(1)
    np.testing.assert_allclose(task.goal_xy, [20.0, 20.0])
    np.testing.assert_allclose(task.z_r, exact.infer_task_latent(1))
    assert exact.task_reward(1).sum() == 4  # the 2 x 2 fine cells of the goal maze cell


def test_nonzero_temperature_is_rejected(exact):
    with pytest.raises(ValueError, match="temperature"):
        exact.low_action(exact.maze.xy[:1], exact.backward(exact.maze.xy[:1]), temperature=1.0)
