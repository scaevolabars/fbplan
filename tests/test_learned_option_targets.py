import jax.numpy as jnp
import numpy as np
import pytest

from fbplan.contracts import normalize_z
from fbplan.learned.option_targets import OptionConfig, direct_value, head_dot, option_terms
from fbplan.option_model import OptionModel, OptionModelConfig
from fbplan.tabular.fake_fb import FakeFBConfig, TabularFB
from fbplan.tabular.mdp import successor_measure
from fbplan.tabular.options import graph_value_iteration, option_tables

NO_CAP = OptionConfig(edge_reward="exact", rho_max=1.0)


def pair_terms(fb, s_obs, w_obs, z, is_goal, config):
    """option_terms for the pairs (s_obs[i], w_obs[i]) of a numpy FBModel."""
    z_w = fb.backward(w_obs)
    f_s_zw, f_w_zw = fb.forward(s_obs, z_w), fb.forward(w_obs, z_w)
    distinct = np.any(s_obs != w_obs, axis=-1)
    return option_terms(f_s_zw, f_w_zw, z_w, z, is_goal, distinct, config), f_w_zw, z_w


def goal_latent(fb, state):
    return np.asarray(normalize_z(fb.backward(fb.maze.xy[[state]]), fb.latent_dim))[0]


@pytest.mark.parametrize("kind", ["exact", "paper"])
def test_terms_equal_approach_a_option_model_on_a_noisy_fake(kind):
    fb = TabularFB(FakeFBConfig(rank=24, rho_noise=0.05))
    rng = np.random.default_rng(0)
    sources = fb.maze.xy[rng.choice(fb.maze.n_states, 12, replace=False)]
    targets = fb.maze.xy[rng.choice(fb.maze.n_states, 8, replace=False)]
    z = goal_latent(fb, 0)
    model = OptionModel(fb, OptionModelConfig(head_agg="min", edge_reward=kind))
    table = model.pair_table(sources, model.targets(targets), z)

    s_obs = np.repeat(sources, len(targets), axis=0)
    w_obs = np.tile(targets, (len(sources), 1))
    zs = np.tile(z, (len(s_obs), 1))
    config = OptionConfig(edge_reward=kind, rho_max=1.0)
    terms, f_w_zw, z_w = pair_terms(fb, s_obs, w_obs, zs, np.ones(len(s_obs), bool), config)
    reachable = np.all(np.asarray(head_dot(f_w_zw, z_w)) > 0, axis=0)
    assert reachable.all()  # the fake's self-measures are positive, so both models divide
    np.testing.assert_allclose(terms.rho_raw, table.rho_raw.ravel(), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(terms.reward, table.reward.ravel(), rtol=1e-4, atol=1e-4)

    f_s_z = fb.forward(sources, np.tile(z, (len(sources), 1)))
    np.testing.assert_allclose(
        direct_value(f_s_z, np.tile(z, (len(sources), 1))),
        model.direct_value(sources, z),
        rtol=1e-4,
    )


def test_terms_do_not_depend_on_the_scale_of_the_subgoal_latent():
    fb = TabularFB(FakeFBConfig(rank=24, rho_noise=0.05))
    s_obs, w_obs = fb.maze.xy[[3, 10, 20]], fb.maze.xy[[30, 5, 21]]
    z = np.tile(goal_latent(fb, 7), (3, 1))
    goal = np.ones(3, bool)
    z_w = fb.backward(w_obs)
    terms = [
        option_terms(
            fb.forward(s_obs, c * z_w), fb.forward(w_obs, c * z_w), c * z_w, z, goal, goal, NO_CAP
        )
        for c in (1.0, 3.0)
    ]
    np.testing.assert_allclose(terms[0].rho_raw, terms[1].rho_raw, rtol=1e-5)
    np.testing.assert_allclose(terms[0].reward, terms[1].reward, rtol=1e-4, atol=1e-4)


def test_a_state_and_itself_is_not_a_valid_edge():
    fb = TabularFB(FakeFBConfig(rank=24, rho_noise=0.05))
    obs = fb.maze.xy[[4, 9]]
    z = np.tile(goal_latent(fb, 1), (2, 1))
    terms, _, _ = pair_terms(fb, obs, obs, z, np.ones(2, bool), OptionConfig())
    np.testing.assert_allclose(terms.rho_raw, 1.0, rtol=1e-6)
    assert not np.asarray(terms.valid).any()
    np.testing.assert_allclose(terms.rho, 0.9)


def test_a_self_loop_stays_invalid_when_rounding_puts_its_rho_below_one():
    f_w_zw = jnp.ones((2, 1, 3))
    f_s_zw = f_w_zw * (1.0 - 1e-7)  # the raw rho a GPU gave for s = w: 0.99999994
    z_w = jnp.ones((1, 3))
    distinct = jnp.array([False])
    terms = option_terms(f_s_zw, f_w_zw, z_w, z_w, jnp.array([True]), distinct, OptionConfig())
    assert float(terms.rho_raw[0]) < 1.0
    assert not bool(terms.valid[0])


def test_goal_zero_drops_the_reward_of_goal_latents_only():
    fb = TabularFB(FakeFBConfig(rank=24, rho_noise=0.05))
    s_obs, w_obs = fb.maze.xy[[3, 3]], fb.maze.xy[[30, 30]]
    z = np.tile(goal_latent(fb, 31), (2, 1))
    is_goal = np.array([True, False])
    exact, _, _ = pair_terms(fb, s_obs, w_obs, z, is_goal, OptionConfig(edge_reward="exact"))
    zero, _, _ = pair_terms(fb, s_obs, w_obs, z, is_goal, OptionConfig(edge_reward="goal_zero"))
    assert float(zero.reward[0]) == 0.0
    assert float(zero.reward[1]) == pytest.approx(float(exact.reward[1]))


def test_nonpositive_self_measure_in_one_head_makes_the_edge_invalid():
    f = jnp.ones((2, 1, 3))
    z_w = jnp.ones((1, 3))
    f_w_zw = f.at[1].set(-1.0)
    yes = jnp.array([True])
    terms = option_terms(0.5 * f, f_w_zw, z_w, z_w, yes, yes, OptionConfig())
    assert not bool(terms.valid[0])


def test_unknown_config_values_are_rejected():
    with pytest.raises(ValueError):
        OptionConfig(edge_reward="zero")
    with pytest.raises(ValueError):
        OptionConfig(rho_max=1.5)


@pytest.fixture(scope="module")
def exact_fake():
    """Full-rank, noiseless fake with the exact pi*_z: F(s, z)^T B(w) = M^{pi*_z}(s, w)."""
    return TabularFB(FakeFBConfig(low_policy="optimal"))


def test_terms_equal_the_exact_option_model(exact_fake):
    fb, maze = exact_fake, exact_fake.maze
    goal, subgoals = 40, np.array([2, 15, 33, 60, 77])
    z = goal_latent(fb, goal)
    reward = fb.reward_of_latent(z)
    policies = [fb.policy(fb.backward(maze.xy[[w]])[0]) for w in subgoals]
    rho, r_before = option_tables(maze, policies, subgoals, reward, fb.discount)

    s_idx = np.repeat(np.arange(maze.n_states), len(subgoals))
    w_idx = np.tile(subgoals, maze.n_states)
    zs = np.tile(z, (len(s_idx), 1))
    terms, _, _ = pair_terms(
        fb, maze.xy[s_idx], maze.xy[w_idx], zs, np.ones(len(s_idx), bool), NO_CAP
    )
    np.testing.assert_allclose(terms.rho_raw, rho.ravel(), atol=1e-5)
    np.testing.assert_allclose(terms.reward, r_before.ravel(), atol=1e-4)

    v_exact = successor_measure(maze, fb.policy(z), fb.discount) @ reward
    f_z = fb.forward(maze.xy, np.tile(z, (maze.n_states, 1)))
    np.testing.assert_allclose(
        direct_value(f_z, np.tile(z, (maze.n_states, 1))), v_exact, atol=1e-4
    )


def test_switch_targets_iterate_to_the_chained_value_of_graph_value_iteration():
    """The residual form V_H = V_1 + gain, gain = max(0, max_w y - V_1), has the fixed point of
    SMDP value iteration over the same options. A low-rank fake makes pi_z poor at long range,
    so chaining gains somewhere (with the exact pi*_z it never would)."""
    fb = TabularFB(FakeFBConfig(rank=16))
    maze, goal = fb.maze, 40
    landmarks = np.unique(np.r_[np.arange(0, maze.n_states, 7), goal])
    z = goal_latent(fb, goal)
    n = len(landmarks)
    s_idx, w_idx = np.repeat(landmarks, n), np.tile(landmarks, n)
    zs = np.tile(z, (n * n, 1))
    config = OptionConfig(edge_reward="exact")
    terms, _, _ = pair_terms(fb, maze.xy[s_idx], maze.xy[w_idx], zs, np.ones(n * n, bool), config)
    rho = np.asarray(terms.rho, np.float64).reshape(n, n)
    reward = np.asarray(terms.reward, np.float64).reshape(n, n)
    v1 = np.asarray(direct_value(fb.forward(maze.xy[landmarks], zs[:n]), zs[:n]), np.float64)

    off_diagonal = ~np.eye(n, dtype=bool)
    gain = np.zeros(n)
    for _ in range(500):
        y = np.where(off_diagonal, reward + rho * (v1 + gain)[None, :], -np.inf)
        gain = np.maximum(0.0, y.max(axis=1) - v1)
    np.testing.assert_allclose(v1 + gain, graph_value_iteration(rho, reward, v1), rtol=1e-6)
    assert (gain > 1e-3 * np.abs(v1).max()).any()
