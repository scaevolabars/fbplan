"""Tests for fbplan.learned.logratio: the log-ratio advantage of T2 and its upstream-equal parts."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fbplan.frozen import forward
from fbplan.learned.checkpoint import FROZEN_MODULES, with_fresh_high_actor
from fbplan.learned.logratio import (
    advantages,
    goal_latent_mask,
    log_ratio,
    logratio_loss,
    logratio_update,
    measure,
)
from fbplan.learned.sampling import HighBatchSpec, make_trajectory_data, sample_high_batch
from fbplan.train_high import parse_args, train_chunk, validation_metrics

OBS_DIM = 5  # observation size of the shared tiny_agent (tests/conftest.py)


def with_mix(agent, mix_prob):
    """The agent with another share of Gaussian latents in high_actor_loss."""
    return agent.replace(config=dict(agent.config) | {"critic_latent_mix_prob": mix_prob})


@pytest.fixture(scope="module")
def agent(tiny_agent):
    return with_fresh_high_actor(tiny_agent.agent, 0)


@pytest.fixture(scope="module")
def batch(agent):
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(400, OBS_DIM)).astype(np.float32)
    terminals = np.zeros(400, np.float32)
    terminals[[99, 249, 399]] = 1.0
    data = make_trajectory_data(obs, terminals)
    return sample_high_batch(
        data, jax.random.PRNGKey(1), HighBatchSpec.from_agent_config(agent.config)
    )


def leaves(agent):
    return jax.tree_util.tree_leaves(agent.network.params)


def test_log_ratio_takes_the_head_mean_before_the_log():
    # head 0: 2 + 0.5 * 10 = 7; head 1: 4 + 0.99 * 10 = 13.9 (rho 2 is clipped to 0.99)
    a = log_ratio(
        jnp.array([[2.0], [4.0]]),
        jnp.array([[10.0], [10.0]]),
        jnp.array([[3.0], [5.0]]),
        jnp.array([[0.5], [2.0]]),
    )
    np.testing.assert_allclose(a, [np.log(10.45) - np.log(4.0)], rtol=1e-6)


def test_log_ratio_clips_rho_at_zero_and_floors_both_values_at_one():
    a = log_ratio(
        jnp.array([[0.5, 30.0], [0.5, 30.0]]),
        jnp.array([[10.0, 0.0], [10.0, 0.0]]),
        jnp.array([[0.2, -4.0], [0.2, -4.0]]),
        jnp.array([[-3.0, 0.5], [-3.0, 0.5]]),
    )
    # sample 0: switch 0.5 -> 1, direct 0.2 -> 1; sample 1: direct -4 -> 1
    np.testing.assert_allclose(a, [0.0, np.log(30.0)], rtol=1e-6)


def test_measure_is_upstream_successor_measure_extract(agent, batch):
    obs, w = batch["observations"], batch["high_actor_targets"]
    z = jax.random.normal(jax.random.PRNGKey(2), (obs.shape[0], agent.config["latent_dim"]))
    b_w = agent.network.select("backward_repr")(w)
    np.testing.assert_array_equal(
        measure(agent, obs, z, b_w), agent.successor_measure_extract(obs, z, b_w)
    )


def t2_advantage(agent, obs, w, g):
    """A of goal latents by the formula of docs/design.md §8 (T2), in float64 from F and B."""
    b_w = np.asarray(agent.network.select("backward_repr")(w), np.float64)
    b_g = np.asarray(agent.network.select("backward_repr")(g), np.float64)
    z = b_g / np.linalg.norm(b_g, axis=-1, keepdims=True) * np.sqrt(b_g.shape[-1])

    def f(x, intent):  # F_h(x, intent) of both heads [2, N, d]
        return np.asarray(forward(agent, x, intent.astype(np.float32)), np.float64)

    f_s_zw, f_w_zw, f_w_z, f_s_z = f(obs, b_w), f(w, b_w), f(w, b_g), f(obs, b_g)
    rho = np.clip(np.sum(f_s_zw * b_w, -1) / np.sum(f_w_zw * b_w, -1), 0.0, 0.99)
    switch = np.mean(np.sum(f_s_zw * z, -1) + rho * np.sum(f_w_z * z, -1), axis=0)
    direct = np.mean(np.sum(f_s_z * z, -1), axis=0)
    return np.log(np.maximum(switch, 1.0)) - np.log(np.maximum(direct, 1.0))


def test_advantages_give_the_t2_formula_computed_from_f_and_b(agent, batch):
    """The wiring of advantages(): which state and which latent enter each F term."""
    obs, w, g = batch["observations"], batch["high_actor_targets"], batch["high_actor_goals"]
    b = agent.network.select("backward_repr")
    z_w, z_raw = b(w), b(g)
    _, adv_log = advantages(agent, obs, w, agent.normalize_z(z_raw), z_raw, z_w)
    expected = t2_advantage(agent, obs, w, g)
    assert np.count_nonzero(expected) >= 5  # the floor at 1 leaves samples to compare (6 of 32)
    np.testing.assert_allclose(adv_log, expected, rtol=1e-4, atol=1e-4)


def test_goal_mask_marks_the_latents_that_sample_latents_takes_from_b(agent, batch):
    goals = batch["high_actor_goals"]
    rng = jax.random.PRNGKey(3)
    _, z_raw = agent.sample_latents(goals, rng, 0.5)
    b_g = agent.network.select("backward_repr")(goals)  # not jitted: equal up to rounding
    from_b = np.all(np.isclose(z_raw, b_g, rtol=1e-5, atol=1e-6), axis=-1)
    mask = np.asarray(goal_latent_mask(rng, goals.shape[0], 0.5))
    np.testing.assert_array_equal(mask, from_b)
    assert 0 < mask.sum() < mask.size


def test_with_gaussian_latents_only_the_update_is_upstream(agent, batch):
    """critic_latent_mix_prob 1: no goal latent, so the log-ratio part is unused."""
    gauss = with_mix(agent, 1.0)
    ours, info = logratio_update(gauss, batch, 2.45)
    theirs, upstream_info = gauss.update(batch)
    for a, b in zip(leaves(ours), leaves(theirs), strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ours.rng, theirs.rng)
    for k in ("actor_loss", "adv", "bc_log_prob", "mse", "saturated"):
        np.testing.assert_array_equal(info[f"high_actor/{k}"], upstream_info[f"high_actor/{k}"])
    assert float(info["high_actor/goal_share"]) == 0.0


def test_goal_latents_use_the_log_ratio_weight(agent, batch):
    """critic_latent_mix_prob 0: all latents are B(g). At goal_alpha 0 every weight is 1."""
    goal = with_mix(agent, 0.0)
    loss, info = logratio_loss(goal, batch, None, 0.0)
    assert float(info["goal_share"]) == 1.0
    np.testing.assert_allclose(loss, -info["bc_log_prob"], rtol=1e-6)
    assert float(info["ess_goal"]) == pytest.approx(1.0)
    sharp, sharp_info = logratio_loss(goal, batch, None, 2.45)
    assert float(sharp) != float(loss) and float(sharp_info["ess_goal"]) < 1.0
    # the upstream proxy is still logged, from the same latents (total_loss is jitted: rounding)
    _, upstream_info = goal.total_loss(batch, grad_params=None)
    np.testing.assert_allclose(info["adv"], upstream_info["high_actor/adv"], rtol=1e-5)


def test_logratio_training_updates_only_pi_h(tiny_agent):
    fresh = with_fresh_high_actor(tiny_agent.agent, 0)
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(400, OBS_DIM)).astype(np.float32)
    terminals = np.zeros(400, np.float32)
    terminals[[99, 249, 399]] = 1.0
    data = make_trajectory_data(obs, terminals)
    spec = HighBatchSpec.from_agent_config(fresh.config)
    key = jax.random.PRNGKey(0)
    trained, info = train_chunk(fresh, data, key, 3, spec, 2.45)
    upstream, _ = train_chunk(fresh, data, key, 3, spec)
    for name in FROZEN_MODULES:
        for a, b in zip(
            jax.tree_util.tree_leaves(trained.network.params[name]),
            jax.tree_util.tree_leaves(fresh.network.params[name]),
            strict=True,
        ):
            np.testing.assert_array_equal(a, b)
    pi_h = [
        jax.tree_util.tree_leaves(a.network.params["modules_high_actor"])
        for a in (trained, upstream)
    ]
    assert not all(np.array_equal(a, b) for a, b in zip(*pi_h, strict=True))
    np.testing.assert_array_equal(trained.rng, upstream.rng)
    assert all(np.isfinite(float(v)) for v in info.values())
    val = validation_metrics(trained, data, key, spec, 2.45)
    assert {"high_actor/actor_loss", "high_actor/adv_log", "high_actor/ess_goal"} <= set(val)


def test_cli_selects_the_log_ratio_advantage():
    base = ["--env", "antmaze-medium-navigate-v0", "--seed", "0"]
    assert parse_args(base).advantage == "upstream"
    args = parse_args([*base, "--name", "logratio", "--advantage", "logratio"])
    assert (args.name, args.advantage, args.goal_alpha) == ("logratio", "logratio", 2.45)
    assert json.dumps(args.goal_alpha) == "2.45"
