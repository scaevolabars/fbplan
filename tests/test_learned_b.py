"""Tests for approach B: switch batches, V_H (stage 1), the new pi_h (stage 2), train_b glue."""

import dataclasses

import flax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fbplan import frozen
from fbplan.frozen import forward, trained_key
from fbplan.learned import train_b
from fbplan.learned.checkpoint import (
    FROZEN_MODULES,
    load_b_model,
    with_fresh_high_actor,
)
from fbplan.learned.high_policy import masked_mean_std, policy_update, switch_advantage
from fbplan.learned.option_targets import OptionConfig, direct_value, option_terms
from fbplan.learned.sampling import (
    HighBatchSpec,
    SwitchBatch,
    make_trajectory_data,
    sample_switch_batch,
    trajectory_window_latents,
)
from fbplan.learned.train_b import (
    Split,
    Stage,
    advantage_stds,
    calibrate_alpha,
    parse_args,
    policy_chunk,
    shared_alpha,
    value_chunk,
)
from fbplan.learned.value import (
    ValueConfig,
    ValueState,
    expectile_loss,
    load_value,
    save_value,
    switch_targets,
    value_update,
)

OBS_DIM, LATENT_DIM = 5, 4  # sizes of the shared tiny_agent (tests/conftest.py)
TINY_VALUE = ValueConfig(hidden_dims=(16, 16))
SPEC = HighBatchSpec(batch_size=64, discount=0.9, actor_geom_sample=False, actor_p_randomgoal=0.3)
STAGE = Stage(SPEC, 0.5, TINY_VALUE)
TRAJ_ENDS = [99, 249, 399]
MEDIUM = "antmaze-medium-navigate-v0"


@pytest.fixture(scope="module")
def agent(tiny_agent):
    """The shared small agent, with head 1 of F copied from head 0.

    With random weights the sign of F(w, z_w)^T z_w is random per head, and a head with negative
    self-measures makes every edge invalid; a trained F has positive ones. With equal heads about
    40 % of the sampled edges are valid.
    """
    agent = tiny_agent.agent
    params = flax.core.unfreeze(agent.network.params)
    params["modules_forward_repr"] = jax.tree_util.tree_map(
        lambda x: x.at[1].set(x[0]), params["modules_forward_repr"]
    )
    return agent.replace(network=agent.network.replace(params=params))


@pytest.fixture(scope="module")
def data():
    obs = np.random.default_rng(0).normal(size=(400, OBS_DIM)).astype(np.float32)
    obs[:, 0] = np.arange(400)  # the first coordinate is the state index
    terminals = np.zeros(400, np.float32)
    terminals[TRAJ_ENDS] = 1.0
    return make_trajectory_data(obs, terminals)


@pytest.fixture(scope="module")
def split(agent, data):
    return Split(data, agent.network.select("backward_repr")(data.observations))


@pytest.fixture(scope="module")
def value():
    return ValueState.create(jax.random.PRNGKey(0), OBS_DIM, LATENT_DIM, TINY_VALUE)


def leaves(tree):
    return jax.tree_util.tree_leaves(tree)


def same(a, b):
    return all(np.array_equal(x, y) for x, y in zip(leaves(a), leaves(b), strict=True))


# ----- sampling and CLI -----


def test_switch_batch_looks_up_latents_and_normalizes_tasks(data):
    table = jnp.asarray(np.random.default_rng(1).normal(size=(400, LATENT_DIM)), jnp.float32)
    batch = sample_switch_batch(data, table, jax.random.PRNGKey(0), SPEC, gaussian_prob=0.5)
    w = np.asarray(batch.subgoals[:, 0]).astype(int)
    np.testing.assert_array_equal(batch.subgoal_latents, np.asarray(table)[w])
    np.testing.assert_allclose(
        np.linalg.norm(batch.task_latents, axis=-1), np.sqrt(LATENT_DIM), rtol=1e-5
    )
    assert 0.3 < float(batch.is_goal.mean()) < 0.7


def test_window_latents_average_within_the_trajectory_only():
    latents = np.arange(12, dtype=np.float32)[:, None]
    terminals = np.zeros(12)
    terminals[[5, 11]] = 1.0
    window = trajectory_window_latents(latents, terminals, half_width=1)[:, 0]
    np.testing.assert_allclose(window[[0, 1, 5, 6]], [0.5, 1.0, 4.5, 6.5])


def test_ablation_runs_get_their_own_folders():
    env = ["--env", MEDIUM, "--seed", "0"]
    main = parse_args(["value", *env])
    ablation = parse_args(["value", *env, "--rho-cap", "1.0", "--name", "value-single-cap1"])
    assert (main.name, ablation.name) == ("value-single", "value-single-cap1")
    assert ablation.rho_cap == 1.0
    policy = parse_args(["policy", *env, "--variant", "region"])
    assert (policy.name, policy.value_name) == ("region", "value-region")


# ----- stage 1 -----


def test_value_is_never_below_the_single_intention_value(agent, value, split):
    batch = split.batch(jax.random.PRNGKey(1), STAGE)
    v1 = direct_value(forward(agent, batch.observations, batch.task_latents), batch.task_latents)
    assert value.gain(batch.observations, batch.task_latents).shape == (2, SPEC.batch_size)
    assert np.all(np.asarray(value.value(v1, batch.observations, batch.task_latents) >= v1))


def test_value_update_moves_params_and_the_target_by_ema(agent, value, split):
    new, info = value_update(value, agent, split.batch(jax.random.PRNGKey(2), STAGE), TINY_VALUE)
    assert np.isfinite(float(info["loss"])) and int(new.step) == 1
    assert float(info["valid"]) > 0.2
    assert not same(new.params, value.params)
    for p, t, t0 in zip(
        leaves(new.params), leaves(new.target_params), leaves(value.target_params), strict=True
    ):
        np.testing.assert_allclose(
            t, TINY_VALUE.ema * p + (1 - TINY_VALUE.ema) * t0, rtol=1e-5, atol=1e-7
        )


def test_switch_targets_assemble_the_option_terms_of_s_and_w(agent, value, split):
    """y = R(s, w) + rho(s, w) V_H(w) - hop_cost, with s and w in their places."""
    config = dataclasses.replace(TINY_VALUE, hop_cost=0.1)
    batch = split.batch(jax.random.PRNGKey(6), STAGE)
    y, v1_s, terms = switch_targets(agent, value, value.params, batch, config)
    s, w, z_w, z = batch.observations, batch.subgoals, batch.subgoal_latents, batch.task_latents
    distinct = jnp.any(s != w, axis=-1)
    expected = option_terms(
        forward(agent, s, z_w),
        forward(agent, w, z_w),
        z_w,
        z,
        batch.is_goal,
        distinct,
        config.option,
    )
    v_w = value.value(direct_value(forward(agent, w, z), z), w, z)
    np.testing.assert_allclose(y, expected.reward + expected.rho * v_w - 0.1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(v1_s, direct_value(forward(agent, s, z), z), rtol=1e-6)
    np.testing.assert_array_equal(terms.valid, expected.valid)


def test_the_v1_control_replaces_v_h_by_v1_in_target_and_advantage(agent, value, split):
    """critic="v_1" (H1 of docs/hypotheses/h-learn.md): the gain is in neither y nor A."""
    control = dataclasses.replace(TINY_VALUE, critic="v_1")
    batch = split.batch(jax.random.PRNGKey(7), STAGE)
    y, v1_s, terms = switch_targets(agent, value, value.params, batch, control)
    w, z = batch.subgoals, batch.task_latents
    v1_w = direct_value(forward(agent, w, z), z)
    np.testing.assert_allclose(y, terms.reward + terms.rho * v1_w, rtol=1e-5, atol=1e-5)
    adv, _ = switch_advantage(agent, value, batch, control)
    np.testing.assert_allclose(adv, y - v1_s, rtol=1e-5, atol=1e-5)
    learned, _ = switch_advantage(agent, value, batch, TINY_VALUE)
    assert not np.allclose(adv, learned)  # the gain is not zero, so the two paths differ
    args = parse_args(
        [
            "policy",
            "--env",
            MEDIUM,
            "--seed",
            "0",
            "--variant",
            "clean",
            "--critic",
            "v_1",
            "--name",
            "clean-v1-awr2x",
        ]
    )
    assert (args.critic, args.value_name) == ("v_1", "value-single")
    stored = dataclasses.asdict(TINY_VALUE)
    del stored["critic"]  # flags.json of a V_H trained before the field existed
    assert ValueConfig.from_dict(stored).critic == "v_h"


def test_expectile_loss_weights_errors_above_more():
    diff = jnp.array([1.0, -1.0])
    np.testing.assert_allclose(expectile_loss(diff, 0.9), [0.9, 0.1])


def test_value_config_round_trips_through_its_json_dict():
    config = ValueConfig(option=OptionConfig(edge_reward="exact", rho_cap=0.95), hop_cost=0.1)
    assert ValueConfig.from_dict(dataclasses.asdict(config)) == config


def test_value_round_trip_through_its_folder(value, tmp_path, monkeypatch):
    monkeypatch.setenv("FBPLAN_APPROACH_B_DIR", str(tmp_path))
    save_value(
        value,
        TINY_VALUE,
        {"seed": 0},
        frozen.trained_dir(trained_key("approach_b", MEDIUM, 0, "v")),
    )
    loaded, config = load_value(MEDIUM, "v", 0, OBS_DIM, LATENT_DIM)
    assert config == TINY_VALUE
    assert same(loaded.params, value.params) and same(loaded.target_params, value.target_params)


# ----- stage 2 -----


def test_policy_update_changes_only_pi_h(agent, value, split):
    fresh = with_fresh_high_actor(agent, 0)
    batch = split.batch(jax.random.PRNGKey(3), STAGE)
    trained, info = policy_update(fresh, value, batch, jnp.float32(0.1), TINY_VALUE)
    for name in FROZEN_MODULES:
        assert same(trained.network.params[name], fresh.network.params[name]), name
    assert not same(
        trained.network.params["modules_high_actor"], fresh.network.params["modules_high_actor"]
    )
    assert all(np.isfinite(float(v)) for v in info.values())


def test_invalid_edges_give_no_policy_gradient(agent, value, data):
    """Subgoal = current state: an untrusted self-loop, so the sample must not move pi_h."""
    fresh = with_fresh_high_actor(agent, 0)
    states = data.observations[:16]
    subgoals = states
    z = jnp.asarray(np.random.default_rng(2).normal(size=(16, LATENT_DIM)), jnp.float32)
    batch = SwitchBatch(
        observations=states,
        subgoals=subgoals,
        subgoal_latents=agent.network.select("backward_repr")(subgoals),
        task_latents=z * 2.0 / jnp.linalg.norm(z, axis=-1, keepdims=True),
        is_goal=jnp.ones(16, bool),
    )
    trained, info = policy_update(fresh, value, batch, jnp.float32(0.1), TINY_VALUE)
    assert float(info["valid"]) == 0.0
    assert same(
        trained.network.params["modules_high_actor"], fresh.network.params["modules_high_actor"]
    )


def test_advantage_spread_counts_valid_edges_only(agent, value, split):
    key = jax.random.PRNGKey(4)
    upstream_std, switch_std = advantage_stds(agent, value, split, key, STAGE)
    _, k_b = jax.random.split(key)
    adv, valid = switch_advantage(agent, value, split.batch(k_b, STAGE), TINY_VALUE)
    assert 0 < valid.sum() < len(valid)
    np.testing.assert_allclose(switch_std, np.std(np.asarray(adv)[np.asarray(valid)]), rtol=1e-4)
    assert np.isfinite(float(upstream_std)) and float(upstream_std) > 0


def test_calibrated_alpha_gives_alpha_times_a_the_upstream_spread(agent, value, split, monkeypatch):
    monkeypatch.setattr(train_b, "CALIBRATION_BATCHES", 2)
    key = jax.random.PRNGKey(5)
    alpha, info = calibrate_alpha(agent, value, split, key, STAGE)
    stds = [advantage_stds(agent, value, split, k, STAGE) for k in jax.random.split(key, 2)]
    upstream = np.mean([float(s[0]) for s in stds])
    switch = np.mean([float(s[1]) for s in stds])
    assert info["upstream_adv_std"] == pytest.approx(upstream)
    assert info["switch_adv_std"] == pytest.approx(switch)
    assert alpha * switch == pytest.approx(float(agent.config["high_alpha"]) * upstream)


def test_shared_alpha_matches_median_spreads_over_several_critics(agent, value, split):
    other = ValueState.create(jax.random.PRNGKey(9), OBS_DIM, LATENT_DIM, TINY_VALUE)
    result = shared_alpha(agent, [value, other], split, jax.random.PRNGKey(0), STAGE, 0.2, 6)
    up, sw = result["upstream_adv_std_median"], result["switch_adv_std_median"]
    assert result["alpha"] * sw == pytest.approx(0.2 * up)
    assert result["upstream_adv_std_q10_q90"][0] <= up <= result["upstream_adv_std_q10_q90"][1]
    assert result["switch_adv_std_q10_q90"][0] <= sw <= result["switch_adv_std_q10_q90"][1]
    single = shared_alpha(agent, [value], split, jax.random.PRNGKey(0), STAGE, 0.2, 6)
    assert single["upstream_adv_std_median"] == up  # the critics enter only the switch spread
    assert single["switch_adv_std_median"] != sw


def test_one_temperature_for_all_seeds_excludes_a_fixed_one():
    env = ["--env", MEDIUM, "--seed", "0", "--variant", "clean"]
    args = parse_args(["policy", *env, "--alpha-from", "cal.json", "--name", "clean-awr2x"])
    assert (str(args.alpha_from), args.alpha, args.name) == ("cal.json", None, "clean-awr2x")
    with pytest.raises(SystemExit):
        parse_args(["policy", *env, "--alpha", "0.1", "--alpha-from", "cal.json"])
    calib = parse_args(["calibrate", "--env", MEDIUM, "--upstream-alpha", "0.2", "--out", "c.json"])
    assert (calib.seeds, calib.batches, calib.value_name) == ([0, 1, 2, 3, 4], 200, "value-single")


def test_masked_mean_std_ignores_masked_samples():
    x = jnp.array([1.0, 3.0, 100.0])
    mean, std = masked_mean_std(x, jnp.array([True, True, False]))
    assert (float(mean), float(std)) == (2.0, 1.0)


# ----- both stages on the real checkpoint -----


@pytest.fixture(scope="module")
def medium(medium_model):
    return medium_model, Split.load(medium_model, "val", "single")


def test_both_stages_run_on_the_real_checkpoint_and_reload(medium, tmp_path, monkeypatch):
    model, split = medium
    stage = Stage(HighBatchSpec.from_agent_config(model.agent.config), 0.5, ValueConfig())
    value = ValueState.create(jax.random.PRNGKey(0), split.obs_dim, model.latent_dim, stage.value)
    value, info = value_chunk(value, model.agent, split, jax.random.PRNGKey(1), 2, stage)
    assert np.isfinite(float(info["loss"])) and 0.5 < float(info["valid"]) <= 1.0

    fresh = with_fresh_high_actor(model.agent, 0)
    alpha = jnp.float32(0.1)
    trained, info = policy_chunk(fresh, value, split, jax.random.PRNGKey(2), alpha, 2, stage)
    assert np.isfinite(float(info["actor_loss"]))
    for name in FROZEN_MODULES:
        assert same(trained.network.params[name], model.agent.network.params[name]), name

    monkeypatch.setenv("FBPLAN_APPROACH_B_DIR", str(tmp_path))
    flags = {"env_name": f"ogbench-{MEDIUM}"}
    frozen.save_trained_checkpoint(
        trained, flags, frozen.trained_dir(trained_key("approach_b", MEDIUM, 0, "clean"))
    )
    loaded = load_b_model(MEDIUM, "clean", 0)
    obs = split.data.observations[:3]
    z = jnp.ones((3, model.latent_dim))
    expected = frozen.FrozenFBModel(MEDIUM, trained).high_latent(obs, z)
    np.testing.assert_array_equal(np.asarray(loaded.high_latent(obs, z)), np.asarray(expected))
