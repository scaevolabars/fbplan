import dataclasses

import flax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fbplan import frozen
from fbplan.learned.high_policy import matched_alpha, policy_update
from fbplan.learned.option_targets import OptionConfig, direct_value
from fbplan.learned.sampling import (
    HighBatchSpec,
    SwitchBatch,
    make_trajectory_data,
    region_latents,
    sample_switch_batch,
)
from fbplan.learned.train_b import (
    Split,
    approach_b_dir,
    config_from_dict,
    example_batch,
    load_b_model,
    load_value,
    policy_chunk,
    save_value,
    value_chunk,
)
from fbplan.learned.value import ValueConfig, ValueState, expectile_loss, forward, value_update
from fbplan.train_high import FROZEN_MODULES, save_checkpoint, with_fresh_high_actor

fbpiswitch = frozen.upstream_module("agents.fbpiswitch")
OBS_DIM, LATENT_DIM = 5, 4
TINY_VALUE = ValueConfig(hidden_dims=(16, 16))
SPEC = HighBatchSpec(batch_size=64, discount=0.9, actor_geom_sample=False, actor_p_randomgoal=0.3)
TRAJ_ENDS = [99, 249, 399]


@pytest.fixture(scope="module")
def agent():
    """Upstream agent with small networks; its F, B and pi_l stand in for the frozen ones.

    Head 1 of F copies head 0. With random weights the sign of F(w, z_w)^T z_w is random per
    head, and a head with negative self-measures makes every edge invalid; a trained F has
    positive ones. With equal heads about 40 % of the sampled edges are valid.
    """
    config = fbpiswitch.get_config().to_dict() | {
        "actor_hidden_dims": (16, 16),
        "forward_repr_hidden_dims": (16, 16),
        "backward_repr_hidden_dims": (16, 16),
        "latent_dim": LATENT_DIM,
        "batch_size": 32,
    }
    agent = fbpiswitch.FBpiSwitchAgent.create(123, example_batch_of(OBS_DIM), config)
    params = flax.core.unfreeze(agent.network.params)
    params["modules_forward_repr"] = jax.tree_util.tree_map(
        lambda x: x.at[1].set(x[0]), params["modules_forward_repr"]
    )
    return agent.replace(network=agent.network.replace(params=params))


def example_batch_of(obs_dim):
    return {
        "observations": np.zeros((1, obs_dim), np.float32),
        "actions": np.zeros((1, 2), np.float32),
    }


@pytest.fixture(scope="module")
def data():
    obs = np.random.default_rng(0).normal(size=(400, OBS_DIM)).astype(np.float32)
    obs[:, 0] = np.arange(400)  # the first coordinate is the state index
    terminals = np.zeros(400, np.float32)
    terminals[TRAJ_ENDS] = 1.0
    return make_trajectory_data(obs, terminals)


@pytest.fixture(scope="module")
def latents(agent, data):
    return agent.network.select("backward_repr")(data.observations)


@pytest.fixture(scope="module")
def value():
    return ValueState.create(jax.random.PRNGKey(0), OBS_DIM, LATENT_DIM, TINY_VALUE)


def leaves(tree):
    return jax.tree_util.tree_leaves(tree)


def same(a, b):
    return all(np.array_equal(x, y) for x, y in zip(leaves(a), leaves(b), strict=True))


# ----- sampling -----


def test_switch_batch_looks_up_latents_and_normalizes_tasks(data):
    table = jnp.asarray(np.random.default_rng(1).normal(size=(400, LATENT_DIM)), jnp.float32)
    batch = sample_switch_batch(data, table, jax.random.PRNGKey(0), SPEC, gaussian_prob=0.5)
    w = np.asarray(batch.subgoals[:, 0]).astype(int)
    np.testing.assert_array_equal(batch.subgoal_latents, np.asarray(table)[w])
    np.testing.assert_allclose(
        np.linalg.norm(batch.task_latents, axis=-1), np.sqrt(LATENT_DIM), rtol=1e-5
    )
    assert 0.3 < float(batch.is_goal.mean()) < 0.7


def test_region_latents_average_within_the_trajectory_only():
    latents = np.arange(12, dtype=np.float32)[:, None]
    terminals = np.zeros(12)
    terminals[[5, 11]] = 1.0
    region = region_latents(latents, terminals, half_width=1)[:, 0]
    np.testing.assert_allclose(region[[0, 1, 5, 6]], [0.5, 1.0, 4.5, 6.5])


# ----- stage 1 -----


def test_value_is_never_below_the_single_intention_value(agent, value, data, latents):
    batch = sample_switch_batch(data, latents, jax.random.PRNGKey(1), SPEC, 0.5)
    v1 = direct_value(forward(agent, batch.observations, batch.task_latents), batch.task_latents)
    assert value.gain(batch.observations, batch.task_latents).shape == (2, SPEC.batch_size)
    assert np.all(np.asarray(value.value(agent, batch.observations, batch.task_latents) >= v1))


def test_value_update_moves_params_and_the_target_by_ema(agent, value, data, latents):
    batch = sample_switch_batch(data, latents, jax.random.PRNGKey(2), SPEC, 0.5)
    new, info = value_update(value, agent, batch, TINY_VALUE)
    assert np.isfinite(float(info["loss"])) and int(new.step) == 1
    assert float(info["valid"]) > 0.2
    assert not same(new.params, value.params)
    for p, t, t0 in zip(
        leaves(new.params), leaves(new.target_params), leaves(value.target_params), strict=True
    ):
        np.testing.assert_allclose(
            t, TINY_VALUE.ema * p + (1 - TINY_VALUE.ema) * t0, rtol=1e-5, atol=1e-7
        )


def test_expectile_loss_weights_errors_above_more():
    diff = jnp.array([1.0, -1.0])
    np.testing.assert_allclose(expectile_loss(diff, 0.9), [0.9, 0.1])


def test_value_config_round_trips_through_json_dict():
    config = ValueConfig(option=OptionConfig(edge_reward="exact", rho_max=0.95), hop_cost=0.1)
    assert config_from_dict(dataclasses.asdict(config)) == config


def test_value_checkpoint_round_trip(value, tmp_path):
    save_value(value, {"value_config": dataclasses.asdict(TINY_VALUE)}, tmp_path)
    loaded, config = load_value(tmp_path, OBS_DIM, LATENT_DIM)
    assert config == TINY_VALUE
    assert same(loaded.params, value.params) and same(loaded.target_params, value.target_params)


# ----- stage 2 -----


def test_policy_update_changes_only_pi_h(agent, value, data, latents):
    fresh = with_fresh_high_actor(agent, 0, example_batch_of(OBS_DIM))
    batch = sample_switch_batch(data, latents, jax.random.PRNGKey(3), SPEC, 0.5)
    trained, info = policy_update(fresh, value, batch, jnp.float32(0.1), TINY_VALUE)
    for name in FROZEN_MODULES:
        assert same(trained.network.params[name], fresh.network.params[name]), name
    assert not same(
        trained.network.params["modules_high_actor"], fresh.network.params["modules_high_actor"]
    )
    assert all(np.isfinite(float(v)) for v in info.values())


def test_invalid_edges_give_no_policy_gradient(agent, value, data):
    """w = s has rho = 1: an untrusted self-loop, so the sample must not move pi_h."""
    fresh = with_fresh_high_actor(agent, 0, example_batch_of(OBS_DIM))
    s = data.observations[:16]
    b_s = agent.network.select("backward_repr")(s)
    z = jnp.asarray(np.random.default_rng(2).normal(size=(16, LATENT_DIM)), jnp.float32)
    batch = SwitchBatch(
        s, s, b_s, z * 2.0 / jnp.linalg.norm(z, axis=-1, keepdims=True), jnp.ones(16, bool)
    )
    trained, info = policy_update(fresh, value, batch, jnp.float32(0.1), TINY_VALUE)
    assert float(info["valid"]) == 0.0
    assert same(
        trained.network.params["modules_high_actor"], fresh.network.params["modules_high_actor"]
    )


def test_matched_alpha_gives_the_upstream_spread():
    alpha = matched_alpha(upstream_adv_std=20.0, switch_adv_std=50.0, upstream_alpha=0.1)
    assert alpha * 50.0 == pytest.approx(0.1 * 20.0)


# ----- both stages on the real checkpoint -----

MEDIUM = "antmaze-medium-navigate-v0"


@pytest.fixture(scope="module")
def medium():
    if not (frozen.checkpoint_dir(MEDIUM) / "params.pkl").exists():
        pytest.skip("medium checkpoint absent: run `uv run python -m fbplan.frozen`")
    if not frozen.dataset_path(MEDIUM, "val").exists():
        pytest.skip("medium validation split absent: call fbplan.frozen.load_dataset once")
    model = frozen.load_model(MEDIUM)
    return model, Split.load(model, "val", "single")


def test_both_stages_run_on_the_real_checkpoint_and_reload(medium, tmp_path, monkeypatch):
    model, split = medium
    spec = HighBatchSpec.from_agent_config(model.agent.config)
    obs_dim = split.data.observations.shape[-1]
    value = ValueState.create(jax.random.PRNGKey(0), obs_dim, model.latent_dim, ValueConfig())
    value, info = value_chunk(
        value,
        model.agent,
        split.data,
        split.latents,
        jax.random.PRNGKey(1),
        2,
        spec,
        0.5,
        ValueConfig(),
    )
    assert np.isfinite(float(info["loss"])) and 0.5 < float(info["valid"]) <= 1.0

    fresh = with_fresh_high_actor(model.agent, 0, example_batch(model.agent, obs_dim))
    trained, info = policy_chunk(
        fresh,
        value,
        split.data,
        split.latents,
        jax.random.PRNGKey(2),
        jnp.float32(0.1),
        2,
        spec,
        0.5,
        ValueConfig(),
    )
    assert np.isfinite(float(info["actor_loss"]))
    for name in FROZEN_MODULES:
        assert same(trained.network.params[name], model.agent.network.params[name]), name

    monkeypatch.setenv("FBPLAN_APPROACH_B_DIR", str(tmp_path))
    save_checkpoint(trained, {}, approach_b_dir(MEDIUM, "clean", 0))
    loaded = load_b_model(MEDIUM, "clean", 0)
    obs = split.data.observations[:3]
    z = jnp.ones((3, model.latent_dim))
    expected = frozen.FrozenFBModel(MEDIUM, trained).high_latent(obs, z)
    np.testing.assert_array_equal(np.asarray(loaded.high_latent(obs, z)), np.asarray(expected))
