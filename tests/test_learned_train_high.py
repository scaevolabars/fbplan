"""Tests for fbplan.train_high: the fresh pi_h, the training chunk and the logged loop."""

import jax
import numpy as np
import pandas as pd
import pytest

from fbplan.learned.checkpoint import FROZEN_MODULES, with_fresh_high_actor
from fbplan.learned.sampling import HighBatchSpec, make_trajectory_data
from fbplan.train_high import train, train_chunk

OBS_DIM = 5  # observation size of the shared tiny_agent (tests/conftest.py)


@pytest.fixture(scope="module")
def frozen_agent(tiny_agent):
    """The shared small agent as fbplan.frozen restores one: its config holds lists, as in
    flags.json."""
    config = {k: list(v) if isinstance(v, tuple) else v for k, v in tiny_agent.config.items()}
    return tiny_agent.agent.replace(config=config)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(400, OBS_DIM)).astype(np.float32)
    terminals = np.zeros(400, np.float32)
    terminals[[99, 249, 399]] = 1.0
    return make_trajectory_data(obs, terminals)


def params_of(agent, name):
    return jax.tree_util.tree_leaves(agent.network.params[name])


def same(a, b) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_fresh_agent_keeps_frozen_modules_and_reinitializes_pi_h(frozen_agent):
    fresh = with_fresh_high_actor(frozen_agent, 0)
    for name in FROZEN_MODULES:
        assert same(params_of(fresh, name), params_of(frozen_agent, name)), name
    assert not same(
        params_of(fresh, "modules_high_actor"), params_of(frozen_agent, "modules_high_actor")
    )


def test_pi_h_init_depends_only_on_seed(frozen_agent):
    a = with_fresh_high_actor(frozen_agent, 0)
    b = with_fresh_high_actor(frozen_agent, 0)
    c = with_fresh_high_actor(frozen_agent, 1)
    assert same(params_of(a, "modules_high_actor"), params_of(b, "modules_high_actor"))
    assert not same(params_of(a, "modules_high_actor"), params_of(c, "modules_high_actor"))


def test_training_updates_only_pi_h(frozen_agent, data):
    fresh = with_fresh_high_actor(frozen_agent, 0)
    spec = HighBatchSpec.from_agent_config(fresh.config)
    trained, info = train_chunk(fresh, data, jax.random.PRNGKey(0), 3, spec)
    for name in FROZEN_MODULES:
        assert same(params_of(trained, name), params_of(fresh, name)), name
    assert not same(
        params_of(trained, "modules_high_actor"), params_of(fresh, "modules_high_actor")
    )
    assert int(trained.network.step) == int(fresh.network.step) + 3
    assert all(np.isfinite(float(v)) for v in info.values())


def test_train_logs_one_row_per_interval(frozen_agent, data, tmp_path):
    fresh = with_fresh_high_actor(frozen_agent, 0)
    train(
        fresh,
        data,
        data,
        steps=4,
        log_every=2,
        key=jax.random.PRNGKey(0),
        log_path=tmp_path / "train.csv",
    )
    log = pd.read_csv(tmp_path / "train.csv")
    assert list(log["step"]) == [2, 4]
    assert {"training/high_actor/actor_loss", "validation/high_actor/actor_loss"} <= set(log)


def test_train_rejects_partial_log_interval(frozen_agent, data, tmp_path):
    fresh = with_fresh_high_actor(frozen_agent, 0)
    with pytest.raises(ValueError):
        train(
            fresh,
            data,
            data,
            steps=5,
            log_every=2,
            key=jax.random.PRNGKey(0),
            log_path=tmp_path / "train.csv",
        )


def test_checkpoint_config_is_the_upstream_pi_h_recipe(medium_model):
    config = medium_model.agent.config
    assert HighBatchSpec.from_agent_config(config) == HighBatchSpec(
        batch_size=1024, discount=0.99, actor_geom_sample=False, actor_p_randomgoal=0.0
    )
    assert config["high_alpha"] == 0.1 and config["lr"] == 3e-4
    assert config["critic_latent_mix_prob"] == 0.5
