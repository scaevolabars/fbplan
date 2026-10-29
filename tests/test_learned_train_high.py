import pickle
import sys
from pathlib import Path

import flax
import jax
import numpy as np
import pandas as pd
import pytest

from fbplan.learned.sampling import HighBatchSpec, make_trajectory_data
from fbplan.train_high import (
    FROZEN_MODULES,
    save_checkpoint,
    train,
    train_chunk,
    with_fresh_high_actor,
)

# TODO(e-learner): use the a-frozen import helper once it is on main.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "ssm"))
from agents.fbpiswitch import FBpiSwitchAgent, get_config  # noqa: E402

OBS_DIM, ACT_DIM = 5, 2


def tiny_config() -> dict:
    """Upstream fbpiswitch config with small networks, as flags.json would give it (lists)."""
    config = get_config().to_dict()
    config |= {
        "actor_hidden_dims": [16, 16],
        "forward_repr_hidden_dims": [16, 16],
        "backward_repr_hidden_dims": [16, 16],
        "latent_dim": 4,
        "batch_size": 32,
    }
    return config


def example_batch() -> dict[str, np.ndarray]:
    return {"observations": np.zeros((1, OBS_DIM), np.float32),
            "actions": np.zeros((1, ACT_DIM), np.float32)}


@pytest.fixture(scope="module")
def frozen_agent():
    """Stands in for the restored checkpoint: a full agent with its own pi_h."""
    config = tiny_config()
    agent = FBpiSwitchAgent.create(123, example_batch(), config)
    # As fbplan.frozen restores it: the stored config may hold lists from flags.json.
    return agent.replace(config=config)


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
    fresh = with_fresh_high_actor(frozen_agent, 0, example_batch())
    for name in FROZEN_MODULES:
        assert same(params_of(fresh, name), params_of(frozen_agent, name)), name
    assert not same(params_of(fresh, "modules_high_actor"),
                    params_of(frozen_agent, "modules_high_actor"))


def test_pi_h_init_depends_only_on_seed(frozen_agent):
    a = with_fresh_high_actor(frozen_agent, 0, example_batch())
    b = with_fresh_high_actor(frozen_agent, 0, example_batch())
    c = with_fresh_high_actor(frozen_agent, 1, example_batch())
    assert same(params_of(a, "modules_high_actor"), params_of(b, "modules_high_actor"))
    assert not same(params_of(a, "modules_high_actor"), params_of(c, "modules_high_actor"))


def test_training_updates_only_pi_h(frozen_agent, data):
    fresh = with_fresh_high_actor(frozen_agent, 0, example_batch())
    spec = HighBatchSpec.from_agent_config(fresh.config)
    trained, info = train_chunk(fresh, data, jax.random.PRNGKey(0), 3, spec)
    for name in FROZEN_MODULES:
        assert same(params_of(trained, name), params_of(fresh, name)), name
    assert not same(params_of(trained, "modules_high_actor"),
                    params_of(fresh, "modules_high_actor"))
    assert int(trained.network.step) == int(fresh.network.step) + 3
    assert all(np.isfinite(float(v)) for v in info.values())


def test_train_logs_one_row_per_interval(frozen_agent, data, tmp_path):
    fresh = with_fresh_high_actor(frozen_agent, 0, example_batch())
    train(fresh, data, data, steps=4, log_every=2, key=jax.random.PRNGKey(0),
          log_path=tmp_path / "train.csv")
    log = pd.read_csv(tmp_path / "train.csv")
    assert list(log["step"]) == [2, 4]
    assert {"training/high_actor/actor_loss", "validation/high_actor/actor_loss"} <= set(log)


def test_train_rejects_partial_log_interval(frozen_agent, data, tmp_path):
    fresh = with_fresh_high_actor(frozen_agent, 0, example_batch())
    with pytest.raises(ValueError):
        train(fresh, data, data, steps=5, log_every=2, key=jax.random.PRNGKey(0),
              log_path=tmp_path / "train.csv")


def test_checkpoint_has_upstream_layout_and_restores(frozen_agent, data, tmp_path):
    fresh = with_fresh_high_actor(frozen_agent, 0, example_batch())
    spec = HighBatchSpec.from_agent_config(fresh.config)
    trained, _ = train_chunk(fresh, data, jax.random.PRNGKey(0), 2, spec)
    save_checkpoint(trained, {"agent": dict(trained.config)}, tmp_path)

    with open(tmp_path / "params.pkl", "rb") as f:
        stored = pickle.load(f)
    assert set(stored) == {"agent"}
    assert set(stored["agent"]["network"]) == {"step", "params", "opt_state"}
    assert all(type(x) is np.ndarray for x in jax.tree_util.tree_leaves(stored["agent"]))

    restored = flax.serialization.from_state_dict(fresh, stored["agent"])
    assert same(jax.tree_util.tree_leaves(restored.network.params),
                jax.tree_util.tree_leaves(trained.network.params))
    assert (tmp_path / "flags.json").exists()
