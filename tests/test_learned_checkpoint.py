"""Tests for fbplan.learned.checkpoint: folders, the trained-modules format, export, loaders."""

import json
import pickle

import flax
import jax
import numpy as np
import pytest

from fbplan import frozen
from fbplan.learned import checkpoint
from fbplan.learned.checkpoint import (
    FROZEN_MODULES,
    cache_relative,
    export_folder,
    load_baseline_model,
    load_trained_file,
    portable_flags,
    save_high_actor,
    trained_dir,
    trained_key,
    with_fresh_high_actor,
    with_high_actor,
)
from fbplan.learned.sampling import HighBatchSpec, make_trajectory_data
from fbplan.train_high import train_chunk

fbpiswitch = frozen.upstream_module("agents.fbpiswitch")
MEDIUM = "antmaze-medium-navigate-v0"
OBS_DIM = 5


@pytest.fixture(scope="module")
def agent():
    config = fbpiswitch.get_config().to_dict() | {
        "actor_hidden_dims": (16, 16),
        "forward_repr_hidden_dims": (16, 16),
        "backward_repr_hidden_dims": (16, 16),
        "latent_dim": 4,
    }
    batch = {
        "observations": np.zeros((1, OBS_DIM), np.float32),
        "actions": np.zeros((1, 2), np.float32),
    }
    return fbpiswitch.FBpiSwitchAgent.create(7, batch, config)


def leaves(tree):
    return jax.tree_util.tree_leaves(tree)


def same(a, b) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(leaves(a), leaves(b), strict=True))


def test_keys_name_the_folders_below_the_cache():
    assert trained_key("baselines", MEDIUM, 0) == "baselines/medium/seed0"
    assert trained_key("approach_b", MEDIUM, 3, "clean") == "approach_b/medium/clean/seed3"


def test_folder_roots_follow_the_environment_variables(monkeypatch, tmp_path):
    monkeypatch.setenv("FBPLAN_APPROACH_B_DIR", str(tmp_path))
    assert trained_dir("approach_b", MEDIUM, 1, "wide") == tmp_path / "medium" / "wide" / "seed1"
    monkeypatch.delenv("FBPLAN_BASELINE_DIR", raising=False)
    assert trained_dir("baselines", MEDIUM, 0) == checkpoint.CACHE / "baselines/medium/seed0"


def test_high_actor_file_holds_only_pi_h_as_numpy(agent, tmp_path):
    save_high_actor(agent, {"seed": 0}, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["flags.json", "high_actor.pkl"]
    stored = frozen.load_numpy_pickle(tmp_path / "high_actor.pkl")
    assert all(type(x) is np.ndarray for x in leaves(stored))
    other = with_fresh_high_actor(agent, 1, OBS_DIM)
    restored = with_high_actor(other, stored)
    assert same(
        restored.network.params["modules_high_actor"], agent.network.params["modules_high_actor"]
    )
    for name in FROZEN_MODULES:
        assert same(restored.network.params[name], other.network.params[name]), name


def test_export_turns_a_full_agent_state_into_the_pi_h_file(agent, tmp_path):
    state = jax.tree_util.tree_map(np.asarray, flax.serialization.to_state_dict(agent))
    (tmp_path / "params.pkl").write_bytes(pickle.dumps({"agent": state}))
    old_flags = {
        "retrained_high_actor": {"from": "/root/.cache/fbplan/checkpoints/medium"},
        "approach_b": {
            "value_dir": "/home/x/.cache/fbplan/approach_b/medium/value-single/seed0",
            "value_config": {"option": {"rho_max": 0.9}},
        },
    }
    (tmp_path / "flags.json").write_text(json.dumps(old_flags))
    export_folder(tmp_path)
    exported = frozen.load_numpy_pickle(tmp_path / "high_actor.pkl")
    assert same(exported, agent.network.params["modules_high_actor"])
    flags = json.loads((tmp_path / "flags.json").read_text())
    assert flags["retrained_high_actor"]["from"] == "checkpoints/medium"
    assert flags["approach_b"]["value_dir"] == "approach_b/medium/value-single/seed0"
    assert flags["approach_b"]["value_config"]["option"] == {"rho_cap": 0.9}


def test_cache_relative_keeps_other_text():
    assert cache_relative("/root/.cache/fbplan/baselines/medium/seed0") == "baselines/medium/seed0"
    assert cache_relative("goal_zero") == "goal_zero"
    assert portable_flags({"a": [1, "x"], "b": None}) == {"a": [1, "x"], "b": None}


def test_a_missing_file_is_downloaded_by_its_key(monkeypatch, tmp_path):
    calls = []

    def fake_download(key, target):
        calls.append((key, target))
        target.mkdir(parents=True, exist_ok=True)
        (target / "value.pkl").write_bytes(pickle.dumps({"w": np.ones(2)}))
        return target

    monkeypatch.setattr(frozen, "download_trained", fake_download)
    key = "approach_b/medium/value-single/seed0"
    loaded = load_trained_file(key, tmp_path / "seed0", "value.pkl")
    np.testing.assert_array_equal(loaded["w"], np.ones(2))
    assert calls == [(key, tmp_path / "seed0")]
    load_trained_file(key, tmp_path / "seed0", "value.pkl")
    assert len(calls) == 1  # present now: no second download


def test_real_checkpoint_retrains_and_reloads_as_a_baseline(tmp_path, monkeypatch):
    if not (frozen.checkpoint_dir(MEDIUM) / "params.pkl").exists():
        pytest.skip("medium checkpoint absent: run `uv run python -m fbplan.frozen`")
    if not frozen.dataset_path(MEDIUM, "val").exists():
        pytest.skip("medium validation split absent: call fbplan.frozen.load_dataset once")
    model, val = frozen.load_model(MEDIUM), frozen.load_dataset(MEDIUM, "val")
    fresh = with_fresh_high_actor(model.agent, 0, val["observations"].shape[-1])
    data = make_trajectory_data(val["observations"], val["terminals"])
    trained, _ = train_chunk(
        fresh, data, jax.random.PRNGKey(0), 2, HighBatchSpec.from_agent_config(fresh.config)
    )
    monkeypatch.setenv("FBPLAN_BASELINE_DIR", str(tmp_path))
    save_high_actor(trained, {}, trained_dir("baselines", MEDIUM, 0))
    loaded = load_baseline_model(MEDIUM, 0)

    obs = val["observations"][:4]
    z = np.random.default_rng(0).normal(size=(4, model.latent_dim)).astype(np.float32)
    expected = frozen.FrozenFBModel(MEDIUM, trained).high_latent(obs, z)
    np.testing.assert_array_equal(np.asarray(loaded.high_latent(obs, z)), np.asarray(expected))
    np.testing.assert_array_equal(
        np.asarray(loaded.forward(obs, z)), np.asarray(model.forward(obs, z))
    )
