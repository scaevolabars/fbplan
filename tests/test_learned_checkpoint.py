"""Tests for fbplan.learned.checkpoint: folders, the trained-modules format, export, loaders."""

import hashlib
import io
import json
import pickle
from pathlib import Path

import flax
import jax
import numpy as np
import pytest

from fbplan import frozen
from fbplan.frozen import load_trained_file, trained_key
from fbplan.learned.checkpoint import (
    FROZEN_MODULES,
    cache_relative,
    export_folder,
    load_baseline_model,
    portable_flags,
    with_fresh_high_actor,
)
from fbplan.learned.sampling import HighBatchSpec, make_trajectory_data
from fbplan.train_high import train_chunk

MEDIUM = "antmaze-medium-navigate-v0"


@pytest.fixture(scope="module")
def agent(tiny_agent):
    return tiny_agent.agent


def leaves(tree):
    return jax.tree_util.tree_leaves(tree)


def same(a, b) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(leaves(a), leaves(b), strict=True))


def test_keys_name_the_folders_below_the_cache():
    assert trained_key("baselines", MEDIUM, 0) == "baselines/medium/seed0"
    assert trained_key("approach_b", MEDIUM, 3, "clean") == "approach_b/medium/clean/seed3"
    assert trained_key("baselines", MEDIUM, 0, "awr2x") == "baselines/medium/awr2x/seed0"


def test_folder_roots_follow_the_environment_variables(monkeypatch, tmp_path):
    monkeypatch.setenv("FBPLAN_APPROACH_B_DIR", str(tmp_path))
    assert (
        frozen.trained_dir(trained_key("approach_b", MEDIUM, 1, "wide"))
        == tmp_path / "medium" / "wide" / "seed1"
    )
    monkeypatch.delenv("FBPLAN_BASELINE_DIR", raising=False)
    expected = Path.home() / ".cache/fbplan/baselines/medium/seed0"
    assert frozen.trained_dir(trained_key("baselines", MEDIUM, 0)) == expected


def test_fresh_pi_h_keeps_the_frozen_modules(agent):
    fresh = with_fresh_high_actor(agent, 1)
    for name in FROZEN_MODULES:
        assert same(fresh.network.params[name], agent.network.params[name]), name
    assert not same(
        fresh.network.params["modules_high_actor"], agent.network.params["modules_high_actor"]
    )


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
    """The real frozen.download_trained runs; only the network call is replaced."""
    key = "approach_b/medium/value-single/seed0"
    files = {
        "value.pkl": pickle.dumps({"w": np.ones(2)}),
        "flags.json": json.dumps({"seed": 0}).encode(),
    }
    pinned = {name: ("v0", hashlib.sha256(data).hexdigest()) for name, data in files.items()}
    by_url = {f"/{frozen._asset_name(key, name)}": data for name, data in files.items()}
    urls = []

    def fake_urlopen(url):
        urls.append(url)
        return io.BytesIO(next(d for suffix, d in by_url.items() if url.endswith(suffix)))

    monkeypatch.setitem(frozen._TRAINED_FILES, key, pinned)
    monkeypatch.setattr(frozen.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("FBPLAN_APPROACH_B_DIR", str(tmp_path))

    loaded = load_trained_file(key, "value.pkl")
    np.testing.assert_array_equal(loaded["w"], np.ones(2))
    assert (tmp_path / "medium/value-single/seed0/value.pkl").exists()
    assert len(urls) == 2
    load_trained_file(key, "value.pkl")
    assert len(urls) == 2  # present now: no second download


def test_real_checkpoint_retrains_and_reloads_as_a_baseline(
    medium_model, medium_val, tmp_path, monkeypatch
):
    model, val = medium_model, medium_val
    fresh = with_fresh_high_actor(model.agent, 0)
    data = make_trajectory_data(val["observations"], val["terminals"])
    trained, _ = train_chunk(
        fresh, data, jax.random.PRNGKey(0), 2, HighBatchSpec.from_agent_config(fresh.config)
    )
    monkeypatch.setenv("FBPLAN_BASELINE_DIR", str(tmp_path))
    flags = {"env_name": f"ogbench-{MEDIUM}"}
    frozen.save_trained_checkpoint(
        trained, flags, frozen.trained_dir(trained_key("baselines", MEDIUM, 0))
    )
    loaded = load_baseline_model(MEDIUM, 0)

    obs = val["observations"][:4]
    z = np.random.default_rng(0).normal(size=(4, model.latent_dim)).astype(np.float32)
    expected = frozen.FrozenFBModel(MEDIUM, trained).high_latent(obs, z)
    np.testing.assert_array_equal(np.asarray(loaded.high_latent(obs, z)), np.asarray(expected))
    np.testing.assert_array_equal(
        np.asarray(loaded.forward(obs, z)), np.asarray(model.forward(obs, z))
    )
