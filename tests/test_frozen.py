"""Tests for fbplan.frozen: checkpoint safety, model queries, env and offline data.

Tests that need a checkpoint or an OGBench file skip when the file is absent.
`uv run python -m fbplan.frozen --datasets antmaze-medium-navigate-v0` downloads all of them.
"""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import re
import urllib.request
from pathlib import Path

import flax
import jax
import numpy as np
import pytest

from fbplan import frozen
from fbplan.contracts import normalize_z

MEDIUM = "antmaze-medium-navigate-v0"
OBS_DIM, ACT_DIM, LATENT_DIM, N_HEADS = 29, 8, 128, 2


@pytest.fixture
def model(medium_model: frozen.FrozenFBModel) -> frozen.FrozenFBModel:
    """The provided medium model, loaded once per session (tests/conftest.py)."""
    return medium_model


@pytest.fixture(scope="module")
def observations() -> np.ndarray:
    """Start and goal observations of the 5 medium tasks: realistic states, no dataset needed.

    Resets follow the recipe of design.md §5, so the states do not depend on the test order.
    """
    env = frozen.make_env(MEDIUM)
    obs = []
    for task_id in range(1, 6):
        np.random.seed(task_id)
        env.action_space.seed(task_id)
        ob, info = env.reset(seed=task_id, options={"task_id": task_id})
        obs += [ob, info["goal"]]
    return np.stack(obs).astype(np.float32)


@pytest.fixture(scope="module")
def latents(observations: np.ndarray) -> np.ndarray:
    return np.random.default_rng(0).normal(size=(len(observations), LATENT_DIM)).astype(np.float32)


def test_numpy_pickle_loader_reads_nested_arrays(tmp_path):
    weights = np.arange(6, dtype=np.float32).reshape(2, 3)
    data = {"agent": {"w": weights}, "rng": np.ones(2, np.uint32)}
    path = tmp_path / "ok.pkl"
    path.write_bytes(pickle.dumps(data, protocol=4))  # the protocol of the provided checkpoints

    loaded = frozen.load_numpy_pickle(path)

    np.testing.assert_array_equal(loaded["agent"]["w"], data["agent"]["w"])
    np.testing.assert_array_equal(loaded["rng"], data["rng"])


def test_numpy_pickle_loader_rejects_other_classes(tmp_path):
    path = tmp_path / "bad.pkl"
    path.write_bytes(pickle.dumps({"f": print}))

    with pytest.raises(pickle.UnpicklingError, match="builtins.print"):
        frozen.load_numpy_pickle(path)


def test_save_numpy_pickle_writes_jax_arrays_as_numpy(tmp_path):
    tree = {"a": jax.numpy.arange(3.0), "b": {"c": np.ones((2, 2), np.float32)}}
    path = tmp_path / "sub" / "tree.pkl"

    frozen.save_numpy_pickle(tree, path)
    loaded = frozen.load_numpy_pickle(path)

    assert isinstance(loaded["a"], np.ndarray)
    np.testing.assert_array_equal(loaded["a"], np.arange(3.0))
    np.testing.assert_array_equal(loaded["b"]["c"], tree["b"]["c"])


def test_save_numpy_pickle_refuses_what_load_numpy_pickle_cannot_read(tmp_path):
    with pytest.raises(TypeError, match="numeric"):
        frozen.save_numpy_pickle({"bad": object()}, tmp_path / "tree.pkl")

    assert list(tmp_path.iterdir()) == []


def test_maze_name_is_the_second_part_of_a_known_env_name():
    assert [frozen.maze_name(env) for env in frozen.ENV_NAMES] == [
        "medium",
        "large",
        "giant",
        "teleport",
    ]
    with pytest.raises(ValueError, match="antmaze-medium-stitch-v0"):
        frozen.maze_name("antmaze-medium-stitch-v0")


def test_cache_dir_is_below_home_unless_its_env_variable_is_set(monkeypatch, tmp_path):
    for variable in ("FBPLAN_CKPT_DIR", "FBPLAN_BASELINE_DIR", "FBPLAN_APPROACH_B_DIR"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FBPLAN_BASELINE_DIR", str(tmp_path / "models"))

    assert (
        frozen.cache_dir("checkpoints") == tmp_path / "home" / ".cache" / "fbplan" / "checkpoints"
    )
    assert frozen.cache_dir("approach_b") == tmp_path / "home" / ".cache" / "fbplan" / "approach_b"
    assert frozen.cache_dir("baselines") == tmp_path / "models"
    assert frozen.trained_dir("baselines/medium/seed3") == tmp_path / "models" / "medium" / "seed3"
    with pytest.raises(ValueError, match="kind"):
        frozen.cache_dir("runs")
    with pytest.raises(ValueError, match="checkpoints/medium"):
        frozen.trained_dir("checkpoints/medium")


def test_upstream_module_imports_from_the_submodule():
    upstream_dir = Path(frozen.__file__).resolve().parents[1] / "third_party" / "ssm"

    module = frozen.upstream_module("agents.fbpiswitch")

    assert Path(module.__file__).resolve().is_relative_to(upstream_dir)
    assert hasattr(module, "FBpiSwitchAgent")


def test_upstream_module_refuses_a_module_from_outside_the_submodule():
    with pytest.raises(ImportError, match="not to"):
        frozen.upstream_module("json")


def test_download_checkpoint_refuses_files_with_a_wrong_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("FBPLAN_CKPT_DIR", str(tmp_path))
    target = frozen.checkpoint_dir(MEDIUM)
    target.mkdir(parents=True)
    for name in ("flags.json", "params.pkl"):
        (target / name).write_text("not the checkpoint")

    with pytest.raises(ValueError, match="sha256"):
        frozen.download_checkpoint(MEDIUM)


def test_download_checkpoint_does_not_use_the_network_when_files_are_present(
    monkeypatch, require_files
):
    require_files.checkpoint(MEDIUM)

    def no_network(*args, **kwargs):
        raise AssertionError("download_checkpoint used the network")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)

    assert frozen.download_checkpoint(MEDIUM) == frozen.checkpoint_dir(MEDIUM)


def test_a_failed_download_leaves_no_file(monkeypatch, tmp_path):
    class BrokenConnection(io.BytesIO):
        def read(self, *args):
            raise ConnectionResetError("connection lost")

    monkeypatch.setattr(urllib.request, "urlopen", lambda url: BrokenConnection(b"x"))

    with pytest.raises(ConnectionResetError):
        frozen._download("https://example.org/params.pkl", tmp_path / "params.pkl", "0" * 64)
    assert list(tmp_path.iterdir()) == []


TRAINED_KEY = "baselines/medium/seed0"
RELEASE = "https://github.com/scaevolabars/fbplan/releases/download/trained-test"
TRAINED_ASSETS = {
    f"{RELEASE}/baselines-medium-seed0-flags.json": b'{"seed": 0}',
    f"{RELEASE}/baselines-medium-seed0-high_actor.pkl": pickle.dumps({"w": np.ones(3)}),
}


class FakeServer:
    """Stands in for urllib.request.urlopen: serves fixed bytes per URL, records requests."""

    def __init__(self, assets: dict[str, bytes]):
        self.assets = assets
        self.requested: list[str] = []

    def urlopen(self, url: str) -> io.BytesIO:
        self.requested.append(url)
        return io.BytesIO(self.assets[url])


def _pin(assets: dict[str, bytes], tag: str) -> dict[str, tuple[str, str]]:
    """Table entry for one trained checkpoint: file name -> (release tag, sha256)."""
    return {
        url.rsplit("-", 1)[-1]: (tag, hashlib.sha256(content).hexdigest())
        for url, content in assets.items()
    }


@pytest.fixture
def release(monkeypatch, tmp_path) -> FakeServer:
    """One published trained checkpoint, a fake GitHub release and an empty HOME."""
    for variable in ("FBPLAN_CKPT_DIR", "FBPLAN_BASELINE_DIR", "FBPLAN_APPROACH_B_DIR"):
        monkeypatch.delenv(variable, raising=False)
    server = FakeServer(dict(TRAINED_ASSETS))
    monkeypatch.setattr(
        frozen, "_TRAINED_FILES", {TRAINED_KEY: _pin(TRAINED_ASSETS, "trained-test")}
    )
    monkeypatch.setattr(urllib.request, "urlopen", server.urlopen)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return server


def test_trained_key_names_the_folder_of_a_trained_checkpoint():
    large = "antmaze-large-navigate-v0"

    assert frozen.trained_key("baselines", MEDIUM, 3) == "baselines/medium/seed3"
    assert frozen.trained_key("approach_b", large, 0, "clean") == "approach_b/large/clean/seed0"
    with pytest.raises(ValueError, match="kind"):
        frozen.trained_key("checkpoints", MEDIUM, 0)
    with pytest.raises(ValueError, match="antmaze-medium-stitch-v0"):
        frozen.trained_key("baselines", "antmaze-medium-stitch-v0", 0)


def test_load_trained_file_downloads_a_missing_checkpoint_once(release):
    loaded = frozen.load_trained_file(TRAINED_KEY, "high_actor.pkl")
    release.requested.clear()
    again = frozen.load_trained_file(TRAINED_KEY, "high_actor.pkl")

    np.testing.assert_array_equal(loaded["w"], np.ones(3))
    np.testing.assert_array_equal(again["w"], np.ones(3))
    assert release.requested == []


def test_save_flags_writes_flags_json_into_a_new_folder(tmp_path):
    flags = {"env_name": f"ogbench-{MEDIUM}", "seed": 2, "agent": {"lr": 0.0003}}

    frozen.save_flags(flags, tmp_path / "new" / "seed2")

    written = (tmp_path / "new" / "seed2" / "flags.json").read_text()
    assert written == json.dumps(flags, indent=2)


def test_download_trained_fetches_the_pinned_release_assets(release):
    target = frozen.download_trained(TRAINED_KEY)

    assert release.requested == list(TRAINED_ASSETS)
    for url, content in TRAINED_ASSETS.items():
        assert (target / url.rsplit("-", 1)[-1]).read_bytes() == content


def test_download_trained_defaults_to_the_cache_layout_of_the_trainers(release, tmp_path):
    target = frozen.download_trained(TRAINED_KEY)

    assert target == tmp_path / "home" / ".cache" / "fbplan" / "baselines" / "medium" / "seed0"
    assert (target / "high_actor.pkl").exists()


def test_download_trained_follows_the_env_variable_of_the_kind(release, monkeypatch, tmp_path):
    monkeypatch.setenv("FBPLAN_BASELINE_DIR", str(tmp_path / "models"))

    assert frozen.download_trained(TRAINED_KEY) == tmp_path / "models" / "medium" / "seed0"


def test_download_trained_refuses_an_asset_with_a_wrong_hash(release):
    url = f"{RELEASE}/baselines-medium-seed0-high_actor.pkl"
    release.assets[url] = b"changed"

    with pytest.raises(ValueError, match=re.escape(f"{url}: sha256")):
        frozen.download_trained(TRAINED_KEY)
    assert [p.name for p in frozen.trained_dir(TRAINED_KEY).iterdir()] == ["flags.json"]


def test_download_trained_checks_present_files_without_the_network(release):
    target = frozen.download_trained(TRAINED_KEY)
    release.requested.clear()

    frozen.download_trained(TRAINED_KEY)
    assert release.requested == []

    (target / "flags.json").write_text('{"seed": 1}')
    with pytest.raises(ValueError, match="sha256"):
        frozen.download_trained(TRAINED_KEY)


def test_download_trained_refuses_an_unpublished_checkpoint(release):
    with pytest.raises(ValueError, match="baselines/medium/seed9 is not published"):
        frozen.download_trained("baselines/medium/seed9")


def test_the_retrained_baselines_are_published_with_complete_entries():
    baselines = {f"baselines/{maze}/seed{k}" for maze in ("medium", "large") for k in range(5)}

    assert baselines <= frozen._TRAINED_FILES.keys()
    for key, files in frozen._TRAINED_FILES.items():
        frozen.trained_dir(key)  # raises for a malformed key
        assert "flags.json" in files, key
        for tag, sha256 in files.values():
            assert tag and re.fullmatch("[0-9a-f]{64}", sha256), key
    for key in baselines:
        assert set(frozen._TRAINED_FILES[key]) == {"flags.json", frozen.HIGH_ACTOR_FILE}


def test_concurrent_downloads_of_one_file_give_the_complete_file(monkeypatch, tmp_path):
    """Parallel eval workers can download the same missing file at the same time."""
    content = b"x" * 100
    path = tmp_path / "params.pkl"
    sha256 = hashlib.sha256(content).hexdigest()

    class SecondDownloadDuringFirst(io.BytesIO):
        started_second = False

        def read(self, *args):
            # The second download starts and finishes after the first has opened its file.
            if not self.started_second:
                self.started_second = True
                monkeypatch.setattr(urllib.request, "urlopen", lambda url: io.BytesIO(content))
                frozen._download("https://example.org/params.pkl", path, sha256)
            return super().read(*args)

    monkeypatch.setattr(urllib.request, "urlopen", lambda url: SecondDownloadDuringFirst(content))
    frozen._download("https://example.org/params.pkl", path, sha256)

    assert path.read_bytes() == content
    assert [p.name for p in tmp_path.iterdir()] == ["params.pkl"]


def _serve_a_frozen_checkpoint(release, monkeypatch, tmp_path) -> None:
    """A fake medium checkpoint on the fake server, downloaded to tmp_path / "frozen"."""
    release.assets |= {
        frozen._DRIVE_URL.format("flags-id"): b"{}",
        frozen._DRIVE_URL.format("params-id"): b"params",
    }
    monkeypatch.setattr(
        frozen,
        "_CHECKPOINT_FILES",
        {
            "medium": {
                "flags.json": ("flags-id", hashlib.sha256(b"{}").hexdigest()),
                "params.pkl": ("params-id", hashlib.sha256(b"params").hexdigest()),
            }
        },
    )
    monkeypatch.setenv("FBPLAN_CKPT_DIR", str(tmp_path / "frozen"))


def test_cli_with_trained_downloads_the_frozen_and_the_trained_checkpoints(
    release, monkeypatch, tmp_path
):
    _serve_a_frozen_checkpoint(release, monkeypatch, tmp_path)

    frozen.main(["--trained", MEDIUM])

    assert (tmp_path / "frozen" / "medium" / "params.pkl").read_bytes() == b"params"
    trained = tmp_path / "home" / ".cache" / "fbplan" / TRAINED_KEY
    assert sorted(p.name for p in trained.iterdir()) == ["flags.json", "high_actor.pkl"]


def test_cli_help_exits_without_an_error(capsys):
    with pytest.raises(SystemExit) as exit_info:
        frozen.main(["--help"])

    assert exit_info.value.code == 0
    assert "--datasets" in capsys.readouterr().out


def test_cli_refuses_an_unknown_env_with_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exit_info:
        frozen.main(["antmaze-medium-stitch-v0"])

    assert exit_info.value.code == 2
    assert "antmaze-medium-stitch-v0" in capsys.readouterr().err


def test_cli_datasets_downloads_both_splits_of_each_env(release, monkeypatch, tmp_path):
    for split in ("train", "val"):
        name = frozen.dataset_path(MEDIUM, split).name
        release.assets[f"{frozen.ogbench.utils.DATASET_URL}/{name}"] = split.encode()
    _serve_a_frozen_checkpoint(release, monkeypatch, tmp_path)

    frozen.main([MEDIUM, "--datasets", MEDIUM])

    for split in ("train", "val"):
        assert frozen.dataset_path(MEDIUM, split).read_bytes() == split.encode()
    assert frozen.dataset_path(MEDIUM, "val").is_relative_to(tmp_path / "home")


def _with_new_high_actor(agent):
    """The agent with pi_h weights + 1: a stand-in for a retrained pi_h."""
    params = flax.core.unfreeze(agent.network.params)
    params["modules_high_actor"] = jax.tree_util.tree_map(
        lambda x: x + 1.0, params["modules_high_actor"]
    )
    return agent.replace(network=agent.network.replace(params=params))


def _trained_flags(env_name: str) -> dict:
    """Smallest flags.json: load_trained_model reads only env_name."""
    return {"env_name": f"ogbench-{env_name}", "seed": 7}


def test_trained_checkpoint_round_trip_keeps_the_frozen_modules(model, monkeypatch, tmp_path):
    monkeypatch.setenv("FBPLAN_BASELINE_DIR", str(tmp_path))
    trained = _with_new_high_actor(model.agent)

    frozen.save_trained_checkpoint(trained, _trained_flags(MEDIUM), tmp_path / "medium" / "seed7")
    loaded = frozen.load_trained_model(MEDIUM, "baselines/medium/seed7")

    assert sorted(p.name for p in (tmp_path / "medium" / "seed7").iterdir()) == [
        "flags.json",
        "high_actor.pkl",
    ]
    for name, expected in trained.network.params.items():
        for a, b in zip(
            jax.tree_util.tree_leaves(loaded.agent.network.params[name]),
            jax.tree_util.tree_leaves(expected),
            strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)


def test_load_trained_model_downloads_a_missing_checkpoint(model, monkeypatch, tmp_path):
    staging = tmp_path / "staging"
    frozen.save_trained_checkpoint(
        _with_new_high_actor(model.agent), _trained_flags(MEDIUM), staging
    )
    assets = {
        f"{RELEASE}/baselines-medium-seed7-{p.name}": p.read_bytes() for p in staging.iterdir()
    }
    monkeypatch.setattr(
        frozen, "_TRAINED_FILES", {"baselines/medium/seed7": _pin(assets, "trained-test")}
    )
    monkeypatch.setattr(urllib.request, "urlopen", FakeServer(assets).urlopen)
    monkeypatch.setenv("FBPLAN_BASELINE_DIR", str(tmp_path / "models"))

    loaded = frozen.load_trained_model(MEDIUM, "baselines/medium/seed7")

    assert (tmp_path / "models" / "medium" / "seed7" / "high_actor.pkl").exists()
    assert not np.allclose(
        np.asarray(loaded.high_latent(np.zeros((1, OBS_DIM)), np.ones((1, LATENT_DIM)))),
        np.asarray(model.high_latent(np.zeros((1, OBS_DIM)), np.ones((1, LATENT_DIM)))),
    )


def test_load_trained_model_refuses_a_checkpoint_of_another_env(model, monkeypatch, tmp_path):
    monkeypatch.setenv("FBPLAN_BASELINE_DIR", str(tmp_path))
    flags = _trained_flags("antmaze-large-navigate-v0")
    frozen.save_trained_checkpoint(model.agent, flags, tmp_path / "medium" / "seed7")

    with pytest.raises(ValueError, match="antmaze-large-navigate-v0"):
        frozen.load_trained_model(MEDIUM, "baselines/medium/seed7")


def test_example_batch_has_the_sizes_of_the_saved_weights(model):
    batch = frozen.example_batch(model.agent.network.params)

    assert batch["observations"].shape == (1, OBS_DIM)
    assert batch["actions"].shape == (1, ACT_DIM)


def test_forward_of_the_agent_is_the_model_forward(model, observations, latents):
    """Equal up to float32 rounding: XLA fuses the jitted method differently (about 3e-6)."""
    with jax.default_matmul_precision("highest"):  # TF32 on an Ampere GPU gives about 1e-3
        plain = np.asarray(frozen.forward(model.agent, observations, latents))
        jitted = np.asarray(model.forward(observations, latents))

    np.testing.assert_allclose(plain, jitted, rtol=1e-5, atol=1e-5)


def test_model_puts_a_host_agent_on_the_device(model):
    state = frozen.load_numpy_pickle(frozen.checkpoint_dir(MEDIUM) / "params.pkl")["agent"]
    host_agent = flax.serialization.from_state_dict(model.agent, state)
    assert any(isinstance(x, np.ndarray) for x in jax.tree_util.tree_leaves(host_agent))

    on_device = frozen.FrozenFBModel(MEDIUM, host_agent).agent

    assert all(isinstance(x, jax.Array) for x in jax.tree_util.tree_leaves(on_device))


def test_restored_parameters_equal_the_checkpoint_arrays(model):
    saved = frozen.load_numpy_pickle(frozen.checkpoint_dir(MEDIUM) / "params.pkl")
    saved_params = saved["agent"]["network"]["params"]

    restored = jax.tree_util.tree_leaves_with_path(model.agent.network.params)
    expected = dict(jax.tree_util.tree_leaves_with_path(saved_params))

    assert {path for path, _ in restored} == set(expected)
    for path, leaf in restored:
        np.testing.assert_array_equal(np.asarray(leaf), expected[path])


@pytest.mark.parametrize("env_name", frozen.ENV_NAMES)
def test_every_checkpoint_loads_with_its_training_discount(env_name, require_files):
    require_files.checkpoint(env_name)

    loaded = frozen.load_model(env_name)

    assert loaded.env_name == env_name
    assert loaded.latent_dim == LATENT_DIM
    assert loaded.discount == (0.995 if "giant" in env_name else 0.99)


def test_forward_and_backward_shapes_and_dtypes(model, observations, latents):
    f = np.asarray(model.forward(observations, latents))
    b = np.asarray(model.backward(observations))

    assert f.shape == (N_HEADS, len(observations), LATENT_DIM) and f.dtype == np.float32
    assert b.shape == (len(observations), LATENT_DIM) and b.dtype == np.float32


def test_policy_outputs_have_contract_shapes_and_ranges(model, observations, latents):
    actions = np.asarray(model.low_action(observations, latents))
    intentions = np.asarray(model.high_latent(observations, latents))

    assert actions.shape == (len(observations), ACT_DIM) and actions.dtype == np.float32
    assert np.all(np.abs(actions) <= 1.0)
    assert intentions.shape == (len(observations), LATENT_DIM)
    np.testing.assert_allclose(np.linalg.norm(intentions, axis=-1), np.sqrt(LATENT_DIM), rtol=1e-5)


def test_forward_and_backward_of_a_row_do_not_depend_on_the_other_rows(model, observations):
    """The planner divides F(s, z_w)^T B(w) values from different batches (rho), so a row's
    output must not depend on the batch it is in."""
    order = np.random.default_rng(0).permutation(len(observations))

    with jax.default_matmul_precision("highest"):  # TF32 on an Ampere GPU gives about 1e-3
        b = np.asarray(model.backward(observations))
        shuffled_b = np.asarray(model.backward(observations[order]))
        shuffled_f = np.asarray(model.forward(observations[order], b[order]))
        f = np.asarray(model.forward(observations, b))

    np.testing.assert_allclose(shuffled_b, b[order], rtol=1e-6)
    np.testing.assert_allclose(shuffled_f, f[:, order], rtol=1e-5, atol=1e-6)


def test_self_successor_measure_is_positive(model, observations):
    """F(w, z_w)^T B(w) with z_w = B(w) is the denominator of rho(s, w)."""
    b = np.asarray(model.backward(observations))

    assert np.all(np.sum(np.asarray(model.forward(observations, b)) * b, axis=-1) > 0)


def test_forward_ignores_the_scale_of_z(model, observations, latents):
    # Full float32 matmuls: on an Ampere GPU the default (TF32) gives errors of about 1.6e-3.
    with jax.default_matmul_precision("highest"):
        reference = np.asarray(model.forward(observations, latents))
        for scale in (1e-3, 1e3):
            scaled = np.asarray(model.forward(observations, scale * latents))
            np.testing.assert_allclose(scaled, reference, rtol=1e-4, atol=1e-4)


def test_baseline_policy_matches_upstream_sample_actions(model, observations):
    """pi_l(s, pi_h(s, z_r)) equals upstream FBpiSwitchAgent.sample_actions at temperature 0.

    With a z_r of norm sqrt(d): for the real z_r the two paths differ on purpose (see
    FrozenFBModel.infer_task_latent). Both paths use full float32 matmuls: on an Ampere GPU the
    default (TF32) gives errors of about 1.6e-3.
    """
    z_r = normalize_z(np.random.default_rng(1).normal(size=LATENT_DIM), LATENT_DIM)
    z_r = z_r.astype(np.float32)
    z_task = np.tile(z_r, (len(observations), 1))
    key = jax.random.PRNGKey(0)

    with jax.default_matmul_precision("highest"):
        ours = np.asarray(model.low_action(observations, model.high_latent(observations, z_task)))
        upstream = np.stack(
            [
                np.asarray(model.agent.sample_actions(ob, z_r, seed=key, temperature=0.0))
                for ob in observations
            ]
        )

    np.testing.assert_allclose(ours, upstream, atol=1e-5)


def test_infer_task_latent_is_stable_across_calls(model):
    first, second = model.infer_task_latent(1), model.infer_task_latent(1)

    assert first.shape == (LATENT_DIM,) and first.dtype == np.float32
    np.testing.assert_array_equal(first, second)


def test_task_latent_points_along_the_sum_of_b_over_goal_states(model):
    """z_r = normalize_z(mean(r * B(s))) with r = 1 within the goal radius of the first 100k
    validation states. Its norm is below sqrt(d), see FrozenFBModel.infer_task_latent."""
    env = frozen.make_env(MEDIUM)
    env.reset(options={"task_id": 1})
    val = frozen.load_dataset(MEDIUM, "val")
    first = slice(0, 100_000)
    dists = np.linalg.norm(val["qpos"][first, :2] - env.unwrapped.cur_goal_xy, axis=-1)
    goal_obs = val["observations"][first][dists <= env.unwrapped._goal_tol]

    expected = np.asarray(model.backward(goal_obs)).sum(axis=0)
    z_r = model.infer_task_latent(1)

    cosine = expected @ z_r / (np.linalg.norm(expected) * np.linalg.norm(z_r))
    assert len(goal_obs) > 0
    np.testing.assert_allclose(cosine, 1.0, rtol=1e-5)
    # The upstream value (0.98 sqrt(d) for this task), neither the raw 1e-7 nor exactly sqrt(d).
    assert 0.75 < np.linalg.norm(z_r) / np.sqrt(LATENT_DIM) < 0.99


def test_goal_latent_of_a_task_goal_cell_is_the_task_latent(model):
    maze = frozen.make_env(MEDIUM).unwrapped
    goal_ij = tuple(maze.task_infos[0]["goal_ij"])

    np.testing.assert_array_equal(model.infer_goal_latent(goal_ij), model.infer_task_latent(1))


def test_goal_latent_refuses_a_cell_that_no_validation_state_reaches(model):
    with pytest.raises(ValueError, match=r"\(0, 0\)"):  # a wall cell
        model.infer_goal_latent((0, 0))


def test_task_latents_differ_between_tasks(model):
    assert not np.allclose(model.infer_task_latent(1), model.infer_task_latent(2))


def test_make_env_gives_the_upstream_eval_env():
    env = frozen.make_env(MEDIUM)
    maze = env.unwrapped
    for seed in (0, 1):  # goal noise is off: every reset puts the goal at the task's cell centre
        ob, info = env.reset(seed=seed, options={"task_id": 1})
        np.testing.assert_array_equal(
            maze.cur_goal_xy, maze.ij_to_xy(maze.task_infos[0]["goal_ij"])
        )

    assert ob.shape == (OBS_DIM,) and info["goal"].shape == (OBS_DIM,)
    assert env.action_space.shape == (ACT_DIM,)
    assert env.spec.max_episode_steps == 1000


def test_eval_env_reproduces_the_dataset_dynamics(require_files):
    """One env step from a dataset state with the dataset action gives the dataset next state.

    pi_l was trained on these transitions, so the eval env must have the same physics.
    MuJoCo 3.1.6 reproduces them to float precision (about 4e-7). MuJoCo 3.17 does not (8e-3).
    """
    require_files.dataset(MEDIUM, "val")
    maze = frozen.make_env(MEDIUM).unwrapped
    data = frozen.load_dataset(MEDIUM, "val")
    within = np.flatnonzero(data["terminals"][:-1] == 0)
    rows = np.random.default_rng(0).choice(within, size=100, replace=False)

    errors = []
    for t in rows:
        maze.set_state(data["qpos"][t], data["qvel"][t])
        maze.do_simulation(data["actions"][t], maze.frame_skip)
        errors.append(np.abs(maze.data.qpos - data["qpos"][t + 1]).max())

    assert np.median(errors) < 1e-5


@pytest.mark.parametrize("split", ["train", "val"])
def test_load_dataset_gives_aligned_transitions(split, require_files):
    require_files.dataset(MEDIUM, split)

    data = frozen.load_dataset(MEDIUM, split)
    obs, next_obs, terminals = data["observations"], data["next_observations"], data["terminals"]

    assert {"observations", "actions", "next_observations", "terminals", "qpos"} <= data.keys()
    assert all(len(v) == len(obs) for v in data.values())
    assert obs.shape[1] == OBS_DIM and data["actions"].shape[1] == ACT_DIM
    assert np.all(np.abs(data["actions"]) <= np.float32(1 - 1e-5))
    assert terminals[-1] == 1
    within = terminals[:-1] == 0
    np.testing.assert_array_equal(next_obs[:-1][within], obs[1:][within])


def test_load_dataset_rejects_an_unknown_split():
    with pytest.raises(ValueError, match="split"):
        frozen.load_dataset(MEDIUM, "test")
