"""Tests for fbplan.frozen: checkpoint safety, model queries, env and offline data.

Tests that need a checkpoint or an OGBench file skip when the file is absent.
`uv run python -m fbplan.frozen` downloads the checkpoints; OGBench files download on first use.
"""

from __future__ import annotations

import pickle
import urllib.request
from pathlib import Path

import jax
import numpy as np
import pytest

from fbplan import frozen
from fbplan.contracts import normalize_z

MEDIUM = "antmaze-medium-navigate-v0"
OBS_DIM, ACT_DIM, LATENT_DIM, N_HEADS = 29, 8, 128, 2


def _skip_without_checkpoint(env_name: str) -> None:
    if not (frozen.checkpoint_dir(env_name) / "params.pkl").exists():
        pytest.skip(f"{env_name} checkpoint absent: run `uv run python -m fbplan.frozen`")


def _skip_without_dataset(env_name: str, split: str) -> None:
    path = frozen.dataset_path(env_name, split)
    if not path.exists():
        pytest.skip(f"{path} absent: call fbplan.frozen.load_dataset once to download it")


@pytest.fixture(scope="module")
def model() -> frozen.FrozenFBModel:
    _skip_without_checkpoint(MEDIUM)
    return frozen.load_model(MEDIUM)


@pytest.fixture(scope="module")
def observations() -> np.ndarray:
    """Start and goal observations of the 5 medium tasks: realistic states, no dataset needed."""
    env = frozen.make_env(MEDIUM)
    obs = []
    for task_id in range(1, 6):
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


def test_download_checkpoint_does_not_use_the_network_when_files_are_present(monkeypatch):
    _skip_without_checkpoint(MEDIUM)

    def no_network(*args, **kwargs):
        raise AssertionError("download_checkpoint used the network")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)

    assert frozen.download_checkpoint(MEDIUM) == frozen.checkpoint_dir(MEDIUM)


def test_checkpoint_names_outside_the_four_navigate_envs_are_refused():
    with pytest.raises(ValueError, match="antmaze-medium-stitch-v0"):
        frozen.checkpoint_dir("antmaze-medium-stitch-v0")


def test_restored_parameters_equal_the_checkpoint_arrays(model):
    saved = frozen.load_numpy_pickle(frozen.checkpoint_dir(MEDIUM) / "params.pkl")
    saved_params = saved["agent"]["network"]["params"]

    restored = jax.tree_util.tree_leaves_with_path(model.agent.network.params)
    expected = dict(jax.tree_util.tree_leaves_with_path(saved_params))

    assert len(restored) == len(expected) == 36
    for path, leaf in restored:
        np.testing.assert_array_equal(np.asarray(leaf), expected[path])


@pytest.mark.parametrize("env_name", frozen.ENV_NAMES)
def test_every_checkpoint_loads_with_its_training_discount(env_name):
    _skip_without_checkpoint(env_name)

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


def test_rho_of_a_state_to_itself_is_one(model, observations):
    """rho(w, w) = F(w, z_w)^T B(w) / F(w, z_w)^T B(w) with z_w = B(w).

    Numerator and denominator come from calls with a different batch order, so the test also
    checks that the result for one row does not depend on the other rows of the batch.
    """
    b = np.asarray(model.backward(observations))
    rev = slice(None, None, -1)
    num = np.sum(np.asarray(model.forward(observations, b)) * b, axis=-1)
    den = np.sum(np.asarray(model.forward(observations[rev], b[rev])) * b[rev], axis=-1)[:, rev]

    assert np.all(den > 0), "self successor measure F(w, z_w)^T B(w) must be positive"
    np.testing.assert_allclose(num / den, 1.0, rtol=1e-4)


def test_forward_ignores_the_scale_of_z(model, observations, latents):
    # Full float32 matmuls: on an Ampere GPU the default (TF32) gives errors of about 1.6e-3.
    with jax.default_matmul_precision("highest"):
        reference = np.asarray(model.forward(observations, latents))
        for scale in (1e-3, 1e3):
            scaled = np.asarray(model.forward(observations, scale * latents))
            np.testing.assert_allclose(scaled, reference, rtol=1e-4, atol=1e-4)


def test_low_action_is_deterministic_at_zero_temperature(model, observations, latents):
    without_seed = np.asarray(model.low_action(observations, latents))
    for seed in (1, 2):
        with_seed = model.low_action(observations, latents, 0.0, jax.random.PRNGKey(seed))
        np.testing.assert_array_equal(np.asarray(with_seed), without_seed)


def test_sampling_follows_the_seed(model, observations, latents):
    def sample(seed: int) -> np.ndarray:
        return np.asarray(model.low_action(observations, latents, 1.0, jax.random.PRNGKey(seed)))

    np.testing.assert_array_equal(sample(0), sample(0))
    assert not np.allclose(sample(0), sample(1))


def test_sampling_without_a_seed_is_an_error(model, observations, latents):
    with pytest.raises(ValueError, match="seed"):
        model.low_action(observations, latents, temperature=1.0)
    with pytest.raises(ValueError, match="seed"):
        model.high_latent(observations, latents, temperature=1.0)


def test_baseline_policy_matches_upstream_sample_actions(model, observations):
    """pi_l(s, pi_h(s, z_r)) at temperature 0 equals upstream FBpiSwitchAgent.sample_actions.

    The test uses a z_r of norm sqrt(d). Upstream passes z_r to pi_h without normalizing it,
    and the real z_r is a little shorter (see the task latent test), so with the real z_r the
    two paths differ slightly: high_latent normalizes every input, as the contract says.
    Both paths use full float32 matmuls: on an Ampere GPU the default (TF32) gives errors of
    about 1.6e-3.
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
    _skip_without_dataset(MEDIUM, "val")

    first, second = model.infer_task_latent(1), model.infer_task_latent(1)

    assert first.shape == (LATENT_DIM,) and first.dtype == np.float32
    np.testing.assert_array_equal(first, second)


def test_task_latent_points_along_the_sum_of_b_over_goal_states(model):
    """z_r = normalize_z(mean(r * B(s))) with r = 1 within the goal radius of the first 100k
    validation states. Its norm is a little below sqrt(d): upstream averages twice (uniform
    softmax weights 1/n, then a mean), so the raw vector is tiny and the 1e-8 of normalize_z
    is not negligible. Only the direction is checked here."""
    _skip_without_dataset(MEDIUM, "val")
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
    assert np.linalg.norm(z_r) <= np.sqrt(LATENT_DIM)


def test_task_latents_differ_between_tasks(model):
    _skip_without_dataset(MEDIUM, "val")

    assert not np.allclose(model.infer_task_latent(1), model.infer_task_latent(2))


def test_make_env_gives_the_upstream_eval_env():
    env = frozen.make_env(MEDIUM)
    ob, info = env.reset(seed=0, options={"task_id": 1})
    maze = env.unwrapped

    assert maze._add_noise_to_goal is False
    np.testing.assert_array_equal(maze.cur_goal_xy, maze.ij_to_xy(maze.task_infos[0]["goal_ij"]))
    assert ob.shape == (OBS_DIM,) and info["goal"].shape == (OBS_DIM,)
    assert env.action_space.shape == (ACT_DIM,)
    assert env.spec.max_episode_steps == 1000


def test_eval_env_reproduces_the_dataset_dynamics():
    """One env step from a dataset state with the dataset action gives the dataset next state.

    pi_l was trained on these transitions, so the eval env must have the same physics.
    MuJoCo 3.1.6 reproduces them to float precision (about 4e-7). MuJoCo 3.17 does not (8e-3).
    """
    _skip_without_dataset(MEDIUM, "val")
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
def test_load_dataset_gives_aligned_transitions(split):
    _skip_without_dataset(MEDIUM, split)

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
