"""Tests for fbplan.contracts: latent normalization and the value semantics of TaskSpec."""

import jax.numpy as jnp
import numpy as np

from fbplan.contracts import TaskSpec, normalize_z


def test_normalize_z_gives_norm_sqrt_d_for_numpy_and_jax():
    z = np.random.default_rng(0).normal(size=(4, 128))
    for x in (z, jnp.asarray(z)):
        norms = np.linalg.norm(np.asarray(normalize_z(x, 128)), axis=-1)
        np.testing.assert_allclose(norms, np.sqrt(128), rtol=1e-5)


def test_normalize_z_keeps_float32_for_numpy_and_jax():
    z = np.random.default_rng(0).normal(size=(4, 128)).astype(np.float32)
    for x in (z, jnp.asarray(z)):
        assert normalize_z(x, 128).dtype == np.float32


def test_normalize_z_ignores_input_scale():
    z = np.random.default_rng(1).normal(size=(3, 8))
    np.testing.assert_allclose(normalize_z(z, 8), normalize_z(1e3 * z, 8), rtol=1e-6)


def test_normalize_z_keeps_zero_vector_finite():
    assert np.all(np.isfinite(normalize_z(np.zeros((1, 8)), 8)))


def test_task_specs_compare_and_hash_by_identity():
    """Array fields have no single truth value, so TaskSpec uses identity, not field equality."""

    def spec() -> TaskSpec:
        return TaskSpec("antmaze-medium-navigate-v0", 1, np.ones(128), np.zeros(2))

    first, second = spec(), spec()

    assert first == first and first != second
    assert len({first, second}) == 2
