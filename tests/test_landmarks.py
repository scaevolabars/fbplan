import numpy as np
import pytest
from planner_fakes import GridFB

from fbplan.landmarks import (
    LandmarkConfig,
    farthest_point_sampling,
    region_latents,
    select_landmarks,
)


def test_farthest_point_sampling_spreads_points_on_a_line():
    points = np.arange(10.0)[:, None]
    np.testing.assert_array_equal(farthest_point_sampling(points, 3, first=0), [0, 9, 4])


def test_farthest_point_sampling_never_repeats_a_duplicated_point():
    points = np.zeros((5, 2))
    assert len(set(farthest_point_sampling(points, 5).tolist())) == 5


def test_farthest_point_sampling_rejects_more_landmarks_than_points():
    with pytest.raises(ValueError):
        farthest_point_sampling(np.zeros((3, 2)), 4)


def dataset_obs(fb: GridFB, repeats: int = 5) -> np.ndarray:
    """Every maze cell several times, with two extra observation dims that are not xy."""
    xy = np.tile(np.stack([fb.obs(c) for c in fb.cells]), (repeats, 1))
    extra = np.arange(len(xy) * 2, dtype=np.float32).reshape(-1, 2)
    return np.concatenate([xy, extra], axis=1)


def test_select_landmarks_covers_every_cell_and_keeps_full_observations():
    fb = GridFB()
    obs = dataset_obs(fb)
    marks = select_landmarks(obs, LandmarkConfig(count=fb.latent_dim), fb)
    assert len(set(fb.cell_ids(marks.observations).tolist())) == fb.latent_dim
    rows = {tuple(row) for row in obs}  # the extra dims make every dataset row unique
    assert all(tuple(row) in rows for row in marks.observations)


def test_select_landmarks_is_reproducible_for_a_seed():
    obs = dataset_obs(GridFB())
    cfg = LandmarkConfig(count=4, pool_size=20, seed=3, latent="state")
    np.testing.assert_array_equal(
        select_landmarks(obs, cfg).observations, select_landmarks(obs, cfg).observations
    )


def test_region_latents_average_b_over_the_states_within_the_radius():
    fb = GridFB()
    obs = dataset_obs(fb, repeats=1)
    z = region_latents(fb, obs, np.array([[1.0, 1.0]]), radius=1.0)
    inside = [(1, 1), (2, 1), (1, 2)]  # (1, 1) and its two neighbours in the U maze
    expected = np.mean([np.eye(fb.latent_dim)[fb.index[c]] for c in inside], axis=0)
    np.testing.assert_allclose(z[0], expected)


def test_select_landmarks_gives_region_latents_on_request():
    fb = GridFB()
    obs = dataset_obs(fb)
    marks = select_landmarks(obs, LandmarkConfig(count=3, latent="region", region_xy=1.0), fb)
    expected = region_latents(fb, obs, marks.observations[:, :2], radius=1.0)
    np.testing.assert_allclose(marks.latents, expected)


def test_region_latents_need_a_model():
    with pytest.raises(ValueError):
        select_landmarks(dataset_obs(GridFB()), LandmarkConfig(count=3, latent="region"))
