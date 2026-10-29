import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from fbplan.learned.sampling import HighBatchSpec, make_trajectory_data, sample_high_batch

# Upstream is imported only to compare distributions. TODO(e-learner): use the a-frozen
# import helper once it is on main.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "ssm"))
from agents.fbpiswitch import get_config  # noqa: E402
from utils.datasets import Dataset, HGCDataset  # noqa: E402

TRAJ_LENGTHS = (300, 50, 1000, 7, 600, 120)


def synthetic_dataset() -> dict[str, np.ndarray]:
    """Flat dataset whose first observation coordinate is the global state index."""
    n = sum(TRAJ_LENGTHS)
    terminals = np.zeros(n, dtype=np.float32)
    terminals[np.cumsum(TRAJ_LENGTHS) - 1] = 1.0
    observations = np.zeros((n, 3), dtype=np.float32)
    observations[:, 0] = np.arange(n)
    return {
        "observations": observations,
        "next_observations": observations,
        "actions": np.zeros((n, 2), dtype=np.float32),
        "terminals": terminals,
    }


def upstream_config(**updates):
    config = get_config()
    config.update(updates)
    return config


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic."""
    grid = np.union1d(a, b)
    cdf_a = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    cdf_b = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def test_final_idx_points_to_trajectory_end():
    obs = np.zeros((5, 1), dtype=np.float32)
    data = make_trajectory_data(obs, np.array([0, 0, 1, 0, 1]))
    np.testing.assert_array_equal(np.asarray(data.final_idx), [2, 2, 2, 4, 4])


def test_make_trajectory_data_rejects_open_last_trajectory():
    with pytest.raises(ValueError):
        make_trajectory_data(np.zeros((3, 1), np.float32), np.array([1, 0, 0]))


@pytest.mark.parametrize("geom", [False, True])
def test_goals_and_targets_stay_in_the_future_of_the_same_trajectory(geom):
    ds = synthetic_dataset()
    data = make_trajectory_data(ds["observations"], ds["terminals"])
    spec = HighBatchSpec(batch_size=20_000, discount=0.9, actor_geom_sample=geom,
                         actor_p_randomgoal=0.0)
    batch = sample_high_batch(data, jax.random.PRNGKey(0), spec)
    idx = np.asarray(batch["observations"][:, 0]).astype(int)
    final = np.asarray(data.final_idx)[idx]
    start = np.minimum(idx + 1, final)
    for key in ("high_actor_goals", "high_actor_targets"):
        other = np.asarray(batch[key][:, 0]).astype(int)
        assert np.all((other >= start) & (other <= final)), key


@pytest.mark.parametrize(
    "updates",
    [
        {},  # the fbpiswitch defaults: uniform trajectory goals, no random goals
        {"actor_geom_sample": True, "discount": 0.98},
        {"actor_p_trajgoal": 0.7, "actor_p_randomgoal": 0.3},
    ],
)
def test_offsets_match_upstream_hgcdataset(updates):
    ds = synthetic_dataset()
    config = upstream_config(**updates)
    n = 100_000

    np.random.seed(0)
    upstream = HGCDataset(Dataset.create(**ds), config).sample(n)
    spec = HighBatchSpec.from_agent_config(config.to_dict() | {"batch_size": n})
    ours = sample_high_batch(make_trajectory_data(ds["observations"], ds["terminals"]),
                             jax.random.PRNGKey(0), spec)

    def offsets(batch, key):
        return np.asarray(batch[key][:, 0]) - np.asarray(batch["observations"][:, 0])

    assert ks_statistic(np.asarray(ours["observations"][:, 0]),
                        upstream["observations"][:, 0]) < 0.01
    for key in ("high_actor_goals", "high_actor_targets"):
        assert ks_statistic(offsets(ours, key), offsets(upstream, key)) < 0.01, key
