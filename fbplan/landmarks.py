"""Landmark selection: N observations from the offline data that cover the state space.

Farthest-point sampling (greedy k-center) gives an even cover, so that every state has a
landmark within a short, trustworthy FB edge. Two spaces: maze position xy (obs[:2]) and
the normalized backward embedding B(s), which needs no maze knowledge. Landmarks keep the
full observation, because B(w) and F(w, .) need the whole state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from fbplan.contracts import FBModel, normalize_z

LandmarkSpace = Literal["xy", "backward"]
_B_BATCH = 4096


@dataclass(frozen=True)
class LandmarkConfig:
    """count landmarks by farthest-point sampling in `space`, from a random pool of
    pool_size dataset observations (the pool bounds the cost of sampling)."""

    count: int = 300
    space: LandmarkSpace = "xy"
    pool_size: int = 100_000
    seed: int = 0


@dataclass(frozen=True)
class Landmarks:
    observations: np.ndarray  # [N, obs_dim]
    dataset_index: np.ndarray  # [N] rows of the dataset they come from


def farthest_point_sampling(points: np.ndarray, count: int, first: int = 0) -> np.ndarray:
    """Indices of `count` points; each new point is the farthest from those already chosen."""
    if count > len(points):
        raise ValueError(f"cannot choose {count} of {len(points)} points")
    chosen = np.empty(count, dtype=int)
    chosen[0] = first
    dist = np.linalg.norm(points - points[first], axis=1)
    dist[first] = -np.inf
    for k in range(1, count):
        chosen[k] = int(np.argmax(dist))
        dist = np.minimum(dist, np.linalg.norm(points - points[chosen[k]], axis=1))
        dist[chosen[k]] = -np.inf  # chosen points stay out, also when points repeat
    return chosen


def select_landmarks(
    observations: np.ndarray, config: LandmarkConfig, fb: FBModel | None = None
) -> Landmarks:
    """Choose landmarks from dataset observations [P, obs_dim]."""
    rng = np.random.default_rng(config.seed)
    pool_size = min(config.pool_size, len(observations))
    pool = rng.choice(len(observations), size=pool_size, replace=False)
    features = _features(observations[pool], config.space, fb)
    first = int(rng.integers(pool_size))
    picked = pool[farthest_point_sampling(features, config.count, first)]
    return Landmarks(observations[picked], picked)


def _features(obs: np.ndarray, space: LandmarkSpace, fb: FBModel | None) -> np.ndarray:
    if space == "xy":
        return obs[:, :2]
    if fb is None:
        raise ValueError("landmarks in backward space need an FB model")
    b = np.concatenate(
        [np.asarray(fb.backward(obs[i : i + _B_BATCH])) for i in range(0, len(obs), _B_BATCH)]
    )
    return np.asarray(normalize_z(b, b.shape[-1]))
