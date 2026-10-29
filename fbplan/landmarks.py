"""Landmark selection: N observations from the offline data that cover the state space.

Farthest-point sampling (greedy k-center) gives an even cover, so that every state has a
landmark within a short, trustworthy FB edge. Two spaces: maze position xy (obs[:2]) and
the normalized backward embedding B(s), which needs no maze knowledge. Landmarks keep the
full observation, because F(w, .) needs the whole state.

The latent of a landmark is the intention z_w that the agent follows to reach it: B(w) of the
landmark state (default), or the mean of B over the states within region_xy of w, as upstream
builds z_r. B(w) of one state carries the pose and velocity of w, and with it the privileged
oracle overshoots (d-eval: region means raised it on medium from 58 % to 87 %). The planner
did not gain: on 100 held-out medium episodes 68/100 with B(w), 59/100 with region means.
With region means rho is higher and flatter, so it separates near and far landmarks less.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from fbplan.contracts import FBModel, normalize_z

LandmarkSpace = Literal["xy", "backward"]
LandmarkLatent = Literal["region", "state"]
_B_BATCH = 4096


@dataclass(frozen=True)
class LandmarkConfig:
    """count landmarks by farthest-point sampling in `space`, from a random pool of
    pool_size dataset observations (the pool bounds the cost of sampling).

    latent: "region" gives each landmark the mean of raw B over the pool states within
    region_xy xy units of it; "state" uses B(w) of the landmark state.
    """

    count: int = 300
    space: LandmarkSpace = "xy"
    pool_size: int = 100_000
    seed: int = 0
    latent: LandmarkLatent = "state"
    region_xy: float = 0.5

    def __post_init__(self) -> None:
        if self.space not in ("xy", "backward"):
            raise ValueError(f"unknown space {self.space!r}")
        if self.latent not in ("region", "state"):
            raise ValueError(f"unknown latent {self.latent!r}")


@dataclass(frozen=True)
class Landmarks:
    observations: np.ndarray  # [N, obs_dim]
    dataset_index: np.ndarray  # [N] rows of the dataset they come from
    latents: np.ndarray | None  # [N, d] region-mean B; None: use B(w) of each landmark state


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
    rows = rng.choice(len(observations), size=pool_size, replace=False)
    pool = observations[rows]
    needs_b = config.space == "backward" or config.latent == "region"
    if needs_b and fb is None:
        raise ValueError("backward-space landmarks and region latents need an FB model")
    b = _backward(fb, pool) if needs_b else None
    features = pool[:, :2] if config.space == "xy" else np.asarray(normalize_z(b, b.shape[-1]))
    first = int(rng.integers(pool_size))
    picked = farthest_point_sampling(features, config.count, first)
    latents = None
    if config.latent == "region":
        latents = _region_mean(b, pool[:, :2], pool[picked, :2], config.region_xy)
    return Landmarks(pool[picked], rows[picked], latents)


def region_latents(
    fb: FBModel, observations: np.ndarray, centres_xy: np.ndarray, radius: float
) -> np.ndarray:
    """Mean of raw B over the observations within `radius` xy units of each centre. [N, d]"""
    return _region_mean(_backward(fb, observations), observations[:, :2], centres_xy, radius)


def _region_mean(
    b: np.ndarray, xy: np.ndarray, centres_xy: np.ndarray, radius: float
) -> np.ndarray:
    out = np.empty((len(centres_xy), b.shape[1]))
    for i, centre in enumerate(centres_xy):
        inside = np.sum((xy - centre) ** 2, axis=1) <= radius**2
        if not inside.any():
            raise ValueError(f"no observation within {radius} of {centre}")
        out[i] = b[inside].mean(axis=0)
    return out


def _backward(fb: FBModel, obs: np.ndarray) -> np.ndarray:
    """Raw B(s) in batches. [P, d]"""
    return np.concatenate(
        [np.asarray(fb.backward(obs[i : i + _B_BATCH])) for i in range(0, len(obs), _B_BATCH)]
    )
