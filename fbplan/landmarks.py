"""Landmark selection: N observations from the offline data that cover the state space.

Farthest-point sampling (greedy k-center) on the maze position xy (obs[:2]) gives an even
cover, so that every state has a landmark within a short, trustworthy FB edge. Landmarks keep
the full observation, because F(w, .) needs the whole state.

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

from fbplan.contracts import FBModel
from fbplan.option_model import batched_backward

LandmarkLatent = Literal["region", "state"]


@dataclass(frozen=True)
class LandmarkConfig:
    """count landmarks by farthest-point sampling in xy, from a random pool of pool_size
    dataset observations (the pool bounds the cost of sampling); seed draws the pool and the
    first landmark.

    latent: "region" gives each landmark the mean of raw B over the pool states within
    region_xy xy units of it; "state" uses B(w) of the landmark state.
    """

    count: int = 300
    pool_size: int = 100_000
    seed: int = 0
    latent: LandmarkLatent = "state"
    region_xy: float = 0.5

    def __post_init__(self) -> None:
        if self.latent not in ("region", "state"):
            raise ValueError(f"unknown latent {self.latent!r}")


@dataclass(frozen=True)
class Landmarks:
    """The selected landmark states and their option latents."""

    observations: np.ndarray  # [N, obs_dim]
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
    rows = _draw_pool(rng, len(observations), config.pool_size)
    pool_size = len(rows)
    pool = observations[rows]
    if config.latent == "region" and fb is None:
        raise ValueError("region latents need an FB model")
    first = int(rng.integers(pool_size))
    picked = farthest_point_sampling(pool[:, :2], config.count, first)
    latents = None
    if config.latent == "region":
        latents = region_latents(fb, pool, pool[picked, :2], config.region_xy)
    return Landmarks(pool[picked], latents)


@dataclass(frozen=True, eq=False)
class RegionField:
    """Region-mean latents at any xy point: the mean of raw B over the data states within
    `radius` xy units of it, as upstream builds z_r from a goal region. A point with no state that
    close (a straight route segment can cut a wall corner) takes the region of its nearest state.
    """

    xy: np.ndarray  # [P, 2] xy of the data states
    b: np.ndarray  # [P, d] their raw B
    radius: float

    @classmethod
    def from_data(
        cls,
        fb: FBModel,
        observations: np.ndarray,
        radius: float,
        pool_size: int = 100_000,
        seed: int = 0,
    ) -> RegionField:
        """From the same random pool of dataset states as select_landmarks with this pool_size
        and seed."""
        pool = observations[_draw_pool(np.random.default_rng(seed), len(observations), pool_size)]
        return cls(pool[:, :2], batched_backward(fb, pool), radius)

    def at(self, point: np.ndarray) -> np.ndarray:
        """The region-mean latent around the xy point. [d]"""
        dist2 = np.sum((self.xy - point) ** 2, axis=1)
        if dist2.min() > self.radius**2:
            dist2 = np.sum((self.xy - self.xy[np.argmin(dist2)]) ** 2, axis=1)
        return self.b[dist2 <= self.radius**2].mean(axis=0)


def region_latents(
    fb: FBModel, observations: np.ndarray, centres_xy: np.ndarray, radius: float
) -> np.ndarray:
    """Mean of raw B over the observations within `radius` xy units of each centre. [N, d]

    This is how upstream builds the task latent z_r from a goal region.
    """
    xy, b = observations[:, :2], batched_backward(fb, observations)
    out = np.empty((len(centres_xy), b.shape[1]))
    for i, centre in enumerate(centres_xy):
        inside = np.sum((xy - centre) ** 2, axis=1) <= radius**2
        if not inside.any():
            raise ValueError(f"no observation within {radius} of {centre}")
        out[i] = b[inside].mean(axis=0)
    return out


def _draw_pool(rng: np.random.Generator, n: int, pool_size: int) -> np.ndarray:
    """Rows of a random pool of min(pool_size, n) dataset states, drawn first from rng."""
    return rng.choice(n, size=min(pool_size, n), replace=False)
