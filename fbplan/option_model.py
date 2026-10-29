"""Option model of "follow pi_{z_w} until w is hit", read off a frozen FB model.

With z_w = B(w) and per F head h (docs/design.md §3):

    rho_h(s, w) = F_h(s, z_w)^T B(w) / F_h(w, z_w)^T B(w)              ~ E[gamma^H]
    R_h(s, w)   = F_h(s, z_w)^T z_r - rho_h(s, w) * F_h(w, z_w)^T z_r  reward before the hit
    direct_h(s) = F_h(s, z_r)^T z_r                                    value of pi_{z_r}

Each quantity is computed per head and then aggregated over heads (min or mean). rho is
clipped after aggregation to [0, gamma] for distinct states (H >= 1, so E[gamma^H] <= gamma;
this makes graph value iteration a gamma-contraction, design.md §3) and to [0, 1] for a state
and itself (same observation, H = 0). The clips are counted because they measure how far FB is
from a valid hitting-time discount, and the raw aggregate is kept: trust rules use it, because
an estimate above 1 is an FB error that clipping would hide. rho and R do not depend on the
scale of B(w): the FB model normalizes z_w before F, and the ratio cancels the scale in the
dot products.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np

from fbplan.contracts import FBModel

HeadAgg = Literal["min", "mean"]
EdgeReward = Literal["exact", "paper", "zero"]


@dataclass(frozen=True)
class OptionModelConfig:
    """head_agg: how to combine the F heads; min is the pessimistic guard against wormholes.

    edge_reward: "exact" is R above; "paper" drops the term -rho * F(w, z_w)^T z_r, as the
    paper's pi_h advantage does; "zero" sets R = 0, which turns planning for goal-reaching
    into a shortest path with edge cost -log rho. "zero" is the default: on medium, exact R
    is noise that the max over routes collects (per trusted edge median +2.1, p99 +68, while
    the largest direct value is 96), and goal-reaching has R ~ 0 off the goal (design.md §3).
    batch_size: the most (source, target) pairs per F call.
    """

    head_agg: HeadAgg = "min"
    edge_reward: EdgeReward = "zero"
    batch_size: int = 4096

    def __post_init__(self) -> None:
        if self.head_agg not in ("min", "mean"):
            raise ValueError(f"unknown head_agg {self.head_agg!r}")
        if self.edge_reward not in ("exact", "paper", "zero"):
            raise ValueError(f"unknown edge_reward {self.edge_reward!r}")


@dataclass(frozen=True)
class Targets:
    """Per-target quantities that every (source, target) pair reuses."""

    obs: np.ndarray  # [M, obs_dim]
    z: np.ndarray  # [M, d] raw B(w)
    self_forward: np.ndarray  # [H, M, d] F_h(w, z_w)

    @property
    def self_measure(self) -> np.ndarray:
        """F_h(w, z_w)^T B(w), the denominator of rho. [H, M]"""
        return np.einsum("hmd,md->hm", self.self_forward, self.z)

    def take(self, idx: np.ndarray) -> Targets:
        return Targets(self.obs[idx], self.z[idx], self.self_forward[:, idx])


@dataclass(frozen=True)
class OptionTable:
    """rho and R for every (source, target) pair."""

    rho: np.ndarray  # [N, M] clipped: [0, gamma] for distinct states, [0, 1] for a state and itself
    rho_raw: np.ndarray  # [N, M] aggregated over heads, before clipping
    reward: np.ndarray | None  # [N, M], None when no task latent was given
    n_clipped_low: int  # pairs with rho_raw < 0
    n_clipped_high: int  # pairs with rho_raw above their upper bound (gamma or 1)


class OptionModel:
    """Batched rho, R and direct values from a frozen FBModel."""

    def __init__(self, fb: FBModel, config: OptionModelConfig) -> None:
        self.fb = fb
        self.config = config

    def targets(self, obs: np.ndarray) -> Targets:
        z = np.concatenate([self._backward(obs[c]) for c in self._chunks(len(obs))])
        return Targets(obs, z, self._forward(obs, z))

    def pair_table(
        self, sources: np.ndarray, targets: Targets, z_r: np.ndarray | None = None
    ) -> OptionTable:
        """rho(s_i, w_j), and R(s_i, w_j) when z_r is given, for all i, j.

        Pairs go through F one batch at a time, so memory does not grow with N * M * d.
        """
        n, m = len(sources), len(targets.obs)
        src, tgt = np.divmod(np.arange(n * m), m)
        self_measure = targets.self_measure
        value_after = None if z_r is None else targets.self_forward @ z_r  # [H, M]
        raw, upper = np.empty(n * m), np.empty(n * m)
        reward = None if z_r is None else np.empty(n * m)
        for c in self._chunks(n * m):
            s, z_w = sources[src[c]], targets.z[tgt[c]]
            f = self._call_padded(self.fb.forward, s, z_w)  # [H, P, d]
            rho_h = _safe_ratio(np.einsum("hpd,pd->hp", f, z_w), self_measure[:, tgt[c]])
            raw[c] = self._aggregate(rho_h)
            same = np.all(s == targets.obs[tgt[c]], axis=-1)
            upper[c] = np.where(same, 1.0, self.fb.discount)
            if reward is not None:
                reward[c] = self._reward(f @ z_r, rho_h, value_after[:, tgt[c]])
        return OptionTable(
            rho=np.clip(raw, 0.0, upper).reshape(n, m),
            rho_raw=raw.reshape(n, m),
            reward=None if reward is None else reward.reshape(n, m),
            n_clipped_low=int((raw < 0).sum()),
            n_clipped_high=int((raw > upper).sum()),
        )

    def direct_value(self, obs: np.ndarray, z_r: np.ndarray) -> np.ndarray:
        """F(s, z_r)^T z_r for each observation. [N]"""
        f = self._forward(obs, np.broadcast_to(z_r, (len(obs), z_r.shape[-1])))
        return self._aggregate(f @ z_r)

    def _reward(
        self, value_before: np.ndarray, rho_h: np.ndarray, value_after: np.ndarray
    ) -> np.ndarray:
        """R per pair from F(s, z_w)^T z_r, rho_h and F(w, z_w)^T z_r, all [H, P]."""
        if self.config.edge_reward == "zero":
            return np.zeros(value_before.shape[1])
        if self.config.edge_reward == "paper":
            return self._aggregate(value_before)
        return self._aggregate(value_before - rho_h * value_after)

    def _aggregate(self, per_head: np.ndarray) -> np.ndarray:
        return per_head.min(axis=0) if self.config.head_agg == "min" else per_head.mean(axis=0)

    def _forward(self, obs: np.ndarray, z: np.ndarray) -> np.ndarray:
        """F(obs, z) in batches -> [H, N, d] float64."""
        parts = [self._call_padded(self.fb.forward, obs[c], z[c]) for c in self._chunks(len(obs))]
        return np.concatenate(parts, axis=1)

    def _backward(self, obs: np.ndarray) -> np.ndarray:
        return self._call_padded(self.fb.backward, obs)

    def _chunks(self, n: int) -> Iterator[slice]:
        size = self.config.batch_size
        return (slice(i, min(i + size, n)) for i in range(0, n, size))

    def _call_padded(self, fn, *arrays: np.ndarray) -> np.ndarray:
        """Call fn on a batch padded to a power of two (at most batch_size).

        A jitted FB model compiles once per input shape; buckets keep that number small.
        """
        n = len(arrays[0])
        size = min(self.config.batch_size, 1 << max(n - 1, 0).bit_length())
        padded = [np.concatenate([a, np.repeat(a[-1:], size - n, axis=0)]) for a in arrays]
        out = np.asarray(fn(*padded), dtype=np.float64)
        return out[..., :n, :]


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den, and 0 where den <= 0: a target that FB cannot reach from itself is unusable."""
    ok = den > 0
    return np.where(ok, num / np.where(ok, den, 1.0), 0.0)
