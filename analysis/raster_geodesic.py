"""Independent check of the maze geodesic: an octile Dijkstra on a raster of the free cells.

    uv run python analysis/raster_geodesic.py [--env antmaze-giant-navigate-v0] [--res 0.5]
        [--n 400] [--seed 0] [--no-fb]

The raster has `res` xy units per pixel; a pixel is free if its maze cell is free. Moves go to the
8 neighbours without cutting a wall corner, so the raster distance is at most 8.3 % longer than
the Euclidean geodesic. The script prints:

- the task geodesics (init_xy to goal_xy) of `MazeGeometry.geodesic_distances` and of the raster;
- for the n validation states of `analysis/calibration.py` (same seed and sample), the share of
  pairs more than 4 units apart whose MazeGeometry geodesic is more than 15 % (and 30 %) shorter
  than the raster's: such a geodesic passes through a wall;
- unless --no-fb: the FB distance (head min, `log rho / log gamma`, capped at 600 steps) in 5-unit
  bins of the raster geodesic, with the walk time at the median xy step of the data.

docs/theory.md §11.1-11.2 use these numbers.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra

from fbplan.frozen import ENV_NAMES, load_dataset, load_model, make_env
from fbplan.maze import MazeGeometry
from fbplan.option_model import OptionModel, OptionModelConfig

FB_CAP = 600  # steps; the cap of the FB distance in the bins
BIN = 5  # xy units per bin
MIN_BIN_PAIRS = 20


class Raster:
    """Pixel graph of the free cells of a maze."""

    def __init__(self, geometry: MazeGeometry, res: float) -> None:
        self.geometry, self.res = geometry, res
        k = int(round(geometry.unit / res))
        free = np.kron(geometry.maze_map == 0, np.ones((k, k), dtype=bool))  # pixel (row, col)
        height, width = free.shape
        self.index = -np.ones((height, width), dtype=int)
        self.index[free] = np.arange(free.sum())
        rows, cols, weights = [], [], []
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            r0, r1 = max(0, -dr), height - max(0, dr)
            c0, c1 = max(0, -dc), width - max(0, dc)
            edge = free[r0:r1, c0:c1] & free[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            if dr and dc:  # no corner cutting
                edge &= free[r0 + dr:r1 + dr, c0:c1] & free[r0:r1, c0 + dc:c1 + dc]
            rr, cc = np.nonzero(edge)
            rows += list(self.index[rr + r0, cc + c0])
            cols += list(self.index[rr + r0 + dr, cc + c0 + dc])
            weights += [res * np.hypot(dr, dc)] * len(rr)
        n = int(free.sum())
        self.graph: csr_matrix = coo_matrix(
            (weights + weights, (rows + cols, cols + rows)), shape=(n, n)).tocsr()

    def pixels(self, xy: np.ndarray) -> np.ndarray:
        """Pixel index of each xy point; -1 on a wall pixel."""
        g, (height, width) = self.geometry, self.index.shape
        col = ((xy[:, 0] + g.offset_xy[0] + g.unit / 2) / self.res).astype(int)
        row = ((xy[:, 1] + g.offset_xy[1] + g.unit / 2) / self.res).astype(int)
        return self.index[np.clip(row, 0, height - 1), np.clip(col, 0, width - 1)]

    def distances(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Raster geodesic [len(a), len(b)] between xy points (all on free pixels)."""
        pa, pb = self.pixels(a), self.pixels(b)
        if (pa < 0).any() or (pb < 0).any():
            raise ValueError("a point lies on a wall pixel")
        return dijkstra(self.graph, indices=pa)[:, pb]


def short_shares(geodesic: np.ndarray, raster: np.ndarray, min_dist: float = 4.0) -> dict:
    """Pairs farther apart than min_dist on the raster: median ratio and the shares of pairs
    whose geodesic is more than 15 % and 30 % shorter than the raster distance."""
    keep = np.isfinite(geodesic) & np.isfinite(raster) & (raster > min_dist)
    ratio = geodesic[keep] / raster[keep]
    return {"pairs": int(keep.sum()), "median": float(np.median(ratio)),
            "below_0.85": float(np.mean(ratio < 0.85)), "below_0.7": float(np.mean(ratio < 0.7))}


def sample_states(env_name: str, n: int, seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The validation states of analysis/calibration.py (same rng and call)."""
    data = load_dataset(env_name, "val")
    idx = np.random.default_rng(seed).choice(len(data["observations"]), n, replace=False)
    return data["observations"][idx], data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", default="antmaze-giant-navigate-v0", choices=ENV_NAMES,
                        help="OGBench env")
    parser.add_argument("--res", type=float, default=0.5, help="xy units per raster pixel")
    parser.add_argument("--n", type=int, default=400, help="number of sampled validation states")
    parser.add_argument("--seed", type=int, default=0, help="seed of the state sample")
    parser.add_argument("--no-fb", action="store_true",
                        help="skip the FB distance bins (no model load)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    env = make_env(args.env)
    geometry = MazeGeometry.from_env(env)
    raster = Raster(geometry, args.res)

    tasks = env.unwrapped.task_infos
    init = np.array([t["init_xy"] for t in tasks])
    goal = np.array([t["goal_xy"] for t in tasks])
    task_geo = np.diag(geometry.geodesic_distances(init, goal))
    task_ras = np.diag(raster.distances(init, goal))
    print(f"{args.env}: task geodesics, MazeGeometry {np.round(task_geo, 1)}, "
          f"raster {np.round(task_ras, 1)}")

    obs, data = sample_states(args.env, args.n, args.seed)
    xy = obs[:, :2]
    on = raster.pixels(xy) >= 0
    shares = short_shares(geometry.geodesic_distances(xy[on], xy[on]),
                          raster.distances(xy[on], xy[on]))
    print(f"pairs {shares['pairs']} (states on a wall pixel: {(~on).sum()}); "
          f"MazeGeometry / raster: median {shares['median']:.3f}, "
          f"< 0.85: {shares['below_0.85']:.1%}, < 0.7: {shares['below_0.7']:.1%}")
    if args.no_fb:
        return

    model = load_model(args.env)
    option_model = OptionModel(model, OptionModelConfig(head_agg="min"))
    table = option_model.pair_table(obs, option_model.targets(obs))
    with np.errstate(divide="ignore"):
        steps = np.minimum(np.log(table.rho) / np.log(model.discount), FB_CAP)
    walk = raster.distances(xy, xy)
    pairs = ~np.eye(len(xy), dtype=bool) & np.isfinite(walk)
    speed = np.median(np.linalg.norm(data["next_observations"][:, :2] - data["observations"][:, :2],
                                     axis=1))
    print(f"median xy step {speed:.3f}; raster geodesic max {walk[pairs].max():.1f}")
    print("| raster geodesic | pairs | walk steps at the median step | FB q25 | FB median | FB q75 "
          f"| at the cap ({FB_CAP}) |\n|---|---|---|---|---|---|---|")
    for lo in range(0, int(walk[pairs].max()) + BIN, BIN):
        in_bin = pairs & (walk >= lo) & (walk < lo + BIN)
        if in_bin.sum() >= MIN_BIN_PAIRS:
            q = np.quantile(steps[in_bin], [0.25, 0.5, 0.75])
            print(f"| [{lo}, {lo + BIN}) | {in_bin.sum()} | {(lo + BIN / 2) / speed:.0f} | "
                  f"{q[0]:.0f} | {q[1]:.0f} | {q[2]:.0f} | "
                  f"{np.mean(steps[in_bin] >= FB_CAP):.0%} |")


if __name__ == "__main__":
    main()
