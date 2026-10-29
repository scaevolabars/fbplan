"""FB distance calibration and wormhole count on one maze.

    uv run python analysis/calibration.py [--env antmaze-medium-navigate-v0] [--n 400]
        [--head-agg min] [--rho-min 0.3 0.5 0.7] [--out-dir results/figures]

For n validation states s, w (uniform sample, fixed seed) it compares the FB distance
-log rho(s, w) / -log gamma (steps, rho from c-planner's OptionModel) with the exact maze
geodesic distance between their xy positions (walls solid, agent a point).

A trusted pair (rho >= rho_min) is a wormhole when the FB distance is shorter than the time
the ant needs to walk the geodesic at its top speed (99th percentile of the per-step xy
displacement in the data). The planner's max operator exploits exactly these pairs.
Pairs with rho >= 1 (clipped to 1, FB distance 0) are counted apart: the planner already
distrusts them. Outputs: a calibration figure, a map of the 200 longest wormholes with
rho < 1, a printed Markdown table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from fbplan.controllers.oracle import MazeGeometry  # noqa: E402
from fbplan.frozen import load_dataset, load_model, make_env  # noqa: E402
from fbplan.option_model import OptionModel, OptionModelConfig  # noqa: E402

INK, MUTED, GRID, ACCENT, WARN = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#e34948"
Y_CAP = 600  # steps; larger FB distances (and rho = 0) are drawn at the cap
MAX_DRAWN = 200  # wormholes drawn on the map, longest geodesic first


def fb_distance(rho: np.ndarray, discount: float) -> np.ndarray:
    """Steps H with gamma^H = rho; inf where rho = 0."""
    with np.errstate(divide="ignore"):
        return np.log(rho) / np.log(discount)


def step_speeds(data: dict[str, np.ndarray]) -> np.ndarray:
    """xy distance moved in one env step, for every transition of the dataset."""
    return np.linalg.norm(data["next_observations"][:, :2] - data["observations"][:, :2], axis=1)


def wormhole_mask(
    rho: np.ndarray, geodesic: np.ndarray, discount: float, top_speed: float, rho_min: float
) -> np.ndarray:
    """Trusted pairs whose FB distance is shorter than the geodesic walk at top speed."""
    steps_needed = geodesic / top_speed
    return (rho >= rho_min) & (fb_distance(rho, discount) < steps_needed)


def plot_calibration(
    geodesic: np.ndarray, fb_steps: np.ndarray, speed: float, title: str, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)
    y = np.minimum(fb_steps, Y_CAP)
    ax.hexbin(geodesic, y, gridsize=45, bins="log", cmap="Blues", mincnt=1, linewidths=0)
    edges = np.arange(0.0, geodesic.max() + 2.0, 2.0)
    mids, med, q25, q75 = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (geodesic >= lo) & (geodesic < hi)
        if in_bin.sum() >= 20:
            mids.append((lo + hi) / 2)
            med.append(np.median(y[in_bin]))
            q25.append(np.quantile(y[in_bin], 0.25))
            q75.append(np.quantile(y[in_bin], 0.75))
    ax.fill_between(mids, q25, q75, color=INK, alpha=0.12, lw=0, label="FB distance, 25–75 %")
    ax.plot(mids, med, color=INK, lw=2, label="FB distance, median")
    xs = np.array([0.0, geodesic.max()])
    ax.plot(xs, xs / speed, color=WARN, lw=2, ls="--", label="geodesic / median speed")
    ax.set_ylim(0, Y_CAP * 1.02)
    ax.set_xlabel("maze geodesic distance (xy units)", color=MUTED)
    ax.set_ylabel(f"FB distance −log ρ / −log γ (steps, capped at {Y_CAP})", color=MUTED)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_wormholes(
    maze: MazeGeometry, xy: np.ndarray, pairs: np.ndarray, title: str, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=150)
    for i, j in np.argwhere(maze.maze_map == 1):
        x, y = maze.centre((i, j)) - maze.unit / 2
        ax.add_patch(Rectangle((x, y), maze.unit, maze.unit, color=GRID, lw=0))
    ax.scatter(xy[:, 0], xy[:, 1], s=4, color=ACCENT, lw=0)
    for a, b in pairs:
        ax.plot(*np.stack([xy[a], xy[b]]).T, color=WARN, lw=0.8, alpha=0.6)
    rows, cols = maze.maze_map.shape
    lo = maze.centre((0, 0)) - maze.unit / 2
    hi = maze.centre((rows - 1, cols - 1)) + maze.unit / 2
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", default="antmaze-medium-navigate-v0")
    parser.add_argument("--n", type=int, default=400, help="number of sampled states")
    parser.add_argument("--head-agg", default="min", choices=["min", "mean"])
    parser.add_argument("--rho-min", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    model = load_model(args.env)
    data = load_dataset(args.env, "val")
    idx = np.random.default_rng(args.seed).choice(len(data["observations"]), args.n, replace=False)
    obs = data["observations"][idx]
    option_model = OptionModel(model, OptionModelConfig(head_agg=args.head_agg))
    table = option_model.pair_table(obs, option_model.targets(obs))
    maze = MazeGeometry.from_env(make_env(args.env))
    geodesic = maze.geodesic_distances(obs[:, :2], obs[:, :2])

    off_diag = ~np.eye(args.n, dtype=bool)
    rho, geo = table.rho[off_diag], geodesic[off_diag]
    fb_steps = fb_distance(rho, model.discount)
    speeds = step_speeds(data)
    median_speed, top_speed = float(np.median(speeds)), float(np.quantile(speeds, 0.99))

    tag = f"{args.env}, n = {args.n}, F heads: {args.head_agg}"
    plot_calibration(geo, fb_steps, median_speed, f"FB distance vs. maze geodesic ({tag})",
                     args.out_dir / f"calibration_{args.env}_{args.head_agg}.png")
    print(f"{tag}: {off_diag.sum()} ordered pairs; rho clipped low {table.n_clipped_low}, "
          f"high {table.n_clipped_high}; rho = 0 for {np.mean(rho == 0):.1%}")
    print(f"step speed: median {median_speed:.3f}, p99 {top_speed:.3f} xy units per step")
    print("\n| rho_min | trusted pairs | of which rho = 1 | wormholes with rho < 1 "
          "| share of trusted | longest (xy) |")
    print("|---|---|---|---|---|---|")
    below_one = (table.rho < 1.0) & off_diag
    for rho_min in args.rho_min:
        trusted = (table.rho >= rho_min) & off_diag
        mask = wormhole_mask(table.rho, geodesic, model.discount, top_speed, rho_min) & below_one
        longest = float(geodesic[mask].max()) if mask.any() else 0.0
        print(f"| {rho_min} | {trusted.sum()} | {(trusted & ~below_one).sum()} | {mask.sum()} "
              f"| {mask.sum() / max(trusted.sum(), 1):.1%} | {longest:.1f} |")
    rho_min = min(args.rho_min)
    mask = wormhole_mask(table.rho, geodesic, model.discount, top_speed, rho_min) & below_one
    pairs = np.argwhere(mask)
    drawn = pairs[np.argsort(-geodesic[mask])[:MAX_DRAWN]]
    plot_wormholes(maze, obs[:, :2], drawn,
                   f"{len(drawn)} longest of {len(pairs)} wormholes, rho >= {rho_min} ({tag})",
                   args.out_dir / f"wormholes_{args.env}_{args.head_agg}.png")


if __name__ == "__main__":
    main()
