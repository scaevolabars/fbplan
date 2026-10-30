"""FB distance calibration and wormhole count on one maze.

    uv run python analysis/calibration.py [--env antmaze-medium-navigate-v0] [--n 400]
        [--head-agg min] [--rho-min 0.3 0.5 0.7] [--max-edge-xy 6] [--fig-dir results/figures]
        [--table-dir results/calibration]

For n validation states s, w (uniform sample, fixed seed) it compares the FB distance
-log rho(s, w) / -log gamma (steps, rho from fbplan.option_model.OptionModel, clipped to [0, gamma])
with the exact maze geodesic distance between their xy positions (walls solid, agent a point).

Wormholes by walking speed (walk_speed_wormholes): pairs that the planner trusts
(fbplan.graph.trusted_edges on the raw rho, as in PlannerConfig) whose FB distance is shorter
than the time the ant needs to walk the geodesic at its top speed (99th percentile of the
per-step xy displacement in the data); the planner's max operator exploits exactly these
pairs. The tabular study counts wormholes by another definition (fbplan.tabular.study). The
table also counts the pairs above rho_min that the planner distrusts (raw rho >= rho_max) and
the wormholes that its xy cap (max_edge_xy) leaves. Defaults come from PlannerConfig().
Outputs: a calibration figure, a map of the 200 longest wormholes, and the table, printed as
Markdown and written as CSV (<table-dir>/wormholes_<env>_<head-agg>.csv).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fbplan.controllers.planner import PlannerConfig  # noqa: E402
from fbplan.frozen import ENV_NAMES, load_dataset, load_model, make_env  # noqa: E402
from fbplan.graph import trusted_edges  # noqa: E402
from fbplan.maze import MazeGeometry, draw_walls  # noqa: E402
from fbplan.option_model import OptionModel, OptionModelConfig, OptionTable  # noqa: E402
from fbplan.stats import markdown_table  # noqa: E402

INK, MUTED, GRID, ACCENT, WARN = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#e34948"
Y_CAP = 600  # steps; larger FB distances (and rho = 0) are drawn at the cap
MAX_DRAWN = 200  # wormholes drawn on the map, longest geodesic first
PLANNER = PlannerConfig()


def fb_distance(rho: np.ndarray, discount: float) -> np.ndarray:
    """Steps H with gamma^H = rho; inf where rho = 0."""
    with np.errstate(divide="ignore"):
        return np.log(rho) / np.log(discount)


def step_speeds(data: dict[str, np.ndarray]) -> np.ndarray:
    """xy distance moved in one env step, for every transition of the dataset."""
    return np.linalg.norm(data["next_observations"][:, :2] - data["observations"][:, :2], axis=1)


def walk_speed_wormholes(
    rho_raw: np.ndarray,
    rho: np.ndarray,
    geodesic: np.ndarray,
    discount: float,
    top_speed: float,
    rho_min: float,
) -> np.ndarray:
    """Pairs that the planner trusts (no self-pairs) whose FB distance, from the clipped rho, is
    shorter than the geodesic walk at top speed."""
    trusted = trusted_edges(rho_raw, rho_min, exclude_diagonal=True, rho_max=PLANNER.rho_max)
    return trusted & (fb_distance(rho, discount) < geodesic / top_speed)


def wormhole_table(
    table: OptionTable,
    geodesic: np.ndarray,
    xy_dist: np.ndarray,
    discount: float,
    top_speed: float,
    rho_mins: list[float],
    max_edge_xy: float,
) -> pd.DataFrame:
    """One row per rho_min: trust counts, wormholes by walking speed and their share."""
    off_diagonal = ~np.eye(len(geodesic), dtype=bool)
    walkable = np.isfinite(geodesic)  # a teleport trap exit cannot be reached on foot
    rows = []
    for rho_min in rho_mins:
        above = (table.rho_raw >= rho_min) & off_diagonal
        trusted = trusted_edges(table.rho_raw, rho_min, exclude_diagonal=True,
                                rho_max=PLANNER.rho_max)
        holes = walk_speed_wormholes(table.rho_raw, table.rho, geodesic, discount, top_speed,
                                     rho_min)
        rows.append({
            "rho_min": rho_min,
            "pairs_above_rho_min": int(above.sum()),
            "raw_rho_ge_rho_max": int((above & ~trusted).sum()),
            "trusted_pairs": int(trusted.sum()),
            "wormholes": int(holes.sum()),
            "wormhole_share": holes.sum() / max(trusted.sum(), 1),
            "wormholes_within_xy_cap": int((holes & (xy_dist <= max_edge_xy)).sum()),
            "unreachable_on_foot": int((holes & ~walkable).sum()),
            "longest_geodesic": float(geodesic[holes & walkable].max(initial=0.0)),
        })
    return pd.DataFrame(rows)


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
    draw_walls(ax, maze, GRID)
    ax.scatter(xy[:, 0], xy[:, 1], s=4, color=ACCENT, lw=0)
    for a, b in pairs:
        ax.plot(*np.stack([xy[a], xy[b]]).T, color=WARN, lw=0.8, alpha=0.6)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env", default="antmaze-medium-navigate-v0", choices=ENV_NAMES, help="OGBench env"
    )
    parser.add_argument("--n", type=int, default=400, help="number of sampled states")
    parser.add_argument(
        "--head-agg", default="min", choices=["min", "mean"], help="how the F heads combine"
    )
    parser.add_argument(
        "--rho-min", type=float, nargs="+", default=[PLANNER.rho_min, 0.5, 0.7],
        help="trust thresholds, one table row each",
    )
    parser.add_argument("--max-edge-xy", type=float, default=PLANNER.max_edge_xy,
                        help="the planner's xy cap")
    parser.add_argument("--seed", type=int, default=0, help="seed of the state sample")
    parser.add_argument(
        "--fig-dir", "--out-dir", type=Path, default=Path("results/figures"),
        help="folder of the figures (--out-dir: old name)",
    )
    parser.add_argument(
        "--table-dir", type=Path, default=Path("results/calibration"),
        help="folder of the wormhole tables",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = load_model(args.env)
    data = load_dataset(args.env, "val")
    idx = np.random.default_rng(args.seed).choice(len(data["observations"]), args.n, replace=False)
    obs = data["observations"][idx]
    option_model = OptionModel(model, OptionModelConfig(head_agg=args.head_agg))
    table = option_model.pair_table(obs, option_model.targets(obs))
    maze = MazeGeometry.from_env(make_env(args.env))
    geodesic = maze.geodesic_distances(obs[:, :2], obs[:, :2])
    xy_dist = np.linalg.norm(obs[:, None, :2] - obs[None, :, :2], axis=-1)
    speeds = step_speeds(data)
    median_speed, top_speed = float(np.median(speeds)), float(np.quantile(speeds, 0.99))

    plotted = ~np.eye(args.n, dtype=bool) & np.isfinite(geodesic)
    rho, geo = table.rho[plotted], geodesic[plotted]
    tag = f"{args.env}, n = {args.n}, F heads: {args.head_agg}"
    plot_calibration(geo, fb_distance(rho, model.discount), median_speed,
                     f"FB distance vs. maze geodesic ({tag})",
                     args.fig_dir / f"calibration_{args.env}_{args.head_agg}.png")
    print(f"{tag}: {plotted.sum()} walkable ordered pairs; rho clipped low {table.n_clipped_low}, "
          f"high {table.n_clipped_high}; rho = 0 for {np.mean(rho == 0):.1%}")
    print(f"step speed: median {median_speed:.3f}, p99 {top_speed:.3f} xy units per step\n")

    result = wormhole_table(table, geodesic, xy_dist, model.discount, top_speed, args.rho_min,
                            args.max_edge_xy)
    print(markdown_table(result.round(3)))
    context = {"env": args.env, "head_agg": args.head_agg, "n_states": args.n,
               "top_speed": top_speed, "max_edge_xy": args.max_edge_xy}
    args.table_dir.mkdir(parents=True, exist_ok=True)
    result.assign(**context).to_csv(
        args.table_dir / f"wormholes_{args.env}_{args.head_agg}.csv", index=False
    )

    rho_min = min(args.rho_min)
    mask = walk_speed_wormholes(table.rho_raw, table.rho, geodesic, model.discount, top_speed,
                                rho_min)
    pairs = np.argwhere(mask)
    drawn = pairs[np.argsort(-geodesic[mask])[:MAX_DRAWN]]
    plot_wormholes(maze, obs[:, :2], drawn,
                   f"{len(drawn)} longest of {len(pairs)} wormholes, rho >= {rho_min}\n{tag}",
                   args.fig_dir / f"wormholes_{args.env}_{args.head_agg}.png")


if __name__ == "__main__":
    main()
