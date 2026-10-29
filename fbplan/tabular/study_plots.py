"""Figures for docs/theory.md from the CSV tables of run_study.

uv run python -m fbplan.tabular.study_plots    # reads and writes docs/figures/theory/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fbplan.tabular.fake_fb import FakeFBConfig, TabularFB  # noqa: E402
from fbplan.tabular.run_study import FAR, MAIN_PLANNERS, OUT_DIR  # noqa: E402
from fbplan.tabular.study import LandmarkModel, rho_bound_wormholes  # noqa: E402

# Validated categorical slots (dataviz reference palette, light mode), in fixed order.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
PLANNERS = [planner.name for planner in MAIN_PLANNERS]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.markersize": 5,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    }
)


def _style(i: int) -> dict:
    return {"color": SLOTS[i], "marker": MARKERS[i]}


def far_success(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Success over pairs at distance >= FAR, pooled per seed, then mean and s.e. over seeds."""
    far = df[df.dist_lo >= FAR].assign(hits=lambda d: d.success * d.n_pairs)
    per_seed = far.groupby([*keys, "seed"], dropna=False)[["hits", "n_pairs"]].sum()
    rate = (per_seed.hits / per_seed.n_pairs).rename("rate").reset_index()
    out = rate.groupby(keys, dropna=False).rate.agg(["mean", "sem"]).reset_index()
    return out.fillna({"sem": 0.0})


def _legend_on_top(fig: plt.Figure, axes: np.ndarray, ncol: int) -> None:
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, 1.0))


def plot_success_vs_distance(noise: pd.DataFrame, out: Path) -> None:
    """Success vs. goal distance per planner; rows: controller, columns: rho noise."""
    levels = [0.01, 0.02, 0.04]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True, sharey=True)
    for row, policy in enumerate(["greedy", "optimal"]):
        for col, level in enumerate(levels):
            ax = axes[row, col]
            sub = noise[(noise.low_policy == policy) & np.isclose(noise.rho_noise, level)]
            stats = sub.groupby(["planner", "dist_lo", "dist_hi"]).success.agg(["mean", "sem"])
            for i, name in enumerate(PLANNERS):
                s = stats.loc[name].reset_index()
                x = (s.dist_lo + s.dist_hi) / 2
                ax.errorbar(x, s["mean"], yerr=s["sem"], label=name, capsize=2, **_style(i))
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(f"{policy} controller, rho noise {level}")
            if row == 1:
                ax.set_xlabel("walking distance start -> goal (steps)")
            if col == 0:
                ax.set_ylabel("success in 100 steps")
    _legend_on_top(fig, axes, ncol=6)
    fig.savefig(out / "success_vs_distance.png")
    plt.close(fig)


def plot_far_success_vs_noise(noise: pd.DataFrame, out: Path) -> None:
    """Far-pair success vs. rho noise per planner, one panel per controller."""
    stats = far_success(noise, ["low_policy", "rho_noise", "planner"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, policy in zip(axes, ["greedy", "optimal"], strict=True):
        for i, name in enumerate(PLANNERS):
            s = stats[(stats.low_policy == policy) & (stats.planner == name)]
            ax.errorbar(s.rho_noise, s["mean"], yerr=s["sem"], label=name, capsize=2, **_style(i))
        ax.set_title(f"{policy} controller")
        ax.set_xlabel("rho noise (std of the error on rho_hat)")
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel(f"success, pairs >= {FAR} steps apart")
    _legend_on_top(fig, axes, ncol=3)
    fig.savefig(out / "far_success_vs_noise.png")
    plt.close(fig)


def plot_guards(guards: pd.DataFrame, out: Path) -> None:
    """Far-pair success of each guard ablation at two noise levels, one panel per controller."""
    stats = far_success(guards, ["low_policy", "rho_noise", "planner"])
    names = list(dict.fromkeys(guards.planner))
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.2), sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, policy in zip(axes, ["greedy", "optimal"], strict=True):
        for i, level in enumerate([0.02, 0.04]):
            s = stats[(stats.low_policy == policy) & np.isclose(stats.rho_noise, level)]
            s = s.set_index("planner").loc[names]
            ax.errorbar(
                s["mean"],
                y,
                xerr=s["sem"],
                ls="none",
                label=f"rho noise {level}",
                capsize=2,
                **_style(i),
            )
        ax.set_title(f"{policy} controller")
        ax.set_xlabel(f"success, pairs >= {FAR} steps apart")
        ax.set_xlim(-0.03, 1.03)
    axes[0].set_yticks(y, names)
    _legend_on_top(fig, axes, ncol=2)
    fig.savefig(out / "guards.png")
    plt.close(fig)


def plot_low_rank(rank: pd.DataFrame, holes: pd.DataFrame, out: Path) -> None:
    """Far-pair success vs. rank for both bases, and the wormhole rate vs. rank."""
    stats = far_success(rank, ["basis", "rank", "planner"])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, basis in zip(axes[:2], ["laplacian", "euclidean"], strict=True):
        for i, name in enumerate(PLANNERS):
            s = stats[(stats.basis == basis) & (stats.planner == name)]
            ax.plot(s["rank"], s["mean"], label=name, **_style(i))
        ax.set_title(f"B basis: {basis}" + (" (ignores walls)" if basis == "euclidean" else ""))
        ax.set_xlabel("rank d of B (104 = exact)")
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel(f"success, pairs >= {FAR} steps apart")
    h = holes[(holes["head_agg"] == "mean") & holes["rank"].notna()]
    for i, basis in enumerate(["laplacian", "euclidean"]):
        s = h[h.basis == basis].groupby("rank").wormhole_rate.mean()
        axes[2].plot(s.index, 100 * s.values, label=basis, **_style(i))
    axes[2].set_title("wormholes: rho_hat > 2 gamma^dist")
    axes[2].set_xlabel("rank d of B")
    axes[2].set_ylabel("% of ordered state pairs")
    axes[2].legend()
    _legend_on_top(fig, axes, ncol=6)
    fig.savefig(out / "low_rank.png")
    plt.close(fig)


def _estimated_distance(config: FakeFBConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(walking distance, -log rho_hat / -log gamma, wormhole flag) for all pairs s != w."""
    model = TabularFB(config)
    rho = LandmarkModel(model, np.arange(model.maze.n_states)).table("mean").rho
    dist = model.maze.geodesic_distances()
    off = ~np.eye(model.maze.n_states, dtype=bool)
    est = np.log(np.maximum(rho, 1e-6)) / np.log(model.discount)
    hole = rho_bound_wormholes(rho, dist, model.discount)
    return dist[off], est[off], hole[off]


def plot_calibration(out: Path) -> None:
    """FB distance vs. walking distance for the exact, a noisy and a wall-blind low-rank fake."""
    cases = [
        ("exact", FakeFBConfig()),
        ("rho noise 0.03", FakeFBConfig(rho_noise=0.03)),
        ("rank 32, euclidean", FakeFBConfig(rank=32, basis="euclidean")),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, (title, config) in zip(axes, cases, strict=True):
        dist, est, hole = _estimated_distance(config)
        jitter = np.random.default_rng(0).uniform(-0.25, 0.25, dist.shape)
        ax.scatter(dist[~hole] + jitter[~hole], est[~hole], s=3, alpha=0.25, color=SLOTS[0], lw=0)
        ax.scatter(
            dist[hole] + jitter[hole], est[hole], s=6, color=SLOTS[1], lw=0, label="wormhole"
        )
        ax.plot([0, 23], [0, 23], color=MUTED, lw=1, ls="--", label="rho = gamma^dist")
        ax.set_title(f"{title}: {hole.mean():.1%} wormholes")
        ax.set_xlabel("walking distance (steps)")
        ax.set_ylim(0, 30)
    axes[0].set_ylabel("-log rho_hat / -log gamma")
    axes[2].legend(loc="upper left")
    fig.savefig(out / "calibration.png")
    plt.close(fig)


def plot_dropped_term(reward: pd.DataFrame, out: Path) -> None:
    """Regret and predicted - true value per R_hat variant; deception vs. hops of the route."""
    df = reward.assign(
        regret=reward.v_star - reward.v_true, deception=reward.v_predicted - reward.v_true
    )
    names = list(dict.fromkeys(df.planner))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    y = np.arange(len(names))[::-1]
    for ax, column, label in [
        (axes[0], "regret", "regret V* - V (reward units, from the task start)"),
        (axes[1], "deception", "predicted - true value (log scale)"),
    ]:
        for i, level in enumerate([0.0, 0.01]):
            s = df[np.isclose(df.rho_noise, level)].groupby("planner")[column].agg(["mean", "sem"])
            s = s.loc[names]
            ax.errorbar(
                s["mean"],
                y,
                xerr=s["sem"],
                ls="none",
                capsize=2,
                label=f"rho noise {level}",
                **_style(i),
            )
        ax.set_xlabel(label)
        ax.set_yticks(y, names)
    axes[1].set_xscale("symlog", linthresh=1.0)
    axes[1].set_xlim(-0.5, 500)
    axes[1].set_yticklabels([])
    axes[0].legend(loc="lower right")
    exact = df[np.isclose(df.rho_noise, 0.0) & df.planner.str.startswith("chained")]
    for i, variant in enumerate(["R=exact", "R=paper"]):
        s = exact[exact.planner.str.endswith(variant)]
        axes[2].scatter(s.hops, s.deception, s=14, label=variant, **_style(i + 2))
    axes[2].set_yscale("symlog", linthresh=1.0)
    axes[2].set_xlabel("hops in the planned route (exact model)")
    axes[2].set_ylabel("predicted - true value")
    axes[2].legend()
    fig.savefig(out / "dropped_term.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=OUT_DIR, help="folder with study_*.csv")
    args = parser.parse_args()
    read = {
        name: pd.read_csv(args.dir / f"study_{name}.csv")
        for name in ("noise", "guards", "rank", "wormholes", "reward")
    }
    plot_success_vs_distance(read["noise"], args.dir)
    plot_far_success_vs_noise(read["noise"], args.dir)
    plot_guards(read["guards"], args.dir)
    plot_low_rank(read["rank"], read["wormholes"], args.dir)
    plot_calibration(args.dir)
    plot_dropped_term(read["reward"], args.dir)
    print(f"figures written to {args.dir}")


if __name__ == "__main__":
    main()
