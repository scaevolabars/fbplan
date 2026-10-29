"""Offline check of a trained V_H against the maze geodesic, no rollouts (approach-b.md §7).

For n validation states s and n goal states g (fixed seed) it compares V_1(s, z_g) and
V_H(s, z_g) with the geodesic distance from s to g. A value that serves goal-reaching orders the
states by distance to the goal, so the main number is the rank correlation of the value with
distance (more negative is better), over all pairs and over the far half. Chaining should help
most at long range, where FB values are noisy.

    uv run python -m fbplan.learned.diagnostics --env antmaze-medium-navigate-v0 \\
        --value value-single --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fbplan import frozen  # noqa: E402
from fbplan.contracts import normalize_z  # noqa: E402
from fbplan.controllers.oracle import MazeGeometry  # noqa: E402
from fbplan.learned.option_targets import direct_value  # noqa: E402
from fbplan.learned.train_b import Split, approach_b_dir, load_value  # noqa: E402
from fbplan.learned.value import Agent, ValueState, forward  # noqa: E402

BIN_WIDTH = 2.0  # xy units


@jax.jit
def pair_values(
    agent: Agent, value: ValueState, obs: jax.Array, z: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """(V_1, V_H) for each (obs_i, z_i). [N] each."""
    v1 = direct_value(forward(agent, obs, z), z)
    return v1, v1 + value.gain(obs, z).min(axis=0)


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation (no ties expected: values and distances are continuous)."""
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def value_grid(
    agent: Agent, value: ValueState, s_obs: np.ndarray, z_goals: np.ndarray, chunk: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    """V_1 and V_H for every (state, goal) pair: [n_s, n_g] each."""
    n_s, n_g = len(s_obs), len(z_goals)
    s_idx, g_idx = np.divmod(np.arange(n_s * n_g), n_g)
    parts = [
        pair_values(agent, value, s_obs[s_idx[i : i + chunk]], z_goals[g_idx[i : i + chunk]])
        for i in range(0, len(s_idx), chunk)
    ]
    v1, vh = (np.concatenate([np.asarray(p[k]) for p in parts]) for k in (0, 1))
    return v1.reshape(n_s, n_g), vh.reshape(n_s, n_g)


def summary_rows(geodesic: np.ndarray, v1: np.ndarray, vh: np.ndarray) -> list[str]:
    """Markdown table: per distance bin, the median values and the share of pairs with a gain."""
    rows = [
        "| geodesic | pairs | median V_1 | median V_H | V_H > V_1 + 1 |",
        "|---|---|---|---|---|",
    ]
    for lo in np.arange(0.0, geodesic.max(), BIN_WIDTH * 2):
        m = (geodesic >= lo) & (geodesic < lo + BIN_WIDTH * 2)
        if m.sum() < 20:
            continue
        rows.append(
            f"| {lo:.0f}-{lo + 2 * BIN_WIDTH:.0f} | {m.sum()} | {np.median(v1[m]):.1f} | "
            f"{np.median(vh[m]):.1f} | {np.mean(vh[m] > v1[m] + 1):.2f} |"
        )
    return rows


def plot(geodesic: np.ndarray, v1: np.ndarray, vh: np.ndarray, title: str, path: Path) -> None:
    edges = np.arange(0.0, geodesic.max() + BIN_WIDTH, BIN_WIDTH)
    centres = edges[:-1] + BIN_WIDTH / 2
    fig, ax = plt.subplots(figsize=(7, 4))
    for values, label, color in ((v1, "V_1 (never switch)", "#888888"), (vh, "V_H", "#1f6fb2")):
        q = np.array(
            [
                np.percentile(values[(geodesic >= a) & (geodesic < b)], [25, 50, 75])
                if ((geodesic >= a) & (geodesic < b)).sum() >= 20
                else [np.nan] * 3
                for a, b in zip(edges[:-1], edges[1:], strict=True)
            ]
        )
        ax.plot(centres, q[:, 1], color=color, label=label)
        ax.fill_between(centres, q[:, 0], q[:, 2], color=color, alpha=0.2, lw=0)
    ax.set_xlabel("maze geodesic distance s -> g (xy units)")
    ax.set_ylabel("value (median, 25-75 %)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", default="antmaze-medium-navigate-v0")
    parser.add_argument("--value", default="value-single", choices=["value-single", "value-region"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--n", type=int, default=300, help="states and goals sampled each")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/train_b/diagnostics"))
    args = parser.parse_args()

    model = frozen.load_model(args.env)
    split = Split.load(model, "val", args.value.removeprefix("value-"))
    obs = np.asarray(split.data.observations)
    rng = np.random.default_rng(0)
    s_idx, g_idx = (rng.choice(len(obs), args.n, replace=False) for _ in range(2))
    z_goals = np.asarray(normalize_z(split.latents[g_idx], model.latent_dim))
    geodesic = MazeGeometry.from_env(frozen.make_env(args.env)).geodesic_distances(
        obs[s_idx, :2], obs[g_idx, :2]
    )
    finite = np.isfinite(geodesic)
    far = finite & (geodesic > np.median(geodesic[finite]))

    print(f"{args.env}, {args.value}, {args.n} x {args.n} validation pairs")
    print("| seed | rank corr V_1 vs dist | V_H | far half: V_1 | V_H |")
    print("|---|---|---|---|---|")
    maze = args.env.split("-")[1]
    for seed in args.seeds:
        value_dir = approach_b_dir(args.env, args.value, seed)
        value, _ = load_value(value_dir, obs.shape[1], model.latent_dim)
        v1, vh = value_grid(model.agent, value, obs[s_idx], z_goals)
        corr = [rank_correlation(v[m], geodesic[m]) for m in (finite, far) for v in (v1, vh)]
        print(f"| {seed} | {corr[0]:.3f} | {corr[1]:.3f} | {corr[2]:.3f} | {corr[3]:.3f} |")
        if seed == args.seeds[0]:
            table = summary_rows(geodesic[finite], v1[finite], vh[finite])
            title = f"{maze} {args.value} seed {seed}: value vs. distance to the goal"
            plot(
                geodesic[finite],
                v1[finite],
                vh[finite],
                title,
                args.out_dir / f"{maze}-{args.value}-seed{seed}.png",
            )
    print("\n" + "\n".join(table))


if __name__ == "__main__":
    main()
