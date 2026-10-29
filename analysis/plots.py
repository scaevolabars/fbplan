"""Success-rate figure per env from evaluation CSVs.

    uv run python analysis/plots.py results/medium/*.csv results/medium/planner*/*.csv \
        [--label-root results/medium] [--cluster config_hash] [--out-dir results/figures]

One PNG per env: success (%) per task and overall, one dot per method with its 95 %
stratified bootstrap CI as a whisker. Each method keeps the same color in every figure.
--label-root and --cluster name and pool rows as in analysis/tables.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from fbplan.stats import load_results, pool_label, summary_table  # noqa: E402

# Fixed color per method (reference categorical order). The privileged oracle is neutral gray.
METHOD_COLORS = {"baseline": "#2a78d6", "nonhier": "#eb6834", "oracle": "#7a7974"}
OTHER_COLORS = ["#1baf7a", "#6250d6", "#e87ba4", "#008300", "#e34948", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def method_colors(methods: list[str]) -> dict[str, str]:
    """Known methods get their fixed color; others take the remaining colors in name order."""
    others = sorted(m for m in methods if m not in METHOD_COLORS)
    return {**METHOD_COLORS, **dict(zip(others, OTHER_COLORS, strict=False))}


def plot_env(table: pd.DataFrame, env_name: str, path: Path) -> None:
    methods = list(table["method"].unique())
    colors = method_colors(methods)
    tasks = list(table["task"].unique())
    width = 0.7 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=150)
    for k, method in enumerate(methods):
        rows = table[table["method"] == method].set_index("task").reindex(tasks)
        x = [i - 0.35 + width * (k + 0.5) for i in range(len(tasks))]
        y = 100 * rows["success"]
        err = [y - 100 * rows["ci_low"], 100 * rows["ci_high"] - y]
        label = f"{method} (privileged)" if method == "oracle" else method
        ax.errorbar(x, y, yerr=err, fmt="o", ms=6, lw=2, capsize=0, color=colors[method],
                    label=label)
    ax.set_xticks(range(len(tasks)), [f"task {t}" if t != "all" else "overall" for t in tasks])
    ax.axvline(len(tasks) - 1.5, color=GRID, lw=1)
    ax.set_ylim(-2, 102)
    ax.set_ylabel("success (%)", color=MUTED)
    ax.set_title(f"{env_name}: success with 95 % CI", color=INK, fontsize=10, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    # Below the axis, so it never hides dots near 0 %.
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1),
              ncols=len(methods))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--label-root", type=Path, help="name rows by CSV path below this folder")
    parser.add_argument("--cluster", help="column of a shared random factor, e.g. config_hash")
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    df = load_results(args.csv, label_root=args.label_root)
    if args.cluster == "config_hash":
        df = df.assign(method=df["method"].map(pool_label))
    for env_name, env_df in df.groupby("env_name"):
        path = args.out_dir / f"success_{env_name}.png"
        plot_env(summary_table(env_df, cluster=args.cluster), env_name, path)
        print(path)


if __name__ == "__main__":
    main()
