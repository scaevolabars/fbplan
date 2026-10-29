"""Report figures that the other analysis scripts do not make.

    uv run python analysis/report_figures.py [--results results/medium] \
        [--out-dir docs/figures/report]

time_to_goal_<env>.png: success within a step budget for each planner variant, minus the
baseline's, over budgets 0..1000 (landmark sets pooled). The env ends an episode at the goal, so
the step count of a success is its time to goal. A gap that shrinks as the budget grows means
that the method reaches the goal later, not that it never reaches it.

The printed tables give the numbers that REPORT.md quotes: median steps of the successes,
success within 400, 600, 800 and 1000 steps, how far from the goal the failures end, and the
largest gap of each planner variant to the baseline. All numbers come from the evaluation
CSVs in --results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fbplan.stats import STRATA, load_results, pool_label  # noqa: E402

# Methods with paired start states (upstream and the eval-seed-0 oracle sweep are left out).
METHODS = ["baseline", "baseline_raw", "nonhier", "oracle", "planner", "planner_high",
           "planner_region"]
REFERENCE = "baseline"
# Colors of analysis/plots.py, so a method has one color in every figure. The baseline is the
# zero line here: blue next to the planner's purple fails the color-vision check.
GAP_COLORS = {"planner_high": "#e87ba4", "planner": "#6250d6", "planner_region": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BUDGETS = np.arange(0, 1001, 10)
TABLE_BUDGETS = (400, 600, 800, 1000)
NEAR_GOAL_XY = 3.0  # a failure that ends this close circled the goal rather than got lost


def load_methods(results: Path) -> pd.DataFrame:
    """Episodes of METHODS, one CSV per method or a folder of CSVs per method (landmark sets)."""
    paths = []
    for method in METHODS:
        single = results / f"{method}.csv"
        paths += [single] if single.exists() else sorted((results / method).glob("*.csv"))
    df = load_results(paths, label_root=results)
    return df.assign(method=df["method"].map(pool_label))


def success_within(df: pd.DataFrame, budgets: np.ndarray) -> pd.DataFrame:
    """Success (%) within each step budget: mean over tasks of the per-task rate, as OGBench.
    Rows are methods in METHODS order, columns are budgets."""
    reached = df["success"].to_numpy()[:, None] & (df["steps"].to_numpy()[:, None] <= budgets)
    per_episode = pd.DataFrame(reached, columns=budgets, index=df.index)
    keys = [df["method"], *(df[column] for column in STRATA)]
    per_task = per_episode.groupby(keys).mean()
    curves = 100 * per_task.groupby(level=0).mean()
    return curves.reindex([m for m in METHODS if m in curves.index])


def summary(df: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    """One row per method with the numbers the report quotes."""
    rows = []
    for method, episodes in df.groupby("method"):
        successes, failures = episodes[episodes["success"]], episodes[~episodes["success"]]
        row = {
            "method": method,
            "median steps of successes": int(successes["steps"].median()),
            **{f"success within {b} (%)": round(curves.loc[method, b], 1) for b in TABLE_BUDGETS},
            "failures": len(failures),
            "median final xy distance of failures": round(failures["final_goal_dist"].median(), 1),
            f"failures within {NEAR_GOAL_XY:g} xy (%)": round(
                100 * (failures["final_goal_dist"] < NEAR_GOAL_XY).mean(), 1
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows).set_index("method").reindex(curves.index).reset_index()


def gap_summary(curves: pd.DataFrame) -> pd.DataFrame:
    """Per planner variant: the largest gap to the baseline, its budget, and the gap at the end."""
    gaps = curves.loc[list(GAP_COLORS)] - curves.loc[REFERENCE]
    return pd.DataFrame({
        "method": gaps.index,
        "largest gap (pp)": gaps.min(axis=1).round(1).to_numpy(),
        "at budget": gaps.idxmin(axis=1).to_numpy(),
        f"gap at {BUDGETS[-1]} (pp)": gaps[BUDGETS[-1]].round(1).to_numpy(),
    })


def markdown(df: pd.DataFrame) -> str:
    """A GitHub Markdown table."""
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def plot_gaps(curves: pd.DataFrame, env_name: str, path: Path) -> None:
    """Success within the budget minus the baseline's, one line per planner variant."""
    gaps = curves.loc[list(GAP_COLORS)] - curves.loc[REFERENCE]
    fig, ax = plt.subplots(figsize=(7.0, 3.8), dpi=150)
    ax.axhline(0.0, color=INK, lw=1.0)
    ax.text(BUDGETS[-1] + 15, 0.0, REFERENCE, color=INK, fontsize=8, va="center")
    for method, color in GAP_COLORS.items():
        ax.plot(BUDGETS, gaps.loc[method], color=color, lw=2.0, solid_capstyle="round",
                label=method)
        end = gaps.loc[method].iloc[-1]
        ax.plot(BUDGETS[-1], end, "o", ms=6, color=color, mec="white", mew=1.5, clip_on=False)
        label = f"{method} {end:+.1f}".replace("-", "\u2212")  # the minus sign of the ticks
        ax.text(BUDGETS[-1] + 15, end, label, color=INK, fontsize=8, va="center")
    ax.set_xlim(0, BUDGETS[-1])
    ax.set_xlabel("step budget (an episode counts as a success if it reaches the goal by then)",
                  color=MUTED, fontsize=8)
    ax.set_ylabel("success minus the baseline (pp)", color=MUTED, fontsize=8)
    ax.set_title(f"{env_name}: success within a step budget, minus the baseline",
                 loc="left", fontsize=9, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results/medium"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/figures/report"))
    args = parser.parse_args()

    df = load_methods(args.results)
    (env_name,) = df["env_name"].unique()
    curves = success_within(df, BUDGETS)
    plot_gaps(curves, env_name, args.out_dir / f"time_to_goal_{env_name}.png")
    print(f"## {env_name}\n")
    print(markdown(summary(df, curves)))
    print(f"\nSuccess within the budget minus the {REFERENCE}'s\n")
    print(markdown(gap_summary(curves)))


if __name__ == "__main__":
    main()
