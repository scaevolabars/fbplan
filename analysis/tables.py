"""Markdown result tables from evaluation CSVs.

    uv run python analysis/tables.py results/*.csv [--reference baseline] [--cluster COLUMN]

Per env: success (%) per task and overall with the 95 % stratified bootstrap CI, and the paired
comparison of every method against the reference method on the same episodes.
A method that appears with several config hashes gets one row per config ("name@hash"),
unless `--cluster config_hash` asks to pool them as checkpoints of one method.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fbplan.stats import load_results, paired_table, summary_table

PRIVILEGED = {"oracle"}  # methods that read privileged information


def markdown(df: pd.DataFrame) -> str:
    """A GitHub Markdown table."""
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def label_configs(df: pd.DataFrame) -> pd.Series:
    """Method name, plus "@config_hash" for methods that have more than one config."""
    n_configs = df.groupby("method")["config_hash"].transform("nunique")
    return df["method"].where(n_configs == 1, df["method"] + "@" + df["config_hash"])


def success_markdown(table: pd.DataFrame) -> str:
    """Methods as rows, tasks as columns, cells "mean [low, high]" in percent."""
    cells = table.assign(
        cell=[
            f"{100 * r.success:.1f} [{100 * r.ci_low:.1f}, {100 * r.ci_high:.1f}]"
            for r in table.itertuples()
        ]
    )
    wide = cells.pivot(index="method", columns="task", values="cell")
    wide = wide.reindex(index=table["method"].unique()).reset_index()
    wide["method"] = [f"{m} (privileged)" if m in PRIVILEGED else m for m in wide["method"]]
    return markdown(wide.rename(columns={t: f"task {t}" for t in wide.columns[1:-1]}))


def paired_markdown(df: pd.DataFrame, reference: str) -> str:
    table = paired_table(df, reference)
    table["diff (pp)"] = [
        f"{100 * r.diff:+.1f} [{100 * r.ci_low:+.1f}, {100 * r.ci_high:+.1f}]"
        for r in table.itertuples()
    ]
    table["p (McNemar)"] = table["p_value"].map(lambda p: f"{p:.2g}")
    columns = ["method", "diff (pp)", "only_method", "only_reference", "p (McNemar)", "n_pairs"]
    return markdown(table[columns])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--reference", default="baseline", help="method for paired comparisons")
    parser.add_argument("--cluster", help="column of a shared random factor, e.g. config_hash")
    args = parser.parse_args()

    df = load_results(args.csv)
    if args.cluster != "config_hash":
        df = df.assign(method=label_configs(df))
    for env_name, env_df in df.groupby("env_name"):
        print(f"## {env_name}\n\nSuccess (%), 95 % stratified bootstrap CI\n")
        print(success_markdown(summary_table(env_df, cluster=args.cluster)))
        if args.reference in set(env_df["method"]) and env_df["method"].nunique() > 1:
            print(f"\nPaired difference against {args.reference} (pp = percentage points)\n")
            print(paired_markdown(env_df, args.reference))
        print()


if __name__ == "__main__":
    main()
