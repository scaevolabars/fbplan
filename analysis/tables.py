"""Markdown result tables from evaluation CSVs.

    uv run python analysis/tables.py results/medium/*.csv results/medium/planner*/*.csv \
        [--label-root results/medium] [--reference baseline] [--cluster config_hash]

Per env: success (%) per task and overall with the 95 % stratified bootstrap CI, and the paired
comparison of every method against the reference method on the same episodes.
Rows are named by fbplan.stats.row_labels: the method, the trained pi_h and its training seed
("baseline[retrained]/seed3"), and "@hash" for a method with several configs. With
--label-root, rows are named by CSV path instead ("planner/l0").
--cluster pools the rows of one label up to "/" (or "@") as draws of a random factor:
`--cluster checkpoint_seed` for training seeds, `--cluster config_hash` for landmark sets in
folders. Paired comparisons stay per row label; for a pooled method the table adds the mean and
the range of its per-label differences.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fbplan.stats import load_results, paired_table, pool_label, row_labels, summary_table

PRIVILEGED = {"oracle"}  # methods that read privileged information


def markdown(df: pd.DataFrame) -> str:
    """A GitHub Markdown table."""
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


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


def paired_markdown(table: pd.DataFrame) -> str:
    """The output of fbplan.stats.paired_table, differences in percentage points."""
    table = table.assign(
        **{
            "diff (pp)": [
                f"{100 * r.diff:+.1f} [{100 * r.ci_low:+.1f}, {100 * r.ci_high:+.1f}]"
                for r in table.itertuples()
            ],
            "p (McNemar)": table["p_value"].map(lambda p: f"{p:.2g}"),
        }
    )
    columns = ["method", "diff (pp)", "only_method", "only_reference", "p (McNemar)", "n_pairs"]
    return markdown(table[columns])


def pooled_paired_markdown(table: pd.DataFrame) -> str:
    """Mean and range of the per-config paired differences, for methods with several configs."""
    diffs = table.groupby(table["method"].map(pool_label), sort=False)["diff"].agg(
        ["size", "mean", "min", "max"]
    )
    diffs = diffs[diffs["size"] > 1]
    rows = pd.DataFrame({
        "method": diffs.index,
        "configs": diffs["size"].to_numpy(),
        "mean diff (pp)": [f"{100 * d:+.1f}" for d in diffs["mean"]],
        "range (pp)": [
            f"[{100 * lo:+.1f}, {100 * hi:+.1f}]"
            for lo, hi in zip(diffs["min"], diffs["max"], strict=True)
        ],
    })
    return markdown(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--reference", default="baseline", help="method for paired comparisons")
    parser.add_argument("--cluster", help="column of a shared random factor, e.g. config_hash")
    parser.add_argument("--label-root", type=Path, help="name rows by CSV path below this folder")
    args = parser.parse_args()

    df = load_results(args.csv, label_root=args.label_root)
    if args.label_root is None:
        df = df.assign(method=row_labels(df))
    pooled = args.cluster is not None
    for env_name, env_df in df.groupby("env_name"):
        shown = env_df.assign(method=env_df["method"].map(pool_label)) if pooled else env_df
        kind = f"cluster bootstrap over {args.cluster}" if args.cluster else "stratified bootstrap"
        print(f"## {env_name}\n\nSuccess (%), 95 % {kind} CI\n")
        print(success_markdown(summary_table(shown, cluster=args.cluster)))
        if args.reference in set(env_df["method"]) and env_df["method"].nunique() > 1:
            print(f"\nPaired difference against {args.reference} (pp = percentage points)\n")
            paired = paired_table(env_df, args.reference)
            print(paired_markdown(paired))
            if pooled and shown["method"].nunique() < env_df["method"].nunique():
                print("\nPooled methods: mean and range of the paired differences\n")
                print(pooled_paired_markdown(paired))
        print()


if __name__ == "__main__":
    main()
