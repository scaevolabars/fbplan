"""Markdown result tables from evaluation CSVs.

    uv run python analysis/tables.py results/medium/*.csv results/medium/planner*/*.csv \
        [--label-root results/medium] [--reference baseline] [--cluster config_hash]

Per env: success (%) per task and overall with the 95 % stratified bootstrap CI, and the paired
comparison of every method against each reference method (default: baseline and baseline_raw,
docs/design.md §8) on the same episodes. Runs whose resets are not seeded (UNPAIRED: upstream's
own evaluation loop) have no episode pairs; they get the unpaired comparison instead.
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

from fbplan.controllers import PRIVILEGED
from fbplan.stats import (
    EPISODE_KEY,
    labelled_results,
    markdown_table,
    paired_table,
    pool_label,
    summary_table,
    unpaired_comparison,
)

UNPAIRED = {"upstream"}  # controller of analysis/parity.py: resets not seeded, no episode pairs


def success_markdown(table: pd.DataFrame, privileged: set[str]) -> str:
    """Methods as rows, tasks as columns, cells "mean [low, high]" in percent.

    Rows whose label is in `privileged` get the mark "(privileged)".
    """
    cells = table.assign(
        cell=[
            f"{100 * r.success:.1f} [{100 * r.ci_low:.1f}, {100 * r.ci_high:.1f}]"
            for r in table.itertuples()
        ]
    )
    wide = cells.pivot(index="method", columns="task", values="cell")
    wide = wide.reindex(index=table["method"].unique()).reset_index()
    wide["method"] = [f"{m} (privileged)" if m in privileged else m for m in wide["method"]]
    return markdown_table(wide.rename(columns={t: f"task {t}" for t in wide.columns[1:-1]}))


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
    return markdown_table(table[columns])


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
    return markdown_table(rows)


def unpaired_markdown(df: pd.DataFrame, methods: list[str], reference: str) -> str:
    """fbplan.stats.unpaired_comparison of each method against the reference."""
    rows = []
    for method in methods:
        est = unpaired_comparison(df, method, reference)
        diff = f"{100 * est.mean:+.1f} [{100 * est.low:+.1f}, {100 * est.high:+.1f}]"
        rows.append((method, diff, est.n))
    return markdown_table(pd.DataFrame(rows, columns=["method", "diff (pp)", "episodes"]))


def same_episodes(df: pd.DataFrame, reference: str) -> set[str]:
    """Labels whose episodes (EPISODE_KEY) are exactly the reference's, the reference included."""
    keys = df.groupby("method")[EPISODE_KEY].apply(lambda g: frozenset(map(tuple, g.values)))
    return set(keys.index[keys == keys[reference]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument(
        "--reference", nargs="+", default=["baseline", "baseline_raw"],
        help="methods for paired comparisons; one table per reference present",
    )
    parser.add_argument("--cluster", help="column of a shared random factor, e.g. config_hash")
    parser.add_argument("--label-root", type=Path, help="name rows by CSV path below this folder")
    args = parser.parse_args()

    df = labelled_results(args.csv, label_root=args.label_root)
    for env_name, env_df in df.groupby("env_name"):
        print_env_tables(env_name, env_df, args.reference, args.cluster)


def print_env_tables(
    env_name: str, df: pd.DataFrame, references: list[str], cluster: str | None
) -> None:
    """Success table, then paired (and unpaired) comparisons against each reference."""
    shown = df.assign(method=df["method"].map(pool_label)) if cluster else df
    kind = f"cluster bootstrap over {cluster}" if cluster else "stratified bootstrap"
    print(f"## {env_name}\n\nSuccess (%), 95 % {kind} CI\n")
    privileged = set(shown["method"][shown["controller"].isin(PRIVILEGED)])
    print(success_markdown(summary_table(shown, cluster=cluster), privileged))
    unpaired = df["controller"].isin(UNPAIRED)
    paired_df = df[~unpaired]
    for reference in references:
        if reference not in set(paired_df["method"]):
            print(f"\nNo comparison against {reference}: no seeded run has that label.")
            continue
        pairs_with_reference = same_episodes(paired_df, reference)
        others = sorted(set(paired_df["method"]) - pairs_with_reference)
        if others:
            print(f"\nNot paired with {reference} (other episodes): {', '.join(others)}")
        if len(pairs_with_reference) > 1:
            print(f"\nPaired difference against {reference} (pp = percentage points)\n")
            paired = paired_table(paired_df[paired_df["method"].isin(pairs_with_reference)],
                                  reference)
            print(paired_markdown(paired))
            if cluster and shown["method"].nunique() < df["method"].nunique():
                print("\nPooled methods: mean and range of the paired differences\n")
                print(pooled_paired_markdown(paired))
        if unpaired.any():
            print(f"\nUnpaired difference against {reference} (resets not seeded)\n")
            print(unpaired_markdown(df, sorted(set(df["method"][unpaired])), reference))
    print()


if __name__ == "__main__":
    main()
