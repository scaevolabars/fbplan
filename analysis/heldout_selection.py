"""Seed selection on held-out goals (docs/design.md §8, plan B decision 8).

    uv run python analysis/heldout_selection.py [--results results/medium]
        [--reference retrained] [--out results/medium/heldout_selection.csv]

For each family with several training seeds (retrained baseline, B clean, B wide, B region,
the 2x families, T2's log-ratio baseline):
the held-out success of every seed (results/medium/heldout/<family>/seed<k>.csv, 10 held-out
goals, eval seeds 0 1 2, 20 episodes), the selected seed (highest held-out success, ties to the
lower seed id), and on the 5 eval tasks the selected seed paired against the selected seed of the
reference family (default: the retrained baseline; `retrained_awr2x` for the 2x sharpness
experiment of design.md §8) and against baseline_raw. Families without 5 held-out CSVs are
skipped. Nothing else is tuned on the held-out goals; the pooled rows stay in analysis/tables.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fbplan.stats import (
    labelled_results,
    load_results,
    markdown_table,
    paired_comparison,
    success_ci,
)

# family -> folder of its eval-task CSVs below the results folder
FAMILIES = {
    "retrained": "baseline_retrained",
    "b_clean": "b_clean",
    "b_wide": "b_wide",
    "b_region": "b_region",
    "retrained_awr2x": "baseline_retrained_awr2x",
    "b_clean_awr2x": "b_clean_awr2x",
    "retrained_logratio": "baseline_retrained_logratio",
}
SEEDS = range(5)


def select_seed(heldout_success: dict[int, float]) -> int:
    """The seed with the highest held-out success; ties go to the lower seed id."""
    return min(heldout_success, key=lambda seed: (-heldout_success[seed], seed))


def _diff(df: pd.DataFrame, method: str, reference: str) -> str:
    r = paired_comparison(df, method, reference)
    return f"{100 * r.diff:+.1f} [{100 * r.low:+.1f}, {100 * r.high:+.1f}], p {r.p_value:.2g}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results", type=Path, default=Path("results/medium"),
        help="folder with heldout/<family>/seed<k>.csv and the eval-task CSVs",
    )
    parser.add_argument(
        "--reference", default="retrained", choices=sorted(FAMILIES),
        help="family whose selected seed is the reference of the paired comparisons",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/medium/heldout_selection.csv"),
        help="CSV of the selection table",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    selected, rows = {}, []
    for family, folder in FAMILIES.items():
        paths = [args.results / "heldout" / family / f"seed{k}.csv" for k in SEEDS]
        if not all(p.exists() for p in paths):
            continue
        heldout = {k: success_ci(load_results([p])).mean for k, p in zip(SEEDS, paths, strict=True)}
        selected[family] = f"{folder}/seed{select_seed(heldout)}"
        rows.append({"family": family, **{f"heldout_seed{k}": v for k, v in heldout.items()},
                     "selected_seed": select_seed(heldout)})
    if args.reference not in selected:
        raise SystemExit(f"the held-out CSVs of {args.reference} are missing")

    eval_paths = [args.results / "baseline_raw.csv"]
    eval_paths += [args.results / f"{label}.csv" for label in selected.values()]
    df = labelled_results(eval_paths, label_root=args.results)
    reference = selected[args.reference]
    for row in rows:
        label = selected[row["family"]]
        row["eval_success"] = success_ci(df[df["method"] == label]).mean
        row[f"vs_selected_{args.reference}"] = (
            "–" if label == reference else _diff(df, label, reference)
        )
        row["vs_baseline_raw"] = _diff(df, label, "baseline_raw")
    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    shown = table.copy()
    for column in [c for c in table.columns if c.startswith("heldout_")] + ["eval_success"]:
        shown[column] = (100 * table[column]).round(1)
    print(markdown_table(shown))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
