"""Reproducibility check: the same evaluation runs on the Mac and on the Linux instance.

    uv run python analysis/platform_check.py [--linux results] [--macos results/macos]
        [--out results/platform_check.csv]

For every CSV below --macos that also exists at the same path below --linux: success on each
platform, the unpaired difference Linux minus Mac with its 95 % stratified bootstrap CI (float
results differ between the platforms, so the episodes do not pair), and the share of episodes
with the same outcome and with the same step count. Final numbers come from Linux only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fbplan.stats import EPISODE_KEY, load_results, markdown_table, success_ci, unpaired_comparison


def compare_platforms(linux: pd.DataFrame, macos: pd.DataFrame) -> dict[str, float]:
    """Successes, unpaired difference and episode agreement of one config on both platforms."""
    both = pd.concat([linux.assign(method="linux"), macos.assign(method="macos")])
    diff = unpaired_comparison(both, "linux", "macos")
    joined = linux.merge(macos, on=EPISODE_KEY, suffixes=("_linux", "_macos"))
    return {
        "linux": success_ci(linux).mean,
        "macos": success_ci(macos).mean,
        "diff": diff.mean,
        "ci_low": diff.low,
        "ci_high": diff.high,
        "episodes": len(joined),
        "same_outcome": float((joined["success_linux"] == joined["success_macos"]).mean()),
        "same_steps": float((joined["steps_linux"] == joined["steps_macos"]).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--linux", type=Path, default=Path("results"))
    parser.add_argument("--macos", type=Path, default=Path("results/macos"))
    parser.add_argument("--out", type=Path, default=Path("results/platform_check.csv"))
    args = parser.parse_args()

    rows = []
    for mac_csv in sorted(args.macos.rglob("*.csv")):
        rel = mac_csv.relative_to(args.macos)
        linux_csv = args.linux / rel
        if "failures" in rel.parts or not linux_csv.exists():
            continue
        row = compare_platforms(load_results([linux_csv]), load_results([mac_csv]))
        rows.append({"run": rel.with_suffix("").as_posix(), **row})
    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    shown = table.assign(**{
        c: (100 * table[c]).round(1) for c in ["linux", "macos", "diff", "ci_low", "ci_high"]
    }).round({"same_outcome": 3, "same_steps": 3})
    print(markdown_table(shown))
    print(f"\n{len(table)} runs; written to {args.out}")


if __name__ == "__main__":
    main()
