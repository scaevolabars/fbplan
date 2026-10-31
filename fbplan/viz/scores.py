"""The scoreboard: paired differences in success, with 95 % CIs, from the committed result CSVs.

Every row compares two sets of CSVs on the same episodes with fbplan.stats, as
analysis/tables.py does for REPORT.md: `paired_comparison` for two single runs,
`pooled_paired_comparison` when one side pools several runs (landmark sets of approach A by
config_hash, training seeds of approach B by checkpoint_seed). The rows and their references
are those of REPORT §1, §7.1 and §8.7.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from fbplan.stats import load_results, paired_comparison, pooled_paired_comparison, success_ci

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Row:
    """One comparison: success of `method` minus success of `reference`, on the same episodes.

    group: "route" (the recorded route against the stronger baseline row), "fb" (an order from
    FB values against the same row) or "weight" (the held-out check of the edge weight).
    cluster: the column that pools the runs of `method`, or None for one run.
    """

    key: str
    group: str
    maze: str
    label: str
    episodes: str
    method: tuple[str, ...]
    reference: tuple[str, ...]
    method_name: str
    reference_name: str
    cluster: str | None = None


@dataclass(frozen=True)
class Score:
    """A Row with its result in percentage points; p is None for a pooled comparison."""

    key: str
    group: str
    maze: str
    label: str
    episodes: str
    method_name: str
    reference_name: str
    diff: float
    low: float
    high: float
    p: float | None
    only_method: int | None
    only_reference: int | None
    n_pairs: int
    method_success: float
    reference_success: float
    sources: tuple[str, ...]


M, L = "results/medium", "results/large"
M10, L10, M20, L20 = (
    "results/medium_10seeds", "results/large_10seeds",
    "results/medium_seeds10-19", "results/large_seeds10-19",
)
H1_M = (f"{M}/planner_ablations/h1_eval/window_k150.csv",)
H1_M20 = (f"{M20}/h1_window_k150.csv",)
H1_L = (f"{L}/planner_ablations/h1_eval/high_k150.csv",)
H1_L20 = (f"{L20}/h1_high_k150.csv",)
RAW_M = (f"{M10}/baseline_raw.csv",)
RAW_M20 = (f"{M20}/baseline_raw.csv",)
BASE_L = (f"{L10}/baseline.csv",)
BASE_L20 = (f"{L20}/baseline.csv",)
HELDOUT_M, HELDOUT_L = f"{M}/planner_ablations", f"{L}/planner_ablations"
ROUTE, FB = "recorded route", "FB values"


def _seeds(folder: str) -> tuple[str, ...]:
    return tuple(f"{folder}/seed{k}.csv" for k in range(5))


def _sets(folder: str) -> tuple[str, ...]:
    return tuple(f"{folder}/l{k}.csv" for k in range(3))


ROWS: tuple[Row, ...] = (
    # The recorded route against the stronger baseline row of each maze (REPORT §8.7).
    Row("route_medium_0_9", "route", "medium", "eval seeds 0–9", "eval seeds 0–9",
        H1_M, RAW_M, ROUTE, "baseline_raw"),
    Row("route_medium_10_19", "route", "medium", "eval seeds 10–19, replication",
        "eval seeds 10–19", H1_M20, RAW_M20, ROUTE, "baseline_raw"),
    Row("route_medium_0_19", "route", "medium", "eval seeds 0–19, all", "eval seeds 0–19",
        H1_M + H1_M20, RAW_M + RAW_M20, ROUTE, "baseline_raw"),
    Row("route_large_0_9", "route", "large", "eval seeds 0–9", "eval seeds 0–9",
        H1_L, BASE_L, ROUTE, "baseline"),
    Row("route_large_10_19", "route", "large", "eval seeds 10–19, replication",
        "eval seeds 10–19", H1_L20, BASE_L20, ROUTE, "baseline"),
    Row("route_large_0_19", "route", "large", "eval seeds 0–19, all", "eval seeds 0–19",
        H1_L + H1_L20, BASE_L + BASE_L20, ROUTE, "baseline"),
    # Orders from FB values on medium, against the same row (REPORT §7.1).
    Row("a_planner_high", "fb", "medium", "A, planner_high", "eval seeds 0–4",
        _sets(f"{M}/planner_high"), (f"{M}/baseline_raw.csv",), "approach A", "baseline_raw",
        "config_hash"),
    Row("a_planner", "fb", "medium", "A, planner", "eval seeds 0–4",
        _sets(f"{M}/planner"), (f"{M}/baseline_raw.csv",), "approach A", "baseline_raw",
        "config_hash"),
    Row("a_region", "fb", "medium", "A, region latents", "eval seeds 0–4",
        _sets(f"{M}/planner_region"), (f"{M}/baseline_raw.csv",), "approach A", "baseline_raw",
        "config_hash"),
    Row("b_clean", "fb", "medium", "B-clean", "eval seeds 0–9",
        _seeds(f"{M10}/b_clean"), RAW_M, "approach B", "baseline_raw", "checkpoint_seed"),
    Row("b_wide", "fb", "medium", "B-wide", "eval seeds 0–9",
        _seeds(f"{M10}/b_wide"), RAW_M, "approach B", "baseline_raw", "checkpoint_seed"),
    Row("b_region", "fb", "medium", "B-region", "eval seeds 0–9",
        _seeds(f"{M10}/b_region"), RAW_M, "approach B", "baseline_raw", "checkpoint_seed"),
    # The held-out check of H2: the same certified edges, xy lengths or FB costs (REPORT §8.7).
    Row("weight_medium_xy", "weight", "medium", "xy length (H1)", "held-out goals",
        (f"{HELDOUT_M}/h1_heldout/window_k150.csv",), (f"{HELDOUT_M}/h1_heldout/baseline.csv",),
        ROUTE, "baseline"),
    Row("weight_medium_fb", "weight", "medium", "FB cost (H2)", "held-out goals",
        (f"{HELDOUT_M}/h2_heldout/window_k150.csv",), (f"{HELDOUT_M}/h1_heldout/baseline.csv",),
        "recorded route, FB costs", "baseline"),
    Row("weight_large_xy", "weight", "large", "xy length (H1)", "held-out goals",
        (f"{HELDOUT_L}/h1_heldout/high_k150.csv",), (f"{HELDOUT_L}/h1_heldout/baseline.csv",),
        ROUTE, "baseline"),
    Row("weight_large_fb", "weight", "large", "FB cost (H2)", "held-out goals",
        (f"{HELDOUT_L}/h2_heldout/high_k150.csv",), (f"{HELDOUT_L}/h1_heldout/baseline.csv",),
        "recorded route, FB costs", "baseline"),
)

# H2 minus H1 on the same held-out episodes: the effect of the weight alone.
WEIGHT_EFFECT: tuple[Row, ...] = (
    Row("weight_medium_fb_minus_xy", "weight_effect", "medium", "FB cost minus xy length",
        "held-out goals", (f"{HELDOUT_M}/h2_heldout/window_k150.csv",),
        (f"{HELDOUT_M}/h1_heldout/window_k150.csv",), "recorded route, FB costs", ROUTE),
    Row("weight_large_fb_minus_xy", "weight_effect", "large", "FB cost minus xy length",
        "held-out goals", (f"{HELDOUT_L}/h2_heldout/high_k150.csv",),
        (f"{HELDOUT_L}/h1_heldout/high_k150.csv",), "recorded route, FB costs", ROUTE),
)


def side(paths: Sequence[str], name: str, root: Path = ROOT) -> pd.DataFrame:
    """The episodes of the CSVs at `paths` (relative to root), with `name` as their method."""
    return load_results([root / p for p in paths]).assign(method=name)


def score(row: Row, root: Path = ROOT) -> Score:
    """The paired comparison of one row from the CSVs below root."""
    a, b = side(row.method, "a", root), side(row.reference, "b", root)
    df = pd.concat([a, b], ignore_index=True)
    if row.cluster is None:
        r = paired_comparison(df, "a", "b")
        diff, low, high, p, only = r.diff, r.low, r.high, r.p_value, (r.only_a, r.only_b)
        n_pairs = r.n_pairs
    else:
        est = pooled_paired_comparison(df, "a", "b", row.cluster)
        diff, low, high, p, only, n_pairs = est.mean, est.low, est.high, None, (None, None), est.n
    method_cluster = row.cluster if row.cluster else None
    return Score(
        key=row.key, group=row.group, maze=row.maze, label=row.label, episodes=row.episodes,
        method_name=row.method_name, reference_name=row.reference_name,
        diff=100 * diff, low=100 * low, high=100 * high, p=p,
        only_method=only[0], only_reference=only[1], n_pairs=n_pairs,
        method_success=100 * success_ci(a, cluster=method_cluster).mean,
        reference_success=100 * success_ci(b).mean,
        sources=tuple(row.method) + tuple(row.reference),
    )


def scoreboard(rows: Sequence[Row] = ROWS, root: Path = ROOT) -> list[dict]:
    """Every row's Score as a dict, in the order of `rows`."""
    return [asdict(score(row, root)) for row in rows]


def paired_outcomes(
    method: Sequence[str], reference: Sequence[str], root: Path = ROOT
) -> dict[str, int]:
    """Counts of the four outcomes of the paired episodes: both reach the goal, only the method,
    only the reference, neither."""
    a = side(method, "a", root).set_index(["task_id", "eval_seed", "episode"])["success"]
    b = side(reference, "b", root).set_index(["task_id", "eval_seed", "episode"])["success"]
    if not a.index.sort_values().equals(b.index.sort_values()):
        raise ValueError("the episodes do not pair")
    b = b.reindex(a.index)
    return {
        "both": int((a & b).sum()),
        "only_method": int((a & ~b).sum()),
        "only_reference": int((~a & b).sum()),
        "neither": int((~a & ~b).sum()),
    }
