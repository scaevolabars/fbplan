"""Statistics for evaluation results: success rates with bootstrap CIs, paired comparisons.

Conventions:

- Overall success is the mean over tasks of the per-task success rate, as in OGBench.
  Every task has the same weight, also when the episode counts differ.
- Every bootstrap is stratified by task (env, task_id): it resamples episodes within a task
  and keeps the task set fixed. Differences between tasks add no width to the CI.
- CIs are percentile intervals from 10 000 bootstrap samples with a fixed RNG seed.
  When every episode of a task has the same outcome, that task adds no width (0 % or 100 %
  give a zero-width CI). This is a known limit of the percentile bootstrap.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fbplan.contracts import EpisodeRecord

STRATA = ["env_name", "task_id"]
EPISODE_KEY = [*STRATA, "eval_seed", "episode"]  # one start state, shared by all methods


@dataclass(frozen=True)
class Estimate:
    """Success rate (mean over tasks) with a bootstrap CI."""

    mean: float
    low: float
    high: float
    n: int  # episodes


@dataclass(frozen=True)
class PairedResult:
    """Success of method a minus success of method b on the same episodes."""

    diff: float  # mean over tasks of the per-task difference
    low: float
    high: float
    n_pairs: int
    only_a: int  # pairs where only a succeeds
    only_b: int  # pairs where only b succeeds
    p_value: float  # exact two-sided McNemar test


def load_results(paths: Iterable[Path], label_root: Path | None = None) -> pd.DataFrame:
    """Concatenate result CSVs written by `fbplan.evaluate.write_csv`.

    With label_root, the method column becomes each CSV's path relative to label_root without
    the suffix (for example "planner/l0"). Runs of one controller with different configs then
    get readable names; the config hash stays in its own column.
    """
    paths = list(paths)
    frames = [pd.read_csv(path) for path in paths]
    if label_root is not None:
        frames = [
            frame.assign(method=Path(path).relative_to(label_root).with_suffix("").as_posix())
            for path, frame in zip(paths, frames, strict=True)
        ]
    df = pd.concat(frames, ignore_index=True)
    missing = {f.name for f in dataclasses.fields(EpisodeRecord)} - set(df.columns)
    if missing:
        raise ValueError(f"result CSVs miss columns {sorted(missing)}")
    return df.astype({"success": bool})


def _quantiles(draws: np.ndarray, level: float) -> tuple[float, float]:
    low, high = np.quantile(draws, [(1 - level) / 2, (1 + level) / 2])
    return float(low), float(high)


def _bootstrap_success(n: np.ndarray, k: np.ndarray, n_boot: int, rng: np.random.Generator):
    """Bootstrap draws of the overall success from episode counts n and successes k [task, cluster].

    A resample of n binary outcomes with k successes has a Binomial(n, k/n) success count,
    and m picks of the same cluster give Binomial(m n, k/n). So we draw counts, not episodes.
    """
    n_clusters = n.shape[1]
    picks = rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters), size=n_boot)
    trials = picks[:, None, :] * n  # [boot, task, cluster]
    successes = rng.binomial(trials, k / n)
    return (successes.sum(-1) / trials.sum(-1)).mean(-1)


def success_ci(
    df: pd.DataFrame,
    cluster: str | None = None,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Overall success of the episodes in `df` with a stratified percentile bootstrap CI.

    `cluster` names a column of a shared random factor, for example the checkpoint of a
    retrained pi_h. Each bootstrap sample then resamples clusters first and episodes inside
    each (task, cluster) cell second, so the CI includes the variance between clusters.
    """
    mean, draws, n = _success_draws(df, cluster, n_boot, seed)
    return Estimate(mean, *_quantiles(draws, level), n=n)


def unpaired_comparison(
    df: pd.DataFrame,
    method_a: str,
    method_b: str,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Success of a minus success of b when the episodes do not pair.

    Use it only when the start states differ, for example against upstream's unseeded
    evaluation loop. The two stratified bootstraps are independent.
    """
    a, b = (df[df["method"] == m] for m in (method_a, method_b))
    if set(map(tuple, a[STRATA].values)) != set(map(tuple, b[STRATA].values)):
        raise ValueError(f"{method_a!r} and {method_b!r} must cover the same tasks")
    mean_a, draws_a, n_a = _success_draws(a, None, n_boot, seed)
    mean_b, draws_b, n_b = _success_draws(b, None, n_boot, seed + 1)
    return Estimate(mean_a - mean_b, *_quantiles(draws_a - draws_b, level), n=n_a + n_b)


def _success_draws(
    df: pd.DataFrame, cluster: str | None, n_boot: int, seed: int
) -> tuple[float, np.ndarray, int]:
    """Point estimate, bootstrap draws and episode count of the overall success."""
    cells = (
        df.assign(_cluster=df[cluster] if cluster else 0)
        .groupby([*STRATA, "_cluster"])["success"]
        .agg(["size", "sum"])
    )
    n = cells["size"].unstack(fill_value=0).to_numpy()  # [task, cluster]
    k = cells["sum"].unstack(fill_value=0).to_numpy()
    if (n == 0).any():
        raise ValueError(f"every task needs episodes in every cluster of {cluster!r}")
    draws = _bootstrap_success(n, k, n_boot, np.random.default_rng(seed))
    return float((k.sum(1) / n.sum(1)).mean()), draws, int(n.sum())


def mcnemar_p_value(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar p-value from the discordant pair counts.

    Under H0 each discordant pair favours either method with probability 1/2. The test is
    valid across tasks with different success rates, because it conditions on each pair.
    """
    n = only_a + only_b
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def _paired_success(df: pd.DataFrame, method_a: str, method_b: str) -> pd.DataFrame:
    """One row per episode key with integer columns a and b (1 = success)."""
    a, b = (df[df["method"] == m].set_index(EPISODE_KEY)["success"] for m in (method_a, method_b))
    if a.index.has_duplicates or b.index.has_duplicates:
        raise ValueError("each episode must appear once per method to pair")
    if not a.index.sort_values().equals(b.index.sort_values()):
        raise ValueError(f"episodes of {method_a!r} and {method_b!r} do not pair")
    return pd.DataFrame({"a": a.astype(int), "b": b.reindex(a.index).astype(int)})


def paired_comparison(
    df: pd.DataFrame,
    method_a: str,
    method_b: str,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> PairedResult:
    """Compare two methods on the same episodes (same env, task, eval seed and episode).

    The CI resamples pairs within each task. The p-value is the exact McNemar test.
    """
    pairs = _paired_success(df, method_a, method_b)
    diff = pairs["a"] - pairs["b"]
    # counts[task] = number of pairs with difference -1, 0, +1
    counts = (
        diff.groupby(level=STRATA).value_counts().unstack(fill_value=0)
        .reindex(columns=[-1, 0, 1], fill_value=0).to_numpy()
    )
    n_task = counts.sum(1)
    rng = np.random.default_rng(seed)
    draws = np.stack([rng.multinomial(c.sum(), c / c.sum(), size=n_boot) for c in counts], 1)
    boot_diff = ((draws[..., 2] - draws[..., 0]) / n_task).mean(-1)
    only_a, only_b = int(counts[:, 2].sum()), int(counts[:, 0].sum())
    low, high = _quantiles(boot_diff, level)
    return PairedResult(
        diff=float(((counts[:, 2] - counts[:, 0]) / n_task).mean()),
        low=low,
        high=high,
        n_pairs=len(pairs),
        only_a=only_a,
        only_b=only_b,
        p_value=mcnemar_p_value(only_a, only_b),
    )


def summary_table(df: pd.DataFrame, cluster: str | None = None, seed: int = 0) -> pd.DataFrame:
    """Success per method and task, plus an overall row ("all") per method, for one env."""
    if df["env_name"].nunique() != 1:
        raise ValueError("summary_table expects the results of one env")
    rows = []
    for method, group in df.groupby("method", sort=False):
        parts = [(str(t), g) for t, g in group.groupby("task_id")] + [("all", group)]
        for task, part in parts:
            est = success_ci(part, cluster=cluster, seed=seed)
            rows.append((method, task, est.n, est.mean, est.low, est.high))
    return pd.DataFrame(rows, columns=["method", "task", "n", "success", "ci_low", "ci_high"])


def paired_table(df: pd.DataFrame, reference: str, seed: int = 0) -> pd.DataFrame:
    """`paired_comparison` of every other method against `reference` (method minus reference)."""
    rows = []
    for method in df["method"].unique():
        if method == reference:
            continue
        r = paired_comparison(df, method, reference, seed=seed)
        rows.append((method, r.diff, r.low, r.high, r.only_a, r.only_b, r.p_value, r.n_pairs))
    columns = ["method", "diff", "ci_low", "ci_high", "only_method", "only_reference", "p_value"]
    return pd.DataFrame(rows, columns=[*columns, "n_pairs"])
