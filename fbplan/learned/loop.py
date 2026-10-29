"""Training loop with CSV logging, shared by train_high and approach B."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import jax
import pandas as pd

State = TypeVar("State")
Info = dict[str, Any]


def run_logged(
    state: State,
    chunk: Callable[[State, jax.Array, int], tuple[State, Info]],
    validate: Callable[[State, jax.Array], Info],
    steps: int,
    log_every: int,
    key: jax.Array,
    log_path: Path,
    progress_key: str,
) -> State:
    """Run `steps` updates in chunks of `log_every`. After each chunk, rewrite the CSV at log_path.

    chunk(state, key, n) runs n updates and returns mean training metrics; validate(state, key)
    returns metrics on held-out data. `progress_key` names the metric printed after each chunk.
    """
    if steps % log_every != 0:
        raise ValueError(f"steps={steps} is not a multiple of log_every={log_every}")
    rows: list[dict[str, float]] = []
    start = time.time()
    for i in range(1, steps // log_every + 1):
        key, train_key, val_key = jax.random.split(key, 3)
        state, train_info = chunk(state, train_key, log_every)
        val_info = validate(state, val_key)
        row = {"step": i * log_every, "time_s": time.time() - start}
        row |= {f"training/{k}": float(v) for k, v in train_info.items()}
        row |= {f"validation/{k}": float(v) for k, v in val_info.items()}
        rows.append(row)
        pd.DataFrame(rows).to_csv(log_path, index=False)
        print(
            f"step {row['step']:>7d}  {progress_key} {row[f'training/{progress_key}']:.4f}"
            f"  val {row[f'validation/{progress_key}']:.4f}  {row['time_s']:.0f}s",
            flush=True,
        )
    return state
