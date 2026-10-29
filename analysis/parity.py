"""Parity check: our evaluation loop against upstream `utils/evaluation.evaluate`.

    uv run python analysis/parity.py --ours runs/<baseline csv> [--reps 1] [--workers 5]

Runs the restored upstream agent through upstream `evaluate` (temperature 0, no videos, as
upstream main.py) for every task of the env in --ours, `--reps` times 50 episodes, and
writes the episodes as EpisodeRecords (method "upstream"). Upstream does not seed resets, so
its episodes do not pair with ours: the comparison is an unpaired stratified bootstrap.
Parity holds when the difference CI contains 0 and both CIs overlap.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from fbplan.contracts import EpisodeRecord
from fbplan.evaluate import write_csv
from fbplan.frozen import load_model, make_env, upstream_module
from fbplan.stats import load_results, summary_table, unpaired_comparison


def upstream_episodes(
    env_name: str, task_id: int, rep: int, n_episodes: int
) -> list[EpisodeRecord]:
    """`n_episodes` of upstream evaluate on one task. `rep` seeds NumPy, which upstream uses
    for its policy PRNG key and the env uses for start noise, so a rerun gives the same episodes."""
    model, env = load_model(env_name), make_env(env_name)
    z_r = model.infer_task_latent(task_id)
    goal = np.asarray(env.unwrapped.task_infos[task_id - 1]["goal_xy"])
    np.random.seed(10_000 * rep + task_id)
    start = time.perf_counter()
    _, trajs, _ = upstream_module("utils.evaluation").evaluate(
        agent=model.agent, env=env, task_id=task_id, inferred_latent=z_r,
        num_eval_episodes=n_episodes, num_video_episodes=0, eval_temperature=0,
    )
    per_episode = (time.perf_counter() - start) / n_episodes
    return [
        EpisodeRecord(
            method="upstream", config_hash="upstream", env_name=env_name, task_id=task_id,
            eval_seed=rep, episode=i, success=bool(traj["info"][-1]["success"]),
            steps=len(traj["action"]),
            final_goal_dist=float(np.linalg.norm(traj["info"][-1]["xy"] - goal)),
            wall_time_s=per_episode,
        )
        for i, traj in enumerate(trajs)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ours", type=Path, required=True, help="CSV of our baseline run")
    parser.add_argument("--reps", type=int, default=1, help="repetitions of 50 episodes per task")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("runs/parity_upstream.csv"))
    args = parser.parse_args()

    ours = load_results([args.ours])
    (env_name,) = ours["env_name"].unique()
    jobs = [(task, rep) for task in sorted(ours["task_id"].unique()) for rep in range(args.reps)]
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(upstream_episodes, env_name, int(t), r, 50) for t, r in jobs]
        records = [record for future in futures for record in future.result()]
    write_csv(records, args.out)

    upstream = pd.DataFrame([dataclasses.asdict(r) for r in records])
    both = pd.concat([ours.assign(method="ours"), upstream], ignore_index=True)
    print(summary_table(both).to_string(index=False))
    diff = unpaired_comparison(both, "ours", "upstream")
    print(f"\nours - upstream: {100 * diff.mean:+.1f} pp, "
          f"95 % CI [{100 * diff.low:+.1f}, {100 * diff.high:+.1f}], {diff.n} episodes")
    print(f"upstream CSV: {args.out}")


if __name__ == "__main__":
    main()
