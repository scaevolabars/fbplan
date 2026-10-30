"""Parity check: our evaluation loop against upstream `utils/evaluation.evaluate`.

    uv run python analysis/parity.py --ours results/medium/baseline.csv [--reps 5] \
        [--episodes 50] [--workers 4] [--out results/medium/upstream.csv]

Runs the restored upstream agent through upstream `evaluate` (temperature 0, no videos, as
upstream main.py) for every task of the env in --ours, `--reps` times `--episodes` episodes,
and writes the episodes as EpisodeRecords (method "upstream"). Upstream resets the env without
a seed, so its episodes do not pair with ours and a rerun does not repeat them exactly (NumPy
is seeded per task and rep, which fixes the start-position noise and the policy key only).
The comparison is an unpaired stratified bootstrap; parity holds when its CI contains 0.
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
from fbplan.evaluate import make_tasks, write_csv
from fbplan.frozen import load_model, make_env, upstream_module
from fbplan.stats import load_results, summary_table, unpaired_comparison


def upstream_episodes(
    env_name: str, task_id: int, rep: int, n_episodes: int
) -> list[EpisodeRecord]:
    """`n_episodes` of upstream evaluate on one task. `rep` seeds NumPy (start-position noise and
    upstream's policy key); the env's own reset RNG stays unseeded, as in upstream."""
    model, env = load_model(env_name), make_env(env_name)
    (task,) = make_tasks(env_name, [task_id], model.infer_task_latent)
    np.random.seed(10_000 * rep + task_id)
    start = time.perf_counter()
    _, trajs, _ = upstream_module("utils.evaluation").evaluate(
        agent=model.agent, env=env, task_id=task_id, inferred_latent=task.z_r,
        num_eval_episodes=n_episodes, num_video_episodes=0, eval_temperature=0,
    )
    per_episode = (time.perf_counter() - start) / n_episodes
    return [
        EpisodeRecord(
            method="upstream", config_hash="upstream", env_name=env_name, task_id=task_id,
            eval_seed=rep, episode=i, success=bool(traj["info"][-1]["success"]),
            steps=len(traj["action"]),
            final_goal_dist=float(np.linalg.norm(traj["info"][-1]["xy"] - task.goal_xy)),
            wall_time_s=per_episode,
        )
        for i, traj in enumerate(trajs)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ours", type=Path, required=True, help="CSV of our baseline run")
    parser.add_argument("--reps", type=int, default=1, help="repetitions per task")
    parser.add_argument("--episodes", type=int, default=50, help="episodes per task and rep")
    parser.add_argument("--workers", type=int, default=4, help="worker processes")
    parser.add_argument(
        "--out", type=Path, default=Path("runs/parity_upstream.csv"),
        help="CSV of upstream's own evaluation episodes",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    ours = load_results([args.ours])
    (env_name,) = ours["env_name"].unique()
    jobs = [(task, rep) for task in sorted(ours["task_id"].unique()) for rep in range(args.reps)]
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        futures = [
            pool.submit(upstream_episodes, env_name, int(t), r, args.episodes) for t, r in jobs
        ]
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
