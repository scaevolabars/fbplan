"""Evaluation loop: run a Controller on OGBench tasks, one EpisodeRecord per episode.

Every method goes through this loop, so all numbers are comparable:

- The reset seed of an episode depends only on (eval_seed, task_id, episode). All methods
  start each episode from the same state (paired design, docs/design.md §5).
- Success is OGBench's `info["success"]` at the last step, as in upstream
  `utils/evaluation.evaluate`. The env ends an episode at the goal or at its time limit.

CLI (writes one CSV of EpisodeRecords, prints the success table and the throughput):

    uv run python -m fbplan.evaluate --method baseline --env antmaze-medium-navigate-v0 \
        --tasks 1 2 3 4 5 --seeds 0 --episodes 50 --workers 8 [--opt key=value ...]
"""

from __future__ import annotations

import argparse
import ast
import csv
import dataclasses
import functools
import hashlib
import json
import multiprocessing
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

from fbplan.contracts import Controller, EpisodeRecord, TaskSpec
from fbplan.controllers import resolve_controller
from fbplan.frozen import load_model, make_env
from fbplan.stats import summary_table

EnvFactory = Callable[[str], gym.Env]


def reset_seed(eval_seed: int, task_id: int, episode: int) -> int:
    """Seed for one episode reset. Hashing the key gives unrelated seeds for nearby keys."""
    return int(np.random.SeedSequence([eval_seed, task_id, episode]).generate_state(1)[0])


def config_hash(controller: Controller) -> str:
    """Short hash of the controller name and its `config` dataclass (if it has one)."""
    config = getattr(controller, "config", None)
    fields = dataclasses.asdict(config) if config is not None else {}
    payload = json.dumps({"name": controller.name, **fields}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:8]


def make_tasks(
    env_name: str,
    task_ids: Sequence[int],
    task_latent: Callable[[int], np.ndarray],
    env_factory: EnvFactory = make_env,
) -> list[TaskSpec]:
    """TaskSpecs with z_r from `task_latent` (usually FBModel.infer_task_latent).

    The goal is the noise-free goal of the OGBench task (the eval env adds no goal noise).
    """
    task_infos = env_factory(env_name).unwrapped.task_infos
    return [
        TaskSpec(
            env_name=env_name,
            task_id=task_id,
            z_r=np.asarray(task_latent(task_id)),
            goal_xy=np.asarray(task_infos[task_id - 1]["goal_xy"], dtype=float),
        )
        for task_id in task_ids
    ]


def _reset(env: gym.Env, task_id: int, seed: int) -> np.ndarray:
    # OGBench draws the start-position noise (and teleport exits) from the global NumPy RNG and
    # the settling actions from the action-space RNG. Seed both, or the start state is random.
    np.random.seed(seed)
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed, options={"task_id": task_id})
    return obs


def _check_goal(env: gym.Env, task: TaskSpec) -> None:
    env_goal = np.asarray(env.unwrapped.cur_goal_xy)
    if not np.allclose(env_goal, task.goal_xy, atol=1e-6):
        raise ValueError(
            f"env goal {env_goal} differs from task goal {task.goal_xy}: "
            "the eval env must not add noise to the goal"
        )


def run_episode(
    env: gym.Env,
    controller: Controller,
    task: TaskSpec,
    eval_seed: int,
    episode: int,
    config_hash: str,
) -> EpisodeRecord:
    """Run one episode until the goal or the env time limit."""
    start = time.perf_counter()
    seed = reset_seed(eval_seed, task.task_id, episode)
    obs = _reset(env, task.task_id, seed)
    _check_goal(env, task)
    controller.reset(task, seed)
    steps, done, info = 0, False, {}
    while not done:
        # Upstream clips too; a controller that leaves [-1, 1] must not change the result.
        action = np.clip(np.asarray(controller.act(obs)), -1.0, 1.0)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        done = terminated or truncated
    return EpisodeRecord(
        method=controller.name,
        config_hash=config_hash,
        env_name=task.env_name,
        task_id=task.task_id,
        eval_seed=eval_seed,
        episode=episode,
        success=bool(info["success"]),
        steps=steps,
        final_goal_dist=float(np.linalg.norm(env.unwrapped.get_xy() - task.goal_xy)),
        wall_time_s=time.perf_counter() - start,
    )


def _run_job(
    env: gym.Env, controller: Controller, task: TaskSpec, eval_seed: int, episodes: range
) -> list[EpisodeRecord]:
    """Episodes of one (task, eval_seed) pair."""
    chash = config_hash(controller)
    return [run_episode(env, controller, task, eval_seed, ep, chash) for ep in episodes]


def _single_env_name(tasks: Sequence[TaskSpec]) -> str:
    names = {task.env_name for task in tasks}
    if len(names) != 1:
        raise ValueError(f"tasks must share one env, got {sorted(names)}")
    return names.pop()


def run_episodes(
    controller: Controller,
    tasks: Sequence[TaskSpec],
    eval_seeds: Sequence[int],
    n_episodes: int,
    env_factory: EnvFactory = make_env,
) -> list[EpisodeRecord]:
    """Run every (task, eval_seed, episode) in this process, in that nested order."""
    env = env_factory(_single_env_name(tasks))
    return [
        record
        for task in tasks
        for eval_seed in eval_seeds
        for record in _run_job(env, controller, task, eval_seed, range(n_episodes))
    ]


# Each worker process builds its controller and env once, in _init_worker.
_worker_state: tuple[gym.Env, Controller] | None = None


def _init_worker(
    controller_factory: Callable[[], Controller], env_factory: EnvFactory, env_name: str
) -> None:
    global _worker_state
    _worker_state = (env_factory(env_name), controller_factory())


def _run_worker_job(task: TaskSpec, eval_seed: int, episodes: range) -> list[EpisodeRecord]:
    assert _worker_state is not None, "_init_worker did not run"
    env, controller = _worker_state
    return _run_job(env, controller, task, eval_seed, episodes)


def run_parallel(
    controller_factory: Callable[[], Controller],
    tasks: Sequence[TaskSpec],
    eval_seeds: Sequence[int],
    n_episodes: int,
    workers: int,
    env_factory: EnvFactory = make_env,
    episodes_per_job: int = 10,
) -> list[EpisodeRecord]:
    """Same records as `run_episodes`, with jobs spread over processes.

    A job is a block of `episodes_per_job` episodes of one (task, eval_seed) pair. Small blocks
    keep all workers busy when tasks have different episode lengths.
    `controller_factory` and `env_factory` must be picklable (module-level callables or
    `functools.partial` of them). Workers start with "spawn" because JAX is not fork-safe.
    """
    env_name = _single_env_name(tasks)
    jobs = [
        (task, eval_seed, range(start, min(start + episodes_per_job, n_episodes)))
        for task in tasks
        for eval_seed in eval_seeds
        for start in range(0, n_episodes, episodes_per_job)
    ]
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(controller_factory, env_factory, env_name),
    ) as pool:
        chunks = pool.map(_run_worker_job, *zip(*jobs, strict=True))
        return [record for chunk in chunks for record in chunk]


def write_csv(records: Sequence[EpisodeRecord], path: Path) -> None:
    """Write records with one column per EpisodeRecord field. Creates parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fields = [field.name for field in dataclasses.fields(EpisodeRecord)]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dataclasses.asdict(record) for record in records)


def parse_options(items: Sequence[str]) -> dict[str, object]:
    """Controller options from "key=value" items. A value is a Python literal or a bare string."""
    options: dict[str, object] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"option {item!r} is not key=value")
        try:
            options[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            options[key] = raw
    return options


def build_controller(spec: str, env_name: str, options: dict[str, object]) -> Controller:
    """Load the frozen model and build the controller. Runs once in each worker process."""
    return resolve_controller(spec)(load_model(env_name), **options)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one controller on OGBench tasks.")
    parser.add_argument("--method", required=True, help="baseline, nonhier, oracle or module:attr")
    parser.add_argument("--env", default="antmaze-medium-navigate-v0")
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="eval seeds")
    parser.add_argument("--episodes", type=int, default=50, help="per task and eval seed")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--opt", nargs="*", default=[], help="controller options key=value")
    parser.add_argument("--out", type=Path, help="CSV path (default: runs/<env>_<method>_...)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    factory = functools.partial(build_controller, args.method, args.env, parse_options(args.opt))
    tasks = make_tasks(args.env, args.tasks, load_model(args.env).infer_task_latent)
    start = time.perf_counter()
    if args.workers == 1:
        records = run_episodes(factory(), tasks, args.seeds, args.episodes)
    else:
        records = run_parallel(factory, tasks, args.seeds, args.episodes, args.workers)
    elapsed = time.perf_counter() - start

    first = records[0]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = args.out or Path("runs") / f"{args.env}_{first.method}_{first.config_hash}_{stamp}.csv"
    write_csv(records, out)
    print(summary_table(pd.DataFrame([dataclasses.asdict(r) for r in records])).to_string())
    steps = sum(r.steps for r in records)
    per_worker = steps / sum(r.wall_time_s for r in records)
    print(
        f"{len(records)} episodes, {steps} steps in {elapsed:.0f} s: "
        f"{steps / elapsed:.0f} steps/s overall, {per_worker:.0f} steps/s per worker. CSV: {out}"
    )


if __name__ == "__main__":
    main()
