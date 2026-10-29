"""Evaluation loop: run a Controller on OGBench tasks, one EpisodeRecord per episode.

Every method goes through this loop, so all numbers are comparable:

- The reset seed of an episode depends only on (eval_seed, task_id, episode). All methods
  start each episode from the same state (paired design, docs/design.md §5).
- Success is OGBench's `info["success"]` at the last step, as in upstream
  `utils/evaluation.evaluate`. The env ends an episode at the goal or at its time limit.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import multiprocessing
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gymnasium as gym
import numpy as np

from fbplan.contracts import Controller, EpisodeRecord, TaskSpec
from fbplan.frozen import make_env

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
    env: gym.Env, controller: Controller, task: TaskSpec, eval_seed: int, n_episodes: int
) -> list[EpisodeRecord]:
    """All episodes of one (task, eval_seed) pair."""
    chash = config_hash(controller)
    return [run_episode(env, controller, task, eval_seed, ep, chash) for ep in range(n_episodes)]


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
        for record in _run_job(env, controller, task, eval_seed, n_episodes)
    ]


# Each worker process builds its controller and env once, in _init_worker.
_worker_state: tuple[gym.Env, Controller] | None = None


def _init_worker(
    controller_factory: Callable[[], Controller], env_factory: EnvFactory, env_name: str
) -> None:
    global _worker_state
    _worker_state = (env_factory(env_name), controller_factory())


def _run_worker_job(task: TaskSpec, eval_seed: int, n_episodes: int) -> list[EpisodeRecord]:
    assert _worker_state is not None, "_init_worker did not run"
    env, controller = _worker_state
    return _run_job(env, controller, task, eval_seed, n_episodes)


def run_parallel(
    controller_factory: Callable[[], Controller],
    tasks: Sequence[TaskSpec],
    eval_seeds: Sequence[int],
    n_episodes: int,
    workers: int,
    env_factory: EnvFactory = make_env,
) -> list[EpisodeRecord]:
    """Same records as `run_episodes`, with (task, eval_seed) jobs spread over processes.

    `controller_factory` and `env_factory` must be picklable (module-level callables or
    `functools.partial` of them). Workers start with "spawn" because JAX is not fork-safe.
    """
    env_name = _single_env_name(tasks)
    jobs = [(task, eval_seed) for task in tasks for eval_seed in eval_seeds]
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(controller_factory, env_factory, env_name),
    ) as pool:
        chunks = pool.map(
            _run_worker_job,
            [task for task, _ in jobs],
            [eval_seed for _, eval_seed in jobs],
            [n_episodes] * len(jobs),
        )
        return [record for chunk in chunks for record in chunk]


def write_csv(records: Sequence[EpisodeRecord], path: Path) -> None:
    """Write records with one column per EpisodeRecord field. Creates parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fields = [field.name for field in dataclasses.fields(EpisodeRecord)]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dataclasses.asdict(record) for record in records)
