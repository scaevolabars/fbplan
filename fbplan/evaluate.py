"""Evaluation loop: run a Controller on OGBench tasks, one EpisodeRecord per episode.

Every method goes through this loop, so all numbers are comparable:

- The reset seed of an episode depends only on (eval_seed, task_id, episode). All methods
  start each episode from the same state (paired design, docs/design.md §5).
- Success is OGBench's `info["success"]` at the last step, as in upstream
  `utils/evaluation.evaluate`. The env ends an episode at the goal or at its time limit.

- `--checkpoint` selects the pi_h of the frozen model (F, B and pi_l never change): the
  provided one, a retrained baseline seed or an approach B model. Every CSV row records it in
  the columns of `fbplan.stats.CHECKPOINT_DEFAULTS`, so that stats can pool training seeds.
- Held-out goals (task ids above HELDOUT_OFFSET, listed in results/heldout_goals/) are custom
  OGBench tasks for hyperparameter selection, so that tuning never sees the 5 eval tasks.

CLI (writes one CSV of EpisodeRecords, prints the success table and the throughput):

    uv run python -m fbplan.evaluate --method baseline --env antmaze-medium-navigate-v0 \
        --tasks 1 2 3 4 5 --seeds 0 --episodes 50 --workers 8 [--opt key=value ...] \
        [--checkpoint provided | retrained:<seed> | b:<variant>:<seed>] [--heldout]

With many workers on one CPU, `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false
intra_op_parallelism_threads=1"` gives each worker one XLA thread (about +15 % steps/s with
10 workers on a 12-core Mac; the results do not change).
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
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import pandas as pd

from fbplan.contracts import Controller, EpisodeRecord, FBModel, TaskSpec
from fbplan.controllers import ALIASES, resolve_controller
from fbplan.frozen import ENV_NAMES, load_dataset, load_model, make_env, upstream_module
from fbplan.learned import checkpoint as trained
from fbplan.maze import Cell, MazeGeometry
from fbplan.stats import CHECKPOINT_DEFAULTS, summary_table

EnvFactory = Callable[[str], gym.Env]
CheckpointKind = Literal["provided", "retrained", "b"]

# Held-out goal k (1, 2, ...) has task_id HELDOUT_OFFSET + k; ids 1..5 are the OGBench eval tasks.
HELDOUT_OFFSET = 100
HELDOUT_DIR = Path(__file__).resolve().parents[1] / "results" / "heldout_goals"


@dataclasses.dataclass(frozen=True)
class Checkpoint:
    """Which pi_h the frozen model carries. F, B and pi_l are always the provided modules.

    provided: the paper's checkpoint (fbplan.frozen.load_model).
    retrained: pi_h retrained with the upstream recipe from training seed `seed`
        (fbplan.learned.checkpoint.load_baseline_model).
    b: approach B pi_h of `variant` from training seed `seed`
        (fbplan.learned.checkpoint.load_b_model).
    """

    kind: CheckpointKind = "provided"
    variant: str = ""
    seed: int = -1

    def __post_init__(self) -> None:
        valid = {
            "provided": self.variant == "" and self.seed == -1,
            "retrained": self.variant == "" and self.seed >= 0,
            "b": self.variant != "" and self.seed >= 0,
        }
        if not valid.get(self.kind, False):
            raise ValueError(f"invalid checkpoint {self}")

    @classmethod
    def parse(cls, spec: str) -> Checkpoint:
        """A checkpoint from "provided", "retrained:<seed>" or "b:<variant>:<seed>"."""
        kind, *rest = spec.split(":")
        try:
            if kind == "provided" and not rest:
                return cls()
            if kind == "retrained" and len(rest) == 1:
                return cls("retrained", "", int(rest[0]))
            if kind == "b" and len(rest) == 2:
                return cls("b", rest[0], int(rest[1]))
        except ValueError:
            pass
        raise ValueError(
            f"checkpoint {spec!r} is not provided, retrained:<seed> or b:<variant>:<seed>"
        )

    def load(self, env_name: str) -> FBModel:
        """The frozen model with this pi_h."""
        if self.kind == "retrained":
            return trained.load_baseline_model(env_name, self.seed)
        if self.kind == "b":
            return trained.load_b_model(env_name, self.variant, self.seed)
        return load_model(env_name)

    def columns(self) -> dict[str, object]:
        """The CSV columns of this checkpoint (keys of fbplan.stats.CHECKPOINT_DEFAULTS)."""
        return dict(zip(CHECKPOINT_DEFAULTS, (self.kind, self.variant, self.seed), strict=True))


PROVIDED = Checkpoint()


@dataclasses.dataclass(frozen=True)
class HeldoutGoal:
    """A custom OGBench task: start cell, goal cell and goal xy (the goal cell's centre)."""

    task_id: int
    init_ij: Cell
    goal_ij: Cell
    goal_xy: tuple[float, float]


def select_heldout_goals(
    env: gym.Env,
    dataset_xy: np.ndarray,
    count: int = 10,
    min_cells: int = 2,
    min_states: int = 50,
    seed: int = 0,
) -> list[HeldoutGoal]:
    """Goals far from the 5 eval goals, with start cells as far from them as the eval tasks'.

    A start cell of a goal is a free cell whose distance to the goal (maze moves between
    4-connected cells) lies in the range of the eval tasks' start-to-goal distances. Goal cells:
    `count` free cells, drawn without replacement (seeded) from those that are at least
    `min_cells` moves from every eval goal, have a start cell, and have at least `min_states`
    of `dataset_xy` within the goal radius. dataset_xy: xy of the states that make z (the
    validation split, see goal_latent); few states give a noisy z. Start cell of each goal: a
    seeded draw. The goal is the cell centre, as in OGBench's own tasks, and success uses
    OGBench's goal radius.
    """
    maze, infos = MazeGeometry.from_env(env), env.unwrapped.task_infos
    radius = env.unwrapped._goal_tol

    def states_near(cell: Cell) -> int:
        centre = np.asarray(env.unwrapped.ij_to_xy(cell))
        return int((np.linalg.norm(dataset_xy - centre, axis=1) <= radius).sum())
    to_eval_goals = [maze.bfs_steps(tuple(info["goal_ij"])) for info in infos]
    lengths = [
        steps[tuple(info["init_ij"])] for steps, info in zip(to_eval_goals, infos, strict=True)
    ]
    free = maze.free_cells()

    def starts(goal_ij: Cell) -> list[Cell]:
        steps = maze.bfs_steps(goal_ij)
        return [c for c in free if min(lengths) <= steps[c] <= max(lengths)]

    candidates = [
        c
        for c in free
        if min(steps[c] for steps in to_eval_goals) >= min_cells
        and starts(c)
        and states_near(c) >= min_states
    ]
    if not candidates:
        raise ValueError("no free cell is far enough from the eval goals")
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(candidates), size=min(count, len(candidates)), replace=False)
    goals = []
    for k, index in enumerate(picked, start=1):
        goal_ij = candidates[index]
        options = starts(goal_ij)
        x, y = env.unwrapped.ij_to_xy(goal_ij)
        init_ij = options[rng.integers(len(options))]
        goals.append(HeldoutGoal(HELDOUT_OFFSET + k, init_ij, goal_ij, (float(x), float(y))))
    return goals


def heldout_goals_path(env_name: str) -> Path:
    return HELDOUT_DIR / f"{env_name}.csv"


def write_heldout_goals(goals: Sequence[HeldoutGoal], path: Path) -> None:
    rows = [
        {"task_id": g.task_id, "init_i": g.init_ij[0], "init_j": g.init_ij[1],
         "goal_i": g.goal_ij[0], "goal_j": g.goal_ij[1], "goal_x": g.goal_xy[0],
         "goal_y": g.goal_xy[1]}
        for g in goals
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def load_heldout_goals(env_name: str) -> list[HeldoutGoal]:
    """The saved held-out goals of an env (written by analysis/heldout_goals.py)."""
    df = pd.read_csv(heldout_goals_path(env_name))
    return [
        HeldoutGoal(int(r.task_id), (int(r.init_i), int(r.init_j)), (int(r.goal_i), int(r.goal_j)),
                    (float(r.goal_x), float(r.goal_y)))
        for r in df.itertuples()
    ]


@functools.cache
def _heldout_table(env_name: str) -> dict[int, HeldoutGoal]:
    return {goal.task_id: goal for goal in load_heldout_goals(env_name)}


@functools.cache
def _val_dataset(env_name: str) -> Any:
    return upstream_module("utils.datasets").Dataset.create(**load_dataset(env_name, "val"))


def goal_latent(model: FBModel, goal_ij: Cell) -> np.ndarray:
    """z of the goal at the centre of cell goal_ij, computed as FrozenFBModel.infer_task_latent
    computes z_r: upstream relabels the validation split with OGBench's goal radius (reward 1
    inside, else 0) and averages r * B(s) over its first num_zero_shot_samples rows.
    It needs the upstream agent, so it works with fbplan.frozen.FrozenFBModel only.
    """
    env = make_env(model.env_name)
    env.reset(options={"task_info": {"init_ij": goal_ij, "goal_ij": goal_ij}})
    relabel = upstream_module("utils.env_utils").relabel_dataset
    relabeled = relabel(f"ogbench-{model.env_name}", env, _val_dataset(model.env_name))
    agent: Any = model.agent  # type: ignore[attr-defined]
    n = agent.config["num_zero_shot_samples"]
    rewards = relabeled["rewards"][:n]
    if not rewards.any():
        raise ValueError(f"no validation state within the goal radius of cell {goal_ij}")
    return np.asarray(agent.infer_latent({"observations": relabeled["observations"][:n],
                                          "rewards": rewards}))


def task_latent(model: FBModel, task_id: int) -> np.ndarray:
    """z of an eval task (FBModel.infer_task_latent) or of a held-out goal (goal_latent)."""
    if task_id > HELDOUT_OFFSET:
        return goal_latent(model, _heldout_table(model.env_name)[task_id].goal_ij)
    return model.infer_task_latent(task_id)


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

    The goal is the noise-free goal of the OGBench task (the eval env adds no goal noise), or
    the saved goal of a held-out task id.
    """
    task_infos = env_factory(env_name).unwrapped.task_infos

    def goal_xy(task_id: int) -> np.ndarray:
        if task_id > HELDOUT_OFFSET:
            return np.asarray(_heldout_table(env_name)[task_id].goal_xy, dtype=float)
        return np.asarray(task_infos[task_id - 1]["goal_xy"], dtype=float)

    return [
        TaskSpec(
            env_name=env_name,
            task_id=task_id,
            z_r=np.asarray(task_latent(task_id)),
            goal_xy=goal_xy(task_id),
        )
        for task_id in task_ids
    ]


def reset_env(env: gym.Env, task: TaskSpec, seed: int) -> np.ndarray:
    """Reset the env to the start of `task` for reset seed `seed` (docs/design.md §5).

    OGBench draws the start-position noise (and teleport exits) from the global NumPy RNG; the
    action-space RNG drives the settling steps, which change only the goal observation. Both
    are seeded, so the same seed gives the same start and goal observations.
    """
    options: dict[str, object] = {"task_id": task.task_id}
    if task.task_id > HELDOUT_OFFSET:
        goal = _heldout_table(task.env_name)[task.task_id]
        options = {"task_info": {"init_ij": goal.init_ij, "goal_ij": goal.goal_ij}}
    np.random.seed(seed)
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed, options=options)
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
    chash: str,
) -> EpisodeRecord:
    """Run one episode until the goal or the env time limit. chash: the record's config_hash."""
    start = time.perf_counter()
    seed = reset_seed(eval_seed, task.task_id, episode)
    obs = reset_env(env, task, seed)
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
        config_hash=chash,
        env_name=task.env_name,
        task_id=task.task_id,
        eval_seed=eval_seed,
        episode=episode,
        success=bool(info["success"]),
        steps=steps,
        final_goal_dist=float(np.linalg.norm(env.unwrapped.get_xy() - task.goal_xy)),
        wall_time_s=time.perf_counter() - start,
    )


class RecordingController:
    """Passes every call to `inner` and keeps the observations of the current episode."""

    def __init__(self, inner: Controller) -> None:
        self.inner = inner
        self.name = inner.name
        self.observations: list[np.ndarray] = []

    def reset(self, task: TaskSpec, seed: int) -> None:
        self.observations = []
        self.inner.reset(task, seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        self.observations.append(obs.copy())
        return self.inner.act(obs)

    def step_info(self) -> dict[str, float | int | str]:
        return self.inner.step_info()


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


def write_csv(
    records: Sequence[EpisodeRecord], path: Path, checkpoint: Checkpoint = PROVIDED
) -> None:
    """Write records with one column per EpisodeRecord field, plus the checkpoint columns.

    Creates parent folders.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = checkpoint.columns()
    with path.open("w", newline="") as f:
        fields = [field.name for field in dataclasses.fields(EpisodeRecord)]
        writer = csv.DictWriter(f, fieldnames=[*fields, *columns])
        writer.writeheader()
        writer.writerows({**dataclasses.asdict(record), **columns} for record in records)


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


def build_controller(
    spec: str, env_name: str, options: dict[str, object], checkpoint: Checkpoint = PROVIDED
) -> Controller:
    """Load the frozen model and build the controller. Runs once in each worker process."""
    return resolve_controller(spec)(checkpoint.load(env_name), **options)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one controller on OGBench tasks.")
    parser.add_argument(
        "--method", required=True, help=f"{', '.join(sorted(ALIASES))} or package.module:attr"
    )
    parser.add_argument("--env", default="antmaze-medium-navigate-v0", choices=ENV_NAMES)
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="eval seeds (protocol: 5)"
    )
    parser.add_argument("--episodes", type=int, default=50, help="per task and eval seed")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--opt", nargs="*", default=[], help="controller options key=value")
    parser.add_argument(
        "--checkpoint", default="provided", help="provided, retrained:<seed> or b:<variant>:<seed>"
    )
    parser.add_argument("--heldout", action="store_true", help="all held-out goals, not --tasks")
    parser.add_argument("--out", type=Path, help="CSV path (default: runs/<env>_<method>_...)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    checkpoint = Checkpoint.parse(args.checkpoint)
    options = parse_options(args.opt)
    factory = functools.partial(build_controller, args.method, args.env, options, checkpoint)
    task_ids = [g.task_id for g in load_heldout_goals(args.env)] if args.heldout else args.tasks
    # B is the same in every checkpoint (load_retrained_model checks it), and so is z.
    model = load_model(args.env)
    tasks = make_tasks(args.env, task_ids, functools.partial(task_latent, model))
    start = time.perf_counter()
    if args.workers == 1:
        records = run_episodes(factory(), tasks, args.seeds, args.episodes)
    else:
        records = run_parallel(factory, tasks, args.seeds, args.episodes, args.workers)
    elapsed = time.perf_counter() - start

    first = records[0]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = args.out or Path("runs") / f"{args.env}_{first.method}_{first.config_hash}_{stamp}.csv"
    write_csv(records, out, checkpoint)
    print(summary_table(pd.DataFrame([dataclasses.asdict(r) for r in records])).to_string())
    steps = sum(r.steps for r in records)
    per_worker = steps / sum(r.wall_time_s for r in records)
    print(
        f"{len(records)} episodes, {steps} steps in {elapsed:.0f} s: "
        f"{steps / elapsed:.0f} steps/s overall, {per_worker:.0f} steps/s per worker. CSV: {out}"
    )


if __name__ == "__main__":
    main()
