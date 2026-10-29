import dataclasses
import itertools

import gymnasium as gym
import numpy as np
import ogbench
import pandas as pd
import pytest
from gymnasium.spaces import Box
from gymnasium.wrappers import TimeLimit

from fbplan.contracts import TaskSpec
from fbplan.evaluate import (
    config_hash,
    make_tasks,
    reset_seed,
    run_episode,
    run_episodes,
    run_parallel,
    write_csv,
)

ENV_NAME = "point-test-v0"
TIME_LIMIT = 50


class _PointEnv(gym.Env):
    """2-D point with OGBench-like reset and success rules, for fast loop tests.

    Like OGBench, it draws the start noise from the global NumPy RNG.
    """

    observation_space = Box(-np.inf, np.inf, (2,))
    action_space = Box(-1.0, 1.0, (2,))

    def __init__(self) -> None:
        self.task_infos = [{"goal_xy": (3.0, 0.0)}, {"goal_xy": (0.0, 3.0)}]
        self.cur_goal_xy = np.zeros(2)
        self._xy = np.zeros(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.cur_goal_xy = np.array(self.task_infos[options["task_id"] - 1]["goal_xy"])
        self._xy = np.random.uniform(-0.5, 0.5, size=2)
        return self._xy.copy(), {}

    def step(self, action):
        self._xy = self._xy + 0.5 * np.clip(action, -1.0, 1.0)
        success = float(np.linalg.norm(self._xy - self.cur_goal_xy) <= 0.5)
        return self._xy.copy(), success, bool(success), False, {"success": success}

    def get_xy(self) -> np.ndarray:
        return self._xy.copy()


def _make_point_env(env_name: str) -> gym.Env:
    assert env_name == ENV_NAME
    return TimeLimit(_PointEnv(), max_episode_steps=TIME_LIMIT)


def _task(task_id: int) -> TaskSpec:
    goal = np.array(_PointEnv().task_infos[task_id - 1]["goal_xy"])
    return TaskSpec(env_name=ENV_NAME, task_id=task_id, z_r=np.zeros(4), goal_xy=goal)


class _GreedyController:
    """Moves straight to the goal. Reads the privileged goal, like an oracle."""

    name = "greedy"

    def reset(self, task: TaskSpec, seed: int) -> None:
        self.goal = task.goal_xy
        self.seeds = [*getattr(self, "seeds", []), seed]

    def act(self, obs: np.ndarray) -> np.ndarray:
        delta = self.goal - obs[:2]
        return delta / max(np.linalg.norm(delta), 1e-8)

    def step_info(self) -> dict:
        return {}


class _StillController:
    """Never moves. Records the first observation of each episode."""

    name = "still"

    def __init__(self, act_dim: int = 2) -> None:
        self.first_obs: list[np.ndarray] = []
        self._new_episode = False
        self._act_dim = act_dim

    def reset(self, task: TaskSpec, seed: int) -> None:
        self._new_episode = True

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self._new_episode:
            self.first_obs.append(obs.copy())
            self._new_episode = False
        return np.zeros(self._act_dim)

    def step_info(self) -> dict:
        return {}


def test_reset_seed_is_deterministic():
    assert reset_seed(3, 2, 7) == reset_seed(3, 2, 7)


def test_reset_seed_differs_across_seed_task_and_episode():
    grid = list(itertools.product(range(5), range(1, 6), range(50)))
    assert len({reset_seed(*key) for key in grid}) == len(grid)


def test_run_episode_reports_success_before_time_limit():
    record = run_episode(_make_point_env(ENV_NAME), _GreedyController(), _task(1), 0, 0, "h")
    assert record.success
    assert record.steps < TIME_LIMIT
    assert record.final_goal_dist <= 0.5


def test_run_episode_reports_failure_at_time_limit():
    record = run_episode(_make_point_env(ENV_NAME), _StillController(), _task(1), 0, 0, "h")
    assert not record.success
    assert record.steps == TIME_LIMIT
    assert record.final_goal_dist > 2.0


def test_run_episode_passes_derived_seed_to_controller():
    controller = _GreedyController()
    run_episode(_make_point_env(ENV_NAME), controller, _task(2), 4, 9, "h")
    assert controller.seeds == [reset_seed(4, 2, 9)]


def test_run_episode_rejects_task_goal_that_differs_from_env_goal():
    task = dataclasses.replace(_task(1), goal_xy=np.array([9.0, 9.0]))
    with pytest.raises(ValueError, match="goal"):
        run_episode(_make_point_env(ENV_NAME), _GreedyController(), task, 0, 0, "h")


def test_methods_see_the_same_start_state_for_the_same_episode_key():
    a, b = _StillController(), _StillController()
    tasks = [_task(1), _task(2)]
    for controller in (a, b):
        run_episodes(controller, tasks, [0, 1], 3, env_factory=_make_point_env)
    np.testing.assert_array_equal(np.stack(a.first_obs), np.stack(b.first_obs))
    assert len({tuple(obs) for obs in a.first_obs}) == len(a.first_obs)


def test_run_episodes_returns_one_record_per_episode_in_grid_order():
    records = run_episodes(
        _GreedyController(), [_task(1), _task(2)], [0, 1], 3, env_factory=_make_point_env
    )
    keys = [(r.task_id, r.eval_seed, r.episode) for r in records]
    assert keys == list(itertools.product([1, 2], [0, 1], range(3)))
    assert {r.method for r in records} == {"greedy"}
    assert {r.env_name for r in records} == {ENV_NAME}


def _without_time(records):
    return [dataclasses.replace(r, wall_time_s=0.0) for r in records]


def test_run_parallel_matches_sequential_run():
    tasks = [_task(1), _task(2)]
    sequential = run_episodes(_GreedyController(), tasks, [0, 1], 2, env_factory=_make_point_env)
    parallel = run_parallel(
        _GreedyController, tasks, [0, 1], 2, workers=2, env_factory=_make_point_env
    )
    assert _without_time(parallel) == _without_time(sequential)


def test_csv_round_trip_keeps_records(tmp_path):
    records = run_episodes(_GreedyController(), [_task(1)], [0], 2, env_factory=_make_point_env)
    path = tmp_path / "out" / "result.csv"
    write_csv(records, path)
    expected = pd.DataFrame([dataclasses.asdict(r) for r in records])
    pd.testing.assert_frame_equal(pd.read_csv(path), expected)


@dataclasses.dataclass(frozen=True)
class _Config:
    rho_min: float = 0.1


class _ConfiguredController(_GreedyController):
    def __init__(self, config: _Config) -> None:
        self.config = config


def test_config_hash_depends_only_on_name_and_config():
    first = config_hash(_ConfiguredController(_Config()))
    again = config_hash(_ConfiguredController(_Config()))
    other = config_hash(_ConfiguredController(_Config(rho_min=0.2)))
    assert first == again != other
    assert config_hash(_GreedyController()) != first


def test_make_tasks_reads_goal_from_env_and_latent_from_callable():
    tasks = make_tasks(ENV_NAME, [2], lambda task_id: np.full(4, task_id), _make_point_env)
    assert [t.task_id for t in tasks] == [2]
    np.testing.assert_array_equal(tasks[0].goal_xy, [0.0, 3.0])
    np.testing.assert_array_equal(tasks[0].z_r, np.full(4, 2))


def _ogbench_eval_env(env_name: str) -> gym.Env:
    """The eval env as upstream main.py builds it. Replace with frozen.make_env after a-frozen."""
    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    env.unwrapped._add_noise_to_goal = False
    return env


def test_real_antmaze_env_gives_paired_start_states_and_full_length_failures():
    env_name = "antmaze-medium-navigate-v0"
    tasks = make_tasks(env_name, [1, 4], lambda _: np.zeros(128), _ogbench_eval_env)
    runs = []
    for _ in range(2):
        controller = _StillController(act_dim=8)
        records = run_episodes(controller, tasks, [0], 1, env_factory=_ogbench_eval_env)
        runs.append(controller.first_obs)
        assert [(r.success, r.steps) for r in records] == [(False, 1000)] * 2
    np.testing.assert_array_equal(np.stack(runs[0]), np.stack(runs[1]))
    np.testing.assert_allclose(tasks[0].goal_xy, [20.0, 20.0])
