import dataclasses
import itertools

import gymnasium as gym
import numpy as np
import ogbench
import pandas as pd
import pytest
from gymnasium.spaces import Box
from gymnasium.wrappers import TimeLimit

from fbplan import evaluate, frozen, train_high
from fbplan.contracts import TaskSpec
from fbplan.controllers import resolve_controller
from fbplan.controllers.oracle import MazeGeometry
from fbplan.evaluate import (
    HELDOUT_OFFSET,
    Checkpoint,
    config_hash,
    goal_latent,
    load_heldout_goals,
    main,
    make_tasks,
    parse_options,
    reset_seed,
    run_episode,
    run_episodes,
    run_parallel,
    select_heldout_goals,
    task_latent,
    write_csv,
)
from fbplan.learned import train_b
from fbplan.stats import CHECKPOINT_DEFAULTS, load_results

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


def test_run_parallel_with_one_episode_per_job_matches_sequential_run():
    tasks = [_task(1)]
    sequential = run_episodes(_GreedyController(), tasks, [3], 3, env_factory=_make_point_env)
    parallel = run_parallel(
        _GreedyController, tasks, [3], 3, workers=2, env_factory=_make_point_env,
        episodes_per_job=1,
    )
    assert _without_time(parallel) == _without_time(sequential)


def test_csv_round_trip_keeps_records(tmp_path):
    records = run_episodes(_GreedyController(), [_task(1)], [0], 2, env_factory=_make_point_env)
    path = tmp_path / "out" / "result.csv"
    write_csv(records, path)
    expected = pd.DataFrame([dataclasses.asdict(r) for r in records]).assign(**CHECKPOINT_DEFAULTS)
    pd.testing.assert_frame_equal(pd.read_csv(path, keep_default_na=False), expected)


def test_write_csv_adds_the_checkpoint_to_every_row(tmp_path):
    records = run_episodes(_GreedyController(), [_task(1)], [0], 2, env_factory=_make_point_env)
    write_csv(records, tmp_path / "provided.csv")
    write_csv(records, tmp_path / "b.csv", Checkpoint("b", "clean", 2))
    df = load_results([tmp_path / "provided.csv", tmp_path / "b.csv"])
    assert df[list(CHECKPOINT_DEFAULTS)].values.tolist() == (
        [["provided", "", -1]] * 2 + [["b", "clean", 2]] * 2
    )


def test_checkpoint_parse_reads_provided_retrained_and_b_specs():
    assert Checkpoint.parse("provided") == Checkpoint()
    assert Checkpoint.parse("retrained:3") == Checkpoint("retrained", "", 3)
    assert Checkpoint.parse("b:clean:2") == Checkpoint("b", "clean", 2)
    assert Checkpoint().columns() == CHECKPOINT_DEFAULTS


@pytest.mark.parametrize("spec", ["retrained", "retrained:x", "b:clean", "provided:1", "other:1"])
def test_checkpoint_parse_rejects_malformed_specs(spec):
    with pytest.raises(ValueError, match="checkpoint"):
        Checkpoint.parse(spec)


def test_checkpoint_load_calls_the_loader_of_its_kind(monkeypatch):
    calls = []
    monkeypatch.setattr(evaluate, "load_model", lambda env: calls.append(("provided", env)))
    monkeypatch.setattr(
        train_high, "load_baseline_model", lambda env, seed: calls.append(("retrained", env, seed))
    )
    monkeypatch.setattr(
        train_b, "load_b_model", lambda env, variant, seed: calls.append(("b", env, variant, seed))
    )
    for spec in ("provided", "retrained:3", "b:wide:1"):
        Checkpoint.parse(spec).load("env")
    assert calls == [("provided", "env"), ("retrained", "env", 3), ("b", "env", "wide", 1)]


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


def test_parse_options_reads_python_literals_and_bare_strings():
    options = parse_options(["lookahead=2", "rho_min=0.1", "agg=min", "label='a=b'"])
    assert options == {"lookahead": 2, "rho_min": 0.1, "agg": "min", "label": "a=b"}


def test_parse_options_rejects_items_without_equals_sign():
    with pytest.raises(ValueError, match="key=value"):
        parse_options(["lookahead"])


MEDIUM = "antmaze-medium-navigate-v0"


def test_heldout_goals_are_far_from_eval_goals_and_start_at_eval_task_distances():
    env = frozen.make_env(MEDIUM)
    maze, infos = MazeGeometry.from_env(env), env.unwrapped.task_infos
    to_eval_goal = [maze.bfs_steps(tuple(t["goal_ij"])) for t in infos]
    lengths = [steps[tuple(t["init_ij"])] for steps, t in zip(to_eval_goal, infos, strict=True)]
    goals = select_heldout_goals(env)
    assert [g.task_id for g in goals] == [HELDOUT_OFFSET + k for k in range(1, 11)]
    assert len({g.goal_ij for g in goals}) == len(goals)
    for g in goals:
        assert min(steps[g.goal_ij] for steps in to_eval_goal) >= 2
        assert min(lengths) <= maze.bfs_steps(g.goal_ij)[g.init_ij] <= max(lengths)
        np.testing.assert_array_equal(g.goal_xy, env.unwrapped.ij_to_xy(g.goal_ij))
    assert select_heldout_goals(frozen.make_env(MEDIUM)) == goals


@pytest.mark.parametrize(
    "env_name", [MEDIUM, "antmaze-large-navigate-v0", "antmaze-giant-navigate-v0"]
)
def test_saved_heldout_goals_equal_the_selection(env_name):
    assert load_heldout_goals(env_name) == select_heldout_goals(frozen.make_env(env_name))


def test_real_env_runs_heldout_goals_from_their_start_cells():
    goals = load_heldout_goals(MEDIUM)[:2]
    tasks = make_tasks(MEDIUM, [g.task_id for g in goals], lambda _: np.zeros(128))
    controller = _StillController(act_dim=8)
    records = run_episodes(controller, tasks, [0], 1)
    assert [(r.task_id, r.steps) for r in records] == [(g.task_id, 1000) for g in goals]
    env = frozen.make_env(MEDIUM).unwrapped
    for goal, task, obs in zip(goals, tasks, controller.first_obs, strict=True):
        np.testing.assert_array_equal(task.goal_xy, goal.goal_xy)
        assert np.linalg.norm(obs[:2] - env.ij_to_xy(goal.init_ij)) < 2.0


@pytest.fixture(scope="module")
def medium_model() -> frozen.FrozenFBModel:
    if not (frozen.checkpoint_dir(MEDIUM) / "params.pkl").exists():
        pytest.skip("medium checkpoint absent: run `uv run python -m fbplan.frozen`")
    if not frozen.dataset_path(MEDIUM, "val").exists():
        pytest.skip("medium val split absent: call fbplan.frozen.load_dataset once")
    return frozen.load_model(MEDIUM)


@pytest.mark.parametrize("method", ["baseline", "baseline_raw", "nonhier", "oracle"])
def test_builtin_controllers_run_an_episode_on_the_real_model(medium_model, method):
    controller = resolve_controller(method)(medium_model)
    tasks = make_tasks(MEDIUM, [2], medium_model.infer_task_latent)
    (record,) = run_episodes(controller, tasks, [0], 1)
    assert (record.method, record.env_name, record.task_id) == (method, MEDIUM, 2)
    assert 1 <= record.steps <= 1000


def test_raw_baseline_equals_baseline_for_a_task_latent_of_norm_sqrt_d(medium_model):
    obs = frozen.load_dataset(MEDIUM, "val")["observations"][:: 50_000][:4]
    z = np.random.default_rng(0).normal(size=medium_model.latent_dim)
    task = TaskSpec(MEDIUM, 1, z / np.linalg.norm(z) * np.sqrt(z.size), np.zeros(2))
    baseline, raw = (resolve_controller(m)(medium_model) for m in ("baseline", "baseline_raw"))
    for controller in (baseline, raw):
        controller.reset(task, seed=0)
    for o in obs:
        np.testing.assert_allclose(raw.act(o), baseline.act(o), atol=1e-5)


def test_task_latent_infers_eval_tasks_upstream_and_heldout_goals_the_same_way(medium_model):
    info = frozen.make_env(MEDIUM).unwrapped.task_infos[1]
    z_task = medium_model.infer_task_latent(2)
    np.testing.assert_array_equal(goal_latent(medium_model, tuple(info["goal_ij"])), z_task)
    np.testing.assert_array_equal(task_latent(medium_model, 2), z_task)
    goal = load_heldout_goals(MEDIUM)[0]
    np.testing.assert_array_equal(
        task_latent(medium_model, goal.task_id), goal_latent(medium_model, goal.goal_ij)
    )


def test_cli_runs_a_heldout_goal_with_a_retrained_checkpoint(medium_model, tmp_path):
    if not (train_high.baseline_dir(MEDIUM, 0) / "params.pkl").exists():
        pytest.skip("retrained medium baseline seed 0 absent")
    out = tmp_path / "retrained.csv"
    task_id = HELDOUT_OFFSET + 1
    argv = ["--method", "baseline", "--checkpoint", "retrained:0", "--tasks", str(task_id)]
    main([*argv, "--episodes", "1", "--out", str(out)])
    df = load_results([out])
    assert df[["task_id", *CHECKPOINT_DEFAULTS]].values.tolist() == [[task_id, "retrained", "", 0]]


def test_cli_writes_one_record_per_episode(medium_model, tmp_path):
    out = tmp_path / "nonhier.csv"
    main(["--method", "nonhier", "--tasks", "2", "--episodes", "2", "--out", str(out)])
    df = pd.read_csv(out)
    assert df[["task_id", "eval_seed", "episode"]].values.tolist() == [[2, 0, 0], [2, 0, 1]]
