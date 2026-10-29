import numpy as np

from fbplan.contracts import TaskSpec
from fbplan.tabular.env import TabularMazeEnv, rollout
from fbplan.tabular.fake_fb import FakeFBConfig, TabularFB
from fbplan.tabular.maze import GridMaze


def test_reset_starts_inside_the_task_init_cell_and_reports_the_goal():
    maze = GridMaze("medium", resolution=2)
    env = TabularMazeEnv(maze)
    obs, info = env.reset(seed=0, options={"task_id": 1})
    assert maze.state_index(obs[None])[0] in maze.cell_states((1, 1))
    np.testing.assert_allclose(info["goal"], [20.0, 20.0])


def test_step_moves_by_one_fine_cell():
    maze = GridMaze("medium", resolution=2)
    env = TabularMazeEnv(maze)
    obs, _ = env.reset(seed=0, options={"task_id": 1})
    start = maze.state_index(obs[None])[0]
    obs, *_ = env.step(np.array([1.0, 0.0]))
    assert maze.state_index(obs[None])[0] == maze.next_states[start, 1, 0]


def test_reaching_the_goal_cell_terminates_with_success():
    maze = GridMaze("medium", resolution=1)
    env = TabularMazeEnv(maze)
    env.reset(seed=0, options={"task_id": 3})  # (5, 3) -> (4, 2), around the wall at (4, 3)
    path = [(0, 1), (-1, 0), (-1, 0), (0, -1), (0, -1), (1, 0)]  # via (6, 3), (6, 1), (4, 1)
    done = [env.step(np.array(a, dtype=float)) for a in path]
    assert not any(step[2] for step in done[:-1])
    _, reward, terminated, truncated, info = done[-1]
    assert terminated and not truncated
    assert info["success"] == 1.0 and reward == 1.0


def test_episode_is_truncated_at_the_step_limit():
    env = TabularMazeEnv(GridMaze("medium", resolution=2), max_episode_steps=3)
    env.reset(seed=0, options={"task_id": 1})
    flags = [env.step(np.zeros(2))[3] for _ in range(3)]
    assert flags == [False, False, True]


def test_teleport_outcomes_are_reproducible_with_the_reset_seed():
    maze = GridMaze("teleport", resolution=1)

    def exits(seed: int) -> list[int]:
        env = TabularMazeEnv(maze)
        out = []
        for _ in range(5):
            env.reset(seed=seed, options={"task_id": 3})  # starts at (5, 6), below entry (4, 6)
            obs, *_ = env.step(np.array([0.0, -1.0]))
            out.append(int(maze.state_index(obs[None])[0]))
            seed += 1
        return out

    assert exits(0) == exits(0)
    assert len(set(exits(0) + exits(10))) > 1


class _SubgoalController:
    """Acts with pi_l towards one fixed state."""

    name = "fixed-subgoal"

    def __init__(self, model: TabularFB, target: int) -> None:
        self.model = model
        self.z = model.backward(model.maze.xy[[target]])

    def reset(self, task: TaskSpec, seed: int) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.model.low_action(obs[None], self.z)[0]

    def step_info(self) -> dict[str, float | int | str]:
        return {}


def test_rollout_of_exact_low_policy_reaches_the_goal_in_geodesic_time():
    model = TabularFB(FakeFBConfig())
    maze = model.maze
    goal_cell = maze.cell_states(maze.tasks[0][1])
    controller = _SubgoalController(model, goal_cell[0])
    result = rollout(controller, model, task_id=1, seed=0, max_steps=200)
    start = result.states[0]
    assert result.success
    assert result.steps == maze.geodesic_distances()[start, goal_cell].min()
