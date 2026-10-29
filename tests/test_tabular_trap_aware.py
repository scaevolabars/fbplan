"""The trap-aware controller (fbplan/controllers/trap_aware.py) on the tabular teleport maze.

The fake FB is exact (full rank, no noise, pi_l executes the optimal policy of its latent), and
its pi_h returns z_task. So the controller's intention is normalize(z_r - weight * gate * z_trap)
and every quantity has a ground truth: the trap cell, the walk that avoids the entries, and the
success of each policy under the time limit (docs/theory.md §11.1).
"""

from collections import deque

import numpy as np
import pytest

from fbplan.contracts import normalize_z
from fbplan.controllers.baseline import BaselineController
from fbplan.controllers.trap_aware import (
    TrapAwareConfig,
    TrapAwareController,
    trap_latent,
    trap_mask,
)
from fbplan.tabular.fake_fb import FakeFBConfig, TabularFB
from fbplan.tabular.maze import GridMaze

GAMMA = 0.95
TRAP_CELL = (1, 7)
LENGTH = 120  # steps per offline trajectory
JUMP_XY = 3.0  # 1.5 fine steps at resolution 2 (fine step 2 xy units)
WALKS = {1: 41, 2: 28, 3: 24, 4: 46, 5: 47}  # fine steps of the walk that enters no teleporter


class _TaskHigh(TabularFB):
    """Exact fake whose pi_h returns z_task."""

    def high_latent(self, obs, z_task):
        return np.asarray(normalize_z(np.asarray(z_task, dtype=np.float64), self.latent_dim))


@pytest.fixture(scope="module")
def model() -> _TaskHigh:
    config = FakeFBConfig(maze="teleport", resolution=2, discount=GAMMA, low_policy="optimal")
    return _TaskHigh(config)


@pytest.fixture(scope="module")
def maze(model) -> GridMaze:
    return model.maze


@pytest.fixture(scope="module")
def trap(maze) -> np.ndarray:
    return maze.cell_states(TRAP_CELL)


@pytest.fixture(scope="module")
def data(maze, trap) -> dict[str, np.ndarray]:
    return _offline_data(maze, trap, np.random.default_rng(0))


def _config(**options) -> TrapAwareConfig:
    return TrapAwareConfig(**{"jump_xy": JUMP_XY, "goal_states": 16, **options})


def _offline_data(maze: GridMaze, trap: np.ndarray, rng, n_traj: int = 150, eps: float = 0.2):
    """A noisy expert that walks shortest paths to random goals and ignores the teleporters, as
    OGBench's data collector does: its paths go through the entries, and some jumps end in the
    trap. Arrays as load_dataset returns them (observations are the xy of the states)."""
    dist = maze.geodesic_distances()
    index = {tuple(rc): s for s, rc in enumerate(maze.fine_rc)}
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # (d_row, d_col) of action ids 1..4
    goals = np.setdiff1d(np.arange(maze.n_states), np.r_[maze.entry_states, trap])
    obs, next_obs, terminals = [], [], []
    for _ in range(n_traj):
        s, goal = (int(rng.choice(goals)) for _ in range(2))
        for t in range(LENGTH):
            if s == goal:
                goal = int(rng.choice(goals))
            closer = []
            for a, (dr, dc) in enumerate(moves, start=1):
                target = index.get((maze.fine_rc[s, 0] + dr, maze.fine_rc[s, 1] + dc))
                if target is not None and dist[target, goal] < dist[s, goal]:
                    closer.append(a)
            a = int(rng.integers(1, 5)) if rng.random() < eps or not closer else rng.choice(closer)
            nxt = _step(maze, s, a, rng)
            obs.append(maze.xy[s])
            next_obs.append(maze.xy[nxt])
            terminals.append(t == LENGTH - 1)
            s = nxt
    return {
        "observations": np.array(obs),
        "next_observations": np.array(next_obs),
        "terminals": np.array(terminals, dtype=np.float32),
    }


def _step(maze: GridMaze, s: int, a: int, rng) -> int:
    k = rng.choice(maze.next_states.shape[-1], p=maze.probs[s, a])
    return int(maze.next_states[s, a, k])


def _rollouts(model, controller, task_id, horizon, n, seed=0) -> tuple[float, float]:
    """(success rate, trap rate) of n episodes from the task's start cell."""
    maze, rng = model.maze, np.random.default_rng(seed)
    init_ij, goal_ij = maze.tasks[task_id - 1]
    goal, trap = set(maze.cell_states(goal_ij).tolist()), set(maze.cell_states(TRAP_CELL).tolist())
    task = model.task_spec(task_id)
    success = trapped = 0
    for _ in range(n):
        controller.reset(task, seed=0)
        s = maze.centre_state(init_ij)
        for _ in range(horizon):
            a = int(maze.action_from_vector(controller.act(maze.xy[s])[None])[0])
            s = _step(maze, s, a, rng)
            if s in goal:
                success += 1
                break
        trapped += s in trap
    return success / n, trapped / n


def _safe_walk(maze: GridMaze, start: int, goal_ij) -> int:
    """Fewest fine steps from start to the goal cell without entering a teleport entry."""
    blocked, adjacency = set(maze.entry_states.tolist()), maze.neighbours()
    steps, queue = {start: 0}, deque([start])
    while queue:
        u = queue.popleft()
        for v in adjacency[u]:
            if v not in steps and v not in blocked:
                steps[v] = steps[u] + 1
                queue.append(v)
    return min(steps.get(int(g), np.inf) for g in maze.cell_states(goal_ij))


def _trajectories(*paths):
    """Flat (xy, next_xy, terminals) of paths of positions (the last one is a next state only)."""
    xy = np.concatenate([np.asarray(p[:-1], dtype=float) for p in paths])
    next_xy = np.concatenate([np.asarray(p[1:], dtype=float) for p in paths])
    terminals = np.concatenate([np.arange(len(p) - 1) == len(p) - 2 for p in paths])
    return xy, next_xy, terminals


def test_trap_mask_marks_the_states_after_the_last_jump_when_they_stay_close():
    xy, next_xy, terminals = _trajectories(
        [(0, 0), (0.1, 0), (0.2, 0), (0.3, 0)],  # no jump
        [(0, 0), (0.1, 0), (5, 5), (5.1, 5), (5, 5.1), (5, 5)],  # jump, then stays
        [(0, 0), (5, 5), (6, 5), (7, 5), (8, 5)],  # jump, then walks away
        [(0, 0), (5, 5), (5.1, 5), (9, 9), (9.1, 9), (9, 9)],  # two jumps, stays after the last
        [(0, 0), (0.1, 0), (5, 5)],  # jump at the last step: no state after it
    )
    mask = trap_mask(xy, next_xy, terminals, jump_xy=2.0, radius=0.5)
    expected = [0, 0, 0] + [0, 0, 1, 1, 1] + [0, 0, 0, 0] + [0, 0, 0, 1, 1] + [0, 0]
    np.testing.assert_array_equal(mask, np.array(expected, dtype=bool))


def test_trap_mask_finds_the_tabular_trap_from_the_data_alone(maze, trap, data):
    mask = trap_mask(data["observations"], data["next_observations"], data["terminals"],
                     jump_xy=JUMP_XY, radius=2.5)
    states = maze.state_index(data["observations"]).reshape(-1, LENGTH)
    final = maze.state_index(data["next_observations"]).reshape(-1, LENGTH)[:, -1]
    ends_in_trap = np.isin(final, trap)
    flagged = mask.reshape(-1, LENGTH).any(axis=1)
    assert ends_in_trap.sum() >= 20
    # In a trajectory that ends in the trap, the marked states are exactly its trap states.
    np.testing.assert_array_equal(mask.reshape(-1, LENGTH)[ends_in_trap],
                                  np.isin(states[ends_in_trap], trap))
    # Few false alarms: trajectories that jump to a good exit near their end.
    assert (flagged & ~ends_in_trap).sum() <= 0.1 * flagged.sum()


def test_minus_the_trap_latent_is_a_penalty_on_the_trap_states(model, trap, data):
    mask = trap_mask(data["observations"], data["next_observations"], data["terminals"],
                     jump_xy=JUMP_XY, radius=2.5)
    z_trap = trap_latent(model, data["observations"], mask)
    assert np.linalg.norm(z_trap) == pytest.approx(np.sqrt(model.latent_dim))
    reward = model.reward_of_latent(-z_trap)  # exact in full rank
    assert np.all(reward[trap] < 0)
    assert np.all(np.delete(reward, trap) >= -1e-8)  # B is kept in float32; rewards are ~1e-3


@pytest.mark.parametrize("task_id", sorted(WALKS))
def test_the_gate_counts_the_steps_of_the_walk_that_avoids_the_entries(model, maze, data, task_id):
    controller = TrapAwareController.from_data(model, data, _config(weight=1.0))
    controller.reset(model.task_spec(task_id), seed=0)
    init_ij, goal_ij = maze.tasks[task_id - 1]
    start = maze.centre_state(init_ij)
    assert _safe_walk(maze, start, goal_ij) == WALKS[task_id]
    assert controller.walk_steps(maze.xy[start]) == pytest.approx(WALKS[task_id], abs=1e-6)


@pytest.mark.parametrize("options", [{"weight": 0.0}, {"weight": 2.0, "time_limit": 1}])
def test_without_a_weight_or_with_the_gate_closed_it_acts_as_the_baseline(model, maze, data,
                                                                         options):
    controller = TrapAwareController.from_data(model, data, _config(**options))
    baseline = BaselineController(model)
    for task_id in (1, 5):
        task = model.task_spec(task_id)
        for s in range(maze.n_states):
            controller.reset(task, seed=0)
            baseline.reset(task, seed=0)
            obs = maze.xy[s]
            if options["weight"] > 0 and controller.walk_steps(obs) <= 1:
                continue  # the gate is open next to the goal
            np.testing.assert_array_equal(controller.act(obs), baseline.act(obs))


@pytest.mark.parametrize("task_id", sorted(WALKS))
def test_where_the_walk_fits_it_walks_and_the_baseline_jumps(model, data, task_id):
    horizon = 60  # every walk fits
    controller = TrapAwareController.from_data(model, data, _config(time_limit=horizon))
    success, trapped = _rollouts(model, controller, task_id, horizon, n=40)
    assert (success, trapped) == (1.0, 0.0)
    _, baseline_trapped = _rollouts(model, BaselineController(model), task_id, horizon, n=40)
    assert baseline_trapped >= 0.25


def test_where_the_walk_does_not_fit_the_gate_hands_back_to_the_baseline(model, data):
    task_id, horizon = 5, 40  # the walk takes 47 steps; the jump can still succeed
    ungated = TrapAwareController.from_data(model, data, _config(gate="none"))
    gated = TrapAwareController.from_data(model, data, _config(time_limit=horizon))
    baseline = BaselineController(model)
    assert _rollouts(model, ungated, task_id, horizon, n=100)[0] == 0.0
    gated_success, _ = _rollouts(model, gated, task_id, horizon, n=200)
    baseline_success, _ = _rollouts(model, baseline, task_id, horizon, n=200)
    assert baseline_success >= 0.25
    assert gated_success >= baseline_success - 0.1


def test_data_without_a_trap_is_rejected(model, data):
    with pytest.raises(ValueError, match="no trap"):
        TrapAwareController.from_data(model, data, _config(jump_xy=100.0))


@pytest.mark.parametrize(
    "options",
    [{"weight": -1.0}, {"gate": "always"}, {"trap_radius": 0.0}, {"time_limit": 0},
     {"jump_xy": 0.0}, {"goal_states": 0}, {"pool_size": 0}],
)
def test_bad_settings_are_rejected(options):
    with pytest.raises(ValueError):
        TrapAwareConfig(**options)
