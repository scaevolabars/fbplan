"""Single switch vs. chained switches on the tabular fake FB, with exact evaluation.

The planners use only FB quantities (F, B, pi_l, pi_h through the `FBModel` methods); the
evaluation uses the exact maze. Landmarks are states. For one task latent z_r:

    rho_hat(s, w) = F(s, z_w)^T B(w) / F(w, z_w)^T B(w),  clipped to [0, gamma] for s != w
    R_hat(s, w)   = F(s, z_w)^T z_r - rho_hat(s, w) F(w, z_w)^T z_r     ("full")
                    F(s, z_w)^T z_r                                     ("proxy", as upstream)
                    0                                                   ("zero": shortest path)
    v_d(s)        = F(s, z_r)^T z_r

Planner kinds:
    direct    pi_l(z_r) only.
    baseline  the fake's idealised pi_h (argmax of the upstream proxy advantage), then pi_l.
    single    best single switch: Q(s, w) = R_hat(s, w) + rho_hat(s, w) v_d(w).
    chained   Q(s, w) = R_hat(s, w) + rho_hat(s, w) V_H(w), V_H from graph value iteration.
For single and chained, only edges with rho_hat >= rho_min are used, each hop costs hop_cost,
and the agent acts with pi_l(z_r) when no option beats v_d(s). By default the choice is redone
at every step; with `commit` the agent keeps w* until it is reached or a timeout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from fbplan.contracts import normalize_z
from fbplan.tabular.fake_fb import TabularFB
from fbplan.tabular.options import graph_value_iteration, option_values


@dataclass(frozen=True)
class PlannerConfig:
    """One planner variant. Only `single` and `chained` read the fields after `kind`."""

    name: str
    kind: Literal["direct", "baseline", "single", "chained"] = "chained"
    aggregate: Literal["mean", "min"] = "mean"  # how the F heads are combined
    rho_min: float = 0.0
    hop_cost: float = 0.0
    reward: Literal["full", "proxy", "zero"] = "full"  # the R_hat variant, see the module doc
    commit: bool = False


@dataclass(frozen=True)
class Plan:
    """Option choice per state for one task."""

    choice: np.ndarray  # [n] landmark column, or -1 for "act with z_r"
    value: np.ndarray  # [n] predicted value max(v_d, max_w Q)
    rho: np.ndarray  # [n, L] rho_hat used by the plan


class LandmarkModel:
    """F(s, B(w)) for all states s and landmarks w, computed once per fake model."""

    def __init__(self, model: TabularFB, landmarks: np.ndarray) -> None:
        self.model = model
        self.landmarks = np.asarray(landmarks)
        maze, n, n_land = model.maze, model.maze.n_states, len(landmarks)
        self._z_w = model.backward(maze.xy[self.landmarks])  # [L, d]
        obs, z = np.tile(maze.xy, (n_land, 1)), np.repeat(self._z_w, n, axis=0)  # landmark-major
        forward = model.forward(obs, z).reshape(model.config.n_heads, n_land, n, -1)
        self._forward = forward.transpose(0, 2, 1, 3)  # [H, n, L, d]
        self._m = np.einsum("hsld,ld->hsl", self._forward, self._z_w)  # F(s, z_w)^T B(w)
        self.option_actions = maze.action_from_vector(model.low_action(obs, z)).reshape(n_land, n)

    def estimates(
        self, z_r: np.ndarray, aggregate: str, reward: str = "full"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(rho_hat [n, L], R_hat [n, L], v_d [n]) for task latent z_r."""
        maze, cols = self.model.maze, np.arange(len(self.landmarks))
        v_w = np.einsum("hsld,d->hsl", self._forward, z_r)  # F(s, z_w)^T z_r
        f_r = self.model.forward(maze.xy, np.tile(z_r, (maze.n_states, 1)))
        v_d = f_r @ z_r  # [H, n]
        m = self._m
        if aggregate == "mean":  # F is linear in the heads, so average before the ratio
            m, v_w, v_d = m.mean(0, keepdims=True), v_w.mean(0, keepdims=True), v_d.mean(0)[None]
        rho = m / m[:, self.landmarks, cols][:, None, :]
        r_hat = {
            "full": v_w - rho * v_w[:, self.landmarks, cols][:, None, :],
            "proxy": v_w,
            "zero": np.zeros_like(v_w),
        }[reward]
        rho, r_hat, v_d = rho.min(0), r_hat.min(0), v_d.min(0)  # no-op after "mean"
        own = np.zeros_like(rho, dtype=bool)
        own[self.landmarks, cols] = True
        rho = np.where(own, 1.0, np.clip(rho, 0.0, self.model.discount))  # rho <= gamma, s != w
        return rho, r_hat, v_d


def goal_latent(model: TabularFB, goal: int) -> np.ndarray:
    """Task latent of the one-state goal `goal`, as infer_task_latent would give it."""
    return np.asarray(normalize_z(model.basis[goal], model.latent_dim))


def make_plan(lm: LandmarkModel, cfg: PlannerConfig, z_r: np.ndarray) -> Plan:
    """Option values and choices of a `single` or `chained` planner for task latent z_r."""
    rho, r_hat, v_d = lm.estimates(z_r, cfg.aggregate, cfg.reward)
    r_eff = np.where(rho >= cfg.rho_min, r_hat - cfg.hop_cost, -np.inf)
    land = lm.landmarks
    if cfg.kind == "chained":
        v_land = graph_value_iteration(rho[land], r_eff[land], v_d[land])
    elif cfg.kind == "single":
        v_land = v_d[land]
    else:
        raise ValueError(f"planner kind {cfg.kind!r} has no option plan")
    q = option_values(rho, r_eff, v_land, land)
    best = q.argmax(axis=1)
    best_q = q[np.arange(len(best)), best]
    choice = np.where(best_q > v_d, best, -1)
    return Plan(choice=choice, value=np.maximum(v_d, best_q), rho=rho)


def _direct_actions(lm: LandmarkModel, z_r: np.ndarray) -> np.ndarray:
    maze = lm.model.maze
    return maze.action_from_vector(lm.model.low_action(maze.xy, np.tile(z_r, (maze.n_states, 1))))


def plan_policy(lm: LandmarkModel, cfg: PlannerConfig, z_r: np.ndarray) -> np.ndarray:
    """Action id per state of a planner that re-plans at every step: [n]."""
    model, maze = lm.model, lm.model.maze
    if cfg.kind == "direct":
        return _direct_actions(lm, z_r)
    if cfg.kind == "baseline":
        z_task = np.tile(z_r, (maze.n_states, 1))
        return maze.action_from_vector(
            model.low_action(maze.xy, model.high_latent(maze.xy, z_task))
        )
    plan = make_plan(lm, cfg, z_r)
    states = np.arange(maze.n_states)
    options = lm.option_actions[np.maximum(plan.choice, 0), states]
    return np.where(plan.choice >= 0, options, _direct_actions(lm, z_r))


def plan_route(lm: LandmarkModel, cfg: PlannerConfig, z_r: np.ndarray, start: int) -> list[int]:
    """Landmark states the plan visits from `start` if every option succeeds."""
    plan = make_plan(lm, cfg, z_r)
    route: list[int] = []
    state = start
    while plan.choice[state] >= 0 and int(lm.landmarks[plan.choice[state]]) not in route:
        state = int(lm.landmarks[plan.choice[state]])
        route.append(state)
    return route


def wormhole_mask(rho_hat: np.ndarray, dist: np.ndarray, gamma: float) -> np.ndarray:
    """Pairs whose rho_hat is more than twice the physical bound gamma^dist (true rho <= it)."""
    return rho_hat > 2.0 * gamma**dist


def evaluate_goal_reaching(
    lm: LandmarkModel, cfg: PlannerConfig, goals: np.ndarray, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Success [n_starts, n_goals] and steps (horizon + 1 on failure) from every start state."""
    maze = lm.model.maze
    if maze.next_states.shape[-1] != 1:
        raise ValueError("exact rollouts need a deterministic maze")
    steps = np.empty((maze.n_states, len(goals)), dtype=np.int64)
    for k, goal in enumerate(goals):
        z_r = goal_latent(lm.model, int(goal))
        if cfg.commit:
            steps[:, k] = _committed_rollouts(lm, cfg, z_r, int(goal), horizon)
        else:
            steps[:, k] = _markov_rollouts(lm, plan_policy(lm, cfg, z_r), int(goal), horizon)
    return steps <= horizon, steps


def _markov_rollouts(lm: LandmarkModel, actions: np.ndarray, goal: int, horizon: int) -> np.ndarray:
    maze = lm.model.maze
    states = np.arange(maze.n_states)
    steps = np.where(states == goal, 0, horizon + 1)
    for t in range(1, horizon + 1):
        states = maze.next_states[states, actions[states], 0]
        steps = np.where((states == goal) & (steps > horizon), t, steps)
    return steps


def _committed_rollouts(
    lm: LandmarkModel, cfg: PlannerConfig, z_r: np.ndarray, goal: int, horizon: int
) -> np.ndarray:
    """Keep w* until it is reached or for twice its predicted duration log(rho)/log(gamma)."""
    maze, gamma = lm.model.maze, lm.model.discount
    plan, direct = make_plan(lm, cfg, z_r), _direct_actions(lm, z_r)
    states = np.arange(maze.n_states)
    target = np.full(maze.n_states, -1)
    deadline = np.zeros(maze.n_states)
    steps = np.where(states == goal, 0, horizon + 1)
    for t in range(horizon):
        reached = (target >= 0) & (states == lm.landmarks[np.maximum(target, 0)])
        renew = (target < 0) | reached | (t >= deadline)
        target = np.where(renew, plan.choice[states], target)
        rho = np.maximum(plan.rho[states, np.maximum(target, 0)], 1e-6)
        deadline = np.where(renew, t + 2 * np.log(rho) / np.log(gamma) + 2, deadline)
        option = lm.option_actions[np.maximum(target, 0), states]
        states = maze.next_states[states, np.where(target >= 0, option, direct[states]), 0]
        steps = np.where((states == goal) & (steps > horizon), t + 1, steps)
    return steps
