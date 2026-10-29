"""Approach A: plan over a landmark graph, then follow one waypoint at a time.

Once per env: the rho table between landmarks. Once per task: V_H by value iteration
(fbplan.graph). At a replan, every landmark w is scored from the current state s with
Q(s, w) = R(s, w) + hop_discount * rho(s, w) * V_H(w) over trusted edges, and the best one
becomes the waypoint w*, unless the task policy has a higher value F(s, z_r)^T z_r.
The controller keeps w* until FB says it is reached (rho(s, w*) >= rho_reached) or until
a timeout, and acts with pi_l(a | s, B(w*)) (or pi_l(a | s, z_r) on the task policy).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property

import numpy as np

from fbplan.contracts import FBModel, TaskSpec
from fbplan.graph import (
    DIRECT,
    GraphSolution,
    extract_route,
    q_values,
    trusted_edges,
    value_iteration,
)
from fbplan.landmarks import LandmarkConfig, select_landmarks
from fbplan.option_model import OptionModel, OptionModelConfig, OptionTable


@dataclass(frozen=True)
class PlannerConfig:
    """Every planner value in the report comes from this config.

    rho_min: trust an edge when rho >= rho_min (rho = gamma^k is about k steps).
    top_k: keep at most top_k trusted out-edges per source (None: no limit).
    hop_discount: factor on rho per hop; below 1 it is a cost per hop.
    rho_reached: the waypoint is reached when rho(s, w*) >= rho_reached.
    commit_timeout: replan after this many steps on one waypoint or on the task policy.
    """

    landmarks: LandmarkConfig = field(default_factory=LandmarkConfig)
    options: OptionModelConfig = field(default_factory=OptionModelConfig)
    rho_min: float = 0.3
    top_k: int | None = None
    hop_discount: float = 1.0
    rho_reached: float = 0.9
    commit_timeout: int = 100
    vi_max_iters: int = 1000
    vi_tol: float = 1e-8


@dataclass(frozen=True)
class TaskPlan:
    """The graph solved for one task."""

    z_r: np.ndarray  # [d]
    reward: np.ndarray | None  # R between landmarks [N, N]; None for edge_reward "zero"
    direct: np.ndarray  # F(l, z_r)^T z_r at each landmark [N]
    solution: GraphSolution


class GraphPlannerController:
    """Landmark graph planner; acts with pi_l on the current waypoint."""

    name = "graph_planner"

    def __init__(
        self, fb: FBModel, landmarks: np.ndarray, config: PlannerConfig | None = None
    ) -> None:
        self.fb = fb
        self.config = config or PlannerConfig()
        self.options = OptionModel(fb, self.config.options)
        self.landmarks = landmarks
        self.targets = self.options.targets(landmarks)
        self.plan: TaskPlan | None = None
        self._edges: OptionTable | None = None
        self._plans: dict[bytes, TaskPlan] = {}
        self._waypoint: int | None = None  # None: replan at the next step
        self._steps_on_waypoint = 0
        self._replans = 0
        self._rho_waypoint = np.nan

    @classmethod
    def from_dataset(
        cls, fb: FBModel, observations: np.ndarray, config: PlannerConfig | None = None
    ) -> GraphPlannerController:
        """Select landmarks from dataset observations with config.landmarks."""
        config = config or PlannerConfig()
        return cls(fb, select_landmarks(observations, config.landmarks, fb).observations, config)

    @property
    def edges(self) -> OptionTable:
        """rho between landmarks. It does not depend on the task, so it is computed once."""
        if self._edges is None:
            self._edges = self.options.pair_table(self.landmarks, self.targets)
        return self._edges

    @cached_property
    def trusted(self) -> np.ndarray:
        """Trusted landmark-to-landmark edges [N, N], without self-loops."""
        cfg = self.config
        return trusted_edges(self.edges.rho, cfg.rho_min, cfg.top_k, exclude_diagonal=True)

    def reset(self, task: TaskSpec, seed: int) -> None:
        """Solve the graph for the task (cached per task). The policy is deterministic."""
        z_r = np.asarray(task.z_r, dtype=np.float64)
        key = z_r.tobytes()
        if key not in self._plans:
            self._plans[key] = self._solve(z_r)
        self.plan = self._plans[key]
        self._waypoint, self._steps_on_waypoint, self._replans = None, 0, 0
        self._rho_waypoint = np.nan

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self._needs_replan(obs):
            self._replan(obs)
        self._steps_on_waypoint += 1
        z = self.plan.z_r if self._waypoint == DIRECT else self.targets.z[self._waypoint]
        return self._action(obs, z)

    def step_info(self) -> dict[str, float | int | str]:
        return {
            "waypoint": DIRECT if self._waypoint is None else self._waypoint,
            "replans": self._replans,
            "rho_waypoint": self._rho_waypoint,
        }

    def _action(self, obs: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.asarray(self.fb.low_action(obs[None], z[None]))[0]

    def _solve(self, z_r: np.ndarray) -> TaskPlan:
        reward = None
        if self.config.options.edge_reward != "zero":
            table = self.options.pair_table(self.landmarks, self.targets, z_r)
            if self._edges is None:  # the same pass gives rho; keep it for later tasks
                self._edges = replace(table, reward=None)
            reward = table.reward
        direct = self.options.direct_value(self.landmarks, z_r)
        cfg = self.config
        solution = value_iteration(
            self.edges.rho,
            reward,
            direct,
            self.trusted,
            cfg.hop_discount,
            cfg.vi_max_iters,
            cfg.vi_tol,
        )
        return TaskPlan(z_r, reward, direct, solution)

    def _needs_replan(self, obs: np.ndarray) -> bool:
        if self._waypoint is None or self._steps_on_waypoint >= self.config.commit_timeout:
            return True
        if self._waypoint == DIRECT:
            return False
        waypoint = self.targets.take(np.array([self._waypoint]))
        self._rho_waypoint = float(self.options.pair_table(obs[None], waypoint).rho[0, 0])
        return self._rho_waypoint >= self.config.rho_reached

    def route(self, obs: np.ndarray) -> list[int]:
        """Landmark ids that the plan visits from obs, in order (diagnostics and plots)."""
        first, _ = self._choose(obs)
        return [] if first == DIRECT else extract_route(self.plan.solution, first)

    def _replan(self, obs: np.ndarray) -> None:
        self._waypoint, self._rho_waypoint = self._choose(obs)
        self._steps_on_waypoint = 0
        self._replans += 1

    def _choose(self, obs: np.ndarray) -> tuple[int, float]:
        """Best waypoint from obs and its rho, or (DIRECT, nan).

        Landmarks that count as reached are excluded: the landmark under the agent has
        Q = rho * V_H ~ V_H, and choosing it again would stall the agent.
        """
        cfg, plan = self.config, self.plan
        row = self.options.pair_table(obs[None], self.targets, plan.z_r)
        candidates = np.where(row.rho >= cfg.rho_reached, -np.inf, row.rho)
        mask = trusted_edges(candidates, cfg.rho_min, cfg.top_k)
        q = q_values(row.rho, row.reward, plan.solution.values, mask, cfg.hop_discount)[0]
        if q.max() <= self.options.direct_value(obs[None], plan.z_r)[0]:
            return DIRECT, np.nan
        best = int(q.argmax())
        return best, float(row.rho[0, best])


class GraphPlusHighController(GraphPlannerController):
    """Same route, but B(w*) is the task of the baseline pi_h, which then picks the intention."""

    name = "graph_plus_high"

    def _action(self, obs: np.ndarray, z: np.ndarray) -> np.ndarray:
        intention = self.fb.high_latent(obs[None], z[None])
        return np.asarray(self.fb.low_action(obs[None], intention))[0]
