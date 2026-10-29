"""Approach A: plan over a landmark graph, then follow one waypoint at a time.

Once per env: the rho table between landmarks. Once per task: V_H by value iteration
(fbplan.graph). At a replan, every landmark w is scored from the current state s with
Q(s, w) = R(s, w) + hop_discount * rho(s, w) * V_H(w) over trusted edges, and the best one
becomes the waypoint w*, unless the task policy has a higher value F(s, z_r)^T z_r.
Edges from landmarks and from s follow the same trust rules (see PlannerConfig).
The controller keeps w* until it is reached (see PlannerConfig.reached_xy) or until a
timeout, and acts with pi_l(a | s, B(w*)) (or pi_l(a | s, z_r) on the task policy).
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass, replace
from functools import cached_property
from typing import TypeVar

import numpy as np

from fbplan.contracts import FBModel, TaskSpec
from fbplan.frozen import load_dataset
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

    An edge (u, w) is trusted when rho_min <= rho(u, w) < rho_max and the xy distance from u
    to w is at most max_edge_xy, then top_k keeps the largest rho per source.
    rho_min: rho = gamma^k is about k steps.
    rho_max: rho >= 1 is an FB error (see graph.trusted_edges); above 1 trusts it (ablation).
    max_edge_xy: guard against wormholes with rho < 1. obs[:2] is the agent's xy position,
        which the agent observes. 6 = 1.5 maze cells in OGBench AntMaze. None: no cap.
    two_way: use min(rho(u, w), rho(w, u)) as the edge discount, an FB-only wormhole guard
        (dynamics in the maze are nearly reversible; FB errors often are not).
    top_k: keep at most top_k trusted out-edges per source (None: no limit).
    hop_discount: factor on rho per hop; below 1 it is a cost per hop.
    reached_xy: the waypoint is reached when the agent is within reached_xy xy units of it.
        None: use rho(s, w*) >= rho_reached instead. The xy rule is the default because on
        medium FB rho near a waypoint peaks at 0.88-0.90 (within 0.5 xy units) and passes
        0.9 elsewhere about 2.3 units away; in a 30-episode probe the planner succeeded in
        23/30 episodes with the xy rule and 10/30 with the rho rule.
    rho_reached: threshold of the rho rule.
    commit_timeout: replan after this many steps on one waypoint or on the task policy.
    """

    landmarks: LandmarkConfig = field(default_factory=LandmarkConfig)
    options: OptionModelConfig = field(default_factory=OptionModelConfig)
    rho_min: float = 0.3
    rho_max: float = 1.0
    max_edge_xy: float | None = 6.0
    two_way: bool = False
    top_k: int | None = None
    hop_discount: float = 1.0
    reached_xy: float | None = 1.5
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
    def edge_rho(self) -> np.ndarray:
        """Edge discounts between landmarks [N, N]: rho, or its two-way min."""
        rho = self.edges.rho
        return np.minimum(rho, rho.T) if self.config.two_way else rho

    @cached_property
    def trusted(self) -> np.ndarray:
        """Trusted landmark-to-landmark edges [N, N], without self-loops."""
        return self._trusted(self.edge_rho, self.landmarks, exclude_diagonal=True)

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
            self.edge_rho,
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
        return bool(self._reached(obs, np.array([self._waypoint]))[0])

    def _reached(
        self, obs: np.ndarray, idx: np.ndarray, rho: np.ndarray | None = None
    ) -> np.ndarray:
        """Which landmarks idx count as reached from obs. The rho rule computes rho(obs, idx)
        unless it is given."""
        cfg = self.config
        if cfg.reached_xy is not None:
            return np.linalg.norm(self.landmarks[idx, :2] - obs[:2], axis=-1) <= cfg.reached_xy
        if rho is None:
            rho = self.options.pair_table(obs[None], self.targets.take(idx)).rho[0]
        return rho >= cfg.rho_reached

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
        rho = row.rho
        if cfg.two_way:  # rho(w, s) for every landmark w, with s as the target
            back = self.options.pair_table(self.landmarks, self.options.targets(obs[None]))
            rho = np.minimum(rho, back.rho.T)
        reached = self._reached(obs, np.arange(len(self.landmarks)), row.rho[0])
        candidates = np.where(reached, -np.inf, rho)
        mask = self._trusted(candidates, obs[None])
        q = q_values(rho, row.reward, plan.solution.values, mask, cfg.hop_discount)[0]
        if q.max() <= self.options.direct_value(obs[None], plan.z_r)[0]:
            return DIRECT, np.nan
        best = int(q.argmax())
        return best, float(row.rho[0, best])

    def _trusted(
        self, rho: np.ndarray, sources: np.ndarray, exclude_diagonal: bool = False
    ) -> np.ndarray:
        """Trust rules of PlannerConfig for edges from source observations to all landmarks."""
        cfg = self.config
        if cfg.max_edge_xy is not None:
            dist = np.linalg.norm(sources[:, None, :2] - self.landmarks[None, :, :2], axis=-1)
            rho = np.where(dist <= cfg.max_edge_xy, rho, -np.inf)
        return trusted_edges(rho, cfg.rho_min, cfg.top_k, exclude_diagonal, cfg.rho_max)


class GraphPlusHighController(GraphPlannerController):
    """Same route, but B(w*) is the task of the baseline pi_h, which then picks the intention."""

    name = "graph_plus_high"

    def _action(self, obs: np.ndarray, z: np.ndarray) -> np.ndarray:
        intention = self.fb.high_latent(obs[None], z[None])
        return np.asarray(self.fb.low_action(obs[None], intention))[0]


C = TypeVar("C", bound=GraphPlannerController)


def config_from_options(options: dict[str, object]) -> PlannerConfig:
    """PlannerConfig from flat CLI options; a dotted key ("landmarks.count") sets a nested field.

    Unknown keys raise TypeError, so that a typo cannot silently fall back to a default.
    """
    base = PlannerConfig()
    top: dict[str, object] = {}
    groups: dict[str, dict[str, object]] = {}
    for key, value in options.items():
        group, _, name = key.rpartition(".")
        if group:
            groups.setdefault(group, {})[name] = value
        else:
            top[key] = value
    for group, values in groups.items():
        nested = getattr(base, group, None)
        if not is_dataclass(nested):
            raise TypeError(f"unknown option group {group!r}")
        top[group] = replace(nested, **values)
    return replace(base, **top)


def make_planner(model: FBModel, **options: object) -> GraphPlannerController:
    """Controller factory for the evaluation CLI: landmarks from the env's training data."""
    return _from_train_data(GraphPlannerController, model, options)


def make_planner_high(model: FBModel, **options: object) -> GraphPlusHighController:
    """Controller factory for the evaluation CLI, with pi_h after the waypoint."""
    return _from_train_data(GraphPlusHighController, model, options)


def _from_train_data(cls: type[C], model: FBModel, options: dict[str, object]) -> C:
    observations = load_dataset(model.env_name, "train")["observations"]
    return cls.from_dataset(model, observations, config_from_options(options))
