"""Approach A: plan over a landmark graph, then follow one waypoint at a time.

Once per env: the rho table between landmarks. Once per task: V_H by value iteration
(fbplan.graph). At a replan, every landmark w is scored from the current state s with
Q(s, w) = R(s, w) + hop_discount * rho(s, w) * V_H(w) over trusted edges, and the best one
becomes the waypoint w*, unless the task policy has a higher value F(s, z_r)^T z_r.
Edges from landmarks and from s follow the same trust rule (trusted_edges).
The controller keeps w* until it is reached (see PlannerConfig.reached_xy) or until a
timeout, and acts with pi_l(a | s, z_w*) (or pi_l(a | s, z_r) on the task policy), where
z_w is the landmark's option latent: B(w) by default, or a region mean (fbplan.landmarks).
With lookahead, it moves on along the planned route as soon as the next stop is worth at
least as much from s as the current waypoint, so the agent does not slow down at each one.
When w* is reached, the controller replans, or with switch_steps it takes the plan's next
stop unless a replan saves more than switch_steps FB steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass, replace
from functools import cached_property
from typing import TypeVar

import numpy as np

from fbplan import graph
from fbplan.contracts import FBModel, TaskSpec
from fbplan.frozen import load_dataset
from fbplan.graph import (
    DIRECT,
    TIE_RTOL,
    GraphSolution,
    best_options,
    extract_route,
    q_values,
    value_iteration,
)
from fbplan.landmarks import LandmarkConfig, select_landmarks
from fbplan.option_model import OptionModel, OptionModelConfig, OptionTable


@dataclass(frozen=True)
class PlannerConfig:
    """Every planner value in the report comes from this config.

    landmarks, options: landmark selection (fbplan.landmarks) and the option model
        (fbplan.option_model).
    Trust (trusted_edges): an edge (u, w) is trusted when rho_min <= rho(u, w) < rho_max and
    the xy distance from u to w is at most max_edge_xy. Trust uses the raw FB estimate; values
    use rho clipped to [0, gamma] (see option_model).
    rho_min: rho = gamma^k is about k steps.
    rho_max: rho >= 1 is an FB error (see graph.trusted_edges); above 1 trusts it (ablation).
    max_edge_xy: guard against wormholes with rho < 1. obs[:2] is the agent's xy position,
        which the agent observes. 6 = 1.5 maze cells in OGBench AntMaze. None: no cap.
    two_way: use min(rho(u, w), rho(w, u)) as the edge discount, an FB-only wormhole guard
        (dynamics in the maze are nearly reversible; FB errors often are not).
    hop_discount: factor on rho per hop; below 1 it is a cost per hop.
    reached_xy: the waypoint is reached when the agent is within reached_xy xy units of it.
        None: use rho(s, w*) >= rho_reached instead. The xy rule is the default because on
        medium FB rho near a waypoint peaks at 0.88-0.90 (within 0.5 xy units) and passes
        0.9 elsewhere about 2.3 units away; in a 30-episode probe the planner succeeded in
        23/30 episodes with the xy rule and 10/30 with the rho rule. Keep it well below
        max_edge_xy: landmarks within reached_xy are excluded at a replan.
    rho_reached: threshold of the rho rule.
    lookahead: each step, compare the current waypoint w with the next stop w' on the route
        (best_next[w]: a landmark or the task policy) and move on when Q(s, w') >= Q(s, w)
        and the edge s -> w' is trusted. This is the replan comparison, restricted to the route.
        Defaults reached_xy = 3 with lookahead: on 100 held-out medium episodes (same start
        states for all) 79/100 successes, against 56/100 for reached_xy = 1.5 without it and
        62/100 for reached_xy = 3 without it (baseline pi_h: 84/100).
    switch_steps: route following at a reach. None: when w* is reached, rescore all
        landmarks (full replan). A number k: take the plan's next stop after w* instead
        (best_next, a landmark or the task policy), unless the full replan finds a choice
        worth more than gamma^-k times its Q, that is, one that saves more than k FB steps.
        While that stop is not trusted from s, the agent keeps heading for w* (until the
        timeout). A full replan from near w* often prefers another landmark:
        near the goal the max over many landmark values beats the one task value, and
        between two ways around a loop small FB errors flip the choice at each reach.
        inf: always follow the plan between timeouts.
    commit_timeout: replan after this many steps on one waypoint or on the task policy.
    replan_every_step: rescore all landmarks at every step instead of committing to w*
        (design.md §4.A). With exact values this is never worse (option interruption);
        commitment guards against noisy values.
    vi_max_iters, vi_tol: value iteration stops at relative change vi_tol; a task whose
        value iteration has not converged after vi_max_iters sweeps raises an error.
    """

    landmarks: LandmarkConfig = field(default_factory=LandmarkConfig)
    options: OptionModelConfig = field(default_factory=OptionModelConfig)
    rho_min: float = 0.3
    rho_max: float = 1.0
    max_edge_xy: float | None = 6.0
    two_way: bool = False
    hop_discount: float = 1.0
    reached_xy: float | None = 3.0
    rho_reached: float = 0.9
    lookahead: bool = True
    switch_steps: float | None = None
    commit_timeout: int = 100
    replan_every_step: bool = False
    vi_max_iters: int = 1000
    vi_tol: float = 1e-8

    def __post_init__(self) -> None:
        if self.switch_steps is not None and self.switch_steps < 0:
            raise ValueError("switch_steps must be >= 0 or None")


def trusted_edges(
    rho_raw: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    config: PlannerConfig | None = None,
    exclude_diagonal: bool = False,
) -> np.ndarray:
    """The planner's trust rule for edges from observations sources [S, obs_dim] to targets
    [T, obs_dim]: rho_min <= rho_raw < rho_max and xy distance <= max_edge_xy. [S, T] bool

    rho_raw is the FB estimate before clipping (with two_way, the min over both directions);
    graph.trusted_edges explains the bounds. The xy cap guards against wormholes with rho < 1.
    exclude_diagonal drops self-loops when sources and targets are the same list. config
    defaults to PlannerConfig().
    """
    cfg = config or PlannerConfig()
    trusted = graph.trusted_edges(rho_raw, cfg.rho_min, exclude_diagonal, cfg.rho_max)
    if cfg.max_edge_xy is not None:
        dist = np.linalg.norm(sources[:, None, :2] - targets[None, :, :2], axis=-1)
        trusted &= dist <= cfg.max_edge_xy
    return trusted


@dataclass(frozen=True)
class TaskPlan:
    """The graph solved for one task."""

    z_r: np.ndarray  # [d]
    solution: GraphSolution


class GraphPlannerController:
    """Landmark graph planner; acts with pi_l on the current waypoint."""

    name = "graph_planner"

    def __init__(
        self,
        fb: FBModel,
        landmarks: np.ndarray,
        config: PlannerConfig | None = None,
        latents: np.ndarray | None = None,
    ) -> None:
        """landmarks: observations [N, obs_dim]; latents: their option latents z_w [N, d]
        (default: B of each landmark state)."""
        self.fb = fb
        self.config = config or PlannerConfig()
        self.options = OptionModel(fb, self.config.options)
        self.landmarks = landmarks
        self.targets = self.options.targets(landmarks, latents)
        self.plan: TaskPlan | None = None
        self._edges: OptionTable | None = None
        self._plans: dict[bytes, TaskPlan] = {}
        self._waypoint: int | None = None  # None: replan at the next step
        self._steps_on_waypoint = 0
        self._replans = 0
        self._advances = 0
        self._switches = 0

    @classmethod
    def from_dataset(
        cls, fb: FBModel, observations: np.ndarray, config: PlannerConfig | None = None
    ) -> GraphPlannerController:
        """Select landmarks and their latents from dataset observations (config.landmarks)."""
        config = config or PlannerConfig()
        marks = select_landmarks(observations, config.landmarks, fb)
        return cls(fb, marks.observations, config, marks.latents)

    @property
    def edges(self) -> OptionTable:
        """rho between landmarks. It does not depend on the task, so it is computed once."""
        if self._edges is None:
            self._edges = self.options.pair_table(self.landmarks, self.targets)
        return self._edges

    @cached_property
    def edge_rho(self) -> np.ndarray:
        """Edge discounts between landmarks for values [N, N]: clipped rho (two-way min)."""
        return self._pair_min(self.edges.rho, self.edges.rho.T)

    @cached_property
    def trusted(self) -> np.ndarray:
        """Trusted landmark-to-landmark edges [N, N], without self-loops."""
        raw = self._pair_min(self.edges.rho_raw, self.edges.rho_raw.T)
        return trusted_edges(raw, self.landmarks, self.landmarks, self.config, True)

    def reset(self, task: TaskSpec, seed: int) -> None:
        """Solve the graph for the task (cached per task). The policy is deterministic."""
        z_r = np.asarray(task.z_r, dtype=np.float64)
        key = z_r.tobytes()
        if key not in self._plans:
            self._plans[key] = self._solve(z_r)
        self.plan = self._plans[key]
        self._waypoint, self._steps_on_waypoint = None, 0
        self._replans, self._advances, self._switches = 0, 0, 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self._needs_replan():
            self._replan(obs)
        elif self._waypoint != DIRECT and self._reached(obs, np.array([self._waypoint]))[0]:
            self._on_reach(obs)
        elif self.config.lookahead:
            self._advance(obs)
        self._steps_on_waypoint += 1
        z = self.plan.z_r if self._waypoint == DIRECT else self.targets.z[self._waypoint]
        return self._action(obs, z)

    def step_info(self) -> dict[str, float | int | str]:
        return {
            "waypoint": DIRECT if self._waypoint is None else self._waypoint,
            "replans": self._replans,
            "advances": self._advances,
            "switches": self._switches,
        }

    def _action(self, obs: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.asarray(self.fb.low_action(obs[None], z[None]))[0]

    def _solve(self, z_r: np.ndarray) -> TaskPlan:
        cfg = self.config
        reward = None  # R = 0 unless edge_reward asks for it (an N x N pass per task)
        if self.options.has_reward:
            reward = self.options.pair_table(self.landmarks, self.targets, z_r).reward
        direct = self.options.direct_value(self.landmarks, z_r)
        solution = value_iteration(
            self.edge_rho,
            reward,
            direct,
            self.trusted,
            cfg.hop_discount,
            cfg.vi_max_iters,
            cfg.vi_tol,
        )
        if not solution.converged:
            raise RuntimeError(f"value iteration did not converge in {cfg.vi_max_iters} sweeps")
        return TaskPlan(z_r, solution)

    def _needs_replan(self) -> bool:
        cfg = self.config
        return (
            cfg.replan_every_step
            or self._waypoint is None
            or self._steps_on_waypoint >= cfg.commit_timeout
        )

    def _on_reach(self, obs: np.ndarray) -> None:
        """The waypoint counts as reached: replan, or follow the route (switch_steps)."""
        margin = self.config.switch_steps
        if margin is None:
            self._replan(obs)
            return
        stop = int(self.plan.solution.best_next[self._waypoint])
        (q_stop,) = self._stop_values(obs, np.array([stop]))
        if not np.isfinite(q_stop):
            return  # keep heading for the waypoint until its next stop is trusted
        if np.isfinite(margin):
            best, q_best = self._choose(obs)
            if q_best - q_stop > (self.fb.discount**-margin - 1.0) * abs(q_stop):
                self._set_waypoint(best)
                self._switches += 1
                return
        self._set_waypoint(stop)
        self._advances += 1

    def _set_waypoint(self, waypoint: int) -> None:
        self._waypoint, self._steps_on_waypoint = waypoint, 0

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
        self._set_waypoint(self._choose(obs)[0])
        self._replans += 1

    def _choose(self, obs: np.ndarray) -> tuple[int, float]:
        """Best waypoint from obs and its Q, or (DIRECT, task value). Ties follow
        graph.best_options.

        Landmarks that count as reached are excluded: the landmark under the agent has
        Q = rho * V_H ~ V_H, and choosing it again would stall the agent.
        """
        q = self._q(obs, exclude_reached=True)
        direct = self._task_value(obs)
        best = int(best_options(q[None], self.plan.solution.values, np.array([direct]))[0])
        return (DIRECT, direct) if best == DIRECT else (best, float(q[best]))

    def _advance(self, obs: np.ndarray) -> None:
        """Move on along the route while the next stop is worth at least the current waypoint.

        A route visits each landmark at most once, so the loop has at most N steps; the bound
        also ends a cycle of tied stops.
        """
        for _ in range(len(self.landmarks)):
            if self._waypoint == DIRECT:
                return
            nxt = int(self.plan.solution.best_next[self._waypoint])
            q_now, q_next = self._stop_values(obs, np.array([self._waypoint, nxt]))
            # A tie moves on: the next stop is nearer the goal (TIE_RTOL, see graph.best_options).
            if not (np.isfinite(q_next) and q_next >= q_now - TIE_RTOL * abs(q_now)):
                return
            self._set_waypoint(nxt)
            self._advances += 1

    def _stop_values(self, obs: np.ndarray, stops: np.ndarray) -> np.ndarray:
        """Q(s, w) for route stops; DIRECT stops get the task value. [len(stops)]"""
        q = np.full(len(stops), self._task_value(obs) if (stops == DIRECT).any() else -np.inf)
        landmark = stops != DIRECT
        if landmark.any():
            q[landmark] = self._q(obs, stops[landmark])
        return q

    def _q(
        self, obs: np.ndarray, idx: np.ndarray | None = None, exclude_reached: bool = False
    ) -> np.ndarray:
        """Q(s, w) for the landmarks idx (default: all), -inf behind an untrusted edge and,
        with exclude_reached, for landmarks that count as reached. [len(idx)]"""
        idx = np.arange(len(self.landmarks)) if idx is None else idx
        row, rho, raw = self._row(obs, idx)
        trusted = trusted_edges(raw, obs[None], self.landmarks[idx], self.config)[0]
        if exclude_reached:
            trusted &= ~self._reached(obs, idx, row.rho[0])
        values = self.plan.solution.values[idx]
        return q_values(rho, row.reward, values, trusted[None], self.config.hop_discount)[0]

    def _task_value(self, obs: np.ndarray) -> float:
        """F(s, z_r)^T z_r: the value of switching to the task policy now."""
        return float(self.options.direct_value(obs[None], self.plan.z_r)[0])

    def _row(self, obs: np.ndarray, idx: np.ndarray) -> tuple[OptionTable, np.ndarray, np.ndarray]:
        """The option table from obs to the landmarks idx, and the edge discounts for values
        (clipped rho) and for trust (raw rho), each the two-way min with two_way."""
        row = self.options.pair_table(obs[None], self.targets.take(idx), self.plan.z_r)
        if not self.config.two_way:
            return row, row.rho, row.rho_raw
        back = self.options.pair_table(self.landmarks[idx], self.options.targets(obs[None]))
        return (
            row,
            self._pair_min(row.rho, back.rho.T),
            self._pair_min(row.rho_raw, back.rho_raw.T),
        )

    def _pair_min(self, rho: np.ndarray, reverse: np.ndarray) -> np.ndarray:
        """min(rho(u, w), rho(w, u)) with two_way, else rho(u, w)."""
        return np.minimum(rho, reverse) if self.config.two_way else rho


class GraphPlusHighController(GraphPlannerController):
    """Same route, but z_w* is the task of the baseline pi_h, which then picks the intention."""

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
