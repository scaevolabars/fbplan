"""Single switch vs. chained switches on the tabular fake FB, with exact evaluation.

The planners see only FB quantities, and they get them from the planner's own code:
`fbplan.option_model.OptionModel` (rho_hat, R_hat and the direct value per F head, then min or
mean over the heads; rho_hat clipped to [0, gamma] for distinct states) and `fbplan.graph`
(trusted edges on the raw rho_hat, value iteration, Q-values). The evaluation uses the exact
maze. Landmarks are states. Names: `v_direct` = F(s, z_r)^T z_r (V_d in docs/theory.md),
`v_graph` = V_H, `r_before` = R_hat.

Planner kinds:
    direct    pi_l(z_r) only.
    baseline  the fake's idealised pi_h (upstream proxy advantage), then pi_l.
    single    best single switch: Q(s, w) = R_hat(s, w) + hop_discount rho_hat(s, w) v_direct(w).
    chained   Q(s, w) = R_hat(s, w) + hop_discount rho_hat(s, w) v_graph(w).
For single and chained, an edge is trusted when rho_min <= raw rho_hat < 1, and the agent acts
with pi_l(z_r) when no trusted option beats v_direct(s). By default the choice is redone at
every step; with `commit` the agent keeps w* until it is reached or a timeout.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, get_args

import numpy as np

from fbplan.contracts import normalize_z
from fbplan.graph import DIRECT, best_options, q_values, trusted_edges, value_iteration
from fbplan.option_model import EdgeReward, HeadAgg, OptionModel, OptionModelConfig, OptionTable
from fbplan.tabular.fake_fb import TabularFB

PlannerKind = Literal["direct", "baseline", "single", "chained"]


@dataclass(frozen=True)
class StudyPlannerConfig:
    """One planner variant. Only `single` and `chained` read the fields after `kind`.

    head_agg, edge_reward: as in `OptionModelConfig` ("exact" R, "paper" = the upstream proxy
    F(s, z_w)^T z_r, "zero"). hop_discount: factor on rho_hat per hop, as in `fbplan.graph`.
    """

    name: str
    kind: PlannerKind = "chained"
    head_agg: HeadAgg = "mean"
    edge_reward: EdgeReward = "exact"
    rho_min: float = 0.0
    hop_discount: float = 1.0
    commit: bool = False

    def __post_init__(self) -> None:
        if self.kind not in get_args(PlannerKind):
            raise ValueError(f"unknown planner kind {self.kind!r}")
        OptionModelConfig(head_agg=self.head_agg, edge_reward=self.edge_reward)  # validates


@dataclass(frozen=True)
class Plan:
    """Option choice per state for one task."""

    choice: np.ndarray  # [n] landmark column, or DIRECT (-1) for "act with z_r"
    value: np.ndarray  # [n] predicted value max(v_direct, max_w Q)
    rho: np.ndarray  # [n, L] clipped rho_hat used by the plan


class LandmarkModel:
    """Option latents of the landmarks, their pi_l actions and the task-free rho_hat tables,
    computed once per fake model."""

    def __init__(self, model: TabularFB, landmarks: np.ndarray) -> None:
        self.model = model
        self.landmarks = np.asarray(landmarks)
        maze, n_land = model.maze, len(self.landmarks)
        self._batch = maze.n_states * n_land  # one F call per table
        self.targets = self._option_model("mean", "zero").targets(maze.xy[self.landmarks])
        self.own = np.arange(maze.n_states)[:, None] == self.landmarks[None, :]  # [n, L] s == w
        obs = np.tile(maze.xy, (n_land, 1))  # landmark-major
        z = np.repeat(self.targets.z, maze.n_states, axis=0)
        actions = maze.action_from_vector(model.low_action(obs, z))
        self.option_actions = actions.reshape(n_land, maze.n_states)
        self._rho_tables: dict[str, OptionTable] = {}

    def table(
        self, head_agg: HeadAgg, edge_reward: EdgeReward = "zero", z_r: np.ndarray | None = None
    ) -> OptionTable:
        """rho_hat [n, L] from every state to every landmark, and R_hat [n, L] for z_r.

        With edge_reward "zero" or no z_r, the task-free rho table is computed once per head
        rule; R_hat is then 0 (or None without z_r).
        """
        if edge_reward == "zero" or z_r is None:
            if head_agg not in self._rho_tables:
                option_model = self._option_model(head_agg, "zero")
                self._rho_tables[head_agg] = option_model.pair_table(
                    self.model.maze.xy, self.targets
                )
            rho = self._rho_tables[head_agg]
            return rho if z_r is None else replace(rho, reward=np.zeros_like(rho.rho))
        option_model = self._option_model(head_agg, edge_reward)
        return option_model.pair_table(self.model.maze.xy, self.targets, z_r)

    def v_direct(self, z_r: np.ndarray, head_agg: HeadAgg) -> np.ndarray:
        """F(s, z_r)^T z_r at every state: [n]."""
        return self._option_model(head_agg, "zero").direct_value(self.model.maze.xy, z_r)

    def _option_model(self, head_agg: HeadAgg, edge_reward: EdgeReward) -> OptionModel:
        config = OptionModelConfig(head_agg=head_agg, edge_reward=edge_reward,
                                   batch_size=self._batch)
        return OptionModel(self.model, config)


def goal_latent(model: TabularFB, goal: int) -> np.ndarray:
    """Task latent of the one-state goal `goal`, as infer_task_latent would give it."""
    return np.asarray(normalize_z(model.basis[goal], model.latent_dim))


def make_plan(lm: LandmarkModel, cfg: StudyPlannerConfig, z_r: np.ndarray) -> Plan:
    """Option values and choices of a `single` or `chained` planner for task latent z_r."""
    table = lm.table(cfg.head_agg, cfg.edge_reward, z_r)
    v_direct = lm.v_direct(z_r, cfg.head_agg)
    trusted = trusted_edges(table.rho_raw, cfg.rho_min) & ~lm.own
    land = lm.landmarks
    if cfg.kind == "chained":
        solution = value_iteration(
            table.rho[land], table.reward[land], v_direct[land], trusted[land], cfg.hop_discount
        )
        if not solution.converged:
            raise RuntimeError(f"graph value iteration did not converge for {cfg.name}")
        v_graph = solution.values
    elif cfg.kind == "single":
        v_graph = v_direct[land]
    else:
        raise ValueError(f"planner kind {cfg.kind!r} has no option plan")
    q = q_values(table.rho, table.reward, v_graph, trusted, cfg.hop_discount)
    choice = best_options(q, v_graph, v_direct)  # the planner's tie rules
    return Plan(choice=choice, value=np.maximum(v_direct, q.max(axis=1)), rho=table.rho)


def direct_actions(lm: LandmarkModel, z_r: np.ndarray) -> np.ndarray:
    """Action id per state of pi_l(z_r): [n]."""
    maze = lm.model.maze
    return maze.action_from_vector(lm.model.low_action(maze.xy, np.tile(z_r, (maze.n_states, 1))))


def plan_actions(lm: LandmarkModel, plan: Plan, z_r: np.ndarray) -> np.ndarray:
    """Action id per state of the policy that re-picks the best option at every step: [n]."""
    states = np.arange(lm.model.maze.n_states)
    options = lm.option_actions[np.maximum(plan.choice, 0), states]
    return np.where(plan.choice != DIRECT, options, direct_actions(lm, z_r))


def markov_actions(lm: LandmarkModel, cfg: StudyPlannerConfig, z_r: np.ndarray) -> np.ndarray:
    """Action id per state of a planner that re-plans at every step: [n]."""
    model, maze = lm.model, lm.model.maze
    if cfg.kind == "direct":
        return direct_actions(lm, z_r)
    if cfg.kind == "baseline":
        z_task = np.tile(z_r, (maze.n_states, 1))
        return maze.action_from_vector(
            model.low_action(maze.xy, model.high_latent(maze.xy, z_task))
        )
    return plan_actions(lm, make_plan(lm, cfg, z_r), z_r)


def plan_route(lm: LandmarkModel, plan: Plan, start: int) -> list[int]:
    """Landmark states the plan visits from `start` if every option succeeds."""
    route: list[int] = []
    state = start
    while plan.choice[state] != DIRECT and int(lm.landmarks[plan.choice[state]]) not in route:
        state = int(lm.landmarks[plan.choice[state]])
        route.append(state)
    return route


def rho_bound_wormholes(rho_hat: np.ndarray, dist: np.ndarray, gamma: float) -> np.ndarray:
    """Pairs whose rho_hat is more than twice the bound gamma^dist (true rho <= gamma^dist).

    This is the tabular definition. `analysis/calibration.py` uses another one for the
    ant: a trusted pair whose FB distance is shorter than the time to walk the geodesic at the
    ant's top speed.
    """
    return rho_hat > 2.0 * gamma**dist


def evaluate_goal_reaching(
    lm: LandmarkModel, cfg: StudyPlannerConfig, goals: np.ndarray, horizon: int
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
            steps[:, k] = _markov_rollouts(lm, markov_actions(lm, cfg, z_r), int(goal), horizon)
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
    lm: LandmarkModel, cfg: StudyPlannerConfig, z_r: np.ndarray, goal: int, horizon: int
) -> np.ndarray:
    """Keep w* until it is reached, or re-plan after twice its predicted duration plus 2 steps.

    The predicted duration is log(rho_hat) / log(gamma) steps; the 2 extra steps keep the
    timeout above 0 for rho_hat near 1.
    """
    maze, gamma = lm.model.maze, lm.model.discount
    plan, direct = make_plan(lm, cfg, z_r), direct_actions(lm, z_r)
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
