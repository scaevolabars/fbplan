"""Run the tabular planning study and write CSV tables for docs/theory.md.

    uv run python -m fbplan.tabular.run_study            # full study, about 10 min on 8 cores
    uv run python -m fbplan.tabular.run_study --quick    # 1 seed, a few goals: smoke test

Experiments (medium maze, 104 states, gamma 0.9, landmarks = all states, horizon 100 steps):
    noise      success vs. goal distance for each planner, rho_noise 0..0.04, both controllers
    guards     chained-planner ablations (head min, rho_min, hop cost, commit) at two noise levels
    rank       low-rank B ("laplacian" vs. wall-blind "euclidean") without noise, plus wormholes
    reward     the dropped term on general rewards: regret and self-deception of each R_hat
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from fbplan.tabular.fake_fb import FakeFBConfig, TabularFB
from fbplan.tabular.mdp import optimal_policy, policy_value
from fbplan.tabular.study import (
    LandmarkModel,
    PlannerConfig,
    evaluate_goal_reaching,
    goal_latent,
    make_plan,
    plan_policy,
    plan_route,
    wormhole_mask,
)

OUT_DIR = Path("docs/figures/theory")
HORIZON = 100
DIST_BINS = [(1, 5), (6, 10), (11, 15), (16, 23)]
BASE = FakeFBConfig(high_radius=10)  # pi_h subgoals within 1 / (1 - gamma) steps

GUARDED = PlannerConfig(
    "chained R=0, guarded", reward="zero", aggregate="min", rho_min=0.4, commit=True
)
MAIN_PLANNERS = [
    PlannerConfig("direct", kind="direct"),
    PlannerConfig("pi_h (idealised)", kind="baseline"),
    PlannerConfig("single switch", kind="single"),
    PlannerConfig("chained"),
    PlannerConfig("chained R=0", reward="zero"),
    GUARDED,
]


def guard_planners() -> list[PlannerConfig]:
    """One change at a time from a plain chained planner, for both R_hat variants."""
    planners = []
    for reward in ("full", "zero"):
        base = PlannerConfig(f"R={reward}: plain", reward=reward)
        planners += [
            base,
            replace(base, name=f"R={reward}: head min", aggregate="min"),
            *(replace(base, name=f"R={reward}: rho_min {r}", rho_min=r) for r in (0.2, 0.4, 0.6)),
            *(replace(base, name=f"R={reward}: hop cost {c}", hop_cost=c) for c in (0.05, 0.2)),
            replace(base, name=f"R={reward}: commit", commit=True),
            replace(
                base, name=f"R={reward}: all guards", aggregate="min", rho_min=0.4, commit=True
            ),
        ]
    return planners


def _distance_rows(success: np.ndarray, dist: np.ndarray, goals: np.ndarray) -> Iterator[dict]:
    d = dist[:, goals]
    for lo, hi in DIST_BINS:
        mask = (d >= lo) & (d <= hi)
        yield {"dist_lo": lo, "dist_hi": hi, "success": success[mask].mean(), "n_pairs": mask.sum()}


def goal_reaching_rows(
    config: FakeFBConfig, planners: list[PlannerConfig], goals: np.ndarray
) -> list[dict]:
    """Success per distance bin of every planner on one fake model."""
    model = TabularFB(config)
    lm = LandmarkModel(model, np.arange(model.maze.n_states))
    dist = model.maze.geodesic_distances()
    rows = []
    for planner in planners:
        success, _ = evaluate_goal_reaching(lm, planner, goals, HORIZON)
        for row in _distance_rows(success, dist, goals):
            rows.append({**asdict(config), "planner": planner.name, **row})
    return rows


def wormhole_rows(config: FakeFBConfig) -> list[dict]:
    """Share of ordered pairs whose rho_hat exceeds twice the bound gamma^dist, per head rule."""
    model = TabularFB(config)
    lm = LandmarkModel(model, np.arange(model.maze.n_states))
    dist = model.maze.geodesic_distances()
    off = ~np.eye(model.maze.n_states, dtype=bool)
    rows = []
    for aggregate in ("mean", "min"):
        rho, _, _ = lm.estimates(goal_latent(model, 0), aggregate)
        holes = wormhole_mask(rho, dist, model.discount) & off
        rows.append({**asdict(config), "aggregate": aggregate, "wormhole_rate": holes[off].mean()})
    return rows


def region_reward(model: TabularFB, task_id: int, seed: int) -> np.ndarray:
    """+5 on the task's goal cell, +1 and -1 on two other random maze cells."""
    maze = model.maze
    init_ij, goal_ij = maze.tasks[task_id - 1]
    cells = [tuple(ij) for ij in np.argwhere(~maze.maze_map) if tuple(ij) not in (init_ij, goal_ij)]
    plus, minus = np.random.default_rng(seed).choice(len(cells), size=2, replace=False)
    reward = np.zeros(maze.n_states)
    reward[maze.cell_states(goal_ij)] = 5.0
    reward[maze.cell_states(cells[plus])] = 1.0
    reward[maze.cell_states(cells[minus])] = -1.0
    return reward


def reward_rows(config: FakeFBConfig, planners: list[PlannerConfig], n_rewards: int) -> list[dict]:
    """True value, predicted value and hops of each planner on region rewards (from the start)."""
    model = TabularFB(config)
    maze = model.maze
    lm = LandmarkModel(model, np.arange(maze.n_states))
    rows = []
    for task_id in range(1, 6):
        start = maze.centre_state(maze.tasks[task_id - 1][0])
        for seed in range(n_rewards):
            reward = region_reward(model, task_id, seed)
            z_r = model.basis.T @ reward  # raw scale: F(s, z_r)^T z_r is in reward units
            v_star = optimal_policy(maze, reward, model.discount)[1][start]
            for planner in planners:
                value = policy_value(maze, plan_policy(lm, planner, z_r), reward, model.discount)
                rows.append(
                    {
                        **asdict(config),
                        "planner": planner.name,
                        "task_id": task_id,
                        "reward_seed": seed,
                        "v_star": v_star,
                        "v_true": value[start],
                        "v_predicted": make_plan(lm, planner, z_r).value[start],
                        "hops": len(plan_route(lm, planner, z_r, start)),
                    }
                )
    return rows


def _jobs(quick: bool) -> list[tuple[str, tuple]]:
    seeds = range(1 if quick else 5)
    goals = np.arange(0, 104, 13) if quick else np.arange(104)
    jobs: list[tuple[str, tuple]] = []
    for low_policy in ("greedy", "optimal"):
        for noise in (0.0, 0.01, 0.02, 0.03, 0.04):
            for seed in seeds:
                cfg = replace(BASE, rho_noise=noise, low_policy=low_policy, seed=seed)
                jobs.append(("noise", (cfg, MAIN_PLANNERS, goals)))
                jobs.append(("wormholes", (cfg,)))
                if noise in (0.02, 0.04):
                    jobs.append(("guards", (cfg, guard_planners(), goals)))
    for basis in ("laplacian", "euclidean"):
        for rank in (104, 64, 32, 16):
            cfg = replace(BASE, rank=rank, basis=basis)
            jobs.append(("rank", (cfg, MAIN_PLANNERS, goals)))
            jobs.append(("wormholes", (cfg,)))
    reward_planners = [
        PlannerConfig(f"{kind}, R={reward}", kind=kind, reward=reward)
        for kind in ("single", "chained")
        for reward in ("full", "proxy", "zero")
    ]
    for noise in (0.0, 0.01):
        cfg = replace(BASE, rho_noise=noise, low_policy="optimal")
        jobs.append(("reward", (cfg, reward_planners, 2 if quick else 10)))
    return jobs


_RUNNERS = {
    "noise": goal_reaching_rows,
    "guards": goal_reaching_rows,
    "rank": goal_reaching_rows,
    "wormholes": wormhole_rows,
    "reward": reward_rows,
}


def _run(job: tuple[str, tuple]) -> tuple[str, list[dict]]:
    name, args = job
    return name, _RUNNERS[name](*args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="1 seed and 8 goals")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    tables: dict[str, list[dict]] = {name: [] for name in _RUNNERS}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, rows in pool.map(_run, _jobs(args.quick)):
            tables[name] += rows
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(args.out / f"study_{name}.csv", index=False)
        print(f"wrote {args.out / f'study_{name}.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
