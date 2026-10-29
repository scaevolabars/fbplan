"""Run the tabular planning study and write CSV tables for docs/theory.md.

    uv run python -m fbplan.tabular.run_study            # full study into docs/figures/theory/
                                                        # about 14 min with 4 workers
    uv run python -m fbplan.tabular.run_study --quick    # 1 seed, 8 goals, into runs/run_study/

Experiments (medium maze, 104 states, gamma 0.9, landmarks = all states, horizon 100 steps):
    noise      success vs. goal distance for each planner, rho_noise 0..0.04, both controllers
    guards     chained-planner ablations (head min, rho_min, hop discount, commit) at 2 noise levels
    rank       low-rank B ("laplacian" vs. wall-blind "euclidean") without noise, plus wormholes
    reward     the dropped term on general rewards: regret and self-deception of each R_hat
"""

from __future__ import annotations

import argparse
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
    StudyPlannerConfig,
    evaluate_goal_reaching,
    make_plan,
    plan_actions,
    plan_route,
    rho_bound_wormholes,
)

OUT_DIR = Path("docs/figures/theory")
QUICK_OUT_DIR = Path("runs/run_study")
HORIZON = 100
DIST_BINS = [(1, 5), (6, 10), (11, 15), (16, 23)]
FAR = 11  # "far" pairs: walking distance >= 11 steps (rho <= 0.31)
BASE = FakeFBConfig(high_radius=10)  # pi_h subgoals within 1 / (1 - gamma) steps

GUARDED = StudyPlannerConfig(
    "chained R=0, guarded", edge_reward="zero", head_agg="min", rho_min=0.4, commit=True
)
MAIN_PLANNERS = [
    StudyPlannerConfig("direct", kind="direct"),
    StudyPlannerConfig("pi_h (idealised)", kind="baseline"),
    StudyPlannerConfig("single switch", kind="single"),
    StudyPlannerConfig("chained"),
    StudyPlannerConfig("chained R=0", edge_reward="zero"),
    GUARDED,
]


def guard_planners() -> list[StudyPlannerConfig]:
    """One change at a time from a plain chained planner, for exact R_hat and R_hat = 0."""
    planners = []
    for reward in ("exact", "zero"):
        base = StudyPlannerConfig(f"R={reward}: plain", edge_reward=reward)
        planners += [
            base,
            replace(base, name=f"R={reward}: head min", head_agg="min"),
            *(replace(base, name=f"R={reward}: rho_min {r}", rho_min=r) for r in (0.2, 0.4, 0.6)),
            *(
                replace(base, name=f"R={reward}: hop discount {h}", hop_discount=h)
                for h in (0.95, 0.8)
            ),
            replace(base, name=f"R={reward}: commit", commit=True),
            replace(
                base, name=f"R={reward}: all guards", head_agg="min", rho_min=0.4, commit=True
            ),
        ]
    return planners


def _distance_rows(success: np.ndarray, dist: np.ndarray, goals: np.ndarray) -> Iterator[dict]:
    d = dist[:, goals]
    for lo, hi in DIST_BINS:
        mask = (d >= lo) & (d <= hi)
        yield {"dist_lo": lo, "dist_hi": hi, "success": success[mask].mean(), "n_pairs": mask.sum()}


def goal_reaching_rows(
    config: FakeFBConfig, planners: list[StudyPlannerConfig], goals: np.ndarray
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
    """Share of ordered pairs whose rho_hat exceeds twice the bound gamma^dist, per head rule.

    The landmarks are all states, so the [n, L] table covers every ordered pair.
    """
    model = TabularFB(config)
    lm = LandmarkModel(model, np.arange(model.maze.n_states))
    dist = model.maze.geodesic_distances()
    off = ~np.eye(model.maze.n_states, dtype=bool)
    rows = []
    for head_agg in ("mean", "min"):
        holes = rho_bound_wormholes(lm.table(head_agg).rho, dist, model.discount) & off
        rows.append({**asdict(config), "head_agg": head_agg, "wormhole_rate": holes[off].mean()})
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


def reward_rows(
    config: FakeFBConfig, planners: list[StudyPlannerConfig], n_rewards: int
) -> list[dict]:
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
                plan = make_plan(lm, planner, z_r)
                actions = plan_actions(lm, plan, z_r)
                value = policy_value(maze, actions, reward, model.discount)
                rows.append(
                    {
                        **asdict(config),
                        "planner": planner.name,
                        "task_id": task_id,
                        "reward_seed": seed,
                        "v_star": v_star,
                        "v_true": value[start],
                        "v_predicted": plan.value[start],
                        "hops": len(plan_route(lm, plan, start)),
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
        StudyPlannerConfig(f"{kind}, R={reward}", kind=kind, edge_reward=reward)
        for kind in ("single", "chained")
        for reward in ("exact", "paper", "zero")
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
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--quick", action="store_true", help="smoke test: 1 seed and 8 goals")
    parser.add_argument(
        "--out-dir",
        type=Path,
        help=f"folder for study_*.csv (default {OUT_DIR}, with --quick {QUICK_OUT_DIR})",
    )
    parser.add_argument("--workers", type=int, default=4, help="processes (default 4)")
    args = parser.parse_args()
    out_dir = args.out_dir or (QUICK_OUT_DIR if args.quick else OUT_DIR)
    tables: dict[str, list[dict]] = {name: [] for name in _RUNNERS}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, rows in pool.map(_run, _jobs(args.quick)):
            tables[name] += rows
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(out_dir / f"study_{name}.csv", index=False)
        print(f"wrote {out_dir / f'study_{name}.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
