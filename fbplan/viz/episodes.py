"""Real rollouts for the figures: the recorded route (H1) and `baseline_raw` from the same starts.

Every episode runs through fbplan.evaluate.run_episode, the loop of the committed CSVs, so it has
the reset seed of the CSV row with the same (task, eval seed, episode). The controllers are the
committed ones, checked by config hash against the CSVs. The ant dynamics are chaotic: on another
platform the same reset seed can give another episode (README, "Platform"), so each episode keeps
the outcome of its CSV row next to its own.

The recorded route is traced, not changed: `traced` subclasses the controller only to keep a copy
of the walks that `_plan` passes to `_path`, the route that the controller acts on at each step.

Which episodes: eval seed 0, episodes 0 and 1 of the 5 eval tasks of medium, each with both
methods (20 episodes). `hero_pair` picks the pair of the animation by a rule fixed before the
recording: the first pair, in task then episode order, where only the recorded route reaches the
goal. The page shows every pair, those where only `baseline_raw` reaches the goal included.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fbplan.contracts import Controller, TaskSpec
from fbplan.controllers import resolve_controller
from fbplan.controllers.planner import (
    RecordedRouteConfig,
    RecordedRouteController,
    config_from_options,
    point_along,
)
from fbplan.evaluate import config_hash, make_tasks, reset_seed, run_episode, task_latent
from fbplan.frozen import load_dataset, load_model, make_env
from fbplan.viz.scores import ROOT

ENV = "antmaze-medium-navigate-v0"
TASKS = (1, 2, 3, 4, 5)
EVAL_SEED = 0
EPISODES = (0, 1)
ROUTE_OPTIONS = {"executor": "window", "max_steps": 150}  # selected on medium (REPORT §8.7)
CSV = {
    "recorded_route": "results/medium/planner_ablations/h1_eval/window_k150.csv",
    "baseline_raw": "results/medium_10seeds/baseline_raw.csv",
}
SPACING_XY = 0.25  # spacing of the stored polylines


class RouteTrace:
    """Mixin for a RecordedRouteController class: keeps the walks of the last planned route."""

    walks: list[tuple[int, int]]
    path: np.ndarray | None

    def act(self, obs: np.ndarray) -> np.ndarray:
        self.walks, self.path = [], None
        return super().act(obs)  # type: ignore[misc]

    def _path(self, s: np.ndarray, walks: list[tuple[int, int]]) -> np.ndarray:
        self.path = super()._path(s, walks)  # type: ignore[misc]
        self.walks = [(int(a), int(b)) for a, b in walks]
        return self.path


@functools.cache
def traced(cls: type[RecordedRouteController]) -> type[RecordedRouteController]:
    """`cls` with RouteTrace: same name, config and decisions."""
    return type(f"Traced{cls.__name__}", (RouteTrace, cls), {})


def yaw(obs: np.ndarray) -> float:
    """Heading of the torso in the xy plane, from its quaternion (w, x, y, z) at obs[3:7]."""
    w, x, y, z = (float(v) for v in obs[3:7])
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def upright(obs: np.ndarray) -> float:
    """The z component of the torso's up axis: below 0, the ant is upside down (as in
    analysis/failures.py)."""
    _, x, y, _ = (float(v) for v in obs[3:7])
    return 1.0 - 2.0 * (x * x + y * y)


def flipped_since(up: Sequence[float], tail: int = 50) -> int | None:
    """The first step of the last upside-down stretch if it covers the last `tail` steps (the
    "flipped" class of analysis/failures.py), else None."""
    down = np.asarray(up) < 0
    if len(down) < tail or not down[-tail:].all():
        return None
    upright_steps = np.flatnonzero(~down)
    return int(upright_steps[-1] + 1) if len(upright_steps) else 0


def simplify(points: np.ndarray, spacing: float = SPACING_XY) -> np.ndarray:
    """The polyline with each kept point at least `spacing` from the last kept one; the first and
    the last point stay. [M, 2]"""
    points = np.asarray(points, dtype=float)
    if len(points) <= 2:
        return points
    kept = [points[0]]
    for p in points[1:-1]:
        if np.linalg.norm(p - kept[-1]) >= spacing:
            kept.append(p)
    kept.append(points[-1])
    return np.stack(kept)


class Recorder:
    """Passes every call to `inner`; keeps the xy, heading and (for a traced route) the route of
    each step."""

    def __init__(self, inner: Controller) -> None:
        self.inner = inner
        self.name = inner.name
        self.steps: list[dict[str, Any]] = []

    def reset(self, task: TaskSpec, seed: int) -> None:
        self.steps = []
        self.inner.reset(task, seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        action = self.inner.act(obs)
        step: dict[str, Any] = {
            "xy": obs[:2].astype(float).tolist(), "yaw": yaw(obs), "up": upright(obs),
        }
        if isinstance(self.inner, RouteTrace):
            step.update(route_step(self.inner))
        self.steps.append(step)
        return action

    def step_info(self) -> dict[str, float | int | str]:
        return self.inner.step_info()


def route_step(ctrl: RouteTrace) -> dict[str, Any]:
    """The route of the last step: landmark ids, recorded walks (row ranges) and the act points.

    route: landmark ids, the first hop first and the goal landmark last; walks: (first row, last
    row) of the recorded walk of each hop, in route order (_plan passes every hop to _path, which
    cuts only the polyline); window: the points at the window distances along the path that the
    controller acted on, None past the goal landmark."""
    route = [int(k) for k in ctrl._route]
    walks = list(ctrl.walks)
    window = []
    for distance in ctrl.config.window_xy:
        point = None if ctrl.path is None else point_along(ctrl.path, distance)
        window.append(None if point is None else point.tolist())
    return {"route": route, "walks": walks, "window": window}


def record(env: Any, controller: Controller, task: TaskSpec, episode: int) -> dict[str, Any]:
    """One episode of eval seed EVAL_SEED through fbplan.evaluate.run_episode."""
    recorder = Recorder(controller)
    record_ = run_episode(env, recorder, task, EVAL_SEED, episode, config_hash(controller))
    final = env.unwrapped.get_xy()
    up = [step["up"] for step in recorder.steps]
    goal_landmark = int(controller.goal) if isinstance(controller, RouteTrace) else None
    return {
        "method": controller.name,
        "goal_landmark": goal_landmark,
        "config_hash": record_.config_hash,
        "task_id": task.task_id,
        "eval_seed": EVAL_SEED,
        "episode": episode,
        "reset_seed": reset_seed(EVAL_SEED, task.task_id, episode),
        "success": record_.success,
        "steps": record_.steps,
        "final_goal_dist": record_.final_goal_dist,
        "final_xy": [float(final[0]), float(final[1])],
        "flipped_since": None if record_.success else flipped_since(up),
        "trace": recorder.steps,
    }


def csv_rows(method: str, root: Path = ROOT) -> pd.DataFrame:
    """The committed CSV of a method, indexed by (task_id, eval_seed, episode)."""
    df = pd.read_csv(root / CSV[method], dtype={"config_hash": str})
    return df.set_index(["task_id", "eval_seed", "episode"])


def check_config(controller: Controller, rows: pd.DataFrame) -> None:
    """The controller must have the config hash of its committed CSV."""
    expected = set(rows["config_hash"])
    if {config_hash(controller)} != expected:
        raise ValueError(
            f"{controller.name}: config hash {config_hash(controller)} is not that of its CSV "
            f"{sorted(expected)}"
        )


def build_controllers(model: Any) -> dict[str, Controller]:
    """The recorded route (traced) and baseline_raw, as the evaluation CLI builds them."""
    data = load_dataset(ENV, "train")
    config = config_from_options(dict(ROUTE_OPTIONS), RecordedRouteConfig())
    route = traced(RecordedRouteController).from_dataset(
        model, data["observations"], data["terminals"], config
    )
    return {"recorded_route": route, "baseline_raw": resolve_controller("baseline_raw")(model)}


def record_pairs(
    tasks: Sequence[int] = TASKS, episodes: Sequence[int] = EPISODES, root: Path = ROOT
) -> tuple[list[dict[str, Any]], RecordedRouteController]:
    """Both methods on each (task, episode) of eval seed EVAL_SEED. Returns the pairs, each
    {"task_id", "episode", "csv": the outcomes of the committed rows, "runs": {method: episode}},
    and the traced route controller (for the parts of its route)."""
    model = load_model(ENV)
    controllers = build_controllers(model)
    rows = {method: csv_rows(method, root) for method in controllers}
    for method, controller in controllers.items():
        check_config(controller, rows[method])
    env = make_env(ENV)
    specs = make_tasks(ENV, list(tasks), functools.partial(task_latent, model))
    pairs = []
    for task in specs:
        for episode in episodes:
            key = (task.task_id, EVAL_SEED, episode)
            pairs.append({
                "task_id": task.task_id,
                "episode": episode,
                "csv": {
                    method: {
                        "success": bool(rows[method].loc[key, "success"]),
                        "steps": int(rows[method].loc[key, "steps"]),
                        "file": CSV[method],
                    }
                    for method in controllers
                },
                "runs": {
                    method: record(env, controller, task, episode)
                    for method, controller in controllers.items()
                },
            })
    return pairs, controllers["recorded_route"]


def compact_walks(
    pairs: Sequence[dict[str, Any]], xy: np.ndarray, spacing: float = SPACING_XY
) -> list[np.ndarray]:
    """Replace the row ranges ("walks") of every route step by ids ("walk_ids") into a table of
    the walks' simplified xy polylines; each distinct walk is stored once. Changes `pairs` in
    place and returns the table."""
    ids: dict[tuple[int, int], int] = {}
    table: list[np.ndarray] = []
    for pair in pairs:
        for run in pair["runs"].values():
            for step in run["trace"]:
                if "walks" not in step:
                    continue
                walk_ids = []
                for start, end in step.pop("walks"):
                    key = (int(start), int(end))
                    if key not in ids:
                        ids[key] = len(table)
                        table.append(simplify(xy[start : end + 1], spacing))
                    walk_ids.append(ids[key])
                step["walk_ids"] = walk_ids
    return table


def hero_pair(pairs: Sequence[dict[str, Any]]) -> int:
    """Index of the animated pair: the first, in task then episode order, where only the recorded
    route reaches the goal; else the first where both do; else the first pair."""
    order = sorted(range(len(pairs)), key=lambda i: (pairs[i]["task_id"], pairs[i]["episode"]))

    def outcome(i: int) -> tuple[bool, bool]:
        runs = pairs[i]["runs"]
        return runs["recorded_route"]["success"], runs["baseline_raw"]["success"]

    for wanted in ((True, False), (True, True)):
        for i in order:
            if outcome(i) == wanted:
                return i
    return order[0]
