"""The parts of the recorded route on one maze, read from a built RecordedRouteController:
recorded train trajectories, landmarks, certified edges, and the first hops from a state.
Nothing here plans or acts; it reads what the controller built from the train split."""

from __future__ import annotations

from typing import Any

import numpy as np

from fbplan.controllers.planner import RecordedRouteController
from fbplan.graph import reachable_visits
from fbplan.viz.episodes import simplify

SAMPLE_EVERY = 25  # every 25th train trajectory: 40 of medium's 1,000


def trajectory_starts(ends: np.ndarray) -> np.ndarray:
    """The first row of each trajectory, from fbplan.graph.trajectory_ends of the rows. [T]"""
    last = np.unique(ends)
    return np.r_[0, last[:-1] + 1]


def sample_walks(
    xy: np.ndarray, ends: np.ndarray, every: int = SAMPLE_EVERY, spacing: float = 0.3
) -> list[np.ndarray]:
    """Every `every`-th recorded trajectory, in data order, as a simplified xy polyline."""
    starts = trajectory_starts(ends)[::every]
    return [simplify(xy[s : ends[s] + 1], spacing) for s in starts]


def first_hops(ctrl: RecordedRouteController, xy: np.ndarray) -> np.ndarray:
    """The landmarks that the first hop from the state at `xy` can reach: those that a recorded
    trajectory passes within K rows after a train state near xy (as _plan finds them). [M]"""
    rows = ctrl._near_rows(np.asarray(xy, dtype=float))
    visit, _ = reachable_visits(ctrl.visits, rows, ctrl.ends, ctrl.config.max_steps)
    return np.unique(ctrl.visits.label[visit])


def parts(ctrl: RecordedRouteController) -> dict[str, Any]:
    """Landmarks, certified edges and a sample of the recorded trajectories of the controller."""
    ends = ctrl.ends
    return {
        "landmarks": ctrl.landmarks[:, :2].astype(float),
        "edges": np.stack([ctrl.edges.source, ctrl.edges.target], axis=1).astype(int),
        "walks": sample_walks(ctrl.xy, ends),
        "n_trajectories": len(trajectory_starts(ends)),
        "n_rows": len(ctrl.xy),
        "max_steps": ctrl.config.max_steps,
        "certify_xy": ctrl.config.certify_xy,
        "window_xy": list(ctrl.config.window_xy),
    }
