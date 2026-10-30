"""analysis/raster_geodesic.py, trap_split.py and trap_gate.py: the pure parts and the CLI."""

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from fbplan.maze import MazeGeometry

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"analysis_{name}", ANALYSIS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCRIPTS = ("raster_geodesic", "trap_split", "trap_gate")
raster_geodesic, trap_split, trap_gate = map(_load, SCRIPTS)

# Cells (1, 1) and (1, 3) with the wall (1, 2) between them; row 2 is free. Centre of (i, j):
# (4j - 4, 4i - 4).
WALLED = MazeGeometry(
    np.array([[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]]), 4.0, (4.0, 4.0)
)


def test_the_raster_distance_is_exact_on_a_straight_free_row():
    raster = raster_geodesic.Raster(WALLED, 0.5)
    assert raster.distances(np.array([[0.0, 4.0]]), np.array([[8.0, 4.0]]))[0, 0] == 8.0


def test_the_raster_walks_around_a_wall():
    raster = raster_geodesic.Raster(WALLED, 0.5)
    around = 4 * np.sqrt(2) + 4  # (0, 0) -> (2, 2) -> (6, 2) -> (8, 0)
    distance = raster.distances(np.array([[0.0, 0.0]]), np.array([[8.0, 0.0]]))[0, 0]
    assert around * 0.95 < distance < around * 1.1


def test_points_on_a_wall_pixel_are_rejected():
    raster = raster_geodesic.Raster(WALLED, 0.5)
    assert raster.pixels(np.array([[4.0, 0.0]]))[0] == -1
    with pytest.raises(ValueError, match="wall pixel"):
        raster.distances(np.array([[4.0, 0.0]]), np.array([[0.0, 0.0]]))


def test_short_shares_count_only_finite_pairs_farther_than_the_minimum():
    geodesic = np.array([[1.0, 5.0, 8.0, np.inf]])
    raster = np.array([[1.0, 10.0, 8.5, 9.0]])
    shares = raster_geodesic.short_shares(geodesic, raster)
    assert shares["pairs"] == 2
    assert shares["below_0.85"] == shares["below_0.7"] == 0.5
    assert shares["median"] == pytest.approx((0.5 + 8.0 / 8.5) / 2)


def test_a_failure_that_ends_at_the_trap_distance_is_trapped():
    goals, trap = {1: np.array([0.0, 24.0])}, np.array([24.0, 0.0])  # trap distance 33.9
    df = pd.DataFrame({"task_id": [1, 1, 1, 1], "success": [True, False, False, False],
                       "final_goal_dist": [0.5, 33.0, 36.5, 10.0]})
    assert trap_split.outcomes(df, goals, trap).tolist() == ["success", "trap", "other", "other"]


def test_identical_episodes_need_the_same_steps_and_final_distance():
    keys = {"task_id": [1, 1, 2], "eval_seed": [0, 0, 0], "episode": [0, 1, 0]}
    reference = pd.DataFrame({**keys, "steps": [10, 1000, 500], "final_goal_dist": [0.1, 5.0, 3.0]})
    run = pd.DataFrame({**keys, "steps": [10, 1000, 501], "final_goal_dist": [0.1, 5.1, 3.0]})
    assert trap_split.identical_share(run, reference).to_dict() == {1: 0.5, 2: 0.0}


@pytest.mark.parametrize("module", [raster_geodesic, trap_split, trap_gate])
def test_cli_shows_defaults_and_has_help_for_every_argument(module):
    parser = module.build_parser()
    assert parser.formatter_class is argparse.ArgumentDefaultsHelpFormatter
    for action in parser._actions:
        assert action.help, f"{module.__name__}: {action.option_strings or action.dest} has no help"
