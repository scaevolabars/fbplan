"""Tests of the pure functions in analysis/. The scripts are not a package; they load by path."""

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from fbplan.option_model import OptionTable

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"analysis_{name}", ANALYSIS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCRIPTS = ("plots", "tables", "failures", "calibration", "platform_check", "heldout_selection")
plots, tables, failures, calibration, platform_check, heldout_selection = map(_load, SCRIPTS)


def test_method_colors_give_every_method_a_color_and_keep_the_fixed_ones():
    methods = ["baseline", "oracle", *[f"planner/l{k}" for k in range(9)]]
    colors = plots.method_colors(methods)
    assert all(m in colors for m in methods)
    assert colors["baseline"] == plots.METHOD_COLORS["baseline"]
    others = [colors[f"planner/l{k}"] for k in range(9)]
    assert others[: len(plots.OTHER_COLORS)] == plots.OTHER_COLORS
    assert others[len(plots.OTHER_COLORS)] == plots.OTHER_COLORS[0]


def test_success_markdown_marks_the_given_labels_as_privileged():
    table = pd.DataFrame({
        "method": ["baseline", "oracle@a1"], "task": ["all", "all"], "n": [10, 10],
        "success": [0.5, 0.75], "ci_low": [0.4, 0.6], "ci_high": [0.6, 0.9],
    })
    lines = tables.success_markdown(table, {"oracle@a1"}).splitlines()
    assert lines[2].startswith("| baseline |")
    assert lines[3].startswith("| oracle@a1 (privileged) | 75.0 [60.0, 90.0]")


def _runs(**episodes_by_method: list[int]) -> pd.DataFrame:
    return pd.DataFrame([
        {"method": m, "env_name": "env", "task_id": 1, "eval_seed": 0, "episode": e}
        for m, eps in episodes_by_method.items() for e in eps
    ])


def test_same_episodes_keeps_only_runs_with_the_reference_episodes():
    df = _runs(baseline=[0, 1, 2], planner=[0, 1, 2], sweep=[0, 1])
    assert tables.same_episodes(df, "baseline") == {"baseline", "planner"}


def test_pooled_paired_markdown_reports_mean_and_range_of_pooled_rows_only():
    paired = pd.DataFrame({"method": ["planner/l0", "planner/l1", "oracle"],
                           "diff": [-0.10, -0.04, 0.06]})
    lines = tables.pooled_paired_markdown(paired).splitlines()
    assert lines[2:] == ["| planner | 2 | -7.0 | [-10.0, -4.0] |"]


def _ant(n: int, upright: bool, moving: bool) -> np.ndarray:
    obs = np.zeros((n, 29))
    obs[:, 0] = np.linspace(0.0, 10.0, n) if moving else 0.0
    obs[:, 3:7] = [1.0, 0.0, 0.0, 0.0] if upright else [0.0, 1.0, 0.0, 0.0]
    return obs


def test_classify_separates_flipped_stuck_and_other_episodes():
    assert failures.classify(_ant(1000, upright=False, moving=True)) == "flipped"
    assert failures.classify(_ant(1000, upright=True, moving=False)) == "stuck"
    assert failures.classify(_ant(1000, upright=True, moving=True)) == "other"


def test_failure_class_marks_episodes_that_end_where_the_goal_cannot_be_walked_to():
    moving = _ant(1000, upright=True, moving=True)
    assert failures.failure_class(moving, cells_left=-1) == "trapped"
    assert failures.failure_class(moving, cells_left=3) == "other"
    flipped = _ant(1000, upright=False, moving=True)
    assert failures.failure_class(flipped, cells_left=-1) == "flipped"


def test_failure_table_keeps_tasks_without_failures_and_counts_shares():
    episodes = pd.DataFrame({"task_id": [1, 1, 2, 2], "episode": [0, 1, 0, 1]})
    failed = pd.DataFrame({"task_id": [1, 1], "episode": [0, 1], "failure": ["flipped", "other"]})
    table = failures.failure_table(episodes, failed)
    assert table.loc[2, ["episodes", "failures", "flipped"]].tolist() == [2, 0, 0]
    assert table.loc["all", ["failures", "flipped", "other", "flipped_share"]].tolist() == [
        2, 1, 1, 0.5
    ]
    none = failures.failure_table(episodes, failed.iloc[:0])
    assert none.loc["all", "failures"] == 0
    assert np.isnan(none.loc["all", "flipped_share"])


def test_wormhole_table_counts_unreachable_pairs_apart():
    rho_raw = np.array([[1.0, 0.9], [0.9, 1.0]])
    table = OptionTable(rho=np.minimum(rho_raw, 0.99), rho_raw=rho_raw, reward=None,
                        n_clipped_low=0, n_clipped_high=0)
    geodesic = np.array([[0.0, np.inf], [np.inf, 0.0]])  # a teleport trap: no walk between them
    xy = np.array([[0.0, 1.0], [1.0, 0.0]])
    (row,) = calibration.wormhole_table(table, geodesic, xy, 0.99, 0.1, [0.3], 6.0).to_dict(
        "records"
    )
    assert (row["wormholes"], row["unreachable_on_foot"], row["longest_geodesic"]) == (2, 2, 0.0)


def test_fb_distance_inverts_the_discount():
    assert calibration.fb_distance(np.array([0.99**10, 0.0]), 0.99).tolist() == pytest.approx(
        [10.0, np.inf]
    )


def test_walk_speed_wormholes_need_trust_and_a_too_short_fb_distance():
    # pairs (0, 1): FB says 1 step, walk needs 10; (0, 2): raw rho >= 1, untrusted;
    # (1, 2): FB says 1 step, walk needs 0.5 step; diagonal: never a wormhole
    rho_raw = np.array([[1.0, 0.99, 1.2], [0.99, 1.0, 0.99], [0.99, 0.99, 1.0]])
    rho = np.minimum(rho_raw, 0.99)
    geodesic = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.05], [1.0, 0.05, 0.0]])
    holes = calibration.walk_speed_wormholes(rho_raw, rho, geodesic, 0.99, 0.1, rho_min=0.3)
    assert holes.tolist() == [
        [False, True, False], [True, False, False], [True, False, False]
    ]


def test_wormhole_table_counts_trust_and_the_xy_cap():
    rho_raw = np.array([[1.0, 0.99, 1.2], [0.99, 1.0, 0.99], [0.99, 0.99, 1.0]])
    table = OptionTable(rho=np.minimum(rho_raw, 0.99), rho_raw=rho_raw, reward=None,
                        n_clipped_low=0, n_clipped_high=1)
    geodesic = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.05], [1.0, 0.05, 0.0]])
    xy = np.array([[0.0, 0.5, 9.0], [0.5, 0.0, 0.05], [9.0, 0.05, 0.0]])
    (row,) = calibration.wormhole_table(table, geodesic, xy, 0.99, 0.1, [0.3], 6.0).to_dict(
        "records"
    )
    assert (row["pairs_above_rho_min"], row["raw_rho_ge_rho_max"], row["trusted_pairs"]) == (
        6, 1, 5
    )
    assert (row["wormholes"], row["wormholes_within_xy_cap"]) == (3, 2)
    assert row["wormhole_share"] == pytest.approx(3 / 5)


def _platform_run(successes: list[int], steps: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "method": "m", "env_name": "env", "task_id": 1, "eval_seed": 0,
        "episode": range(len(successes)), "success": [bool(x) for x in successes], "steps": steps,
    })


def test_compare_platforms_gives_successes_the_unpaired_difference_and_identical_episodes():
    linux = _platform_run([1, 1, 0, 0], [10, 20, 1000, 1000])
    mac = _platform_run([1, 0, 0, 0], [10, 30, 1000, 1000])
    row = platform_check.compare_platforms(linux, mac)
    assert (row["linux"], row["macos"], row["episodes"]) == (0.5, 0.25, 4)
    assert row["diff"] == pytest.approx(0.25)
    assert row["same_outcome"] == 0.75 and row["same_steps"] == 0.75


def test_select_seed_takes_the_highest_held_out_success_and_the_lower_id_on_ties():
    assert heldout_selection.select_seed({0: 0.5, 1: 0.7, 2: 0.7, 3: 0.6}) == 1
    assert heldout_selection.select_seed({4: 0.9, 2: 0.9}) == 2


CLI_SCRIPTS = (*SCRIPTS, "parity", "heldout_goals")


@pytest.mark.parametrize("name", CLI_SCRIPTS)
def test_cli_follows_the_convention_help_for_every_argument_defaults_shown_4_workers(name):
    # docs/design.md §7, CLI convention, items 6 and 7
    parser = _load(name).build_parser()
    assert parser.formatter_class is argparse.ArgumentDefaultsHelpFormatter
    for action in parser._actions:
        assert action.help, f"{name}: {action.option_strings or action.dest} has no help"
        if action.dest == "workers":
            assert action.default == 4


def test_calibration_writes_figures_to_fig_dir_and_keeps_out_dir_as_its_old_name():
    parser = calibration.build_parser()
    assert parser.parse_args(["--fig-dir", "a"]).fig_dir == Path("a")
    assert parser.parse_args(["--out-dir", "b"]).fig_dir == Path("b")


def test_heldout_goals_writes_its_tables_to_table_dir():
    parser = _load("heldout_goals").build_parser()
    assert parser.parse_args([]).table_dir == Path("results/heldout_goals")
    assert parser.parse_args(["--table-dir", "t"]).table_dir == Path("t")
