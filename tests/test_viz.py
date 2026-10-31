"""Tests of fbplan.viz: the data that the figures, the page and the video show.

The score rows run on the committed CSVs and must give REPORT.md's numbers. The traced route
runs on the exact grid model of the planner tests. The renderers run once on a small synthetic
bundle, so that a broken figure fails here and not in the regeneration.
"""

import re
from pathlib import Path

import numpy as np
import pytest
from planner_fakes import GOAL, START, GridFB

from fbplan.contracts import EpisodeRecord, TaskSpec
from fbplan.controllers.planner import (
    FBRouteController,
    RecordedRouteConfig,
    RecordedRouteController,
    point_along,
)
from fbplan.evaluate import config_hash, write_csv
from fbplan.landmarks import RegionField
from fbplan.viz import episodes, explain, figures, page, reel, scores
from fbplan.viz.theme import DARK, LIGHT

# The U corridor walked from START up, across and down to GOAL (tests/test_planner.py).
U_WALK = [(1, 4), (1, 3), (1, 2), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (5, 2), (5, 3), (5, 4)]


def grid_route(cls=RecordedRouteController, walks=(U_WALK,), max_steps=20):
    """A recorded-route controller on the grid model, landmarks at START and GOAL."""
    fb = GridFB()
    xy = np.concatenate([np.asarray(w, dtype=float) for w in walks])
    terminals = np.concatenate([np.r_[np.zeros(len(w) - 1), 1.0] for w in walks])
    field = RegionField(fb.all_obs()[:, :2], fb.backward(fb.all_obs()), radius=0.5)
    landmarks = np.stack([fb.obs(cell) for cell in (START, GOAL)])
    ctrl = cls(fb, landmarks, xy, terminals, field, RecordedRouteConfig(max_steps=max_steps))
    ctrl.reset(TaskSpec(fb.env_name, 1, fb.infer_task_latent(1), np.full(2, np.nan)), seed=0)
    return fb, ctrl


# ----- the traced route -----


@pytest.mark.parametrize("cls", [RecordedRouteController, FBRouteController])
def test_traced_route_has_the_name_config_and_actions_of_the_controller(cls):
    fb, plain = grid_route(cls)
    _, traced = grid_route(episodes.traced(cls))
    assert traced.name == plain.name and config_hash(traced) == config_hash(plain)
    for cell in [START, (1, 2), (3, 1), (5, 2)]:
        np.testing.assert_array_equal(traced.act(fb.obs(cell)), plain.act(fb.obs(cell)))


def test_route_step_gives_the_route_its_recorded_walks_and_the_act_points():
    fb, ctrl = grid_route(episodes.traced(RecordedRouteController))
    ctrl.act(fb.obs(START))
    step = episodes.route_step(ctrl)
    assert step["route"] == list(ctrl._route) and step["route"][-1] == ctrl.goal
    assert len(step["walks"]) == len(step["route"])
    rows = np.concatenate([np.arange(a, b + 1) for a, b in step["walks"]])
    np.testing.assert_array_equal(ctrl.xy[rows[-1]], GOAL)  # the walks end at the goal landmark
    # The act points lie on the path that the controller acted on; 12 xy is past its end (10).
    np.testing.assert_allclose(step["window"][0], point_along(ctrl.path, 4.0))
    np.testing.assert_allclose(step["window"][1], point_along(ctrl.path, 8.0))
    assert step["window"][2] is None


def test_route_step_is_empty_without_a_certified_route():
    # Only the walk from GOAL to START is recorded, so no certified route leads to GOAL.
    fb, ctrl = grid_route(episodes.traced(RecordedRouteController), walks=(U_WALK[::-1],))
    ctrl.act(fb.obs(START))
    assert episodes.route_step(ctrl) == {"route": [], "walks": [], "window": [None] * 3}


class FakeAnt:
    """A controller that returns zeros; its observations carry a torso quaternion."""

    name = "fake"

    def reset(self, task, seed):
        pass

    def act(self, obs):
        return np.zeros(8)

    def step_info(self):
        return {}


def ant_obs(x, y, quaternion):
    return np.r_[x, y, 0.5, quaternion, np.zeros(22)]


def test_recorder_keeps_xy_heading_and_upright_of_each_step():
    recorder = episodes.Recorder(FakeAnt())
    recorder.reset(None, 0)
    half = np.pi / 8  # a yaw of pi / 4
    recorder.act(ant_obs(1.0, 2.0, [np.cos(half), 0, 0, np.sin(half)]))
    recorder.act(ant_obs(3.0, 4.0, [0, 1, 0, 0]))  # turned over about x
    first, second = recorder.steps
    assert first["xy"] == [1.0, 2.0] and second["xy"] == [3.0, 4.0]
    assert first["yaw"] == pytest.approx(np.pi / 4) and first["up"] == pytest.approx(1.0)
    assert second["up"] == pytest.approx(-1.0)
    assert "route" not in first


@pytest.mark.parametrize(
    ("up", "expected"),
    [
        ([1.0] * 10 + [-1.0] * 60, 10),
        ([-1.0] * 70, 0),
        ([1.0] * 40 + [-1.0] * 30, None),  # down for fewer than 50 steps
        ([-1.0] * 60 + [1.0], None),
    ],
)
def test_flipped_since_is_the_start_of_the_last_upside_down_stretch(up, expected):
    assert episodes.flipped_since(up) == expected


def test_simplify_keeps_both_ends_and_drops_points_closer_than_the_spacing():
    line = np.c_[np.linspace(0, 1, 11), np.zeros(11)]
    kept = episodes.simplify(line, spacing=0.25)
    np.testing.assert_allclose(kept[[0, -1]], line[[0, -1]])
    assert (np.linalg.norm(np.diff(kept[:-1], axis=0), axis=1) >= 0.25 - 1e-12).all()
    assert len(kept) < len(line)


def test_compact_walks_stores_each_distinct_walk_once():
    xy = np.c_[np.arange(10.0), np.zeros(10)]
    steps = [{"walks": [(0, 3), (3, 6)]}, {"walks": [(1, 3), (3, 6)]}, {"xy": [0, 0]}]
    pairs = [{"runs": {"recorded_route": {"trace": steps}}}]
    table = episodes.compact_walks(pairs, xy, spacing=0.0)
    assert [s.get("walk_ids") for s in steps] == [[0, 1], [2, 1], None]
    assert "walks" not in steps[0]
    np.testing.assert_array_equal(table[1], xy[3:7])
    np.testing.assert_array_equal(table[2], xy[1:4])


def pair(task_id, ep, route_ok, base_ok):
    return {"task_id": task_id, "episode": ep, "runs": {
        "recorded_route": {"success": route_ok}, "baseline_raw": {"success": base_ok}}}


def test_hero_pair_is_the_first_where_only_the_recorded_route_reaches_the_goal():
    pairs = [pair(2, 0, True, False), pair(1, 1, True, False), pair(1, 0, True, True)]
    assert episodes.hero_pair(pairs) == 1  # task 1 before task 2
    assert episodes.hero_pair([pair(1, 0, False, True), pair(1, 1, True, True)]) == 1
    assert episodes.hero_pair([pair(1, 1, False, False), pair(1, 0, False, True)]) == 1


# ----- the parts of the route -----


def test_trajectory_starts_and_sample_walks_follow_the_trajectory_ends():
    ends = np.array([2, 2, 2, 4, 4, 7, 7, 7])
    np.testing.assert_array_equal(explain.trajectory_starts(ends), [0, 3, 5])
    xy = np.c_[np.arange(8.0), np.zeros(8)]
    walks = explain.sample_walks(xy, ends, every=2, spacing=0.0)
    assert len(walks) == 2
    np.testing.assert_array_equal(walks[1], xy[5:8])


def test_first_hops_are_the_landmarks_that_recorded_walks_near_the_agent_reach():
    _, ctrl = grid_route()
    np.testing.assert_array_equal(explain.first_hops(ctrl, np.array(START, float)), [0, 1])
    np.testing.assert_array_equal(explain.first_hops(ctrl, np.array((5, 2), float)), [1])


# ----- the scores -----


@pytest.mark.parametrize(
    ("row", "diff", "low", "high"),
    [
        ("route_medium_0_19", 4.0, 2.5, 5.5),  # REPORT §8.7
        ("route_large_10_19", 1.8, -0.7, 4.4),
        ("a_planner", -12.4, -15.8, -9.2),  # REPORT §7.1, pooled over landmark sets
        ("b_clean", -0.6, -3.5, 2.2),  # pooled over training seeds
        ("weight_medium_fb", -3.3, -7.0, 0.5),
    ],
)
def test_score_rows_give_the_numbers_of_the_report(row, diff, low, high):
    every = {r.key: r for r in scores.ROWS + scores.WEIGHT_EFFECT}
    s = scores.score(every[row])
    assert (round(s.diff, 1), round(s.low, 1), round(s.high, 1)) == (diff, low, high)


def test_the_weight_effect_rows_give_h2_minus_h1_of_the_report():
    medium, large = (scores.score(r) for r in scores.WEIGHT_EFFECT)
    rounded = (round(medium.diff, 1), round(medium.low, 1), round(medium.high, 1))
    assert rounded == (-6.2, -9.8, -2.3)
    assert round(large.diff, 1) == -10.7


def test_every_score_file_is_committed():
    for row in scores.ROWS + scores.WEIGHT_EFFECT:
        for path in row.method + row.reference:
            assert (scores.ROOT / path).is_file(), path


def test_paired_outcomes_of_the_medium_runs_match_the_report():
    counts = scores.paired_outcomes(scores.H1_M + scores.H1_M20, scores.RAW_M + scores.RAW_M20)
    assert counts == {"both": 3384, "only_method": 823, "only_reference": 622, "neither": 171}


def write_run(path: Path, method: str, successes: list[bool]) -> str:
    records = [
        EpisodeRecord(method, "h", "env", 1 + k % 2, 0, k, ok, 10, 0.0, 0.1)
        for k, ok in enumerate(successes)
    ]
    write_csv(records, path)
    return path.name


def test_score_and_paired_outcomes_on_synthetic_runs(tmp_path):
    a = write_run(tmp_path / "a.csv", "a", [True, True, True, False])
    b = write_run(tmp_path / "b.csv", "b", [True, False, False, False])
    row = scores.Row("x", "route", "m", "label", "e", (a,), (b,), "a", "b")
    s = scores.score(row, root=tmp_path)
    assert s.diff == pytest.approx(50.0)  # task 1: 100 - 50, task 2: 50 - 0
    assert (s.only_method, s.only_reference, s.n_pairs) == (2, 0, 4)
    counts = scores.paired_outcomes((a,), (b,), root=tmp_path)
    assert counts == {"both": 1, "only_method": 2, "only_reference": 0, "neither": 1}


@pytest.mark.parametrize(
    ("p", "text"),
    [(0.16246, "0.16"), (0.044867, "0.045"), (0.0027303, "0.003"), (9.4759e-06, "1e-5"),
     (1.368e-07, "1e-7"), (4.77e-11, "5e-11"), (None, "")],
)
def test_format_p_writes_p_as_the_report_does(p, text):
    assert figures.format_p(p) == text


# ----- the page and the renderers -----


def test_i16_round_trips_scaled_values_and_none():
    encoded = page.i16([1.234, -2.5, np.nan], 100)
    decoded = np.frombuffer(__import__("base64").b64decode(encoded), dtype="<i2")
    np.testing.assert_array_equal(decoded, [123, -250, page.NONE])
    with pytest.raises(ValueError):
        page.i16([400.0], 100)


def test_polylines_store_the_start_offset_of_each_line():
    enc = page.polylines([[[0, 0], [1, 1]], [[2, 2]]])
    assert enc["offsets"] == [0, 2, 3]


def test_the_page_template_loads_nothing_from_the_network():
    text = page.TEMPLATE.read_text()
    assert not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", text)
    assert "<link" not in text and "@import" not in text
    for marker in ("/*LIGHT*/", "/*DARK*/", "/*FONTS*/", "/*DATA*/"):
        assert marker in text


def test_schedule_and_cut():
    assert figures.schedule([10, 100], slow=2, fast=20, extra=4)[:8] == [0, 2, 4, 6, 8, 10, 12, 14]
    assert figures.schedule([10, 100], slow=2, fast=20, extra=4)[-1] == 100
    line = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
    np.testing.assert_allclose(reel.cut(line, 0.25), [[0, 0], [1, 0]])
    np.testing.assert_allclose(reel.cut(line, 1.0), line)
    assert reel.signed(-6.2) == "−6.2" and reel.signed(1.8) == "+1.8"


def run(method, steps, success, route=False):
    trace = []
    for t in range(steps):
        step = {"xy": [0.5 + 0.1 * t, 0.5], "yaw": 0.0, "up": 1.0}
        if route:
            step.update(route=[0, 1], walk_ids=[0], window=[[1.0, 0.5], None, None])
        trace.append(step)
    return {"method": method, "success": success, "steps": steps, "trace": trace,
            "final_xy": [0.5 + 0.1 * steps, 0.5], "flipped_since": None if success else 1,
            "goal_landmark": 1 if route else None, "final_goal_dist": 0.0, "reset_seed": 7}


def score_dict(key, group="route", label="eval seeds 0–9"):
    return {"key": key, "group": group, "maze": "medium", "label": label, "episodes": "e",
            "method_name": "a", "reference_name": "b", "diff": 1.0, "low": -1.0, "high": 3.0,
            "p": 0.2, "only_method": 3, "only_reference": 2, "n_pairs": 10,
            "method_success": 80.0, "reference_success": 79.0, "sources": ["x.csv"]}


@pytest.fixture
def bundle():
    """A synthetic bundle in the format of fbplan.viz.__main__.record, on a 4 x 4 maze."""
    keys = ["route_medium_0_19", "route_large_10_19", "weight_medium_fb_minus_xy",
            "weight_large_fb_minus_xy", "a_planner_high", "b_clean"]
    return {
        "env": "antmaze-medium-navigate-v0", "platform": "test", "commit": "abc", "hero": 0,
        "hero_rule": "rule", "sample_every": 1,
        "maze": {"map": [[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]], "unit": 1.0,
                 "offset": [0.0, 0.0], "goal_radius": 0.2},
        "tasks": {"1": {"init_xy": [1.0, 1.0], "goal_xy": [2.0, 2.0]}},
        "parts": {"landmarks": [[1.0, 1.0], [2.0, 2.0]], "edges": [[0, 1]],
                  "walks": [[[1, 1], [2, 2]]], "n_trajectories": 1, "n_rows": 2,
                  "max_steps": 150, "certify_xy": 0.5, "window_xy": [4.0, 8.0, 12.0]},
        "first_hops": [[1]],
        "walk_table": [[[1.0, 1.0], [2.0, 2.0]]],
        "pairs": [{"task_id": 1, "episode": 0,
                   "csv": {m: {"success": True, "steps": 5, "file": "f.csv"}
                           for m in ("recorded_route", "baseline_raw")},
                   "runs": {"recorded_route": run("recorded_route", 5, True, route=True),
                            "baseline_raw": run("baseline_raw", 9, False)}}],
        "scores": [score_dict(k, "fb" if k.startswith(("a_", "b_")) else "route") for k in keys],
        "weight_effect": [],
        "outcomes": {"medium": {"both": 5, "only_method": 3, "only_reference": 2, "neither": 1}},
    }


def test_the_renderers_draw_a_synthetic_bundle(bundle, tmp_path):
    for theme in (LIGHT, DARK):
        figures.hero_poster(bundle, theme, tmp_path / "hero.png", 3)
        figures.explainer_png(bundle, theme, tmp_path / "explainer.png")
        figures.scoreboard_png(bundle, theme, tmp_path / "scoreboard.png")
    text = page.html(bundle)
    assert "/*DATA*/" not in text and '"platform":"test"' in text
    video = reel.Reel(bundle)
    for scene in video.scenes():
        video.reset()
        scene.draw(1.0)
        assert video.frame().shape == (reel.H, reel.W, 3)
