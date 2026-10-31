"""A vertical video (Instagram Reel: 1080 x 1920, 30 fps, H.264) of the recorded route.

It shows only what the bundle holds (fbplan.viz.__main__): the parts of the route, the hero
episode pair (real rollouts) and the scores of the committed CSVs. The key content stays inside
the centre 4:5 area (y 285-1635), which the feed and the profile grid show.

Scenes: the task; recorded walks; landmarks and certified edges; the route; the act points;
the episode pair; 5,000 paired episodes; where it does not help; sources.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyBboxPatch

from fbplan.maze import draw_walls
from fbplan.viz.figures import (
    PI_L,
    format_p,
    math_font,
    maze_of,
    outcome_text,
    position,
    route_polyline,
    trail,
    use_font,
)
from fbplan.viz.theme import DARK, Theme

W, H, FPS = 1080, 1920, 30
PT = 72 / 100  # points per pixel at dpi 100
MAZE = (90, 470, 860)  # left, top, size in pixels
FADE = 8  # frames of each cross-fade between scenes
REPO = "github.com/scaevolabars/fbplan"


def signed(v: float, digits: int = 1) -> str:
    """+1.8 or −6.2, with a minus sign rather than a hyphen."""
    return f"{v:+.{digits}f}".replace("-", "−")


def short_status(run: dict[str, Any], t: int) -> str:
    """outcome_text, short enough for one line of the video."""
    if t >= run["steps"] and not run["success"] and run.get("flipped_since") is not None:
        return f"upside down since step {run['flipped_since']}, time limit"
    return outcome_text(run, t)


def ease(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def phase(u: float, start: float, end: float) -> float:
    """Progress of u within [start, end], eased, clipped to [0, 1]."""
    return ease((u - start) / (end - start))


def cut(line: np.ndarray, fraction: float) -> np.ndarray:
    """The first `fraction` of a polyline by arc length. [M, 2]"""
    line = np.asarray(line, dtype=float)
    if fraction >= 1 or len(line) < 2:
        return line
    steps = np.linalg.norm(np.diff(line, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(steps)]
    target = fraction * arc[-1]
    k = int(np.searchsorted(arc, target, side="right"))
    if k == 0:
        return line[:1]
    if k >= len(line):
        return line
    t = (target - arc[k - 1]) / max(steps[k - 1], 1e-12)
    return np.vstack([line[:k], line[k - 1] + t * (line[k] - line[k - 1])])


@dataclass
class Scene:
    name: str
    frames: int
    draw: Callable[[float], None]


class Reel:
    """All artists of the video on one figure; each scene sets them for its progress u."""

    def __init__(self, bundle: dict[str, Any], theme: Theme = DARK) -> None:
        math_font(use_font())
        self.b, self.th = bundle, theme
        pair = bundle["pairs"][bundle["hero"]]
        self.pair = pair
        self.route_run = pair["runs"]["recorded_route"]
        self.base_run = pair["runs"]["baseline_raw"]
        self.marks = np.asarray(bundle["parts"]["landmarks"])
        self.edges = np.asarray(bundle["parts"]["edges"])
        self.goal = np.asarray(bundle["tasks"][str(pair["task_id"])]["goal_xy"])
        self.first = self.route_run["trace"][0]
        self.hub = self.first["route"][0]
        self.out = self.edges[self.edges[:, 0] == self.hub, 1]
        self.scores = {s["key"]: s for s in bundle["scores"] + bundle["weight_effect"]}
        self._figure()

    # ----- layout -----

    def _text(self, x: float, y: float, s: str, px: float, **kw: Any) -> Any:
        kw.setdefault("color", self.th.ink)
        kw.setdefault("va", "baseline")
        return self.fig.text(x / W, 1 - y / H, s, fontsize=px * PT, **kw)

    def _figure(self) -> None:
        th = self.th
        self.fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=th.page)
        left, top, size = MAZE
        ax = self.fig.add_axes([left / W, 1 - (top + size) / H, size / W, size / H])
        ax.set_facecolor(th.surface)
        draw_walls(ax, maze_of(self.b), th.wall)
        self.ax = ax
        self.goal_ring = ax.add_patch(Circle(self.goal, self.b["maze"]["goal_radius"],
                                             fc=th.surface, ec=th.ink, lw=2.6, zorder=6))
        self.goal_label = ax.text(self.goal[0], self.goal[1] - 1.1, "goal", fontsize=27 * PT,
                                  color=th.ink, ha="center", va="top", zorder=9)
        start = np.asarray(self.first["xy"])
        self.start_mark, = ax.plot(*start, marker="s", ms=17, mfc="none", mec=th.ink2, mew=2.2,
                                   zorder=4)
        self.start_label = ax.text(start[0] + 1.0, start[1] + 1.0, "start", fontsize=27 * PT,
                                   color=th.ink2, zorder=9)
        self.walks = LineCollection([], colors=th.walk, linewidths=1.3, zorder=2)
        ax.add_collection(self.walks)
        self.landmarks, = ax.plot([], [], ls="none", marker="o", ms=4.2, mew=0,
                                  color=th.landmark, zorder=3)
        self.edge_lines = LineCollection([], colors=th.route, linewidths=1.1, alpha=0.6, zorder=3)
        ax.add_collection(self.edge_lines)
        self.hub_mark, = ax.plot([], [], marker="o", ms=14, color=th.route, mec=th.surface,
                                 mew=2.4, ls="none", zorder=7)
        self.route_line, = ax.plot([], [], color=th.ink, lw=2.4, ls=(0, (3, 2.2)), zorder=5)
        self.hops, = ax.plot([], [], ls="none", marker="o", ms=9, mfc=th.surface, mec=th.ink,
                             mew=2, zorder=6)
        self.goal_mark, = ax.plot([], [], ls="none", marker="D", ms=12, color=th.ink, zorder=6)
        self.spokes = LineCollection([], colors=th.ink2, linewidths=1.4, zorder=6)
        ax.add_collection(self.spokes)
        self.rings, = ax.plot([], [], ls="none", marker="o", ms=17, mfc=th.surface, mec=th.route,
                              mew=4, zorder=8)
        self.trail = {}
        self.ant = {}
        self.head = {}
        self.tag = {}
        for key, color in (("base", th.baseline), ("route", th.route)):
            self.trail[key], = ax.plot([], [], color=color, lw=4, solid_capstyle="round",
                                       solid_joinstyle="round", zorder=6)
            self.ant[key], = ax.plot([], [], ls="none", marker="o", ms=21, color=color,
                                     mec=th.surface, mew=3, zorder=10)
            self.head[key], = ax.plot([], [], color=th.ink, lw=3, solid_capstyle="round",
                                      zorder=11)
            self.tag[key] = ax.text(0, 0, "", fontsize=26 * PT, color=color, weight="bold",
                                    zorder=12, va="center")
        # Text slots: title, subtitle, caption under the maze, small print, clock.
        self.title = self._text(60, 318, "", 60, weight="bold")
        self.sub = self._text(60, 372, "", 32, color=th.ink2, va="top", linespacing=1.3)
        self.caption = self._text(60, 1376, "", 34, va="top", linespacing=1.35)
        self.small = self._text(60, 1590, "", 23, color=th.ink2, va="top", linespacing=1.35)
        self.clock = self._text(950, 402, "", 34, ha="right", weight="bold")
        self.status = [self._text(60, 1376 + 52 * k, "", 34, va="top") for k in range(2)]
        # A second axes in pixel coordinates for the number scenes.
        self.board = self.fig.add_axes([0, 0, 1, 1])
        self.board.set_xlim(0, W)
        self.board.set_ylim(H, 0)
        self.board.axis("off")
        self.board.set_zorder(-1)

    def reset(self) -> None:
        """Hide every dynamic artist and empty every text slot."""
        for line in (self.landmarks, self.hub_mark, self.route_line, self.hops, self.goal_mark,
                     self.rings, *self.trail.values(), *self.ant.values(), *self.head.values()):
            line.set_data([], [])
        for coll in (self.walks, self.edge_lines, self.spokes):
            coll.set_segments([])
        for text in (self.title, self.sub, self.caption, self.small, self.clock, *self.status,
                     *self.tag.values()):
            text.set_text("")
        self.ax.set_visible(True)
        self.walks.set_alpha(1.0)
        self.landmarks.set_alpha(1.0)
        self.caption.set_position((60 / W, 1 - 1376 / H))
        self.small.set_position((60 / W, 1 - 1590 / H))
        self.board.cla()
        self.board.set_xlim(0, W)
        self.board.set_ylim(H, 0)
        self.board.axis("off")

    def frame(self) -> np.ndarray:
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    # ----- pieces -----

    def _walks(self, u: float) -> None:
        self.walks.set_segments([cut(np.asarray(w), u) for w in self.b["parts"]["walks"]])

    def _landmarks(self, u: float) -> None:
        k = int(round(u * len(self.marks)))
        self.landmarks.set_data(*self.marks[:k].T)

    def _route(self, u: float) -> None:
        line = route_polyline(self.first, self.b["walk_table"])
        self.route_line.set_data(*cut(line, u).T)
        hops = self.marks[self.first["route"][:-1]]
        self.hops.set_data(*(hops.T if u > 0.3 else ([], [])))
        self.goal_mark.set_data(*(self.marks[[self.route_run["goal_landmark"]]].T
                                  if u > 0.95 else ([], [])))

    def _agent(self, key: str, run: dict[str, Any], t: int) -> None:
        xy, heading = position(run, t)
        down = outcome_text(run, t).startswith("upside down")
        color = self.trail[key].get_color()
        self.ant[key].set_data([xy[0]], [xy[1]])
        self.ant[key].set_markerfacecolor(self.th.surface if down else color)
        self.ant[key].set_markeredgecolor(color if down else self.th.surface)
        tip = xy + 1.0 * np.array([np.cos(heading), np.sin(heading)])
        self.head[key].set_data([xy[0], tip[0]], [xy[1], tip[1]])

    def _act_points(self, step: dict[str, Any]) -> None:
        xy = np.asarray(step["xy"])
        pts = np.asarray([p for p in step.get("window", []) if p is not None]).reshape(-1, 2)
        self.rings.set_data(*pts.T)
        self.spokes.set_segments([np.stack([xy, p]) for p in pts])

    # ----- scenes -----

    def s_task(self, u: float) -> None:
        self.title.set_text("Recorded-route tracking")
        self.sub.set_text("A frozen forward-backward (FB) agent on OGBench AntMaze\n"
                          "follows routes that its training data already walked.")
        self.caption.set_text("The task: the ant starts at the square and must\n"
                              "reach the goal ring within 1,000 steps.")
        self._agent("route", self.route_run, 0)

    def s_walks(self, u: float) -> None:
        n = self.b["parts"]["n_trajectories"]
        self.title.set_text("1  Recorded walks")
        self.sub.set_text("The offline train data of antmaze-medium.")
        self._walks(phase(u, 0.0, 0.8))
        self.caption.set_text(f"{n:,} recorded trajectories ({len(self.b['parts']['walks'])} "
                              "shown).\nNothing is trained: the FB model and its\n"
                              "policies stay frozen.")

    def s_edges(self, u: float) -> None:
        parts = self.b["parts"]
        self.title.set_text("2  Landmarks and edges")
        self.sub.set_text(f"{len(self.marks)} landmarks spread over the maze.")
        self._walks(1.0)
        self.walks.set_alpha(1 - 0.7 * phase(u, 0.0, 0.3))
        self._landmarks(phase(u, 0.0, 0.35))
        grow = phase(u, 0.4, 0.85)
        hub = self.marks[self.hub]
        if grow > 0:
            self.edge_lines.set_segments(
                [np.stack([hub, hub + grow * (self.marks[w] - hub)]) for w in self.out])
            self.hub_mark.set_data([hub[0]], [hub[1]])
        self.caption.set_text(
            f"An edge u → w counts only if one recorded trajectory\n"
            f"went from u to w within K = {parts['max_steps']} steps: "
            f"{len(self.edges):,} edges.\nShown: the {len(self.out)} edges out of one landmark.")

    def s_route(self, u: float) -> None:
        self.title.set_text("3  The shortest route")
        self.sub.set_text("To the goal landmark (diamond) that F picks for the task.")
        self.walks.set_alpha(0.3)
        self._walks(1.0)
        self._landmarks(1.0)
        self._agent("route", self.route_run, 0)
        self._route(phase(u, 0.05, 0.7))
        self.caption.set_text("Shortest by xy length, laid along the recorded\n"
                              "walks. No FB value orders the route.")

    def s_act(self, u: float) -> None:
        self.title.set_text("4  Act on the route ahead")
        self.sub.set_text("Then plan the whole route again, at every step.")
        self._landmarks(1.0)
        self._route(1.0)
        self._agent("route", self.route_run, 0)
        if u > 0.15:
            self._act_points(self.first)
        self.caption.set_text(f"The frozen policy {PI_L} acts on the mean B latent\n"
                              "of the points 4, 8 and 12 xy ahead (rings).")

    def s_episode(self, t: int) -> None:
        th, pair = self.th, self.pair
        self.title.set_text("Same start, two controllers")
        self.sub.set_text(f"medium, task {pair['task_id']}, eval seed 0, episode "
                          f"{pair['episode']}")
        self.clock.set_text(f"step {t}")
        self._landmarks(1.0)
        self.landmarks.set_alpha(0.45)
        for key, run in (("base", self.base_run), ("route", self.route_run)):
            self.trail[key].set_data(*trail(run, min(t, run["steps"])).T)
            self._agent(key, run, t)
            xy, _ = position(run, t)
            self.tag[key].set_position((xy[0] + 0.9, xy[1] + (0.9 if key == "route" else -0.9)))
        self.tag["route"].set_text("H1")
        self.tag["base"].set_text("baseline")
        run = self.route_run
        if t < run["steps"]:
            step = run["trace"][t]
            line = route_polyline(step, self.b["walk_table"])
            self.route_line.set_data(*(line.T if line is not None else ([], [])))
            self.goal_mark.set_data(*self.marks[[run["goal_landmark"]]].T)
            self._act_points(step)
        names = (("route", "recorded route (H1)", run), ("base", "baseline_raw", self.base_run))
        for k, (key, name, r) in enumerate(names):
            self.status[k].set_text(f"{name}: {short_status(r, t)}")
            self.status[k].set_color(th.route if key == "route" else th.baseline)
        self.small.set_position((60 / W, 1 - 1490 / H))
        self.small.set_text(
            "Picked by a rule fixed before recording: the first of 10 recorded pairs\n"
            "where only the recorded route reached the goal. In 1 of the 10, only\n"
            f"baseline_raw did. Real rollouts on {self.b['platform']}; Linux runs of this seed\n"
            "give other episodes (the dynamics are chaotic).")

    def s_numbers(self, u: float) -> None:
        th, b = self.th, self.board
        self.ax.set_visible(False)
        s = self.scores["route_medium_0_19"]
        o = self.b["outcomes"]["medium"]
        self.title.set_text("5,000 paired episodes")
        self.sub.set_text("antmaze-medium, eval seeds 0–19, 5 tasks: both\n"
                          "controllers start from the same 5,000 states.")
        grow = phase(u, 0.05, 0.4)
        rows = ((s["method_success"], "recorded route (H1)", th.route, 600),
                (s["reference_success"], "baseline_raw, the stronger baseline", th.baseline, 790))
        for value, name, color, y in rows:
            b.text(60, y - 60, name, fontsize=32 * PT, color=th.ink2)
            b.add_patch(FancyBboxPatch((60, y - 30), 960, 60, fc=th.grid, ec="none",
                                       boxstyle="round,pad=0,rounding_size=8"))
            width = 960 * value / 100 * grow
            if width > 1:
                b.add_patch(FancyBboxPatch((60, y - 30), width, 60,
                                           boxstyle="round,pad=0,rounding_size=8", fc=color,
                                           ec="none"))
            b.text(60 + max(width, 0) - 18, y + 14, f"{value * grow:.1f} %", ha="right",
                   fontsize=40 * PT, weight="bold", color=th.page if grow > 0.25 else th.ink)
        if u > 0.42:
            b.text(60, 940, f"{signed(s['diff'])} pp", fontsize=96 * PT, weight="bold",
                   color=th.ink)
            b.text(60, 995, f"95 % CI [{signed(s['low'])}, {signed(s['high'])}], "
                   f"p = {format_p(s['p'])} (paired)", fontsize=32 * PT, color=th.ink2)
        split = phase(u, 0.5, 0.85)
        if split > 0:
            parts = ((o["both"], "both", th.landmark), (o["only_method"], "only H1", th.route),
                     (o["only_reference"], "only baseline_raw", th.baseline),
                     (o["neither"], "neither", th.grid))
            total = sum(n for n, _, _ in parts)
            b.text(60, 1110, "Who reaches the goal, pair by pair", fontsize=34 * PT,
                   weight="bold", color=th.ink)
            x = 60.0
            for n, _, color in parts:
                w = 960 * n / total * split - 4
                if w > 0:
                    b.add_patch(FancyBboxPatch((x, 1140), w, 64,
                                               boxstyle="round,pad=0,rounding_size=6", fc=color,
                                               ec="none"))
                x += 960 * n / total * split
            if split > 0.95:
                for k, (n, name, color) in enumerate(parts):
                    x, y = 60 + 480 * (k % 2), 1270 + 56 * (k // 2)
                    b.add_patch(FancyBboxPatch((x, y - 24), 26, 26, fc=color, ec=th.axis,
                                               lw=1, boxstyle="round,pad=0,rounding_size=4"))
                    b.text(x + 40, y, f"{name} {n:,}", fontsize=31 * PT, color=th.ink)
        if u > 0.88:
            self.caption.set_position((60 / W, 1 - 1420 / H))
            self.caption.set_text(
                f"When only one reaches the goal, it is H1 in {o['only_method']:,}\n"
                f"pairs and baseline_raw in {o['only_reference']:,}.")
        self.small.set_text("Source: results/medium*/ (committed CSVs), fbplan.stats.")

    def s_limits(self, u: float) -> None:
        th, b = self.th, self.board
        self.ax.set_visible(False)
        self.title.set_text("Where it does not help")
        self.sub.set_text("Paired differences, percentage points, 95 % CI.")
        sc = self.scores
        rows = [
            (sc["route_large_10_19"], "Large maze, new eval seeds 10–19:\n"
             "recorded route vs baseline", th.route),
            (sc["weight_medium_fb_minus_xy"], "FB values as edge costs instead of xy\n"
             "length, medium (held-out goals)", th.fb),
            (sc["weight_large_fb_minus_xy"], "The same, large (held-out goals)", th.fb),
            (sc["a_planner_high"], "Approach A, FB graph planner (best variant):\n"
             "medium, vs baseline_raw", th.fb),
            (sc["b_clean"], "Approach B, learned multi-switch critic\n"
             "(B-clean): medium, vs baseline_raw", th.fb),
        ]
        lo, hi, x0, x1 = -20.0, 10.0, 80.0, 1000.0
        X = lambda v: x0 + (max(min(v, hi), lo) - lo) / (hi - lo) * (x1 - x0)  # noqa: E731
        top, step = 560, 170
        bottom = top + step * len(rows) - 60
        for v in range(-20, 11, 5):
            b.plot([X(v), X(v)], [top - 40, bottom], color=th.ink if v == 0 else th.grid,
                   lw=2.2 if v == 0 else 1.2, zorder=1)
            b.text(X(v), bottom + 42, signed(v, 0) if v else "0", ha="center",
                   fontsize=26 * PT, color=th.muted)
        for k, (s, label, color) in enumerate(rows):
            show = phase(u, 0.05 + 0.1 * k, 0.15 + 0.1 * k)
            if show <= 0:
                continue
            y = top + step * k
            b.text(x0, y - 8, label, fontsize=29 * PT, color=th.ink2, va="bottom",
                   linespacing=1.25)
            lo_v, hi_v = s["low"], s["high"]
            b.plot([X(lo_v), X(lo_v) + show * (X(hi_v) - X(lo_v))], [y + 40, y + 40],
                   color=color, lw=5, solid_capstyle="round", zorder=3)
            b.plot([X(s["diff"])], [y + 40], marker="o", ms=19, color=color, mec=th.page,
                   mew=3, zorder=4)
            p = format_p(s["p"])
            b.text(x1, y - 8, signed(s["diff"]) + (f", p = {p}" if p else ", pooled"),
                   ha="right",
                   va="bottom", fontsize=29 * PT, color=th.ink, weight="bold")
        if u > 0.6:
            self.caption.set_position((60 / W, 1 - 1440 / H))
            self.caption.set_text("Where FB values set the order, the result was worse\n"
                                  "(A, FB edge costs) or level (B). FB gives the goal,\n"
                                  "the intentions and the policy; the order comes\n"
                                  "from the recorded walks.")
        self.small.set_position((60 / W, 1 - 1680 / H))
        self.small.set_text("Source: REPORT.md §7–§8.7; results/ (committed CSVs).")

    def s_end(self, u: float) -> None:
        th, b = self.th, self.board
        self.ax.set_visible(False)
        b.text(W / 2, 760, "Code, report and every CSV", ha="center", fontsize=40 * PT,
               color=th.ink2)
        b.text(W / 2, 850, REPO, ha="center", fontsize=54 * PT, weight="bold", color=th.ink)
        b.text(W / 2, 1000,
               "Numbers: the committed CSVs in results/, with bootstrap\n"
               "CIs and exact McNemar tests (fbplan.stats).\n"
               "Episode: a real rollout of the committed code with the\n"
               f"published checkpoints, recorded on {self.b['platform']}.\n"
               "This video: uv run --group viz python -m fbplan.viz",
               ha="center", va="top", fontsize=30 * PT, color=th.ink2, linespacing=1.5)

    # ----- the cut -----

    def scenes(self) -> list[Scene]:
        steps = [r["steps"] for r in (self.base_run, self.route_run)]
        end, first = max(steps), min(steps)
        # 30 steps per second until 1 s after the first episode ends, then 25 steps per frame;
        # the last step stays for 2.5 s.
        switch = min(first + 30, end)
        clock = list(range(0, switch)) + list(range(switch, end, 25)) + [end] * 76
        return [
            Scene("task", 90, self.s_task),
            Scene("walks", 120, self.s_walks),
            Scene("edges", 165, self.s_edges),
            Scene("route", 120, self.s_route),
            Scene("act", 120, self.s_act),
            Scene("episode", len(clock), lambda u: self.s_episode(clock[min(int(u * len(clock)),
                                                                            len(clock) - 1)])),
            Scene("numbers", 270, self.s_numbers),
            Scene("limits", 240, self.s_limits),
            Scene("end", 120, self.s_end),
        ]

    def frames(self):
        """Every frame of the video in order, with cross-fades between scenes."""
        previous = None
        for scene in self.scenes():
            for k in range(scene.frames):
                self.reset()
                scene.draw(k / max(scene.frames - 1, 1))
                image = self.frame()
                if previous is not None and k < FADE:
                    a = (k + 1) / (FADE + 1)
                    image = (a * image + (1 - a) * previous).astype(np.uint8)
                yield image
            previous = image


def write_reel(bundle: dict[str, Any], path: Path, theme: Theme = DARK) -> int:
    """The video as H.264 MP4 (yuv420p, +faststart, no audio). Returns the frame count."""
    import imageio_ffmpeg  # only the video needs the encoder (dependency group viz)

    reel = Reel(bundle, theme)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(path), (W, H), fps=FPS, codec="libx264", macro_block_size=1, quality=None,
        pix_fmt_out="yuv420p",
        output_params=["-crf", "17", "-preset", "slow", "-profile:v", "high", "-movflags",
                       "+faststart"],
    )
    writer.send(None)
    count = 0
    for image in reel.frames():
        writer.send(np.ascontiguousarray(image))
        count += 1
    writer.close()
    plt.close(reel.fig)
    return count


def write_cover(bundle: dict[str, Any], path: Path, theme: Theme = DARK, t: int = 150) -> None:
    """One frame of the episode scene as a PNG, for the cover of the post."""
    reel = Reel(bundle, theme)
    reel.reset()
    reel.s_episode(t)
    plt.imsave(path, reel.frame())
    plt.close(reel.fig)
