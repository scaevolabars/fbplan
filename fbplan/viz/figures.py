"""The raster outputs of fbplan.viz, one per theme: the hero GIF, the explainer strip and the
scoreboard. They draw only what the bundle holds (fbplan.viz.__main__): recorded episodes,
the parts of the route and the scores from the committed CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.transforms import blended_transform_factory
from PIL import Image

from fbplan.maze import MazeGeometry, draw_walls
from fbplan.viz.theme import FONTS, Theme

DPI = 100
ROUTE_NAME = "recorded route (H1)"
BASE_NAME = "baseline_raw (FB π-Switch)"


def use_font() -> str:
    """The first font of FONTS that matplotlib finds."""
    found = {f.name for f in font_manager.fontManager.ttflist}
    name = next((f for f in FONTS if f in found), "DejaVu Sans")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [name]})
    return name


def maze_of(bundle: dict[str, Any]) -> MazeGeometry:
    m = bundle["maze"]
    return MazeGeometry(np.asarray(m["map"]), float(m["unit"]), tuple(m["offset"]))


def route_polyline(step: dict[str, Any], table: list) -> np.ndarray | None:
    """The agent's xy, then the recorded walk of each hop: the route of one step. [M, 2]"""
    if not step.get("walk_ids"):
        return None
    pieces = [np.asarray(step["xy"])[None]] + [np.asarray(table[k]) for k in step["walk_ids"]]
    return np.vstack(pieces)


def position(run: dict[str, Any], t: int) -> tuple[np.ndarray, float]:
    """xy and heading of the agent at step t; past its last step, the final state."""
    trace = run["trace"]
    if t < len(trace):
        return np.asarray(trace[t]["xy"]), trace[t]["yaw"]
    return np.asarray(run["final_xy"]), trace[-1]["yaw"]


def trail(run: dict[str, Any], t: int) -> np.ndarray:
    xy = np.asarray([s["xy"] for s in run["trace"][: t + 1]])
    if t >= len(run["trace"]):
        xy = np.vstack([xy, run["final_xy"]])
    return xy


def outcome_text(run: dict[str, Any], t: int) -> str:
    """The state of an episode at step t, from its trace: running, upside down, or the end."""
    if t < run["steps"]:
        return "upside down" if run["trace"][t].get("up", 1.0) < 0 else "running"
    if run["success"]:
        return f"reached the goal at step {run['steps']}"
    flipped = run.get("flipped_since")
    if flipped is not None:
        return f"upside down since step {flipped}; time limit, not at the goal"
    return f"time limit: not at the goal after {run['steps']} steps"


def schedule(steps: list[int], slow: int = 2, fast: int = 20, extra: int = 40) -> list[int]:
    """Frame steps: every `slow` steps until `extra` steps after the first episode ends, then
    every `fast` steps until the last one ends, then the last step."""
    end, switch = max(steps), min(min(steps) + extra, max(steps))
    return list(range(0, switch, slow)) + list(range(switch, end, fast)) + [end]


def math_font(name: str) -> None:
    """Subscripts such as $\\pi_l$ in the text font."""
    plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": name,
                         "mathtext.it": f"{name}:italic", "mathtext.bf": f"{name}:bold"})


PI_L, PI_H = r"$\pi_l$", r"$\pi_h$"


class Hero:
    """The two episodes of the hero pair, side by side, as matplotlib artists per frame."""

    W, H = 880, 640

    def __init__(self, bundle: dict[str, Any], theme: Theme) -> None:
        math_font(use_font())
        self.b, self.th = bundle, theme
        pair = bundle["pairs"][bundle["hero"]]
        self.pair = pair
        self.runs = {"base": pair["runs"]["baseline_raw"], "route": pair["runs"]["recorded_route"]}
        self.table = bundle["walk_table"]
        self.landmarks = np.asarray(bundle["parts"]["landmarks"])
        self.goal = np.asarray(bundle["tasks"][str(pair["task_id"])]["goal_xy"])
        steps = [r["steps"] for r in self.runs.values()]
        self.frames = schedule(steps)
        self.switch = min(min(steps) + 40, max(steps))
        self._build()

    def _y(self, y: float) -> float:
        return 1 - y / self.H

    def _build(self) -> None:
        th, maze, W = self.th, maze_of(self.b), self.W
        self.fig = plt.figure(figsize=(W / DPI, self.H / DPI), dpi=DPI, facecolor=th.surface)
        f, y = self.fig, self._y
        task, ep = self.pair["task_id"], self.pair["episode"]
        f.text(24 / W, y(36), "Same start, two controllers", fontsize=18, color=th.ink,
               weight="bold")
        f.text(24 / W, y(58), f"antmaze-medium, task {task}, eval seed 0, episode {ep}: "
               "both episodes start from the same reset seed", fontsize=10.5, color=th.ink2)
        self.clock = f.text(1 - 24 / W, y(36), "", fontsize=18, color=th.ink, ha="right",
                            weight="bold")
        self.speed = f.text(1 - 24 / W, y(58), "", fontsize=10.5, color=th.ink2, ha="right")
        heads = {
            "base": (BASE_NAME, f"{PI_H} picks a new intention at every step", th.baseline),
            "route": (ROUTE_NAME, f"{PI_L} acts on the route 4, 8 and 12 xy ahead", th.route),
        }
        self.art: dict[str, dict[str, Any]] = {}
        for key, x0 in (("base", 24), ("route", 456)):
            name, sub, color = heads[key]
            f.text(x0 / W, y(94), name, fontsize=12.5, color=th.ink, weight="bold")
            f.text(x0 / W, y(113), sub, fontsize=10.5, color=th.ink2)
            ax = f.add_axes([x0 / W, y(524), 400 / W, 400 / self.H])
            ax.set_facecolor(th.surface)
            draw_walls(ax, maze, th.wall)
            if key == "route":
                ax.plot(*self.landmarks.T, ls="none", marker="o", ms=2.2, mew=0,
                        color=th.landmark, zorder=1)
            start = np.asarray(self.runs[key]["trace"][0]["xy"])
            ax.plot(*start, marker="s", ms=8, mfc="none", mec=th.ink2, mew=1.4, zorder=3)
            ax.text(start[0] + 0.9, start[1] + 0.9, "start", fontsize=9.5, color=th.ink2, zorder=6)
            ax.add_patch(Circle(self.goal, self.b["maze"]["goal_radius"], fc=th.surface,
                                ec=th.ink, lw=1.8, zorder=4))
            ax.text(self.goal[0] - 1.0, self.goal[1] + 0.2, "goal", fontsize=9.5, color=th.ink,
                    ha="right", va="center", zorder=6)
            a: dict[str, Any] = {"ax": ax}
            if key == "route":
                a["route"], = ax.plot([], [], color=th.ink, lw=1.3, ls=(0, (3, 2.2)), zorder=3)
                a["hops"], = ax.plot([], [], ls="none", marker="o", ms=4.5, mfc=th.surface,
                                     mec=th.ink, mew=1.1, zorder=3)
                a["goal_mark"], = ax.plot([], [], ls="none", marker="D", ms=6, color=th.ink,
                                          zorder=4)
                a["window"], = ax.plot([], [], ls="none", marker="o", ms=8, mfc=th.surface,
                                       mec=color, mew=2.2, zorder=7)
            a["trail"], = ax.plot([], [], color=color, lw=2.2, solid_capstyle="round",
                                  solid_joinstyle="round", zorder=5)
            a["head"], = ax.plot([], [], color=th.ink, lw=1.8, solid_capstyle="round", zorder=8)
            a["ant"], = ax.plot([], [], ls="none", marker="o", ms=10, color=color,
                                mec=th.surface, mew=1.8, zorder=8)
            a["status"] = f.text(x0 / W, y(544), "", fontsize=11, color=th.ink)
            self.art[key] = a
        handles = [
            Line2D([], [], color=th.ink, lw=1.3, ls=(0, (3, 2.2))),
            Line2D([], [], ls="none", marker="o", ms=8, mfc=th.surface, mec=th.route, mew=2.2),
            Line2D([], [], ls="none", marker="D", ms=6, color=th.ink),
        ]
        labels = ["route (recorded walks)", "act points", "goal landmark"]
        f.legend(handles, labels, loc="upper left", bbox_to_anchor=(452 / W, y(552)), ncol=3,
                 frameon=False, fontsize=9.5, labelcolor=th.ink2, handlelength=1.8,
                 columnspacing=1.2, handletextpad=0.5, borderaxespad=0)
        out = self.b["outcomes"]["medium"]
        rule = ("Pair chosen by a fixed rule: the first of the 10 recorded pairs (tasks 1–5, "
                "episodes 0–1) where only the recorded route reaches the goal.")
        stats = ("In the 5,000 committed paired episodes (eval seeds 0–19) only the recorded "
                 f"route reaches the goal in {out['only_method']:,}, only baseline_raw in "
                 f"{out['only_reference']:,}.")
        where = (f"Recorded on {self.b['platform']} with the code at {self.b['commit']}. The "
                 "dynamics are chaotic: the Linux CSV row of this seed is another episode.")
        for k, line in enumerate((rule, stats, where)):
            f.text(24 / W, y(588 + 18 * k), line, fontsize=9.5, color=th.ink2)

    def draw(self, t: int) -> Image.Image:
        th = self.th
        self.clock.set_text(f"step {t}")
        self.speed.set_text("2 steps per frame" if t < self.switch else "20 steps per frame")
        for key, run in self.runs.items():
            a = self.art[key]
            xy, heading = position(run, t)
            path = trail(run, min(t, run["steps"]))
            a["trail"].set_data(*path.T)
            a["ant"].set_data([xy[0]], [xy[1]])
            text = outcome_text(run, t)
            down = text.startswith("upside down")
            a["ant"].set_markerfacecolor(th.surface if down else a["trail"].get_color())
            a["ant"].set_markeredgecolor(a["trail"].get_color() if down else th.surface)
            tip = xy + 0.9 * np.array([np.cos(heading), np.sin(heading)])
            a["head"].set_data([xy[0], tip[0]], [xy[1], tip[1]])
            a["status"].set_text(text)
            a["status"].set_color(th.ink2 if text == "running" else th.ink)
            if key != "route":
                continue
            live = t < run["steps"]
            step = run["trace"][min(t, run["steps"] - 1)]
            line = route_polyline(step, self.table) if live else None
            route = step.get("route", []) if live else []
            a["route"].set_data(*(line.T if line is not None else ([], [])))
            hops = self.landmarks[route[:-1]] if len(route) > 1 else np.empty((0, 2))
            a["hops"].set_data(*hops.T)
            a["goal_mark"].set_data(*self.landmarks[[run["goal_landmark"]]].T)
            points = [p for p in step.get("window", []) if p is not None] if live else []
            window = np.asarray(points).reshape(-1, 2)
            a["window"].set_data(*window.T)
        self.fig.canvas.draw()
        return Image.fromarray(np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy())


def write_gif(
    frames: list[Image.Image], durations: list[int], path: Path, colors: int = 48
) -> None:
    """Frames on one shared palette (no dithering), so that unchanged pixels stay equal."""
    probe = Image.fromarray(np.vstack([np.asarray(frames[k]) for k in (0, len(frames) // 2, -1)]))
    palette = probe.quantize(
        colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )
    quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(path, save_all=True, append_images=quantized[1:], duration=durations,
                      loop=0, optimize=False, disposal=1)


def hero_gif(bundle: dict[str, Any], theme: Theme, path: Path) -> None:
    hero = Hero(bundle, theme)
    frames = [hero.draw(t) for t in hero.frames]
    durations = [40] * len(frames)
    durations[0], durations[-1] = 1500, 4000
    plt.close(hero.fig)
    write_gif(frames, durations, path)


def hero_poster(bundle: dict[str, Any], theme: Theme, path: Path, t: int) -> None:
    """One frame of the hero as a PNG."""
    hero = Hero(bundle, theme)
    hero.draw(t).save(path, optimize=True)
    plt.close(hero.fig)


# ----- the explainer strip -----

def explainer_png(bundle: dict[str, Any], theme: Theme, path: Path) -> None:
    """Four panels: recorded walks; landmarks and the certified edges of one landmark; the route
    at the first step of the hero episode; the act points along the executed episode."""
    math_font(use_font())
    th, maze, parts = theme, maze_of(bundle), bundle["parts"]
    marks = np.asarray(parts["landmarks"])
    pair = bundle["pairs"][bundle["hero"]]
    run = pair["runs"]["recorded_route"]
    goal = np.asarray(bundle["tasks"][str(pair["task_id"])]["goal_xy"])
    first = run["trace"][0]
    W, H = 1400, 540
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=th.surface)
    panel, gap, x0 = 320, 26, 24
    edges = np.asarray(parts["edges"])
    hub = first["route"][0]
    out = edges[edges[:, 0] == hub, 1]
    texts = [
        ("Recorded walks",
         f"{parts['n_trajectories']:,} train trajectories, {parts['n_rows']:,} states.\n"
         f"Shown: every {bundle['sample_every']}th trajectory."),
        ("Landmarks and certified edges",
         f"{len(marks)} landmarks. u → w is certified when one\ntrajectory passes u, then w "
         f"within K = {parts['max_steps']} steps:\n{len(edges):,} edges. Shown: the {len(out)} "
         "out of one landmark,\nas straight lines; each has a recorded walk."),
        ("Shortest certified route",
         "From the agent to the goal landmark that F picks,\nweighted by xy length, laid "
         "along the recorded\nwalks. No FB value weighs an edge."),
        ("Act on the route ahead, plan again",
         f"{PI_L} acts on the mean latent of the points 4, 8\nand 12 xy ahead. The route is "
         "planned again\nat every step. Shown: every 40th step."),
    ]
    for k, (title, body) in enumerate(texts):
        left = x0 + k * (panel + gap)
        fig.text(left / W, 1 - 30 / H, str(k + 1), fontsize=15, color=th.route, weight="bold")
        fig.text((left + 20) / W, 1 - 30 / H, title, fontsize=13, color=th.ink, weight="bold")
        ax = fig.add_axes([left / W, 1 - (48 + panel) / H, panel / W, panel / H])
        ax.set_facecolor(th.surface)
        draw_walls(ax, maze, th.wall)
        fig.text(left / W, 1 - (48 + panel + 20) / H, body, fontsize=10.5, color=th.ink2,
                 va="top", linespacing=1.4)
        ax.add_patch(Circle(goal, bundle["maze"]["goal_radius"], fc=th.surface, ec=th.ink,
                            lw=1.4, zorder=6))
        if k == 0:
            lines = [np.asarray(w) for w in parts["walks"]]
            ax.add_collection(LineCollection(lines, colors=th.walk, linewidths=0.8, alpha=0.9))
        if k == 1:
            ax.plot(*marks.T, ls="none", marker="o", ms=2.6, mew=0, color=th.landmark)
            segs = [np.stack([marks[hub], marks[w]]) for w in out]
            ax.add_collection(LineCollection(segs, colors=th.route, linewidths=0.7, alpha=0.55))
            ax.plot(*marks[hub], marker="o", ms=7, color=th.route, mec=th.surface, mew=1.4)
        if k >= 2:
            ax.plot(*marks.T, ls="none", marker="o", ms=2.0, mew=0, color=th.landmark)
        if k == 2:
            line = route_polyline(first, bundle["walk_table"])
            ax.plot(*line.T, color=th.ink, lw=1.4, ls=(0, (3, 2.2)))
            hops = marks[first["route"][:-1]]
            ax.plot(*hops.T, ls="none", marker="o", ms=5, mfc=th.surface, mec=th.ink, mew=1.1)
            ax.plot(*marks[run["goal_landmark"]], marker="D", ms=6.5, color=th.ink)
            ax.plot(*np.asarray(first["xy"]), marker="o", ms=9, color=th.route, mec=th.surface,
                    mew=1.6, zorder=8)
        if k == 3:
            executed = trail(run, run["steps"])
            ax.plot(*executed.T, color=th.route, lw=2.0, alpha=0.9, solid_capstyle="round")
            for t in range(0, run["steps"], 40):
                step = run["trace"][t]
                pts = np.asarray([p for p in step["window"] if p is not None]).reshape(-1, 2)
                xy = np.asarray(step["xy"])
                if len(pts):
                    ax.plot([xy[0], *pts[:, 0]], [xy[1], *pts[:, 1]], color=th.ink2, lw=0.8)
                    ax.plot(*pts.T, ls="none", marker="o", ms=6, mfc=th.surface, mec=th.route,
                            mew=1.6, zorder=7)
                ax.plot(*xy, marker="o", ms=5, color=th.ink, mec=th.surface, mew=1.0, zorder=8)
    fig.text(x0 / W, 16 / H,
             f"antmaze-medium, K = {parts['max_steps']} (the medium configuration). Panels 3 and "
             "4: the recorded route in the episode of the hero animation (task "
             f"{pair['task_id']}, eval seed 0, episode {pair['episode']}), recorded on "
             f"{bundle['platform']}.", fontsize=10, color=th.ink2)
    fig.savefig(path, facecolor=th.surface, dpi=DPI)
    plt.close(fig)


# ----- the scoreboard -----

GROUPS = {
    "route": "The recorded route against the stronger baseline row (medium: baseline_raw; "
             "large: baseline)",
    "fb": "Orders from FB values, against baseline_raw on medium",
    "weight": "The same certified edges, weighted by xy length or by FB cost; held-out goals, "
              "against baseline",
}


def format_p(p: float | None) -> str:
    """A McNemar p as REPORT.md writes it: 0.16, 0.045, 0.003, 1e-5 (from 9.5e-06); "" for
    none."""
    if p is None:
        return ""
    if p >= 0.01:
        return f"{p:.2g}"
    if p >= 0.001:
        return f"{p:.1g}"
    two, exponent = f"{p:.1e}".split("e")  # 9.4759e-06 -> 9.5, -6, as analysis/tables.py
    mantissa, exponent = int(np.floor(float(two) + 0.5)), int(exponent)  # then half up: 1e-5
    if mantissa == 10:
        mantissa, exponent = 1, exponent + 1
    return f"{mantissa}e{exponent}"


def is_fb(score: dict[str, Any]) -> bool:
    """Whether FB values order the route of the row's method."""
    return score["group"] == "fb" or "FB" in score["label"]


def scoreboard_png(bundle: dict[str, Any], theme: Theme, path: Path) -> None:
    use_font()
    th, scores = theme, bundle["scores"]
    rows: list[tuple[str, Any]] = []
    for group in GROUPS:
        rows.append(("head", group))
        rows += [("row", s) for s in scores if s["group"] == group]
    W, row_h = 880, 23
    top, bottom = 112, 66
    H = top + bottom + row_h * len(rows)
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=th.surface)
    left, right = 290, 200
    ax = fig.add_axes([left / W, bottom / H, (W - left - right) / W, (H - top - bottom) / H])
    ax.set_facecolor(th.surface)
    ticks = range(-25, 11, 5)
    ax.set_xlim(-25.5, 11.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    for x in ticks:
        ax.axvline(x, color=th.grid, lw=0.8, zorder=0)
    ax.axvline(0, color=th.ink, lw=1.2, zorder=1)
    mixed = blended_transform_factory(fig.transFigure, ax.transData)
    for y, (kind, item) in enumerate(rows):
        if kind == "head":
            ax.text(24 / W, y + 0.15, GROUPS[item], transform=mixed, fontsize=10.5,
                    color=th.ink, weight="bold", va="center", zorder=5,
                    bbox={"fc": th.surface, "ec": "none", "pad": 2})
            continue
        s = item
        color = th.fb if is_fb(s) else th.route
        main = s["key"] == "route_medium_0_19"
        ax.text(36 / W, y, f"{s['maze']}, {s['label']}", transform=mixed, fontsize=10.5,
                va="center", color=th.ink if main else th.ink2,
                weight="bold" if main else "normal")
        ax.plot([s["low"], s["high"]], [y, y], color=color, lw=2.0, solid_capstyle="round",
                zorder=3)
        ax.plot(s["diff"], y, marker="o", ms=7.5, color=color, mec=th.surface, mew=1.6, zorder=4)
        p = format_p(s["p"])
        value = f"{s['diff']:+.1f} [{s['low']:+.1f}, {s['high']:+.1f}]"
        value += f"  p = {p}" if p else ""
        ax.text((W - right + 14) / W, y, value, transform=mixed, fontsize=10, va="center",
                color=th.ink if main else th.ink2, weight="bold" if main else "normal")
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", colors=th.muted, labelsize=9.5, length=0)
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([f"{x:+d}" if x else "0" for x in ticks])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("success minus the reference, percentage points (95 % CI; p: exact McNemar)",
                  fontsize=10, color=th.ink2)
    fig.text(24 / W, 1 - 32 / H, "Paired differences in success", fontsize=17, color=th.ink,
             weight="bold")
    fig.text(24 / W, 1 - 55 / H,
             "Each row compares two controllers on the same start states. The A and B rows pool "
             "3 landmark sets or 5 training seeds (no p).", fontsize=10.5, color=th.ink2)
    fig.text(24 / W, 1 - 74 / H, "Source: the committed CSVs in results/, with fbplan.stats; "
             "fbplan/viz/scores.py lists the files of every row.", fontsize=10.5, color=th.ink2)
    handles = [Line2D([], [], color=c, lw=2, marker="o", ms=7, mec=th.surface, mew=1.4)
               for c in (th.route, th.fb)]
    fig.legend(handles, ["order from recorded walks and xy length", "order from FB values"],
               loc="upper left", bbox_to_anchor=(24 / W, 1 - 82 / H), ncol=2, frameon=False,
               fontsize=10.5, labelcolor=th.ink, handlelength=2.2, columnspacing=2.0,
               borderaxespad=0)
    fig.savefig(path, facecolor=th.surface, dpi=DPI)
    plt.close(fig)
