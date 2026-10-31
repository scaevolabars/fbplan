"""Regenerate every output of fbplan.viz in docs/figures/viz/.

    uv run --group viz python -m fbplan.viz [record] [render] [--out docs/figures/viz] \
        [--cache runs/viz]

record: runs the 20 episodes of fbplan.viz.episodes (about 1 min on a laptop CPU), reads the
parts of the route and the scoreboard from the committed CSVs, and writes runs/viz/bundle.json.
It needs the medium checkpoint and train split (`python -m fbplan.frozen --datasets
antmaze-medium-navigate-v0`). render: the GIFs, the PNGs, the HTML page and the vertical video
(Instagram Reel, with its cover) from that bundle. With no stage, both run.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures" / "viz"
CACHE = ROOT / "runs" / "viz"
HERO_RULE = (
    "Pair shown, by a rule fixed before the recording: the first of 10 recorded pairs (tasks "
    "1–5, episodes 0–1) where only the recorded route reaches the goal."
)


def platform_name() -> str:
    system = {"Darwin": "macOS"}.get(platform.system(), platform.system())
    return f"{system} {platform.machine()}"


def commit() -> str:
    """The short commit of the code; "+changes" when fbplan/ differs from it."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    try:
        head = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--", "fbplan")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("+changes" if dirty else "")


def plain(obj: Any, digits: int = 5) -> Any:
    """JSON-ready: numpy to lists and numbers, floats to `digits` significant digits."""
    if isinstance(obj, dict):
        return {str(k): plain(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v, digits) for v in obj]
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist(), digits)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(f"{float(obj):.{digits}g}")
    return obj


def record(cache: Path) -> Path:
    """Roll out the episodes and collect everything that the outputs show into one bundle."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")  # as scripts/reproduce.sh: CPU, like the CSVs
    from fbplan.frozen import make_env
    from fbplan.maze import MazeGeometry
    from fbplan.viz import episodes, explain, scores

    pairs, ctrl = episodes.record_pairs()
    table = episodes.compact_walks(pairs, ctrl.xy)
    hero = episodes.hero_pair(pairs)
    env = make_env(episodes.ENV)
    maze = MazeGeometry.from_env(env)
    infos = env.unwrapped.task_infos
    bundle = {
        "env": episodes.ENV,
        "platform": platform_name(),
        "commit": commit(),
        "hero": hero,
        "hero_rule": HERO_RULE,
        "sample_every": explain.SAMPLE_EVERY,
        "maze": {
            "map": maze.maze_map, "unit": maze.unit, "offset": maze.offset_xy,
            "goal_radius": float(env.unwrapped._goal_tol),
        },
        "tasks": {
            k + 1: {"init_xy": info["init_xy"], "goal_xy": info["goal_xy"]}
            for k, info in enumerate(infos)
        },
        "parts": explain.parts(ctrl),
        "first_hops": [
            explain.first_hops(ctrl, np.asarray(pair["runs"]["recorded_route"]["trace"][0]["xy"]))
            for pair in pairs
        ],
        "walk_table": table,
        "pairs": pairs,
        "scores": scores.scoreboard(),
        "weight_effect": scores.scoreboard(scores.WEIGHT_EFFECT),
        "outcomes": {
            "medium": scores.paired_outcomes(scores.H1_M + scores.H1_M20,
                                             scores.RAW_M + scores.RAW_M20),
            "large": scores.paired_outcomes(scores.H1_L + scores.H1_L20,
                                            scores.BASE_L + scores.BASE_L20),
        },
    }
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "bundle.json"
    path.write_text(json.dumps(plain(bundle), separators=(",", ":")))
    return path


def render(cache: Path, out: Path) -> list[Path]:
    from fbplan.viz import figures, page, reel
    from fbplan.viz.theme import THEMES

    bundle = json.loads((cache / "bundle.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for theme in THEMES:
        for name, draw in (("hero", figures.hero_gif), ("explainer", figures.explainer_png),
                           ("scoreboard", figures.scoreboard_png)):
            suffix = "gif" if name == "hero" else "png"
            path = out / f"{name}_{theme.name}.{suffix}"
            draw(bundle, theme, path)
            written.append(path)
    path = out / "recorded_route.html"
    path.write_text(page.html(bundle))
    written.append(path)
    path = out / "reel.mp4"
    reel.write_reel(bundle, path)
    reel.write_cover(bundle, out / "reel_cover.png")
    written += [path, out / "reel_cover.png"]
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stages", nargs="*", help="record, render (default: both)")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--cache", type=Path, default=CACHE)
    args = parser.parse_args()
    unknown = set(args.stages) - {"record", "render"}
    if unknown:
        parser.error(f"unknown stages {sorted(unknown)}: use record, render")
    stages = args.stages or ["record", "render"]
    if "record" in stages:
        print(f"bundle: {record(args.cache)}")
    if "render" in stages:
        for path in render(args.cache, args.out):
            print(f"{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}: "
                  f"{path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
