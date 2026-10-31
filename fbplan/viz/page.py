"""The self-contained HTML page of fbplan.viz: a replay of every recorded pair, the parts of the
route step by step, and the scoreboard. No network access: the data, the styles and the script
are inside the file (template: page.html next to this module).

Numbers go into the page as base64 of little-endian int16, scaled (xy by 100, angles by 1000);
-32768 stands for "none". The script decodes them with `i16`.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np

from fbplan.viz.figures import format_p, is_fb
from fbplan.viz.theme import CSS_FONTS, DARK, LIGHT, css_variables

TEMPLATE = Path(__file__).with_name("page.html")
NONE = -32768


def i16(values: Any, scale: float = 1.0) -> str:
    """Values (NaN for none) as base64 of int16 after scaling and rounding."""
    a = np.asarray(values, dtype=float).ravel() * scale
    if np.nanmax(np.abs(a), initial=0.0) > 32767:
        raise ValueError("a value does not fit in int16 at this scale")
    out = np.where(np.isnan(a), NONE, np.round(np.nan_to_num(a))).astype("<i2")
    return base64.b64encode(out.tobytes()).decode("ascii")


def polylines(lines: list[Any], scale: float = 100.0) -> dict[str, Any]:
    """Polylines as one encoded array of points and the start offset of each (in points)."""
    arrays = [np.asarray(line, dtype=float).reshape(-1, 2) for line in lines]
    offsets = np.cumsum([0] + [len(a) for a in arrays]).tolist()
    points = np.vstack(arrays) if arrays else np.empty((0, 2))
    return {"offsets": offsets, "points": i16(points, scale)}


def run_data(run: dict[str, Any]) -> dict[str, Any]:
    trace = run["trace"]
    out = {
        key: run[key]
        for key in ("method", "success", "steps", "flipped_since", "goal_landmark",
                    "final_goal_dist", "reset_seed")
    }
    out["xy"] = i16([s["xy"] for s in trace] + [run["final_xy"]], 100)
    out["yaw"] = i16([s["yaw"] for s in trace], 1000)
    out["up"] = i16([s["up"] for s in trace], 1000)
    if "route" in trace[0]:
        out["route"] = [s["route"] for s in trace]
        out["walks"] = [s["walk_ids"] for s in trace]
        window = [[c for p in s["window"] for c in (p if p is not None else (np.nan, np.nan))]
                  for s in trace]
        out["window"] = i16(window, 100)
    return out


def page_data(bundle: dict[str, Any]) -> dict[str, Any]:
    """What the page script reads: the bundle with its arrays encoded."""
    parts = bundle["parts"]
    scores = [
        {**s, "p_text": format_p(s["p"]), "fb": is_fb(s)}
        for s in bundle["scores"] + bundle["weight_effect"]
    ]
    return {
        "env": bundle["env"],
        "platform": bundle["platform"],
        "commit": bundle["commit"],
        "hero": bundle["hero"],
        "maze": bundle["maze"],
        "tasks": bundle["tasks"],
        "landmarks": np.round(np.asarray(parts["landmarks"]), 2).tolist(),
        "edges": i16(parts["edges"]),
        "sample": polylines(parts["walks"]),
        "sample_every": bundle["sample_every"],
        "counts": {k: parts[k] for k in ("n_trajectories", "n_rows", "max_steps", "certify_xy")},
        "window_xy": parts["window_xy"],
        "first_hops": bundle["first_hops"],
        "walk_table": polylines(bundle["walk_table"]),
        "pairs": [
            {
                "task_id": pair["task_id"],
                "episode": pair["episode"],
                "csv": pair["csv"],
                "runs": {m: run_data(r) for m, r in pair["runs"].items()},
            }
            for pair in bundle["pairs"]
        ],
        "scores": scores,
        "outcomes": bundle["outcomes"],
    }


def html(bundle: dict[str, Any]) -> str:
    data = json.dumps(page_data(bundle), separators=(",", ":"), ensure_ascii=False)
    replacements = {
        "/*LIGHT*/": css_variables(LIGHT),
        "/*DARK*/": css_variables(DARK),
        "/*FONTS*/": CSS_FONTS,
        "/*DATA*/": data.replace("</", "<\\/"),
    }
    text = TEMPLATE.read_text()
    for marker, value in replacements.items():
        if marker not in text:
            raise ValueError(f"the template has no {marker}")
        text = text.replace(marker, value)
    return text
