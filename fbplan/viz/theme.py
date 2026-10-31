"""One palette for every output of fbplan.viz: the GIF, the PNGs and the HTML page.

The series colors are slots of the categorical palette that analysis/plots.py also uses:
orange for the recorded route, aqua for `baseline_raw` (its color in analysis/plots.py), violet
for the orders that come from FB values. Checked as a set, in both modes, with the dataviz
validator (all pairs): worst CVD dE 9.2 light, 9.4 dark. Aqua is below 3:1 on the light surface,
so every aqua mark has a text label next to it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Theme:
    name: str
    page: str
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    wall: str
    landmark: str
    walk: str  # recorded train trajectories
    route: str  # the recorded route (H1)
    baseline: str  # baseline_raw
    fb: str  # orders from FB values


LIGHT = Theme(
    name="light", page="#f9f9f7", surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e",
    muted="#898781", grid="#e1e0d9", axis="#c3c2b7", wall="#d9d8d2", landmark="#9c9b95",
    walk="#b9b8b1", route="#eb6834", baseline="#1baf7a", fb="#6250d6",
)
DARK = Theme(
    name="dark", page="#0d0d0d", surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7",
    muted="#898781", grid="#2c2c2a", axis="#383835", wall="#3a3a36", landmark="#77766f",
    walk="#5a5954", route="#d95926", baseline="#199e70", fb="#9085e9",
)
THEMES = (LIGHT, DARK)

# A humanist sans from the system: no font file ships with the outputs.
FONTS = ["Seravek", "Gill Sans Nova", "Gill Sans", "Ubuntu", "Calibri", "DejaVu Sans"]
CSS_FONTS = 'Seravek, "Gill Sans Nova", "Gill Sans", Ubuntu, Calibri, "DejaVu Sans", sans-serif'


def css_variables(theme: Theme) -> str:
    """The theme as CSS custom properties (--page, --surface, ...)."""
    return "".join(f"--{key}:{value};" for key, value in asdict(theme).items() if key != "name")
