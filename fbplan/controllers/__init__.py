"""Controllers: complete agents that follow `fbplan.contracts.Controller`.

A controller factory has the signature `factory(model: FBModel, **options) -> Controller`.
The evaluation CLI finds a factory by alias or by a "package.module:attribute" path, so a
new method needs no change here.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

from fbplan.contracts import Controller

ControllerFactory = Callable[..., Controller]

ALIASES = {
    "baseline": "fbplan.controllers.baseline:BaselineController",
    "nonhier": "fbplan.controllers.baseline:NonHierController",
    "oracle": "fbplan.controllers.oracle:make_oracle",
    "planner": "fbplan.controllers.planner:make_planner",
    "planner_high": "fbplan.controllers.planner:make_planner_high",
}


def resolve_controller(spec: str) -> ControllerFactory:
    """The factory for an alias in ALIASES or a "package.module:attribute" path."""
    path = ALIASES.get(spec, spec)
    module_name, sep, attribute = path.partition(":")
    if not sep:
        raise ValueError(f"unknown controller {spec!r}: use {sorted(ALIASES)} or module:attribute")
    return getattr(importlib.import_module(module_name), attribute)
