"""The trap-aware gate at the start states of the teleport eval tasks (controller reset only).

    uv run python analysis/trap_gate.py [--weights 0 0.5 1 2 4] [--trap-radius 1.5]
        [--reset-seeds 5]

For each weight in z_c and each eval task, at the env reset states of `reset_seed(0, task, s)`,
s < reset-seeds (no episode is run):

- the gate's walk estimate `TrapAwareController.walk_steps` (inf where rho_walk <= 0): median,
  minimum and maximum over the reset states;
- F's trap occupancy `F(s, z_c)^T z_trap / d`, the mean over the F heads and the reset states.

docs/theory.md §11.4 uses these numbers.
"""

from __future__ import annotations

import argparse
import functools

import numpy as np
import pandas as pd

from fbplan.controllers.trap_aware import TrapAwareConfig, TrapAwareController
from fbplan.evaluate import make_tasks, reset_env, reset_seed, task_latent
from fbplan.frozen import load_dataset, load_model, make_env
from fbplan.stats import markdown_table

ENV = "antmaze-teleport-navigate-v0"
EVAL_SEED = 0


def _steps(x: float) -> str:
    return "inf" if not np.isfinite(x) else f"{x:.0f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0],
                        help="weights of z_trap in z_c (and in the intention)")
    parser.add_argument("--trap-radius", type=float, default=1.5,
                        help="trap rule radius (1.5: the selected config)")
    parser.add_argument("--reset-seeds", type=int, default=5,
                        help="reset states per task: episodes 0 .. reset-seeds - 1 of eval seed 0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model, env = load_model(ENV), make_env(ENV)
    data = load_dataset(ENV, "train")
    tasks = make_tasks(ENV, [1, 2, 3, 4, 5], functools.partial(task_latent, model))
    rows = []
    for weight in args.weights:
        config = TrapAwareConfig(weight=weight, trap_radius=args.trap_radius)
        controller = TrapAwareController.from_data(model, data, config)
        for task in tasks:
            controller.reset(task, seed=0)
            obs = np.stack([reset_env(env, task, reset_seed(EVAL_SEED, task.task_id, s))
                            for s in range(args.reset_seeds)])
            steps = np.array([controller.walk_steps(o) for o in obs])
            forward = np.asarray(model.forward(obs, np.tile(controller.z_c, (len(obs), 1))))
            occupancy = float((forward @ controller.z_trap).mean()) / model.latent_dim
            rows.append({"weight": weight, "task": task.task_id,
                         "walk steps, median": _steps(float(np.median(steps))),
                         "min": _steps(steps.min()), "max": _steps(steps.max()),
                         "trap occupancy": f"{occupancy:+.2f}"})
    print(markdown_table(pd.DataFrame(rows)))


if __name__ == "__main__":
    main()
