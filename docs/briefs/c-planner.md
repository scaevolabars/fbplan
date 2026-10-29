# Brief: c-planner (approach A: landmark graph planner)

Read `docs/design.md` first, especially §3 and §4.A.

## Goal

Implement approach A as controllers that follow the `Controller` protocol.

## Owns

`fbplan/option_model.py`, `fbplan/landmarks.py`, `fbplan/graph.py`,
`fbplan/controllers/planner.py`, `tests/test_option_model.py`, `tests/test_graph.py`,
`tests/test_planner.py`, `docs/status/c-planner.md`.

## Modules

- `option_model.py`: batched `ρ(s,w)`, `R(s,w;z_r)`, direct value `F(s,z_r)ᵀz_r` from an `FBModel`.
  Options: head aggregation (min / mean), with or without the dropped term.
  Clip `ρ` to [0, 1] and record how often clipping happens.
- `landmarks.py`: select N landmark observations from the offline data.
  At least: farthest-point sampling on xy (`obs[:2]`) and on `B(s)`. Keep full observations.
- `graph.py` (pure numpy): edge table N×N computed in batches; pruning (`ρ_min`, top-k);
  value iteration with an optional cost per hop; route extraction.
- `controllers/planner.py`:
  - `GraphPlannerController`: `reset` computes `V_H` for the task (cache the edge table per env);
    `act` scores all landmarks from `s`, picks `w*`, keeps `w*` until it is reached
    (`ρ(s,w*) ≥ ρ_reached`) or a timeout, then acts with `π_l(a|s,B(w*))`.
  - `GraphPlusHighController`: same route, but the action comes from `π_h(s, B(w*))` then `π_l`.
  - Configs as frozen dataclasses with defaults; every value in the report must come from a config.

## Tests

Develop against `fbplan/tabular/fake_fb.py` from b-theory. If it is not merged yet, write a
minimal fake in `tests/` and switch later. Test: VI on a known graph, pruning, route extraction,
commitment logic, invariance to the scale of `B(w)`.

## Real model (after a-frozen merges)

Route plots on the medium maze (xy) for the 5 tasks; FB edge count and clipping statistics;
a quick smoke run (5 tasks × 10 episodes) against the baseline with a minimal loop.
The full evaluation belongs to d-eval.

## Done when

Tests pass, route plots look sensible, smoke run done, numbers in the status file.

## Stop and ask the user

Before you change the math in design.md, and when a design choice has a real trade-off
(for example landmark count vs. per-step cost).
