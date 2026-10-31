"""Figures and an interactive page of the recorded route (REPORT §8.7), from committed data only.

    uv run python -m fbplan.viz            # every output in docs/figures/viz/

- `scores`: paired differences with CIs from the committed result CSVs (fbplan.stats).
- `episodes`: real rollouts of the recorded route and `baseline_raw` through the evaluation
  loop of the CSVs, with the route that the controller planned at every step.
- `explain`: the parts of the route (recorded walks, landmarks, certified edges) of one maze.
- `figures`, `page`: the GIF, the PNGs and the self-contained HTML page.
"""
