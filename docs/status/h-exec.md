# Status: h-exec

## Done

- `docs/hypotheses/h-exec.md`: 11 cheap measurements (M1–M11), 5 ranked hypotheses (H1–H5) and
  10 rejected ones. All runs were cheap checks on macOS, 4 workers, at most about 2 minutes each,
  on held-out goals or custom teleport pairs; none on the eval tasks. No code changes.
- Main findings:
  - On medium, no test-time choice of `z` lowers the flips: EMA smoothing, holding `z`, and a
    safety reward in `z` leave flips unchanged or raise them. The evaluation flip hazard equals
    the train data's (0.23 per 1,000 steps); the action and the position in the data barely
    predict flips (AUC 0.65–0.66).
  - On teleport, a trap latent from the offline data halves the trap share (52.5 % → 26.5 %), but
    success does not move (15.0 % → 14.5 %): the saved episodes time out. H1 (rank 1) adds a time
    gate.
  - Unfreezing `π_l`: worth testing for pace on large and giant (H2), not for flips (H5); fair only
    if every row runs on the new `π_l`.
  - Paired episodes are almost independent (31 % discordant against 34 % for independent coin
    flips), so a gain must be about +4 pp to pass the rule.

## Next

- None. Waiting for the coordinator.

## Blockers

- None.

## Requests to the coordinator

- H1 needs teleport held-out goals (d-eval, `select_heldout_goals`).
- A Linux run of `baseline_raw` on the medium held-out goals would show whether its +3.8 pp on the
  eval tasks is goal-specific: on the held-out goals (Mac) it is −3.2 pp [−8.0, +1.5].
- Decision on scope for H2 and H5: is an unfrozen `π_l`, given to every row, in scope?
