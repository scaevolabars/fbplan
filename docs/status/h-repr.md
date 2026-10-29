# Status: h-repr

Lens: representations and the task latent (brief `docs/briefs/hypotheses.md`).

## Done

- Read the brief, design.md, REPORT.md, theory.md §11, approach-b.md and the status files.
- Cheap checks with the frozen model (scratch scripts outside the repo, CPU, ≤ 4 workers;
  details in the appendix of `docs/hypotheses/h-repr.md`):
  - The direction of `z_r` is stable and robust (radius, weighting, train split, pose filters):
    not a lever.
  - The norm of `z_r` at `π_h`'s input is a monotone look-ahead knob offline. On 600 paired
    medium held-out episodes, success peaks at the training norm (`c = 1`, 85.7 %); `c = 0.7`
    is −4.3 pp (p = 0.048).
  - `baseline_raw` on the same held-out episodes: −0.7 pp [−4.5, +3.3] against `baseline`. The
    +3.8 pp of the medium eval tasks does not replicate.
  - Absorbing states are in the offline data (flips: 21 % of medium train trajectories end
    flipped; the teleport trap, found without the map). The FB flip occupancy predicts a flip
    with AUC 0.64.
  - Large held-out goals 102 and 103: 74–77 % of the goal-ball states are flipped ants, so their
    `z_r` mostly encodes a flipped ant (for d-eval: filter the ball to upright states).
- `docs/hypotheses/h-repr.md`: 6 ranked hypotheses (region-mean intention targets first), a
  note on the medium bar, and the rejected ideas.

## Next

- Nothing. The coordinator closed this session (2026-10-30).

## Blockers

- None.

## Requests to the coordinator

- None open. Answered (2026-10-30): both baseline rows and every evaluated family get eval
  seeds 5-9 on Linux, and every final test uses 10 eval seeds, paired against both rows
  (design.md §8, main e3f7438). The note on large held-out goals 102 and 103 went to d-eval.
