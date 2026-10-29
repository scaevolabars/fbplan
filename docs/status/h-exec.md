# Status: h-exec

## Done

- Read the brief, design.md, REPORT.md, theory.md §11 and the failure CSVs.
- Offline data check: 10 % of the medium train states are flipped; 210 of 1,000 train episodes
  flip and none recovers. `B` separates flipped states (AUC 0.985).
- Flip hazard per 1,000 steps: data 0.23; non-hier 0.20–0.24; baseline 0.29–0.41 (from the CSVs).
- Rollout diagnostic (Mac, held-out goals, 400 episodes): `F(s,z)ᵀz_flip` predicts flips
  (AUC 0.71), but not better for the chosen `z` than for `z_r`.

## Next

- Small intervention runs on held-out goals (intention smoothing, hold, safety composition).
- Write docs/hypotheses/h-exec.md.

## Blockers

- None.

## Requests to the coordinator

- None.
