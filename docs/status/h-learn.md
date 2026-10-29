# Status: h-learn

## Done

- Read the brief, design.md, REPORT.md, theory.md, approach-b.md, all status files and the
  committed results.
- Offline check of the `π_h` AWR weights on medium (CPU, about 1 minute per run; scripts in the
  h-learn scratchpad, not committed): where the weight goes by goal distance, which advantage term
  carries the progress signal, and the effective sharpness (ESS) of each family. Results:
  `docs/hypotheses/h-learn.md` §0.
  - Upstream `π_h` (α = 0.1): 56 % of the goal-latent AWR weight is on samples already within the
    0.5 success radius of the goal; 4 % on goals 12 or more xy units away.
  - The long-range progress signal is the term `F(s,z_w)ᵀz`; B's `R = 0` removes it, and B's
    switch advantage anti-correlates with progress at 16–40 xy units (−0.16, −0.20).
  - B's trained temperatures gave goal-latent ESS 0.027–0.052 against the baseline's 0.168.
- `docs/hypotheses/h-learn.md`: 6 ranked hypotheses (H1 sharp one-switch without `V_H`, H2
  log-ratio proxy advantage, H3 test-time ensemble, H4 ESS-matched temperature, H5 distance-space
  B critic, H6 HIQL value), predictions P1–P3 for the 2× experiment (written at 23:45 MSK, before
  its result), a decision table by 2× outcome, and 11 rejected ideas.

- Coordinator decision (2026-10-29, after 23:45 MSK): H1 is the attribution control of the 2×
  experiment. e-learner trains it in the same overnight batch after the 2× families, and it is
  evaluated with them. H2–H6 wait for the shortlist after h-exec and h-repr.

## Next

- Stopped on the coordinator's instruction. When asked: compare the 2× results with P1–P3; run
  the cheap tests of H1 and H4 (CPU, about 2–5 minutes each) with the 2× calibration file.

## Blockers

- None.

## Requests to the coordinator

1. Answered (H1 added). The 2× experiment matches the std of `α·A`, not the effective sharpness. My offline estimate:
   goal-latent ESS about 0.03 for B at 2× against 0.35 for the baseline at 2× (h-learn.md §0 C,
   P3). So "B at 2× beats the baseline at 2×" would not show a gain of chaining by itself. I propose
   H1 (B with `V_H = V_1` at the same temperature, one GPU batch) as the attribution control.
2. The scratch scripts of §0 could become `analysis/awr_weights.py` (d-eval) if the report uses the
   numbers. I do not own `analysis/`.
