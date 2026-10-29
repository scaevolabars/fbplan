# Brief: hypothesis sessions (h-exec, h-learn, h-repr)

Read first: `docs/design.md` (§8 has the decisions and the gain rule), `REPORT.md`,
`docs/theory.md`, `docs/plans/approach-b.md`, every `docs/status/*.md`, and the canonical Linux
results in `results/` (`results/macos/` is the cross-platform check).

## Goal

Find the hypotheses most likely to give a **gain over the baseline** under the gain rule of
design.md §8: paired on the 5 eval tasks, 5 eval seeds, significant against both the normalized
baseline and `baseline_raw`, with no selection on the eval tasks (select on the held-out goals).

## State of evidence (short; details in the files above)

- Medium (Linux): baseline 77.0 %, `baseline_raw` 80.8 %, 5 retrained baselines 78.3 %,
  privileged oracle 81.3 % (+0.5 pp against `baseline_raw`). Large: baseline 37.4 %, oracle
  +1.9 pp (not significant). Giant: oracle 2.5 %. Teleport: bounded by the trap and the time limit.
- About 71 % of baseline failures on medium are flips (a `π_l` failure); most others are slow walks
  that hit the time limit.
- Approach A (landmark graph planner) is 4–12 pp below the baseline; ablations confirm its guards.
- Approach B (learned multi-switch value + AWR `π_h`): B clean 79.3 %, no gain. Its 5 seeds split
  (75.5–83.0 %), explained by a per-seed AWR temperature; an experiment at 2× AWR sharpness for
  both the baseline and B runs overnight.
- The task latent matters: on medium the raw upstream `z_r` beats the normalized one by +3.8 pp.

## Constraints

No generative model of states or trajectories. No new environment interaction for training
(offline OGBench data only). `F`, `B`, `π_l` frozen by default; a hypothesis that unfreezes one
must say why and what it costs, and must keep the comparison fair (the baseline gets the same
modules).

## Your output

`docs/hypotheses/<your-branch>.md`: 5–8 hypotheses, ranked by expected value
(probability of a gain × size of the gain / cost). For each:

1. **Claim**, one sentence.
2. **Mechanism**: why it could beat the stronger baseline, not only the weaker one.
3. **Evidence** for and against, from the committed results (cite the files).
4. **Cheap falsification test**: an offline check or a small run you can describe exactly
   (you may run it if it takes at most 10 minutes of CPU with at most 4 workers).
5. **Full test**: the exact runs, their cost in GPU and CPU hours, and how the gain rule decides.
6. **Constraint check**: frozen modules, offline data, no generative model.
7. **Prior work**, if a known method does this (for example HIQL, SoRB, QRL, CRL).

End with a list of the hypotheses you rejected, each with a one-line reason.

## Rules

- You own only `docs/hypotheses/<your-branch>.md` and `docs/status/<your-branch>.md`.
  No code changes, no training, no cloud jobs. Scratch scripts stay outside the repository.
- Do not repeat the lenses of the other two sessions; your lens is in your start message.
- Questions go to the coordinator: end your message with a line that starts with
  `QUESTION FOR COORDINATOR:` and end your turn. No question dialogs.
- Do not add a git remote and do not push. Time box: about 2 hours. Commit when done.
