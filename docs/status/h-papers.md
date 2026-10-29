# Status: h-papers

Lens: the papers and repos that the assignment lists (Barreto et al. 2016; Touati and Ollivier
2021; Touati, Rapin and Ollivier 2022 with `controllable_agent`; Meta Motivo with `metamotivo`;
the FB π-Switch paper's future work). Brief: `docs/briefs/hypotheses.md`.

## Done

- Read the brief, design.md, REPORT.md, theory.md §11.4, the status files of c-planner, d-eval,
  e-learner and h-*, and the three other hypothesis files.
- Read the five papers in full (PDF text in the scratchpad) and the inference code of both repos
  (cloned to the scratchpad, not run). The Meta Motivo id of the start message (2410.20092) is the
  OGBench paper; Meta Motivo is 2504.11054.
- Two cheap checks (Mac, 4 workers, about 4 minutes of wall time; scratch scripts outside the repo;
  held-out goals and offline data only, no eval task):
  - M1: Meta Motivo's tracking window on the true route (privileged oracle). Window over the next
    1–3 cells: +1.0 pp [−2.2, +4.3] over the oracle's single waypoint (600 paired held-out
    episodes); longer windows score lower. Not a lever of its own.
  - M2: landmark edges certified by recorded trajectories, weighted by xy length, give near-true
    routes without the map: medium stretch median 1.00–1.01, 90 % 1.03–1.06; large median 1.02.
- `docs/hypotheses/h-papers.md`: gate G0, 5 ranked hypotheses (H1 recorded-route tracking first,
  P 0.10), 17 rejected ideas, a one-block summary per paper.

- G0 answered by the coordinator (d-eval 36a10c0): on 10 eval seeds the oracle is +1.0 pp
  [−1.2, +3.2] over `baseline_raw` on medium, so H1–H3 are dropped on medium by the fixed reading.
  The same gate for large is in design.md §8; d-eval runs it.

## Next

- None. The coordinator closed this session (2026-10-30).

## Blockers

- None.

## Requests to the coordinator

- None open. G0: answered (measured by d-eval).
- If the large gate opens H1: c-planner builds the controller (data-certified routes, about 100 lines).
