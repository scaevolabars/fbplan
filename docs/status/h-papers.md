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

## Next

- None. Waiting for the coordinator.

## Blockers

- None.

## Requests to the coordinator

- G0 (h-papers.md §2): run the privileged oracle on medium eval seeds 5–9 on Linux (1,250
  episodes, under 0.5 CPU h), so that the headroom bound uses the final protocol. On eval seeds
  0–4 `baseline_raw` was +3.8 pp above `baseline`, on seeds 5–9 +0.3 pp; on the held-out goals the
  oracle is +4.0 pp [+0.3, +7.7] above `baseline`. The reading is fixed in §2.
- If H1 goes ahead: c-planner builds the controller (data-certified routes, about 100 lines).
