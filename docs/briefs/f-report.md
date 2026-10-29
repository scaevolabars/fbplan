# Brief: f-report (the report)

Read `docs/design.md` first, then `REPORT.md` (the skeleton), `docs/theory.md`,
`docs/plans/approach-b.md` and every `docs/status/*.md`.

## Goal

`REPORT.md`: the most important deliverable of the assignment. The reviewers read it to judge
the analysis, not only the numbers: why each behaviour happens, what worked and what did not.
The author must be able to defend every claim on a call.

## Owns

`REPORT.md`, `docs/figures/report/`, `analysis/report_figures.py` (only if the existing
analysis scripts cannot make a figure that the report needs), `docs/status/f-report.md`.

## Rules for the content

1. Every number comes from a committed file in `results/` (or `docs/figures/theory/`) and names
   it. No number from a status file or a chat goes into the report until its CSV is committed.
2. Every comparison against the baseline is paired, with the 95 % CI from `analysis/tables.py`.
   Say when a difference is not significant.
3. State negative results plainly (for example: approach A is below the baseline on medium) and
   explain them with evidence: failure classes, calibration, route plots, ablations.
4. Claims about the method come with the reason (theory section, tabular study, or a measured
   diagnostic). Mark a hypothesis as a hypothesis.
5. Prefer a figure or a table to long prose. English, short sentences, one term per thing,
   the notation of `docs/design.md` §3.
6. The report describes the method and the evidence. It does not describe the multi-session
   workflow, the coordinator, or GPU account details.

## Steps

1. Fill the sections whose inputs are final now: §2 setting, §3 theory summary, §4 approach A
   method, §5 approach B method, §6 protocol, and in §8 the gap to the paper, the medium
   headroom (oracle, failure classes), and the FB calibration and wormholes.
2. Keep a list in your status file of every result the report still waits for, with its owner
   (`docs/design.md` §6).
3. Fill §1, §7, §9 and §10 when the results arrive. Write §1 last.

## Done when

All sections are filled, every number traces to a committed file, and `uv run ruff check .`
passes if you added code.

## Stop and ask the user

Before a claim that changes the story of the report (for example a new headline result), and
when a needed result has no owner.
