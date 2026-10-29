# Brief: g-release (reproducibility, repository hygiene, code review)

Read `docs/design.md` first, then `README.md` and every `docs/status/*.md`.

## Goal

A reviewer clones the public repository, follows `README.md`, and gets every table and figure
of the report. The code is clean enough to pass a strict review: the assignment rejects
untidy solutions.

## Owns

`README.md`, `scripts/reproduce.sh`, `devbox.json`, `.github/`, `docs/status/g-release.md`.

## Steps

1. **One reproduce entry point.** `scripts/reproduce.sh` runs the pipeline in order:
   checkpoints, evaluations, tables, figures. Training is a separate stage, because it needs a
   GPU; by default the script uses the published trained checkpoints (a-frozen adds their
   download). A `--smoke` mode runs every stage on CPU in a few minutes (few episodes, few
   training steps), so a reviewer can check that the pipeline works. Add `devbox` scripts
   that call it. Keep `README.md` consistent with it.
2. **CI.** A GitHub Actions workflow that runs `uv sync`, `uv run ruff check .` and
   `uv run pytest` on CPU. Tests that need checkpoints or datasets skip. Ask the user before
   you add a workflow that downloads data.
3. **Code review, first pass now.** Review the whole repository against the code rules of
   `docs/design.md` §7: small typed functions, docstrings that say what and why, no dead code,
   no duplicated logic across components, consistent names, clear CLIs, tests that test
   behaviour. Do not edit files that other components own. Write each finding in your status
   file with the file, the line, the problem and a proposed fix, grouped by owner, and tell the
   user, so the coordinator can send them out.
4. **Process files.** `docs/briefs/` and `docs/status/` are workflow notes. Propose to the user
   what happens to them before submission (keep, move to `docs/dev/`, or remove), with a
   reason. Do not remove them yourself.
5. **Final pass before submission.** Repeat steps 3 and 4, and check `README.md` on a clean
   clone.

## Rules

- Do not add a git remote and do not push. The coordinator publishes the repository.
- `uv run ruff check .` and `uv run pytest` pass before each commit.

## Done when

`scripts/reproduce.sh --smoke` passes from a clean clone, CI is green, and the review findings
are resolved or accepted by the user.

## Stop and ask the user

Before a CI workflow that downloads data, before any change to files of other components, and
before the final submission.
