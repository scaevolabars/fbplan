# fbplan

Planning over sequences of intentions with frozen forward-backward (FB) representations,
on OGBench AntMaze. Baseline: FB π-Switch ([paper](https://arxiv.org/abs/2605.13207),
code in `third_party/ssm`).

Status: work in progress. Design and notation: `docs/design.md`.

## Setup

```bash
git clone --recurse-submodules <this-repo> && cd fbplan
uv sync                 # or: devbox run setup
uv run pytest           # or: devbox run test
```

On a Linux machine with CUDA, `uv sync` installs the CUDA build of JAX.
