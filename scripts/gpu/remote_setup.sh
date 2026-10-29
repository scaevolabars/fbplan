#!/usr/bin/env bash
# Prepare a fresh GPU instance: tools, the locked Python env with CUDA JAX, OGBench data.
# Runs on the instance (scripts/gpu/vast.sh push calls it). Safe to run again.
#
#   bash scripts/gpu/remote_setup.sh [maze ...]     default mazes: medium large
set -euo pipefail
cd "$(dirname "$0")/../.."

if ! command -v tmux >/dev/null || ! command -v curl >/dev/null; then
  apt-get update -qq && apt-get install -y -qq tmux curl ca-certificates >/dev/null
fi
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# On Linux, pyproject.toml selects jax[cuda12], so this installs the CUDA build.
uv sync --frozen
uv run python -c "import jax; d = jax.devices(); assert d[0].platform == 'gpu', d; print(d)"

uv run python - "$@" <<'EOF'
import sys

import ogbench

mazes = sys.argv[1:] or ["medium", "large"]
ogbench.download_datasets([f"antmaze-{m}-navigate-v0" for m in mazes])
EOF

uv run pytest -q
