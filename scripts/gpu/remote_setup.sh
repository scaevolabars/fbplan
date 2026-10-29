#!/usr/bin/env bash
# Prepare a fresh GPU instance: tools, the locked Python env with CUDA JAX, OGBench data.
# Runs on the instance (scripts/gpu/vast.sh push calls it). Safe to run again.
#
#   bash scripts/gpu/remote_setup.sh [maze ...]     default mazes: medium large
set -euo pipefail
cd "$(dirname "$0")/../.."

packages=(tmux curl ca-certificates libosmesa6)
if ! dpkg -s "${packages[@]}" >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq "${packages[@]}" >/dev/null
fi
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
# The OGBench maze env builds a MuJoCo renderer, which needs an OpenGL context. The instance has
# no display and its CUDA image has no NVIDIA EGL, so use software OpenGL (OSMesa).
export MUJOCO_GL=osmesa

# On Linux, pyproject.toml selects jax[cuda12], so this installs the CUDA build.
uv sync --frozen
uv run python -c "import jax; d = jax.devices(); assert d[0].platform == 'gpu', d; print(d)"

uv run python - "$@" <<'EOF'
import sys

import ogbench

mazes = sys.argv[1:] or ["medium", "large"]
ogbench.download_datasets([f"antmaze-{m}-navigate-v0" for m in mazes])
EOF

# The tests compare with upstream at CPU tolerances. On Ampere and newer GPUs, JAX multiplies
# float32 matrices in TF32 by default (differences of about 1e-3), so test at full precision.
# Training keeps the JAX default, as upstream main.py does (it does not set the precision).
JAX_DEFAULT_MATMUL_PRECISION=highest uv run pytest -q
