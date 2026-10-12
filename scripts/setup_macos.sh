#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
  echo "UV is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

git submodule update --init --recursive
uv sync --locked

echo
echo "DeepGroebner is installed. Run Python commands with: uv run"
echo "Run the baseline checks with: scripts/check_baseline.sh"
