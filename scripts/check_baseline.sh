#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Run scripts/setup_macos.sh first." >&2
  exit 1
fi

TF_CPP_MIN_LOG_LEVEL=1 .venv/bin/python -m pytest -q
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure

if ! command -v M2 >/dev/null 2>&1 || ! M2 --version >/dev/null 2>&1; then
  echo "Macaulay2 is unavailable; see docs/SETUP.md." >&2
  exit 1
fi

M2 --no-debug --stop --silent -q \
  -e 'needsPackage("Ideals", FileName => "m2/Ideals.m2"); needsPackage("SelectionStrategies", FileName => "m2/SelectionStrategies.m2"); check Ideals; check SelectionStrategies; exit 0'
