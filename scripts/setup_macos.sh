#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-/Library/Frameworks/Python.framework/Versions/3.10/bin/python3}"
if [[ ! -x "$python_bin" ]]; then
  echo "Python 3.10 was not found at $python_bin" >&2
  echo "Set PYTHON_BIN to a Python 3.10 executable and run this script again." >&2
  exit 1
fi

git submodule update --init --recursive
"$python_bin" -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements/macos-arm64.txt
PATH="$repo_root/.venv/bin:$PATH" \
  .venv/bin/python -m pip install -r requirements/macos-box2d.txt
.venv/bin/python -m pip install --no-build-isolation -e .

echo
echo "GeoGB is installed. Activate it with: source .venv/bin/activate"
echo "Run the baseline checks with: scripts/check_baseline.sh"
