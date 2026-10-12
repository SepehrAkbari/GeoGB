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
if [[ -e .venv ]]; then
  echo ".venv already exists. Remove it before running a clean setup." >&2
  exit 1
fi

git submodule update --init --recursive
"$python_bin" -m venv .venv
PATH="$repo_root/.venv/bin:$PATH" \
  .venv/bin/python -m pip install \
    pip==25.2 setuptools==75.8.2 wheel==0.45.1 swig==4.1.1
PATH="$repo_root/.venv/bin:$PATH" \
  .venv/bin/python -m pip install -r requirements/macos-arm64.lock.txt
.venv/bin/python -m pip install --no-build-isolation -e .

echo
echo "DeepGroebner is installed. Activate it with: source .venv/bin/activate"
echo "Run the baseline checks with: scripts/check_baseline.sh"
