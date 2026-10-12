#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

episodes="${1:-100}"
run_name="${2:-archived-checkpoint-${episodes}}"
policy="data/eval/3-20-10-weighted/3-20-10-weighted/policy.h5"
reference="data/eval/3-20-10-weighted/3-20-10-weighted/results.csv"
output="artifacts/eval/${run_name}"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Run scripts/setup_macos.sh first." >&2
  exit 1
fi
if [[ ! -f "$policy" || ! -f "$reference" ]]; then
  echo "Archived data is unavailable. Mount the external drive and check the data symlink." >&2
  exit 1
fi
if [[ -e "$output" ]]; then
  echo "Output already exists: $output" >&2
  echo "Choose a different run name as the second argument." >&2
  exit 1
fi

TF_CPP_MIN_LOG_LEVEL=1 .venv/bin/python scripts/eval.py \
  --environment RandomBinomialIdeal \
  --env_seed 123 \
  --distribution 3-20-10-weighted \
  --elimination gebauermoeller \
  --rewards additions \
  --k 2 \
  --use_cython True \
  --agent_seed 123 \
  --policy_model pmlp \
  --policy_kwargs '{"hidden_layers": [128]}' \
  --policy_weights "$policy" \
  --episodes "$episodes" \
  --max_episode_length 500 \
  --use_gpu False \
  --name "$run_name" \
  --datetag False \
  --logdir artifacts/eval

.venv/bin/python - "$reference" "$output/results.csv" <<'PY'
import csv
import statistics
import sys


def read_results(path):
    with open(path, newline="") as stream:
        return [(float(row["Return"]), int(row["Length"])) for row in csv.DictReader(stream)]


reference = read_results(sys.argv[1])
current = read_results(sys.argv[2])

for label, rows in (("Archived reference", reference), ("Current run", current)):
    returns = [row[0] for row in rows]
    lengths = [row[1] for row in rows]
    print(
        f"{label}: episodes={len(rows)}, "
        f"mean return={statistics.fmean(returns):.3f}, "
        f"mean length={statistics.fmean(lengths):.3f}"
    )
PY
