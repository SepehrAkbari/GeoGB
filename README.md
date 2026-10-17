# GeoGB

Structure-aware reinforcement learning research for S-pair selection, based on
Dylan Peifer's [DeepGroebner](https://github.com/dylanpeifer/deepgroebner).
The `deepgroebner` package retains the original environment and TensorFlow code;
the new `geogb` package provides PyTorch MLP, contextual Deep Sets, and
full-attention Set Transformer policies. See the
[PyTorch interface and setup notes](agent/PYTORCH_FOUNDATION.md) for usage and
validation, and the [set policy notes](agent/SET_POLICIES.md) for Phase 2A.
PyTorch PPO, ISAB, and graph policies are deferred to later passes.

## Setup

Requires UV, CMake, Macaulay2, and Python.

```bash
git clone --recurse-submodules https://github.com/SepehrAkbari/GeoGB.git
cd GeoGB
uv sync --locked
scripts/check_baseline.sh
```

## Data

Keep the extracted DeepGroebner archive outside this repository. Link its
`data` directory into the repository root

```bash
ln -s /absolute/path/to/deepgroebner/data data
```

## Run

```bash
scripts/run_archived_eval.sh 100 test-run
uv run python scripts/train.py --help
uv run python scripts/eval.py --help
```

## Citation

Dylan Peifer, Michael Stillman, and Daniel Halpern-Leistner. “Learning
Selection Strategies in Buchberger's Algorithm.” ICML 2020.
[arXiv:2005.01917](https://arxiv.org/abs/2005.01917)
