# GeoGB

Based on Dylan Peifer's
[DeepGroebner](https://github.com/dylanpeifer/deepgroebner) repository at commit
`94f3183efd290a400ab97908e52ac04de7b259de`.

## Setup

Requires UV, CMake, and Macaulay2.

```bash
git clone --recurse-submodules https://github.com/SepehrAkbari/GeoGB.git
cd GeoGB
uv sync --locked
scripts/check_baseline.sh
```

## Data

Keep the extracted DeepGroebner archive outside this repository. Link its
`data` directory into the repository root:

```bash
ln -s /absolute/path/to/deepgroebner/data data
```

The local `data` link and generated `artifacts` are ignored by Git. See
`docs/DATA.md` for the archive URL and checksum.

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
