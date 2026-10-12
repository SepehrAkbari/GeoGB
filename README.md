# DeepGroebner baseline

This branch contains Dylan Peifer's
[DeepGroebner](https://github.com/dylanpeifer/deepgroebner) code at commit
`94f3183efd290a400ab97908e52ac04de7b259de`, with macOS compatibility changes.

## Setup

Requires Apple Silicon macOS, Python 3.10, CMake, and Macaulay2.

```bash
git clone --branch deepgroebner --recurse-submodules https://github.com/SepehrAkbari/GeoGB.git
cd GeoGB
scripts/setup_macos.sh
source .venv/bin/activate
ln -s /Volumes/Sepehr-SSD/deepgroebner-data/deepgroebner/data data
scripts/check_baseline.sh
```

Evaluate the archived checkpoint:

```bash
scripts/run_archived_eval.sh 100 test-run
```

Training and evaluation options:

```bash
python scripts/train.py --help
python scripts/eval.py --help
```

## Citation

Dylan Peifer, Michael Stillman, and Daniel Halpern-Leistner. “Learning
Selection Strategies in Buchberger's Algorithm.” ICML 2020.
[arXiv:2005.01917](https://arxiv.org/abs/2005.01917)

Original repository: <https://github.com/dylanpeifer/deepgroebner>
