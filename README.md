# GeoGB

GeoGB is a research continuation of Dylan Peifer's
[DeepGroebner](https://github.com/dylanpeifer/deepgroebner) project. It studies
reinforcement-learning selection strategies for Buchberger's algorithm and
will contain the experiments and extensions developed for Sepehr Akbari's
thesis.

The imported upstream baseline is DeepGroebner commit
`94f3183efd290a400ab97908e52ac04de7b259de` (the v3-era thesis code). The
original Git history, GPL-3.0 license, and authorship are preserved. The
associated work is:

- Dylan Peifer, Michael Stillman, and Daniel Halpern-Leistner, "Learning
  selection strategies in Buchberger's algorithm," ICML 2020,
  [arXiv:2005.01917](https://arxiv.org/abs/2005.01917).
- Dylan Peifer's 2021 Cornell University PhD thesis.
- [DeepGroebner thesis data, Zenodo 4724144](https://doi.org/10.5281/zenodo.4724144).

## Quick start on Apple Silicon macOS

```bash
git clone --recurse-submodules https://github.com/SepehrAkbari/GeoGB.git
cd GeoGB
scripts/setup_macos.sh
source .venv/bin/activate
scripts/check_baseline.sh
```

The maintained native environment uses Python 3.10 and TensorFlow 2.13.1. The
original `deepgroebner.yml` remains unchanged as a record of Dylan's Python 3.7
environment.

See [docs/SETUP.md](docs/SETUP.md) for platform details and Macaulay2 setup.
See [docs/DATA.md](docs/DATA.md) before downloading the 17.2 GB thesis archive.
See [docs/BASELINE.md](docs/BASELINE.md) for the current validation record.

## Repository relationship

The expected Git remotes are:

```text
origin    https://github.com/SepehrAkbari/GeoGB.git
upstream  https://github.com/dylanpeifer/deepgroebner.git
```

`main` is the stable GeoGB branch. Compatibility work and research changes
should be developed on focused branches and merged only after the baseline
checks pass.

## Running experiments

The primary entry points are:

```bash
python scripts/train.py --help
python scripts/eval.py --help
```

For example, the original degree-strategy experiment is:

```bash
python scripts/train.py --distribution 3-20-10-weighted --value_model degree
```

Generated datasets, model checkpoints, and logs belong under ignored data or
artifact directories rather than in Git. Every thesis experiment should record
its command-line arguments, random seeds, source commit, dependency versions,
and summarized result.

To evaluate Dylan's archived `3-20-10-weighted` checkpoint without retraining:

```bash
scripts/run_archived_eval.sh 100 first-replication-100
```

Use a new run name each time. Results are written under the ignored
`artifacts/eval/` directory; the archived data remains unchanged.

## License and attribution

GeoGB retains DeepGroebner's GNU General Public License v3.0. Preserve upstream
authorship and cite the paper, thesis, code, and dataset when using or extending
the original work.
