# GeoGB setup

## Native Apple Silicon environment

The maintained development baseline uses Python 3.10, TensorFlow 2.13.1,
Gym 0.23.1, and the versions in `requirements/macos-arm64.lock.txt`. Dylan's
original `deepgroebner.yml` is retained as an archival description of the
2021 environment.

From the repository root:

```bash
scripts/setup_macos.sh
source .venv/bin/activate
scripts/check_baseline.sh
```

The setup script initializes the pinned GoogleTest submodule, creates `.venv`,
installs the dependencies, and builds the package in editable mode.

## Macaulay2

Macaulay2 is required for the `m2/` packages and statistics-generation scripts,
but not for the Python or C++ unit tests. Verify it with:

```bash
M2 --version
```

Then start `M2` from the repository root and run:

```macaulay2
needsPackage("Ideals", FileName => "m2/Ideals.m2");
needsPackage("SelectionStrategies", FileName => "m2/SelectionStrategies.m2");
check Ideals
check SelectionStrategies
```

On the initial GeoGB setup machine, Homebrew's Macaulay2 installation could
not load `/opt/homebrew/opt/libffi/lib/libffi.8.dylib`. Repair the Homebrew
installation before running these checks, normally with:

```bash
brew reinstall libffi macaulay2
```

## Notebook use

TensorFlow 2.13 constrains `typing-extensions`, so JupyterLab itself is not
installed into the research runtime. Register the environment as a kernel:

```bash
.venv/bin/python -m ipykernel install --user --name geogb --display-name "GeoGB"
```

Then use that kernel from an existing JupyterLab installation.
