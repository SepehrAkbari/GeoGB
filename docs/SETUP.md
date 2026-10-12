# GeoGB setup

## Native Apple Silicon environment

The baseline uses Python 3.10, TensorFlow 2.13.1, and Gym 0.23.1. Dependencies
are declared in `pyproject.toml` and locked in `uv.lock`.

From the repository root:

```bash
scripts/setup_macos.sh
scripts/check_baseline.sh
```

The setup script initializes GoogleTest and runs `uv sync --locked`.

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

## Notebook use

TensorFlow 2.13 constrains `typing-extensions`, so JupyterLab itself is not
installed into the research runtime. Register the environment as a kernel:

```bash
uv run python -m ipykernel install --user --name geogb --display-name "GeoGB"
```

Then use that kernel from an existing JupyterLab installation.
