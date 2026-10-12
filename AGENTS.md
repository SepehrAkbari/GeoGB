# Repository rules

- Use UV for Python dependencies and commands.
- Declare dependencies in `pyproject.toml` and commit `uv.lock` changes.
- Keep datasets and generated artifacts out of Git.
- Preserve the `deepgroebner` branch as the baseline.
- Run `scripts/check_baseline.sh` before merging changes.
