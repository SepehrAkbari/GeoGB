# Baseline validation

## Upstream source

- Repository: `dylanpeifer/deepgroebner`
- Imported commit: `94f3183efd290a400ab97908e52ac04de7b259de`
- GoogleTest submodule: `b4999a1e2ba8afad6f76e7bb2d16c9abaeb9411e`

## Initial native macOS baseline

The baseline was established on arm64 macOS 26.3 with Python 3.10.7,
TensorFlow 2.13.1, CMake 4.3.2, Apple Clang 17.0.0, and Macaulay2 1.26.06.
Python dependencies are resolved by UV and recorded in `uv.lock`.

- Python: 148 tests passed.
- C++: 51 tests passed across the polynomial, ideal, and Buchberger suites.
- Cython extension: built and imported successfully in editable mode.
- Algebra environment: a seeded one-epoch training smoke test completed.
- CartPole-v0: a seeded one-epoch training smoke test completed.
- LunarLander-v2: a seeded, length-limited training smoke test completed.
- Macaulay2: both local packages passed their complete package checks.
- Full thesis dataset: downloaded to the external SSD and exposed through the
  ignored repository-local `data` symbolic link.

## First archived-checkpoint evaluation

The archived policy for the `3-20-10-weighted` distribution was evaluated for
100 episodes with the original evaluation seeds and settings:

```bash
scripts/run_archived_eval.sh 100 first-replication-100
```

- Archived 10,000-episode mean return: `-84.847`.
- New 100-episode mean return: `-86.670`.
- New 100-episode mean episode length: `36.800`.

The mean-return difference is `-1.823`, which is smaller than one standard
error of the 100-episode run. Episode rows are not expected to be bit-for-bit
identical because this baseline uses newer numerical libraries and a different
platform. A 10,000-episode run is the next comparison before retraining.

## Compatibility-only changes

These changes do not alter the intended algorithms:

1. Replaced removed NumPy `np.float` aliases with the built-in `float` type.
2. Selected TensorFlow's legacy Adam optimizer, which matches the optimizer
   implementation used by the older TensorFlow code and avoids the slower new
   optimizer path on Apple Silicon.
3. Compared C++ probability vectors with a numerical tolerance rather than
   exact floating-point equality.
4. Replaced fixed seeded C++ random outputs with same-seed reproducibility and
   structural checks. The C++ standard does not guarantee identical
   `std::discrete_distribution` output across Linux `libstdc++` and macOS
   `libc++` implementations.

## Interpretation

Passing this baseline establishes that the imported implementation builds and
runs on the target Mac. It does not yet reproduce the thesis tables or plots.
That requires the archived Zenodo data and explicit reruns of the corresponding
v3 experiment configurations. Small numerical differences are expected from
the newer TensorFlow stack, Apple Silicon arithmetic, and platform-specific
random-number distribution implementations.
