# GeoGB PyTorch foundation

`deepgroebner` retains Dylan Peifer's algebra engine and TensorFlow training
implementation. `geogb` is the new PyTorch research package in the same installed
distribution. Phase 1 established an independent per-pair MLP, observations,
batching, an environment adapter, and a masked categorical utility. Phase 2A adds
contextual Deep Sets and full-attention Set Transformer policies; Phase 2B adds
an Induced Set Transformer. See [set policy notes](SET_POLICIES.md).
PPO and graph policies remain deferred.
The original DeepGroebner sources and the `deepgroebner` branch are unchanged.

## Environment contract

`geogb.environments.BuchbergerEnv` delegates to
`deepgroebner.wrapped.CLeadMonomialsEnv`; it does not reimplement arithmetic,
pair elimination, or rewards. The mathematical state is (G, P). Observation row
i concatenates the k leading monomial exponent vectors for the two polynomials
in P[i]. Its shape is (len(P), 2*k*n). Missing monomials within a polynomial are
zero-filled by the existing engine. Zero feature rows are valid actions.

`reset(seed=...)` returns `(numpy_observation, info)`. `step(action)` returns
`(numpy_observation, reward, terminated, truncated, info)`. Integer row indices
pass through unchanged, with bounds checks before entering C++. NumPy int32
observations are returned intact; no tensors or GPU operations enter the engine.
Empty terminal observations retain shape (0, d). Reset and step info include
`num_actions`. Stepping before reset or after either episode boundary is rejected.

Only `elimination='gebauermoeller'` and `rewards='additions'` are supported:
the existing Cython constructor accepts other strings but ignores them. GeoGB
rejects them explicitly. Rewards remain `-1 - reduction_steps`. The sorting
flags and k are forwarded. Native binomial strings such as `3-3-3-weighted`,
general polynomial strings such as `3-3-3-0.5-uniform`, and native generator
suffixes are passed through. Distribution strings must be valid for the native
engine: its parser and recursive resampling are not hardened against malformed
or degenerate distributions. In particular a generator that can never produce
an initial S-pair cannot reset successfully. Do not use such distributions.

The native fixed/cyclic generator currently infers n from a zero-based maximum
variable index (in `FixedIdealGenerator`), so its feature width can be smaller
than the named cyclic variable count. The adapter preserves native output; use
random distributions for foundation validation. Repairing that upstream issue
and testing cyclic observations should precede cyclic research experiments.

Natural termination means P is empty. Optional `max_episode_steps` imposes an
adapter truncation limit; termination takes precedence at the same step.
Truncation observations still contain available actions. `value(strategy, gamma)`
forwards the native rollout estimator, including on the final underlying state.
It estimates the algebraic continuation, without applying the adapter time limit.
Deterministic strategies include first, degree, normal, and sugar. Random/sample
value strategies use native randomness not fully controlled by reset seeding.

`reset(seed=None)` continues the generator stream. Explicit seeds are nonnegative
signed 32-bit integers, matching the Cython seed interface. Seed the environment
and PyTorch separately. Native C++ random streams may differ between platforms;
fixed seeds guarantee reproducibility within a given software/platform setup,
not cross-platform identical ideals or trajectories.

## Observations and policies

`PairObservation(array)` validates an unpadded, nonnegative integer matrix and
retains the original array without copying. Treat it as read-only while in use.
`collate_pairs([observation, ...], device='cpu')` constructs a `PairBatch`:

- `features`: torch.float32, shape (B, P_max, d), zero-padded.
- `mask`: torch.bool, shape (B, P_max), True for available actions.

Rows retain action order. Feature dimensions must agree. `.to(device)` moves
both tensors. Collation converts integers to float32 only for neural inputs;
very large exponents can round, while the raw observation remains lossless.
Terminal states may be stored/batched but cannot be sent to a policy or action
distribution. Historical -1-padded arrays are not accepted as raw observations:
remove their known padding before collation, or construct a float32 `PairBatch`
with the explicitly known Boolean mask. Masks are never inferred from embeddings.

`PairMLP(feature_dim, hidden_dims=(128,), activation=torch.nn.ReLU)` is an ordinary
`nn.Module`. `model(batch)` returns raw (B, P_max) logits. Shared rowwise weights
make it permutation equivariant and independent of other pairs and padding.
Invalid output scores are zero, but still require the mask. There is no weight
conversion or numerical-equivalence claim relative to the historical model.
Standard `.to`, `.train`, `.eval`, and `state_dict` work normally.

`masked_categorical(logits, batch.mask)` returns PyTorch's Categorical with
invalid logits replaced by -inf. Invalid probability and gradient are exactly
zero. All-invalid states and nonfinite valid scores raise ValueError. PyTorch's
entropy implementation safely handles the zero-probability padded entries.
Future PPO should reuse this component. A future graph policy can take its own
structured observation and still return one raw logit per action; no shared
base class forces it through a dense pair representation.

## Installation and devices

Keep Python 3.10 and all historical direct dependency pins. PyTorch 2.1.2 is a
compatibility pin: it accepts typing-extensions 4.5.0, required by TensorFlow
2.13.1; newer PyTorch releases require a newer typing-extensions. This is an older
PyTorch release, not the latest maintained release. Modernizing the combined
TensorFlow/PyTorch dependency constraints is a separate prerequisite for using
newer PyTorch or GPUs unsupported by these wheels. No torch-geometric or RL
framework is installed. No unrelated existing dependencies were upgraded.

On Apple Silicon (CPU execution requires no GPU):

```bash
uv sync --locked
uv run --locked pytest -q tests/geogb
```

On remote Linux x86_64 with NVIDIA GPU, use the same commands. The locked PyPI
PyTorch 2.1.2 Linux wheel already depends on CUDA 12.1 runtime libraries and
Triton; no extra CUDA index or locally compiled CUDA toolkit is needed for
these operations. The machine still needs a compatible NVIDIA driver and GPU.
Check before selecting CUDA explicitly:

```bash
nvidia-smi
uv sync --locked
uv run --locked python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
uv run --locked pytest -q tests/geogb
```

CPU execution works even when CUDA is unavailable. CUDA tests run automatically
when available, otherwise skip. Linux CPU use still installs the locked NVIDIA
runtime dependencies. Linux ARM GPU environments are not validated by this pass.
The remote machine was not available for validation here.

For a future CUDA 11.8 wheel variant, PyTorch publishes the cu118 index. UV must
bind torch to that explicit index, gated to Linux, for example:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cu118", marker = "sys_platform == 'linux'" }]

[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true
```

This is an alternative project configuration, not needed for the committed
CUDA 12.1 PyPI lock. It requires `uv lock` followed by `uv sync --locked` and a
reviewed lock change; do not silently replace wheels underneath a locked
environment. A CPU-only Linux variant likewise needs a reviewed explicit CPU
index/source configuration. Keep macOS on PyPI, since it has no CUDA wheels.

Installation references:
[PyTorch wheel versions](https://pytorch.org/get-started/previous-versions/),
[UV PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/),
[PyTorch 2.1.2 dependency declarations](https://github.com/pytorch/pytorch/blob/v2.1.2/setup.py).

## Run an untrained policy

Run this with `uv run --locked python` from the repository. Set device to
`'cuda'` only after verifying CUDA availability; use `'cpu'` on the Mac.

```python
import torch
from geogb.environments import BuchbergerEnv
from geogb.observations import PairObservation, collate_pairs
from geogb.models import PairMLP
from geogb.rl import masked_categorical

# Change to torch.device('cuda') on a verified CUDA machine.
device = torch.device('cpu')
torch.manual_seed(42)
env = BuchbergerEnv('3-3-3-weighted', k=2)
observation, info = env.reset(seed=7)
model = PairMLP(observation.shape[1]).to(device).eval()
while len(observation):
    batch = collate_pairs([PairObservation(observation)], device=device)
    with torch.no_grad():
        action = masked_categorical(model(batch), batch.mask).sample().item()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

This selects actions without updating parameters. Preserve model configuration
alongside state_dict checkpoints in future experiments; do not commit checkpoints.

## Validation and next passes

```bash
uv sync --locked
uv run --locked pytest -q tests/geogb
scripts/check_baseline.sh
```

Focused tests cover native transition/reward parity with nontrivial action
indices, seeded resets, terminal/truncation boundaries, masks, malformed batches,
distribution gradients, MLP permutation/padding/batch consistency, in-memory
serialization, three short sampled episodes, and a fresh-process check that
GeoGB imports and inference never import TensorFlow. CPU tests are mandatory;
CUDA tests skip cleanly without a device. Baseline checks also exercise the
historical TensorFlow code, C++ tests, and Macaulay2 packages.

Before PPO, specify rollout storage, critic observations, termination versus
truncation bootstrapping, generator splits, RNG recording and checkpoint metadata.
Before GNNs, define their state context
and structured features while preserving P order. No training, architecture
comparisons, reproduction claims, or benchmarks belong to this foundation pass.
