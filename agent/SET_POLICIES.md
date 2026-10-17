# Phases 2A and 2B: contextual set policies

GeoGB now includes four native PyTorch policies: independent `PairMLP`,
`DeepSetsPolicy`, `SetTransformerPolicy`, and `InducedSetTransformerPolicy`.
All accept `PairBatch`, return raw
(B, P) action logits in the original row order, and use the existing
`masked_categorical(logits, batch.mask)`. Invalid scores are zero; they must
still be masked when selecting actions. Every policy rejects any batch member
with no valid actions. The algebra environment and its action indices are unchanged.
No dependencies or lock entries changed in Phases 2A or 2B.

## Contextual Deep Sets

For pair rows x_i, the policy computes

    h_i = phi(x_i)
    z = sum(mask_i * h_i) / p             # default mean aggregation
    logits_i = rho(concat(h_i, z, log(1 + p)))

Here p counts valid actions, excluding padding. `aggregation='sum'` omits the
division. Embeddings are masked after phi, so encoder biases cannot contribute
through padding. Phi applies the chosen activation after every Linear layer;
rho uses it only between Linear layers and returns an unrestricted scalar.

Constructor options:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `feature_dim` | required | Features per pair |
| `embed_dim` | 64 | Output width of phi |
| `encoder_depth` | 2 | Linear layers in phi, including its output |
| `scorer_depth` | 2 | Linear layers in rho, including scalar output |
| `hidden_dim` | 64 | Intermediate widths in phi and rho |
| `activation` | `nn.ReLU` | Zero-argument activation module constructor |
| `aggregation` | `'mean'` | `'mean'` or `'sum'` |

Depth one is valid. Mean context describes the average embedded pair; sum
context scales with cardinality. Both include log(1+p) explicitly. Unlike the
independent MLP, another pair can affect an action's score through shared context.
Pooling requires O(P) work for fixed network widths.

## Full-attention Set Transformer

Input rows are linearly projected to embed_dim, passed through stacked Set
Attention Blocks, and scored by a shared scalar Linear layer. There are no
positional embeddings, inducing points, or pooling decoder. Each SAB is
MAB(X, X), where this implementation uses post-normalized residuals:

    H = LayerNorm(Q + Dropout(MultiheadAttention(Q, K, V)))
    MAB(Q, K) = LayerNorm(H + Dropout(FF(H)))

K and V are the same context set. FF is Linear → activation → Linear. The
attention primitive also applies dropout to attention weights. LayerNorm acts
within each row's features, never across actions or batch members.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `feature_dim` | required | Features per pair |
| `embed_dim` | 64 | Projection and attention width |
| `num_heads` | 4 | Must divide embed_dim |
| `depth` | 2 | Number of SABs |
| `feedforward_dim` | 128 | FF intermediate width |
| `dropout` | 0.1 | Attention-weight and residual-branch probability |
| `activation` | `nn.ReLU` | FF activation constructor |

Full attention supplies a distinct, content-dependent context to each query,
rather than the single pooled context in Deep Sets. It has O(P²) interactions
per block. This pass makes no training-quality or runtime benchmark claims.

Masks use True for valid rows. `MultiheadAttentionBlock` converts context masks
to PyTorch's inverse `key_padding_mask`, excluding padded keys and values from
attention normalization. It sanitizes padded queries/context before projection
(including arbitrary NaN padding), and zeros invalid query outputs. Policies
also clear padding before input projection and after scoring.

## Induced Set Transformer (Phase 2B)

`InducedSetTransformerPolicy` replaces each SAB with a reusable
`InducedSetAttentionBlock`:

    U = MAB(I, X)
    X_prime = MAB(X, U)

Each ISAB owns an (m, d) `nn.Parameter` named `inducing_vectors`, initialized
with Xavier uniform (bounds ±sqrt(6/(m+d))). It is shared across observations
using `unsqueeze(0).expand(B, -1, -1)`, so gradients accumulate into one parameter
matrix. Different blocks have independent inducing vectors. The `induce` and
`distribute` MABs also have independent parameters. Neither stage collapses the
action dimension of the policy output or introduces positional embeddings.

Stage one uses all-valid inducing queries and the candidate validity mask for
keys/values. Stage two uses candidate queries with their original mask and an
all-valid induced context. Existing MAB padding sanitization protects both stages
from NaNs in padded rows. Both stages reuse the full-attention policy's residual,
LayerNorm, feedforward, and dropout conventions.

Constructor options match `SetTransformerPolicy`, plus `num_inducing=16`.
Defaults are embed_dim=64, num_heads=4, depth=2, feedforward_dim=128,
dropout=0.1, activation=nn.ReLU. Any positive inducing count is allowed, including
m < P, m = P, and m > P. Input projection and final per-action linear scoring
are unchanged. The factory name is `'induced_set_transformer'`:

```python
model = make_policy('induced_set_transformer', feature_dim=12, num_inducing=16)
```

Each block performs attention only with query/key lengths (m, P) and (P, m).
Attention work is approximately O(P*m*d), compared with O(P²*d) for SAB, plus
projection/feedforward costs. There is no candidate-to-candidate P×P attention
operation. For small P or m comparable to P, two cross-attention stages can cost
more than one SAB; lower asymptotic interaction counts alone do not establish
runtime or memory improvements on a particular device/kernel.

## Usage, reproducibility, and checks

Use `uv sync --locked` and the existing Python 3.10/PyTorch 2.1.2 setup described
in [foundation notes](PYTORCH_FOUNDATION.md). CPU requires no GPU. Move both
model and batch to the same device; use CUDA only when available.

```python
import torch
from geogb.models import make_policy
from geogb.observations import collate_pairs
from geogb.rl import masked_categorical

# observations is a list of nonterminal NumPy pair matrices from the adapter.
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch = collate_pairs(observations, device=device)
model = make_policy('set_transformer', batch.features.shape[-1]).to(device).eval()
# Alternatives: 'induced_set_transformer', 'deep_sets', or 'mlp'.
with torch.no_grad():
    actions = masked_categorical(model(batch), batch.mask).sample()
```

Model-specific keywords pass directly through the factory. Save constructor
settings with `state_dict` in future experiments. Seed environment generation
separately. Evaluate in `.eval()` mode: dropout is disabled, and permutation,
padding, and batch/single predictions agree up to floating-point rounding.
Training-mode dropout draws independent masks, so individual stochastic forward
passes need not be permutation equivariant or match different batch layouts.
Deep Sets has no built-in dropout. No training loop is implemented here.

```bash
uv run --locked pytest -q tests/geogb
scripts/check_baseline.sh
```

Tests check the Deep Sets formula directly, non-prefix masks, arbitrary padding,
key/value exclusion, cross-attention with unequal query/context lengths,
context dependence, single-action states, terminal rejection, permutation
and batch consistency, gradients, dropout, serialization, distribution use,
and seeded short native Buchberger episodes. ISAB tests include m=1, 3, and 5
against three-action states, exact stage-mask/shape contracts, independent stage
parameters, shared inducing-gradient accumulation, and nonzero attention and
inducing gradients in stacked policies. CUDA cases skip without a device.

## Optional forward-pass diagnostic

The diagnostic is opt-in and is not run by pytest:

```bash
uv run --locked python scripts/diagnose_attention.py --sizes 32 128 512
uv run --locked python scripts/diagnose_attention.py --device cuda --sizes 128 512 2048 --num-inducing 16
```

It compares matching widths, heads, depth, and feedforward dimensions with dropout
disabled, using synthetic all-valid float32 batches. Configuration/version/device
metadata goes to stderr; CSV results go to stdout. Options include batch size,
embedding dimensions, inducing count, warmups, repeats, seed, and CPU threads.
Timing is the median synchronized forward duration, excluding construction,
transfer, warmup, and profiling. No wall-clock assertions are made.

On CPU, `cpu_allocation_bytes` is the sum of positive self-allocation events in
one separately profiled forward. It measures allocation traffic, including
temporary tensors; it is **not** peak live memory or process RSS. On CUDA,
`cuda_incremental_peak_bytes` is peak allocated tensor memory above the warmed-up
model/input baseline, excluding cached reserved memory and non-PyTorch allocations.
The unused device-specific memory column is blank. `attention_interactions` is a
theoretical B×heads×depth×P² (SAB) or B×heads×depth×2Pm (ISAB) count, not measured
allocated attention weights; PyTorch may use fused attention kernels.

Large sizes can exhaust memory; choose sizes for the available machine. Results
describe untrained evaluation forwards only and do not predict PPO backward-pass
memory or speed. This pass only smoke-checks tiny CPU inputs, not large benchmarks.

## Reusable components and deferred work

`models/attention.py` provides `MultiheadAttentionBlock` with separate query and
context tensors/masks, `SetAttentionBlock` for self-attention, and
`InducedSetAttentionBlock` for two-stage inducing attention. `models/common.py`
provides rowwise feedforward construction and pair-policy input validation.

PPO, training infrastructure, and graph networks remain deferred.
The existing PyTorch-version and native cyclic-generator caveats still apply.
Remote CUDA execution and scaling on large pair sets remain unverified here.
