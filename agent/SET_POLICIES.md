# Phase 2A: contextual set policies

GeoGB now includes three native PyTorch policies: independent `PairMLP`,
`DeepSetsPolicy`, and `SetTransformerPolicy`. All accept `PairBatch`, return raw
(B, P) action logits in the original row order, and use the existing
`masked_categorical(logits, batch.mask)`. Invalid scores are zero; they must
still be masked when selecting actions. Every policy rejects any batch member
with no valid actions. The algebra environment and its action indices are unchanged.
No dependencies or lock entries changed in Phase 2A.

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
# Alternatives: make_policy('deep_sets', d, aggregation='sum') or 'mlp'.
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
and seeded short native Buchberger episodes. CUDA cases skip without a device.

## Reuse for ISAB and deferred work

`models/attention.py` provides `MultiheadAttentionBlock` with separate query and
context tensors/masks and `SetAttentionBlock` for self-attention. The former can
be reused for both future ISAB stages: inducing queries attend to pair context,
then pair queries attend to the induced context. Unequal lengths and separate
mask gradients are already tested. `models/common.py` provides rowwise
feedforward construction and pair-policy input validation. Inducing parameters,
ISAB assembly, and its factory entry are intentionally absent.

PPO, training infrastructure, graph networks, and Phase 2B remain deferred.
The existing PyTorch-version and native cyclic-generator caveats still apply.
Remote CUDA execution and scaling on large pair sets remain unverified here.
