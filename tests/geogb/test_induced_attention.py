"""ISAB-specific mathematical, masking, and attention-shape contracts."""

import pytest
import torch

from geogb.models import InducedSetTransformerPolicy
from geogb.models.attention import InducedSetAttentionBlock
from geogb.observations import PairBatch


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("num_inducing", [1, 3, 5])
def test_two_stage_masks_and_compact_equivalence(device, num_inducing):
    torch.manual_seed(26)
    block = InducedSetAttentionBlock(8, 2, num_inducing=num_inducing, dropout=0).to(device).eval()
    features = torch.randn(2, 5, 8, device=device)
    mask = torch.tensor([[True, False, True, False, True],
                         [False, False, True, False, False]], device=device)
    features[~mask] = torch.nan
    calls = []

    def capture(module, args):
        calls.append(args)

    handles = [block.induce.register_forward_pre_hook(capture),
               block.distribute.register_forward_pre_hook(capture)]
    try:
        result = block(features, mask)
    finally:
        for handle in handles:
            handle.remove()
    inducing, original, inducing_mask, original_mask = calls[0]
    queries, context, query_mask, context_mask = calls[1]
    assert inducing.shape == (2, num_inducing, 8)
    assert inducing.stride(0) == 0  # both states share one expanded parameter
    torch.testing.assert_close(inducing[0], block.inducing_vectors)
    assert original is features and queries is features
    assert original_mask is mask and query_mask is mask
    assert inducing_mask.all() and context_mask.all()
    assert context.shape == (2, num_inducing, 8)
    assert torch.isfinite(context).all() and torch.isfinite(result).all()
    assert (result[~mask] == 0).all()
    for i in range(2):
        compact = features[i:i+1, mask[i]]
        expected = block(compact, torch.ones(compact.shape[:2], dtype=torch.bool, device=device))
        torch.testing.assert_close(result[i, mask[i]], expected[0], atol=2e-6, rtol=2e-5)
    # Nonzero gradients through both stages and the shared inducing parameters.
    weights = torch.randn_like(result)
    (result * weights).sum().backward()
    assert block.inducing_vectors.grad.abs().sum() > 0
    for stage in [block.induce, block.distribute]:
        for name, parameter in stage.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert stage.attention.in_proj_weight.grad.abs().sum() > 0
    assert not ({id(p) for p in block.induce.parameters()}
                & {id(p) for p in block.distribute.parameters()})


@pytest.mark.parametrize("device", DEVICES)
def test_shared_inducing_gradient_accumulates_across_states(device):
    torch.manual_seed(7)
    block = InducedSetAttentionBlock(8, 2, num_inducing=2, dropout=0).to(device).eval()
    features = torch.randn(2, 4, 8, device=device)
    mask = torch.tensor([[True, True, False, True], [False, True, False, False]], device=device)
    weights = torch.randn_like(features)
    batched_loss = (block(features, mask) * weights).sum()
    batched_grad = torch.autograd.grad(batched_loss, block.inducing_vectors)[0]
    separate_grads = []
    for i in range(2):
        loss = (block(features[i:i+1], mask[i:i+1]) * weights[i:i+1]).sum()
        separate_grads.append(torch.autograd.grad(loss, block.inducing_vectors)[0])
    torch.testing.assert_close(batched_grad, sum(separate_grads), atol=2e-6, rtol=2e-5)
    assert torch.isfinite(batched_grad).all() and batched_grad.abs().sum() > 0


@pytest.mark.parametrize("device", DEVICES)
def test_attention_never_uses_candidate_self_attention(device):
    torch.manual_seed(12)
    model = InducedSetTransformerPolicy(4, 8, num_heads=2, num_inducing=2,
                                        depth=3, dropout=0).to(device).eval()
    calls = []

    def capture(module, args):
        queries, keys, values = args
        calls.append((queries.shape[1], keys.shape[1], values.shape[1]))

    handles = []
    for block in model.blocks:
        handles.extend([block.induce.attention.register_forward_pre_hook(capture),
                        block.distribute.attention.register_forward_pre_hook(capture)])
    try:
        batch = PairBatch(torch.rand(2, 7, 4, device=device),
                          torch.ones(2, 7, dtype=torch.bool, device=device))
        result = model(batch)
    finally:
        for handle in handles:
            handle.remove()
    assert result.shape == (2, 7)
    assert calls == [(2, 7, 7), (7, 2, 2)] * 3
    assert not any(q == k == 7 for q, k, _ in calls)
    for left, right in zip(model.blocks, model.blocks[1:]):
        assert left.inducing_vectors is not right.inducing_vectors
        assert not ({id(p) for p in left.parameters()} & {id(p) for p in right.parameters()})


@pytest.mark.parametrize("num_inducing", [0, -1, True, 1.5])
def test_invalid_inducing_counts(num_inducing):
    for constructor in [InducedSetAttentionBlock, InducedSetTransformerPolicy]:
        with pytest.raises(ValueError, match="num_inducing"):
            constructor(8, num_inducing=num_inducing)


def test_inducing_initialization_and_empty_context():
    torch.manual_seed(2)
    block = InducedSetAttentionBlock(8, 2, num_inducing=3)
    assert isinstance(block.inducing_vectors, torch.nn.Parameter)
    assert block.inducing_vectors.requires_grad
    bound = (6 / (3 + 8)) ** .5
    assert block.inducing_vectors.abs().max() <= bound
    assert block.inducing_vectors.std() > 0
    torch.manual_seed(2)
    restored = InducedSetAttentionBlock(8, 2, num_inducing=3)
    torch.testing.assert_close(block.inducing_vectors, restored.inducing_vectors)
    for rows, mask in [(torch.zeros(2, 0, 8), torch.zeros(2, 0, dtype=torch.bool)),
                       (torch.zeros(2, 3, 8), torch.zeros(2, 3, dtype=torch.bool))]:
        with pytest.raises(ValueError, match="valid key"):
            block(rows, mask)
