"""Context, equivariance, masking, and trajectory tests without training."""

import io

import numpy as np
import pytest
import torch
from torch import nn

from geogb.environments import BuchbergerEnv
from geogb.models import (DeepSetsPolicy, InducedSetTransformerPolicy, PairMLP,
                         SetTransformerPolicy, make_policy)
from geogb.models.attention import MultiheadAttentionBlock, SetAttentionBlock
from geogb.observations import PairBatch, collate_pairs
from geogb.rl import masked_categorical


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))]
POLICIES = ["mean", "sum", "attention", "induced_1", "induced_3", "induced_5"]


def policy(kind, feature_dim=4, **kwargs):
    if kind.startswith("induced_"):
        return InducedSetTransformerPolicy(feature_dim, num_inducing=int(kind.split("_")[1]), **kwargs)
    if kind == "attention":
        return SetTransformerPolicy(feature_dim, **kwargs)
    return DeepSetsPolicy(feature_dim, aggregation=kind, **kwargs)


def observations():
    return [np.array([[0, 1, 2, 3], [4, 0, 3, 1], [2, 2, 0, 5]], dtype=np.int32),
            np.zeros((1, 4), dtype=np.int32),
            np.array([[3, 1, 4, 2], [1, 5, 0, 1]], dtype=np.int32)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", POLICIES)
def test_policy_contract(kind, device):
    torch.manual_seed(72)
    batch = collate_pairs(observations(), device=device)
    model = policy(kind).to(device).eval()
    with torch.no_grad():
        scores = model(batch)
        assert scores.shape == (3, 3) and scores.device.type == device
        assert torch.isfinite(scores).all() and (scores[~batch.mask] == 0).all()
        # Both no-grad inference and ordinary forward are checked.
        for i, observation in enumerate(observations()):
            single = model(collate_pairs([observation], device=device))[0]
            torch.testing.assert_close(scores[i, :len(observation)], single, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(model(batch), scores, atol=2e-6, rtol=2e-5)
    dist = masked_categorical(scores, batch.mask)
    torch.testing.assert_close(dist.probs.sum(-1), torch.ones(3, device=device))
    assert (dist.probs[~batch.mask] == 0).all()
    assert dist.probs[1, 0] == 1
    actions = dist.sample((64,))
    assert batch.mask.expand(64, -1, -1).gather(2, actions.unsqueeze(-1)).all()
    assert torch.isfinite(dist.log_prob(actions)).all() and torch.isfinite(dist.entropy()).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", POLICIES)
def test_equivariance_and_padding(kind, device):
    torch.manual_seed(32)
    model = policy(kind).to(device).eval()
    batch = collate_pairs(observations(), device=device)
    # NaN and huge padding must have no effect, including in non-prefix masks.
    features = torch.cat((batch.features, torch.full((3, 4, 4), torch.nan, device=device)), dim=1)
    mask = torch.cat((batch.mask, torch.zeros(3, 4, dtype=torch.bool, device=device)), dim=1)
    features[~mask] = torch.nan
    extended = PairBatch(features, mask)
    permutations = torch.tensor([[4, 2, 0, 5, 1, 6, 3],
                                 [3, 0, 6, 2, 4, 1, 5],
                                 [1, 3, 6, 0, 5, 2, 4]], device=device)
    permuted = PairBatch(features.gather(1, permutations.unsqueeze(-1).expand(-1, -1, 4)),
                         mask.gather(1, permutations))
    for no_grad in [False, True]:
        with torch.set_grad_enabled(not no_grad):
            original = model(batch)
            padded = model(extended)
            torch.testing.assert_close(padded[:, :3], original, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(model(permuted), padded.gather(1, permutations),
                                       atol=2e-6, rtol=2e-5)
            huge = features.masked_fill(~mask.unsqueeze(-1), 1e20)
            torch.testing.assert_close(model(PairBatch(huge, mask)), padded,
                                       atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", POLICIES)
def test_gradients_and_serialization(kind, device):
    torch.manual_seed(35)
    batch = collate_pairs(observations(), device=device)
    features = batch.features.clone().requires_grad_()
    model = policy(kind).to(device).train()
    logits = model(PairBatch(features, batch.mask))
    dist = masked_categorical(logits, batch.mask)
    # Also exercise scalar-score gradients, whose common shift cancels in a categorical.
    loss = (-dist.log_prob(torch.tensor([1, 0, 1], device=device)).mean()
            - .1 * dist.entropy().mean() + .01 * logits.square().mean())
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    if kind.startswith("induced_"):
        for block in model.blocks:
            assert block.inducing_vectors.grad.abs().sum() > 0
            assert block.induce.attention.in_proj_weight.grad.abs().sum() > 0
            assert block.distribute.attention.in_proj_weight.grad.abs().sum() > 0
    assert torch.isfinite(features.grad).all()
    assert (features.grad[~batch.mask] == 0).all()
    assert features.grad[batch.mask].abs().sum() > 0
    model.eval()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = policy(kind).to(device).eval()
    restored.load_state_dict(torch.load(buffer, map_location=device, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(restored(batch), model(batch))


@pytest.mark.parametrize("kind", POLICIES)
def test_reject_terminal_and_wrong_features(kind):
    model = policy(kind)
    terminal = np.empty((0, 4), dtype=np.int32)
    for arrays in [[terminal], [observations()[0], terminal]]:
        with pytest.raises(ValueError, match="no valid actions"):
            model(collate_pairs(arrays))
    with pytest.raises(ValueError, match="feature dimension"):
        model(collate_pairs([np.zeros((1, 2), dtype=np.int32)]))


@pytest.mark.parametrize("aggregation", ["mean", "sum"])
def test_deep_sets_formula(aggregation):
    # Hand-set phi to the identity and rho to a known linear expression.
    model = DeepSetsPolicy(2, embed_dim=2, encoder_depth=1, scorer_depth=1,
                           activation=nn.Identity, aggregation=aggregation)
    with torch.no_grad():
        model.encoder[0].weight.copy_(torch.eye(2))
        model.encoder[0].bias.zero_()
        model.scorer[0].weight.copy_(torch.tensor([[1., 2., 3., 4., 5.]]))
        model.scorer[0].bias.fill_(.5)
    arrays = [np.array([[1, 2], [3, 4]], dtype=np.int32), np.array([[2, 1]], dtype=np.int32)]
    batch = collate_pairs(arrays)
    scores = model(batch)
    for i, x in enumerate(arrays):
        context = x.mean(0) if aggregation == "mean" else x.sum(0)
        expected = x @ np.array([1., 2.]) + context @ np.array([3., 4.]) + 5 * np.log1p(len(x)) + .5
        torch.testing.assert_close(scores[i, :len(x)], torch.tensor(expected, dtype=torch.float32))
    assert scores[1, 1] == 0


@pytest.mark.parametrize("device", DEVICES)
def test_attention_keys_values_and_cross_attention(device):
    torch.manual_seed(22)
    block = MultiheadAttentionBlock(8, 2, feedforward_dim=16, dropout=0).to(device).eval()
    queries = torch.randn(2, 3, 8, device=device, requires_grad=True)
    context = torch.randn(2, 5, 8, device=device, requires_grad=True)
    query_mask = torch.tensor([[True, False, True], [False, True, False]], device=device)
    context_mask = torch.tensor([[False, True, True, False, True],
                                 [True, False, False, False, False]], device=device)
    result = block(queries, context, query_mask, context_mask)
    assert result.shape == (2, 3, 8)
    assert (result[~query_mask] == 0).all()
    for i in range(2):
        compact_q = queries[i:i+1, query_mask[i]]
        compact_kv = context[i:i+1, context_mask[i]]
        expected = block(compact_q, compact_kv,
                         torch.ones(compact_q.shape[:2], dtype=torch.bool, device=device),
                         torch.ones(compact_kv.shape[:2], dtype=torch.bool, device=device))
        torch.testing.assert_close(result[i, query_mask[i]], expected[0], atol=2e-6, rtol=2e-5)
    # Changing masked keys/values to arbitrary nonfinite values must not matter.
    poisoned = context.masked_fill(~context_mask.unsqueeze(-1), torch.nan)
    bad_queries = queries.masked_fill(~query_mask.unsqueeze(-1), torch.nan)
    torch.testing.assert_close(block(bad_queries, poisoned, query_mask, context_mask), result)
    changed = context.clone()
    changed[0, 1, 0] += 10
    assert not torch.allclose(block(queries, changed, query_mask, context_mask)[0], result[0])
    weights = torch.randn_like(result)
    (result * weights).sum().backward()
    assert torch.isfinite(queries.grad).all() and torch.isfinite(context.grad).all()
    assert (queries.grad[~query_mask] == 0).all()
    assert (context.grad[~context_mask] == 0).all()
    assert context.grad[context_mask].abs().sum() > 0
    for name, parameter in block.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["attention", "induced_3"])
def test_dropout(device, kind):
    torch.manual_seed(15)
    batch = collate_pairs(observations(), device=device)
    model = policy(kind, dropout=.5).to(device).train()
    first, second = model(batch), model(batch)
    assert not torch.allclose(first[batch.mask], second[batch.mask])
    assert (first[~batch.mask] == 0).all() and (second[~batch.mask] == 0).all()
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(batch), model(batch), rtol=0, atol=0)
    deterministic = policy(kind, dropout=0).to(device).train()
    torch.testing.assert_close(deterministic(batch), deterministic(batch), rtol=0, atol=0)
    # Confirm dropout reaches both attention weights and residual branches.
    blocks = ([model.blocks[0].block] if kind == "attention"
              else [model.blocks[0].induce, model.blocks[0].distribute])
    for block in blocks:
        assert block.attention.dropout == .5
        assert block.attention_dropout.p == .5 and block.feedforward_dropout.p == .5


@pytest.mark.parametrize("kind", POLICIES)
def test_context_reaches_other_actions(kind):
    torch.manual_seed(19)
    model = policy(kind).eval()
    batch = collate_pairs([observations()[0]])
    x = batch.features.clone().requires_grad_()
    model(PairBatch(x, batch.mask))[0, 0].backward()
    assert x.grad[0, 1:].abs().sum() > 0
    changed = batch.features.clone()
    changed[0, 1, 0] += 10
    assert not torch.allclose(model(PairBatch(changed, batch.mask))[0, 0],
                              model(batch)[0, 0], atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", POLICIES)
def test_short_buchberger_episodes(kind, device):
    torch.manual_seed(42)
    env = BuchbergerEnv("3-3-3-weighted", k=2)
    model = policy(kind, feature_dim=12).to(device).eval()
    for seed in [1, 2, 3]:
        observation, _ = env.reset(seed=seed)
        for _ in range(300):
            batch = collate_pairs([observation], device=device)
            with torch.no_grad():
                dist = masked_categorical(model(batch), batch.mask)
                action = dist.sample().item()
            observation, reward, terminated, truncated, _ = env.step(action)
            assert np.isfinite(reward) and not truncated
            if terminated:
                assert observation.shape == (0, 12)
                break
        else:
            pytest.fail("small policy episode did not terminate")


def test_factory_and_configuration():
    assert isinstance(make_policy("mlp", 4, hidden_dims=(8,)), PairMLP)
    assert isinstance(make_policy("deep_sets", 4), DeepSetsPolicy)
    transformer = make_policy("set_transformer", 4)
    assert len(transformer.blocks) == 2
    assert transformer.projection.out_features == 64
    assert transformer.blocks[0].block.attention.num_heads == 4
    assert transformer.blocks[0].block.feedforward[0].out_features == 128
    induced = make_policy("induced_set_transformer", 4)
    assert isinstance(induced, InducedSetTransformerPolicy)
    assert len(induced.blocks) == 2 and induced.projection.out_features == 64
    for block in induced.blocks:
        assert block.inducing_vectors.shape == (16, 64)
        assert block.induce.attention.num_heads == 4
        assert block.distribute.feedforward[0].out_features == 128
        assert block.induce.attention.dropout == .1
    batch = collate_pairs(observations())
    for model in [DeepSetsPolicy(4, 12, hidden_dim=17, encoder_depth=3, scorer_depth=4, activation=nn.Tanh),
                  SetTransformerPolicy(4, 12, num_heads=3, depth=3, feedforward_dim=19,
                                       activation=nn.GELU, dropout=0),
                  InducedSetTransformerPolicy(4, 12, num_heads=3, num_inducing=7,
                                              depth=3, feedforward_dim=19,
                                              activation=nn.GELU, dropout=0)]:
        assert model(batch).shape == (3, 3)
    with pytest.raises(ValueError, match="policy name"):
        make_policy("isab", 4)
    with pytest.raises(TypeError):
        make_policy("deep_sets", 4, typo=True)


@pytest.mark.parametrize("kwargs", [{"feature_dim": 0}, {"embed_dim": 0},
                                     {"encoder_depth": 0}, {"scorer_depth": 0},
                                     {"hidden_dim": 1.5}, {"encoder_depth": True},
                                     {"aggregation": "max"}])
def test_bad_deep_sets_configuration(kwargs):
    config = {"feature_dim": 4, **kwargs}
    with pytest.raises(ValueError):
        DeepSetsPolicy(**config)


@pytest.mark.parametrize("kwargs", [{"feature_dim": 0}, {"embed_dim": 0},
                                     {"num_heads": 0}, {"num_heads": 3},
                                     {"depth": 0}, {"depth": 1.5},
                                     {"feedforward_dim": -1}, {"dropout": -.1},
                                     {"dropout": 1.1}, {"dropout": float("nan")}])
@pytest.mark.parametrize("constructor", [SetTransformerPolicy, InducedSetTransformerPolicy])
def test_bad_transformer_configuration(kwargs, constructor):
    with pytest.raises(ValueError):
        constructor(**{"feature_dim": 4, **kwargs})


def test_attention_mask_validation_and_sab():
    torch.manual_seed(4)
    block = MultiheadAttentionBlock(8, 2, dropout=0)
    rows = torch.randn(2, 3, 8)
    mask = torch.ones(2, 3, dtype=torch.bool)
    for bad_mask in [mask.float(), mask[:, :2], torch.zeros_like(mask)]:
        with pytest.raises(ValueError):
            block(rows, rows, mask, bad_mask)
    with pytest.raises(ValueError):
        block(rows, rows[:1], mask, mask[:1])
    with pytest.raises(ValueError):
        block(rows[:, :, :4], rows, mask, mask)
    sab = SetAttentionBlock(8, 2, dropout=0)
    torch.testing.assert_close(sab(rows, mask), sab.block(rows, rows, mask, mask))
