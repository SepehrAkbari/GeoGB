"""Small CPU/CUDA contract tests; no policy training or benchmark runs."""

import io
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from deepgroebner.wrapped import CLeadMonomialsEnv
from geogb.environments import BuchbergerEnv
from geogb.models import PairMLP
from geogb.observations import PairBatch, PairObservation, collate_pairs
from geogb.rl import masked_categorical


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))]


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("distribution", ["3-3-3-weighted", "3-3-3-0.5-uniform", "3-3-3-maximum-homog"])
def test_environment_matches_native(distribution, k):
    env = BuchbergerEnv(distribution, k=k)
    native = CLeadMonomialsEnv(distribution, k=k)
    state, info = env.reset(seed=17)
    native.seed(17)
    expected = native.reset()
    np.testing.assert_array_equal(state, expected)
    assert state.dtype == np.int32 and state.shape[1] == 6 * k
    assert info["num_actions"] == len(state)
    np.testing.assert_array_equal(env.reset(seed=17)[0], state)
    assert env.value() == native.value()
    for t in range(300):
        action = (t * 3 + 1) % len(state)
        state, reward, terminated, truncated, info = env.step(np.int64(action))
        expected, expected_reward, done, _ = native.step(action)
        np.testing.assert_array_equal(state, expected)
        assert reward == expected_reward and np.isfinite(reward) and reward <= -1
        assert terminated == done and not truncated
        assert state.shape == (info["num_actions"], 6 * k)
        if terminated:
            assert state.shape == (0, 6 * k)
            assert env.value() == 0
            with pytest.raises(RuntimeError, match="reset"):
                env.step(0)
            break
    else:
        pytest.fail("small native episode did not terminate")


def test_environment_guards_and_truncation():
    for kwargs in [{"elimination": "none"}, {"rewards": "reductions"}, {"k": 0},
                   {"max_episode_steps": 0}]:
        with pytest.raises(ValueError):
            BuchbergerEnv(**kwargs)
    env = BuchbergerEnv("3-5-5-weighted", max_episode_steps=1)
    with pytest.raises(RuntimeError):
        env.step(0)
    with pytest.raises(RuntimeError):
        env.value()
    for seed in [-1, 2**31, 1.5, True]:
        with pytest.raises(ValueError, match="seed"):
            env.reset(seed=seed)
    state, _ = env.reset(seed=9)
    for action in [-1, len(state), 0.5, True]:
        with pytest.raises(ValueError, match="action"):
            env.step(action)
    state, _, terminated, truncated, _ = env.step(0)
    assert truncated == (not terminated)
    assert len(state) > 0 and truncated
    with pytest.raises(RuntimeError):
        env.step(0)
    with pytest.raises(ValueError):
        env.value(strategy="unknown")


def test_natural_termination_at_time_limit():
    reference = BuchbergerEnv("3-3-3-weighted")
    reference.reset(seed=7)
    count = 0
    while True:
        _, _, done, _, _ = reference.step(0)
        count += 1
        if done:
            break
    env = BuchbergerEnv("3-3-3-weighted", max_episode_steps=count)
    env.reset(seed=7)
    for _ in range(count):
        state, _, terminated, truncated, _ = env.step(0)
    assert terminated and not truncated and len(state) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_collation(device):
    x = np.arange(12, dtype=np.int32).reshape(3, 4)[::-1]
    y = np.zeros((1, 4), dtype=np.int64)
    observation = PairObservation(x)
    assert observation.features is x
    batch = collate_pairs([observation, y, np.empty((0, 4), dtype=np.int32)], device=device)
    assert batch.features.shape == (3, 3, 4)
    assert batch.features.dtype == torch.float32 and batch.mask.dtype == torch.bool
    assert batch.features.device.type == device
    assert batch.mask.cpu().tolist() == [[True] * 3, [True, False, False], [False] * 3]
    np.testing.assert_array_equal(batch.to("cpu").features[0].numpy(), x)
    assert batch.mask[1, 0]  # all-zero features are still a valid action
    assert collate_pairs([np.empty((0, 4), dtype=np.int32)]).features.shape == (1, 0, 4)


@pytest.mark.parametrize("x", [np.zeros(3, dtype=int), np.zeros((2, 0), dtype=int),
                                  np.array([[-1, -1]]), np.array([[1.5, 2.0]]),
                                  np.array([[True, False]])])
def test_bad_observation(x):
    with pytest.raises(ValueError):
        PairObservation(x)


def test_bad_batch():
    with pytest.raises(ValueError, match="empty"):
        collate_pairs([])
    with pytest.raises(ValueError, match="inconsistent"):
        collate_pairs([np.zeros((1, 2), int), np.zeros((2, 3), int)])
    x = torch.zeros(2, 3, 4)
    for features, mask in [(x.double(), torch.ones(2, 3, dtype=torch.bool)),
                           (x, torch.ones(2, 3)), (x, torch.ones(3, 2, dtype=torch.bool)),
                           (x[0], torch.ones(2, 3, dtype=torch.bool)),
                           (x - 1, torch.ones(2, 3, dtype=torch.bool))]:
        with pytest.raises(ValueError):
            PairBatch(features, mask)


@pytest.mark.parametrize("device", DEVICES)
def test_distribution(device):
    torch.manual_seed(3)
    logits = torch.tensor([[1., 2., 900.], [4., -900., 6.]], device=device, requires_grad=True)
    mask = torch.tensor([[True, True, False], [False, True, False]], device=device)
    dist = masked_categorical(logits, mask)
    torch.testing.assert_close(dist.probs.sum(-1), torch.ones(2, device=device))
    assert (dist.probs[~mask] == 0).all()
    actions = dist.sample((256,))
    assert mask.expand(256, -1, -1).gather(2, actions.unsqueeze(-1)).all()
    assert torch.isfinite(dist.log_prob(actions)).all()
    assert torch.isfinite(dist.entropy()).all()
    assert dist.entropy()[1] == 0
    loss = -dist.log_prob(actions).mean() - 0.1 * dist.entropy().mean()
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[~mask] == 0).all()
    assert logits.grad[0, :2].abs().sum() > 0
    assert torch.isneginf(dist.log_prob(torch.tensor([2, 0], device=device))).all()


@pytest.mark.parametrize("logits,mask", [
    (torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.bool)),
    (torch.zeros(1, 0), torch.zeros(1, 0, dtype=torch.bool)),
    (torch.zeros(2, 3), torch.ones(2, 3)),
    (torch.zeros(2, 3), torch.ones(2, 2, dtype=torch.bool)),
    (torch.tensor([[float('nan')]]), torch.ones(1, 1, dtype=torch.bool)),
    (torch.tensor([[float('inf')]]), torch.ones(1, 1, dtype=torch.bool)),
    (torch.ones(1, 1, dtype=torch.int32), torch.ones(1, 1, dtype=torch.bool)),
])
def test_bad_distribution(logits, mask):
    with pytest.raises(ValueError):
        masked_categorical(logits, mask)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("hidden_dims", [(), (128,), (16, 8)])
def test_model(device, hidden_dims):
    torch.manual_seed(31)
    a = np.arange(12, dtype=np.int32).reshape(3, 4)
    b = np.ones((1, 4), dtype=np.int32)
    batch = collate_pairs([a, b], device=device)
    model = PairMLP(4, hidden_dims).to(device).eval()
    scores = model(batch)
    assert scores.shape == (2, 3)
    permutation = [2, 0, 1]
    permuted = collate_pairs([a[permutation], b], device=device)
    torch.testing.assert_close(model(permuted)[0], scores[0, permutation])
    torch.testing.assert_close(model(collate_pairs([b], device=device))[0, 0], scores[1, 0])
    noisy = batch.features.clone()
    noisy[~batch.mask] = torch.nan
    torch.testing.assert_close(model(PairBatch(noisy, batch.mask)), scores)
    model.train()
    dist = masked_categorical(model(batch), batch.mask)
    loss = -dist.log_prob(torch.tensor([0, 0], device=device)).mean() - .01 * dist.entropy().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = PairMLP(4, hidden_dims).to(device)
    restored.load_state_dict(torch.load(buffer, map_location=device, weights_only=True))
    torch.testing.assert_close(restored(batch), model(batch))
    with pytest.raises(ValueError, match="no valid"):
        model(collate_pairs([np.empty((0, 4), dtype=np.int32)], device=device))
    with pytest.raises(ValueError, match="dimension"):
        model(collate_pairs([np.zeros((1, 3), dtype=np.int32)], device=device))


@pytest.mark.parametrize("device", DEVICES)
def test_short_trajectories(device):
    torch.manual_seed(42)
    model = PairMLP(12).to(device).eval()
    env = BuchbergerEnv("3-3-3-weighted", k=2)
    for seed in [1, 2, 3]:
        observation, _ = env.reset(seed=seed)
        for _ in range(300):
            batch = collate_pairs([PairObservation(observation)], device=device)
            with torch.no_grad():
                dist = masked_categorical(model(batch), batch.mask)
                action = dist.sample().item()
            observation, reward, terminated, truncated, _ = env.step(action)
            assert np.isfinite(reward) and not truncated
            if terminated:
                break
        else:
            pytest.fail("small integration episode did not finish")


def test_imports_do_not_load_tensorflow():
    code = """
import sys
from geogb.environments import BuchbergerEnv
from geogb.observations import collate_pairs
from geogb.models import PairMLP
from geogb.rl import masked_categorical
env = BuchbergerEnv('3-3-3-weighted')
x, _ = env.reset(seed=1)
batch = collate_pairs([x])
masked_categorical(PairMLP(x.shape[1])(batch), batch.mask).sample()
assert not any(k == 'tensorflow' or k.startswith('tensorflow.') for k in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())
