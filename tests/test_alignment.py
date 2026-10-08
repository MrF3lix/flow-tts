import numpy as np
import pytest
import torch

from test_tts.data.lengths import sequence_mask
from test_tts.models import alignment
from test_tts.models.alignment import gaussian_log_likelihood, generate_path, maximum_path, pool_path


def reference_path(value, t_x, t_y):
    """Literal port of Matcha's Cython maximum_path_each."""
    value = value.copy()
    path = np.zeros_like(value, dtype=np.int32)
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            v_cur = -1e9 if x == y else value[x, y - 1]
            v_prev = (0.0 if y == 0 else -1e9) if x == 0 else value[x - 1, y - 1]
            value[x, y] = max(v_cur, v_prev) + value[x, y]
    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[index, y] = 1
        if index != 0 and (index == y or value[index, y - 1] < value[index - 1, y - 1]):
            index -= 1
    return path


def random_case(seed):
    g = torch.Generator().manual_seed(seed)
    t_x = torch.tensor([7, 12, 3, 12])
    t_y = torch.tensor([30, 40, 3, 12])
    value = torch.randn(4, 12, 40, generator=g)
    mask = sequence_mask(t_x, 12)[:, :, None] & sequence_mask(t_y, 40)[:, None, :]
    return value, mask, t_x, t_y


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference(seed):
    value, mask, t_x, t_y = random_case(seed)
    path = maximum_path(value, mask).numpy()
    for b in range(4):
        ref = reference_path((value[b] * mask[b]).numpy(), int(t_x[b]), int(t_y[b]))
        np.testing.assert_array_equal(path[b], ref)


@pytest.mark.parametrize("seed", [0, 1])
def test_numpy_fallback_matches_numba(seed, monkeypatch):
    value, mask, _, _ = random_case(seed)
    fast = maximum_path(value, mask)
    monkeypatch.setattr(alignment, "HAVE_NUMBA", False)
    slow = maximum_path(value, mask)
    torch.testing.assert_close(fast, slow)


def test_path_is_monotonic_and_complete():
    value, mask, t_x, t_y = random_case(3)
    path = maximum_path(value, mask)
    for b in range(4):
        p = path[b, : t_x[b], : t_y[b]]
        assert torch.all(p.sum(0) == 1)  # every frame has exactly one token
        assert torch.all(p.sum(1) >= 1)  # every token has a frame
        token_of_frame = p.argmax(0)
        assert torch.all(token_of_frame.diff() >= 0) and torch.all(token_of_frame.diff() <= 1)


def test_generate_path_inverts_durations():
    durations = torch.tensor([[2, 1, 3, 0], [1, 1, 1, 1]])
    mask = sequence_mask(torch.tensor([3, 4]), 4)[:, :, None] & sequence_mask(durations.sum(1), 8)[:, None, :]
    path = generate_path(durations, mask.float())
    torch.testing.assert_close(path.sum(-1), (durations * sequence_mask(torch.tensor([3, 4]), 4)).float())


def test_pool_path_columns_sum_to_one():
    durations = torch.tensor([[3, 2, 4, 1, 1]])  # 11 mel frames -> 3 latent frames (12 with padding)
    frame_mask = sequence_mask(torch.tensor([11]), 12)
    path = generate_path(durations, torch.ones(1, 5, 12))
    pooled = pool_path(path, frame_mask, 4)
    torch.testing.assert_close(pooled.sum(1), torch.ones(1, 3))
    torch.testing.assert_close(pooled[0, :, 0], torch.tensor([0.75, 0.25, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(pooled[0, :, 2], torch.tensor([0.0, 0.0, 1 / 3, 1 / 3, 1 / 3]))


def test_gaussian_log_likelihood():
    mu, y = torch.randn(2, 5, 3), torch.randn(2, 5, 7)
    ll = gaussian_log_likelihood(mu, y)
    ref = torch.distributions.Normal(mu[:, :, :, None], 1.0).log_prob(y[:, :, None, :]).sum(1)
    torch.testing.assert_close(ll, ref, atol=1e-4, rtol=1e-5)
