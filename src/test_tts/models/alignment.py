"""Monotonic alignment search (Glow-TTS / Matcha) and the text -> latent expansion.

Why alignment happens at the *mel* rate: at the latent rate (93.75 / 4 = 23.4 frames/s) the
Bernese corpus has up to 2 characters per latent frame (4 with blanks), and a monotonic
alignment needs at least one frame per token. At the mel rate the worst case is 0.5. So tokens
are aligned to mel frames, and the hard mel-rate path is average-pooled by the VAE compression
`r` into a soft token -> latent-frame assignment (`pool_path`). A latent frame then sees the
average of the text features of the 4 mel frames it covers; tokens shorter than a latent
frame are mixed in proportionally instead of being dropped.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from test_tts.data.lengths import sequence_mask

_NEG = -1e9

try:
    import numba

    @numba.njit(cache=True, nogil=True)
    def _maximum_path_each(path, value, t_x, t_y):  # pragma: no cover - compiled
        index = t_x - 1
        for y in range(t_y):
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                v_cur = _NEG if x == y else value[x, y - 1]
                if x == 0:
                    v_prev = 0.0 if y == 0 else _NEG
                else:
                    v_prev = value[x - 1, y - 1]
                value[x, y] = max(v_cur, v_prev) + value[x, y]
        for y in range(t_y - 1, -1, -1):
            path[index, y] = 1
            if index != 0 and (index == y or value[index, y - 1] < value[index - 1, y - 1]):
                index = index - 1

    @numba.njit(cache=True, parallel=True)
    def _maximum_path_batch(paths, values, t_xs, t_ys):  # pragma: no cover - compiled
        for b in numba.prange(paths.shape[0]):
            _maximum_path_each(paths[b], values[b], t_xs[b], t_ys[b])

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False


def _maximum_path_numpy(value: np.ndarray, t_x: np.ndarray, t_y: np.ndarray) -> np.ndarray:
    """Same DP, vectorised over batch and text positions (loop over frames only)."""
    B, Tx, Ty = value.shape
    xs = np.arange(Tx)[None, :]
    acc = np.empty_like(value)
    prev = None
    for y in range(Ty):
        if y == 0:
            stay = np.full((B, Tx), _NEG, dtype=value.dtype)
            move = np.where(xs == 0, 0.0, _NEG).astype(value.dtype).repeat(B, 0)
        else:
            stay = prev.copy()
            move = np.concatenate([np.full((B, 1), _NEG, dtype=value.dtype), prev[:, :-1]], axis=1)
        stay[:, y : y + 1] = _NEG if y < Tx else stay[:, y : y + 1]
        cur = np.maximum(stay, move) + value[:, :, y]
        lo = np.maximum(0, t_x + y - t_y)[:, None]
        hi = np.minimum(t_x, y + 1)[:, None]
        cur = np.where((xs >= lo) & (xs < hi), cur, _NEG)
        acc[:, :, y] = cur
        prev = cur

    path = np.zeros((B, Tx, Ty), dtype=np.int32)
    index = t_x - 1
    rows = np.arange(B)
    for y in range(Ty - 1, -1, -1):
        active = y < t_y
        path[rows[active], index[active], y] = 1
        if y == 0:
            break
        idx_prev = np.maximum(index - 1, 0)
        down = (index != 0) & ((index == y) | (acc[rows, index, y - 1] < acc[rows, idx_prev, y - 1]))
        index = np.where(active & down, index - 1, index)
    return path


@torch.no_grad()
def maximum_path(value: Tensor, mask: Tensor) -> Tensor:
    """Most likely monotonic hard alignment.

    `value`: `(B, Tx, Ty)` log-likelihood of frame y under token x; `mask`: `(B, Tx, Ty)`.
    Returns a `{0, 1}` path of the same shape, dtype and device: every frame is assigned to
    exactly one token, every token gets at least one frame, and the path is monotonic.
    Requires `Ty >= Tx` for every item.
    """
    device, dtype = value.device, value.dtype
    mask = mask.to(torch.bool)
    v = (value.float() * mask).cpu().numpy().astype(np.float32)
    t_x = mask[:, :, 0].sum(1).cpu().numpy().astype(np.int32)
    t_y = mask[:, 0, :].sum(1).cpu().numpy().astype(np.int32)
    if (t_x > t_y).any():
        raise ValueError(f"alignment needs at least as many frames as tokens; got tokens {t_x.tolist()} frames {t_y.tolist()}")
    if HAVE_NUMBA:
        path = np.zeros(v.shape, dtype=np.int32)
        _maximum_path_batch(path, v, t_x, t_y)
    else:
        path = _maximum_path_numpy(v, t_x, t_y)
    return torch.from_numpy(path).to(device=device, dtype=dtype)


def generate_path(durations: Tensor, mask: Tensor) -> Tensor:
    """Hard path `(B, N, T)` from integer durations `(B, N)`; `mask` `(B, N, T)`."""
    B, N, T = mask.shape
    cum = torch.cumsum(durations, dim=1).long()
    path = sequence_mask(cum.reshape(-1), T).reshape(B, N, T).to(mask.dtype)
    path = path - torch.nn.functional.pad(path, (0, 0, 1, 0))[:, :-1]
    return path * mask


def pool_path(path: Tensor, frame_mask: Tensor, r: int) -> Tensor:
    """Mel-rate path `(B, N, T)` -> soft latent-rate assignment `(B, N, T / r)`.

    Column j holds the fraction of the valid mel frames `r*j .. r*j + r - 1` that belong to
    each token, so every valid latent column sums to 1. `T` must be a multiple of `r`.
    """
    B, N, T = path.shape
    if T % r:
        raise ValueError(f"frame count {T} is not a multiple of {r}")
    L = T // r
    counts = frame_mask.to(path.dtype).reshape(B, 1, L, r).sum(-1).clamp_min(1.0)
    return path.reshape(B, N, L, r).sum(-1) / counts


def gaussian_log_likelihood(mu: Tensor, y: Tensor) -> Tensor:
    """`log N(y_t; mu_x, I)` for all token/frame pairs: `mu (B, C, N)`, `y (B, C, T)` -> `(B, N, T)`."""
    c = mu.shape[1]
    y_sq = y.square().sum(1, keepdim=True)  # (B, 1, T)
    mu_sq = mu.square().sum(1).unsqueeze(-1)  # (B, N, 1)
    cross = torch.matmul(mu.transpose(1, 2), y)  # (B, N, T)
    return -0.5 * (y_sq - 2 * cross + mu_sq) - 0.5 * c * np.log(2 * np.pi)
