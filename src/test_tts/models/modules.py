"""Transformer building blocks shared by the text encoder and the latent decoders.

Channels-last `(B, T, D)` inside; boolean masks `(B, T)` are True on valid positions. Padded
positions are excluded as attention keys and re-zeroed after every block, so outputs on valid
positions do not depend on how much padding a batch has (tests/test_latent_fm.py checks this).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"rotary embedding needs an even head dim, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, length: int, device) -> tuple[Tensor, Tensor]:
        t = torch.arange(length, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


class SelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"dim {dim} not divisible by {n_heads} heads")
        self.n_heads, self.head_dim, self.dropout = n_heads, dim // n_heads, dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: Tensor, mask: Tensor, rope: tuple[Tensor, Tensor] | None = None) -> Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if rope is not None:
            q, k = apply_rotary(q, *rope), apply_rotary(k, *rope)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask[:, None, None, :], dropout_p=self.dropout if self.training else 0.0
        )
        return self.out(out.transpose(1, 2).reshape(B, T, D))


class FeedForward(nn.Module):
    """GELU MLP; with `kernel_size > 1` the two projections are 1-D convs over time (Matcha's
    text encoder uses kernel 3), with the hidden activations masked in between."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0, kernel_size: int = 1):
        super().__init__()
        hidden = dim * mult
        self.kernel_size = kernel_size
        if kernel_size == 1:
            self.w1, self.w2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)
        else:
            self.w1 = nn.Conv1d(dim, hidden, kernel_size, padding=kernel_size // 2)
            self.w2 = nn.Conv1d(hidden, dim, kernel_size, padding=kernel_size // 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        if self.kernel_size == 1:
            return self.w2(self.drop(F.gelu(self.w1(x))))
        m = mask[:, None, :].to(x.dtype)
        h = self.drop(F.gelu(self.w1(x.transpose(1, 2) * m))) * m
        return self.w2(h).transpose(1, 2)


class TransformerBlock(nn.Module):
    """Pre-LN transformer block (text encoder)."""

    def __init__(self, dim: int, n_heads: int, ff_mult: int = 4, kernel_size: int = 1, dropout: float = 0.1):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, n_heads, dropout)
        self.ff = FeedForward(dim, ff_mult, dropout, kernel_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor, rope=None) -> Tensor:
        x = x + self.drop(self.attn(self.norm1(x), mask, rope))
        x = x + self.drop(self.ff(self.norm2(x), mask))
        return x * mask[..., None]


class DiTBlock(nn.Module):
    """Transformer block with adaLN-zero conditioning on a global vector `c` (B, D):
    shift/scale before attention and MLP, gates on both residuals, all zero at init."""

    def __init__(self, dim: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(dim, n_heads, dropout)
        self.ff = FeedForward(dim, ff_mult, dropout)
        self.drop = nn.Dropout(dropout)
        self.ada = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x: Tensor, c: Tensor, mask: Tensor, rope=None) -> Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(F.silu(c)).unsqueeze(1).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1) + shift1
        x = x + gate1 * self.drop(self.attn(h, mask, rope))
        h = self.norm2(x) * (1 + scale2) + shift2
        x = x + gate2 * self.drop(self.ff(h, mask))
        return x * mask[..., None]


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, out_dim)
        for layer in (self.ada, self.proj):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.ada(F.silu(c)).unsqueeze(1).chunk(2, dim=-1)
        return self.proj(self.norm(x) * (1 + scale) + shift)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_dim: int = 256, scale: float = 1000.0):
        super().__init__()
        self.freq_dim, self.scale = freq_dim, scale
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: Tensor) -> Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = self.scale * t.float()[:, None] * freqs[None]
        return self.mlp(torch.cat([args.cos(), args.sin()], dim=-1))


class ConvPositionEmbedding(nn.Module):
    """Two grouped convs over time (F5-TTS): local context and relative position for free."""

    def __init__(self, dim: int, kernel_size: int = 31, groups: int = 16):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=groups)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=groups)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        m = mask[:, None, :].to(x.dtype)
        h = F.mish(self.conv1(x.transpose(1, 2) * m)) * m
        h = F.mish(self.conv2(h)) * m
        return h.transpose(1, 2)


class ConvPrenet(nn.Module):
    """Conv -> LayerNorm -> ReLU -> dropout stack with a zero-initialised residual projection
    (Matcha's ConvReluNorm), applied to token embeddings before the transformer."""

    def __init__(self, dim: int, n_layers: int = 3, kernel_size: int = 5, dropout: float = 0.5):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        m = mask[..., None].to(x.dtype)
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = conv((h * m).transpose(1, 2)).transpose(1, 2)
            h = self.drop(F.relu(norm(h)))
        return (x + self.proj(h * m)) * m
