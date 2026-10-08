"""Character encoder with a mel-space prior (for alignment) and a duration predictor.

As in Matcha-TTS / Glow-TTS: the encoder predicts one Gaussian mean per token in the
(normalised) mel space; monotonic alignment search finds the token -> mel-frame path that
maximises the likelihood of the real mel under those means; the resulting per-token frame
counts are the duration predictor's targets. The encoder's hidden states `h` are what the
latent decoder is conditioned on (expanded along the alignment).
"""

from __future__ import annotations

import math

from torch import Tensor, nn

from test_tts.data.lengths import sequence_mask
from test_tts.models.modules import ConvPrenet, RotaryEmbedding, TransformerBlock


class TextEncoder(nn.Module):
    def __init__(self, n_vocab: int, n_mels: int, hidden: int = 192, n_layers: int = 6, n_heads: int = 2,
                 ff_mult: int = 4, kernel_size: int = 3, prenet_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.emb = nn.Embedding(n_vocab, hidden)
        nn.init.normal_(self.emb.weight, 0.0, hidden**-0.5)
        self.prenet = ConvPrenet(hidden, prenet_layers) if prenet_layers > 0 else None
        self.rope = RotaryEmbedding(hidden // n_heads)
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden, n_heads, ff_mult, kernel_size, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(hidden)
        self.proj_mel = nn.Linear(hidden, n_mels)

    def forward(self, x: Tensor, x_lengths: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """`x (B, N)` token ids -> `h (B, N, D)`, `mu_mel (B, n_mels, N)`, `mask (B, N)`."""
        mask = sequence_mask(x_lengths, x.shape[1])
        h = self.emb(x) * math.sqrt(self.hidden) * mask[..., None]
        if self.prenet is not None:
            h = self.prenet(h, mask)
        rope = self.rope(h.shape[1], h.device)
        for block in self.blocks:
            h = block(h, mask, rope)
        h = self.norm(h) * mask[..., None]
        mu_mel = self.proj_mel(h).transpose(1, 2) * mask[:, None, :]
        return h, mu_mel, mask


class DurationPredictor(nn.Module):
    """Two conv layers -> log mel-frame duration per token (FastSpeech / Matcha)."""

    def __init__(self, in_dim: int, filter_channels: int = 256, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_dim, filter_channels, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=pad)
        self.norm1, self.norm2 = nn.LayerNorm(filter_channels), nn.LayerNorm(filter_channels)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(self, h: Tensor, mask: Tensor) -> Tensor:
        """`h (B, N, D)`, `mask (B, N)` -> `log_durations (B, N)`, zero on padding."""
        m = mask[:, None, :].to(h.dtype)
        x = h.transpose(1, 2)
        for conv, norm in ((self.conv1, self.norm1), (self.conv2, self.norm2)):
            x = conv(x * m).relu()
            x = self.drop(norm(x.transpose(1, 2)).transpose(1, 2))
        return self.proj(x * m).squeeze(1) * mask
