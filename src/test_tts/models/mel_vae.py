"""Mel VAE (NOTES.md §3): 1-D conv encoder/decoder over time, latent `z` of shape `(T/r, C)`.

Layout (channels-first, `(B, C, T)` throughout):

    mel (B, n_mels, T) ─ conv_in ─ [blocks]₀ ─ down ─ [blocks]₁ ─ down ─ … ─ 1x1 ─► mu, logvar (B, C_z, T/r)
    z   (B, C_z, T/r)  ─ 1x1 ─ [blocks]_L ─ up ─ … ─ [blocks]₀ ─ conv_out ─► mel_hat (B, n_mels, T)

Blocks are ConvNeXt-style: depthwise conv (k=7) → per-frame LayerNorm → pointwise MLP → scaled
residual. LayerNorm over channels per frame has no cross-time statistics, so zero padding does
not leak into valid frames the way GroupNorm/BatchNorm would. Padded frames are re-zeroed after
every stage. Losses are masked means over valid frames.

Latent normalisation (NOTES.md §2): `z_mean`/`z_std` buffers, per channel, filled in after
training by `tts-vae-latent-stats`. `encode(..., normalize=True)` returns unit-variance latents
for the energy model; `decode(..., normalized=True)` undoes it. Both default to the raw latent
until the stats are fitted.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from test_tts.data.lengths import sequence_mask


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dim of a `(B, C, T)` tensor."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ConvNeXtBlock1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, mlp_ratio: int = 4, layer_scale: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.norm = ChannelLayerNorm(channels)
        self.pw1 = nn.Conv1d(channels, mlp_ratio * channels, 1)
        self.pw2 = nn.Conv1d(mlp_ratio * channels, channels, 1)
        self.gamma = nn.Parameter(layer_scale * torch.ones(channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        h = self.dwconv(x)
        h = self.norm(h)
        h = self.pw2(F.gelu(self.pw1(h)))
        return x + self.gamma * h


class Stage(nn.Module):
    def __init__(self, channels: int, n_blocks: int, kernel_size: int):
        super().__init__()
        self.blocks = nn.Sequential(*[ConvNeXtBlock1d(channels, kernel_size) for _ in range(n_blocks)])

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        x = self.blocks(x)
        return x * mask if mask is not None else x


class Downsample(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.norm = ChannelLayerNorm(c_in)
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.norm(x))


class Upsample(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.norm = ChannelLayerNorm(c_in)
        self.conv = nn.ConvTranspose1d(c_in, c_out, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.norm(x))


class Encoder(nn.Module):
    def __init__(self, n_mels: int, latent_dim: int, channels: list[int], blocks_per_level: int, kernel_size: int):
        super().__init__()
        self.conv_in = nn.Conv1d(n_mels, channels[0], kernel_size, padding=kernel_size // 2)
        self.stages = nn.ModuleList([Stage(c, blocks_per_level, kernel_size) for c in channels])
        self.downs = nn.ModuleList([Downsample(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
        self.norm_out = ChannelLayerNorm(channels[-1])
        self.head = nn.Conv1d(channels[-1], 2 * latent_dim, 1)

    def forward(self, mel: Tensor, masks: list[Tensor] | None) -> tuple[Tensor, Tensor]:
        x = self.conv_in(mel)
        for level, stage in enumerate(self.stages):
            mask = masks[level] if masks is not None else None
            x = stage(x * mask if mask is not None else x, mask)
            if level < len(self.downs):
                x = self.downs[level](x)
        mask = masks[-1] if masks is not None else None
        out = self.head(self.norm_out(x))
        if mask is not None:
            out = out * mask
        mu, logvar = out.chunk(2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, n_mels: int, latent_dim: int, channels: list[int], blocks_per_level: int, kernel_size: int):
        super().__init__()
        rev = list(reversed(channels))
        self.conv_in = nn.Conv1d(latent_dim, rev[0], 1)
        self.stages = nn.ModuleList([Stage(c, blocks_per_level, kernel_size) for c in rev])
        self.ups = nn.ModuleList([Upsample(rev[i], rev[i + 1]) for i in range(len(rev) - 1)])
        self.norm_out = ChannelLayerNorm(rev[-1])
        self.conv_out = nn.Conv1d(rev[-1], n_mels, kernel_size, padding=kernel_size // 2)

    def forward(self, z: Tensor, masks: list[Tensor] | None) -> Tensor:
        # masks are ordered by level (0 = mel rate); the decoder walks them backwards
        x = self.conv_in(z)
        n = len(self.stages)
        for i, stage in enumerate(self.stages):
            mask = masks[n - 1 - i] if masks is not None else None
            x = stage(x * mask if mask is not None else x, mask)
            if i < len(self.ups):
                x = self.ups[i](x)
        mask = masks[0] if masks is not None else None
        out = self.conv_out(F.gelu(self.norm_out(x)))
        return out * mask if mask is not None else out


class MelVAE(nn.Module):
    def __init__(
        self,
        n_mels: int = 96,
        latent_dim: int = 32,
        compression: int = 4,
        channels: list[int] = (192, 256, 384),
        blocks_per_level: int = 2,
        kernel_size: int = 7,
        beta: float = 5e-3,
    ):
        super().__init__()
        channels = list(channels)
        if compression != 2 ** (len(channels) - 1):
            raise ValueError(f"compression={compression} needs {int(math.log2(compression)) + 1} channel levels, got {len(channels)}")
        self.n_mels = n_mels
        self.latent_dim = latent_dim
        self.compression = compression
        self.beta = beta
        self.n_levels = len(channels)

        self.encoder = Encoder(n_mels, latent_dim, channels, blocks_per_level, kernel_size)
        self.decoder = Decoder(n_mels, latent_dim, channels, blocks_per_level, kernel_size)

        # Per-channel latent statistics, fitted after training (tts-vae-latent-stats).
        self.register_buffer("z_mean", torch.zeros(latent_dim, 1))
        self.register_buffer("z_std", torch.ones(latent_dim, 1))
        self.register_buffer("latent_stats_fitted", torch.tensor(False))

    # ------------------------------------------------------------------ masks / lengths
    def latent_lengths(self, lengths: Tensor) -> Tensor:
        return -(-lengths // self.compression)  # ceil

    def level_masks(self, lengths: Tensor | None, t_max: int) -> list[Tensor] | None:
        """One `(B, 1, T_level)` mask per resolution level, level 0 at the mel frame rate."""
        if lengths is None:
            return None
        if t_max % self.compression:
            raise ValueError(f"mel length {t_max} must be a multiple of compression={self.compression}")
        masks = []
        for level in range(self.n_levels):
            factor = 2**level
            masks.append(sequence_mask(-(-lengths // factor), t_max // factor).unsqueeze(1).to(torch.float32))
        return masks

    # ------------------------------------------------------------------ latent normalisation
    def normalize_latent(self, z: Tensor) -> Tensor:
        return (z - self.z_mean) / self.z_std

    def denormalize_latent(self, z: Tensor) -> Tensor:
        return z * self.z_std + self.z_mean

    @torch.no_grad()
    def set_latent_stats(self, mean: Tensor, std: Tensor):
        self.z_mean.copy_(mean.reshape(self.latent_dim, 1))
        self.z_std.copy_(std.reshape(self.latent_dim, 1).clamp_min(1e-5))
        self.latent_stats_fitted.fill_(True)

    # ------------------------------------------------------------------ encode / decode
    def encode(self, mel: Tensor, lengths: Tensor | None = None, normalize: bool = False):
        """`mel (B, n_mels, T)` normalised with the data stats -> `(mu, logvar)`, each `(B, C_z, T/r)`."""
        masks = self.level_masks(lengths, mel.shape[-1])
        mu, logvar = self.encoder(mel, masks)
        if normalize:
            mu = self.normalize_latent(mu)
            logvar = logvar - 2 * torch.log(self.z_std)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: Tensor, z_lengths: Tensor | None = None, normalized: bool = False) -> Tensor:
        """`z (B, C_z, L)` -> `mel_hat (B, n_mels, L * r)` in normalised mel space."""
        if normalized:
            z = self.denormalize_latent(z)
        lengths = None if z_lengths is None else z_lengths * self.compression
        masks = self.level_masks(lengths, z.shape[-1] * self.compression)
        return self.decoder(z, masks)

    # ------------------------------------------------------------------ training
    def forward(self, mel: Tensor, lengths: Tensor | None = None, sample: bool = True) -> dict:
        """Returns `mel_hat`, `mu`, `logvar`, `z` and the losses `loss`, `recon`, `kl`."""
        mu, logvar = self.encode(mel, lengths)
        z = self.reparameterize(mu, logvar) if sample else mu
        z_lengths = None if lengths is None else self.latent_lengths(lengths)
        mel_hat = self.decode(z, z_lengths)

        if lengths is None:
            recon = (mel_hat - mel).abs().mean()
            kl = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).mean()
        else:
            frame_mask = sequence_mask(lengths, mel.shape[-1]).unsqueeze(1).to(mel.dtype)
            recon = ((mel_hat - mel).abs() * frame_mask).sum() / (frame_mask.sum() * self.n_mels)
            z_mask = sequence_mask(z_lengths, mu.shape[-1]).unsqueeze(1).to(mu.dtype)
            kl_elem = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar)
            kl = (kl_elem * z_mask).sum() / (z_mask.sum() * self.latent_dim)

        loss = recon + self.beta * kl
        return {"mel_hat": mel_hat, "mu": mu, "logvar": logvar, "z": z, "loss": loss, "recon": recon, "kl": kl}

    def training_step(self, batch: dict) -> dict:
        out = self(batch["mel"], batch["lengths"])
        return {k: out[k] for k in ("loss", "recon", "kl")}
