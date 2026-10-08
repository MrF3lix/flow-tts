"""Text-conditioned conditional flow matching in the mel VAE's latent space.

    text ─ TextEncoder ─► h (N, D), mu_mel (N, n_mels)
                           │
          MAS vs. mel ─────┤  hard token -> mel-frame path, durations (training)
          DurationPredictor┘  log mel-frame durations (inference)
                           │
          path pooled by r ─► soft token -> latent-frame assignment  ─►  c = h expanded  (L, D)
                                                                          │
    z_0 ~ N(0, I) ─ Euler ODE with  v_θ(z_t, t, c)  (LatentDiT) ─────────► z_1 (L, C)  ─► VAE decoder ─► mel ─► vocoder

Losses (all masked means): duration MSE in the log domain, the mel prior NLL that drives
the alignment (Matcha's "prior loss"), and the conditional flow matching loss with the OT
path `z_t = (1 - (1 - σ) t) z_0 + t z_1`, target `z_1 - (1 - σ) z_0`.

The latents are the VAE posterior means, normalised per channel (`encode(..., normalize=True)`),
so `z_0 ~ N(0, I)` is on the same scale as the data. The decoder is time-conditioned here; the
energy model (NOTES.md §5) reuses the encoder, aligner and conditioning unchanged and swaps
the decoder for a time-independent scalar potential.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from test_tts.data.lengths import sequence_mask
from test_tts.models.alignment import gaussian_log_likelihood, generate_path, maximum_path, pool_path
from test_tts.models.modules import (
    ConvPositionEmbedding,
    DiTBlock,
    FinalLayer,
    RotaryEmbedding,
    TimestepEmbedding,
)
from test_tts.models.text_encoder import DurationPredictor, TextEncoder


class LatentDiT(nn.Module):
    """Velocity network `v(z_t, t, c)`: frames are tokens, `[z_t; c]` is projected in, time
    enters through adaLN-zero, positions through rotary attention and a conv embedding."""

    def __init__(self, latent_dim: int, cond_dim: int, d_model: int = 256, depth: int = 8, n_heads: int = 4,
                 ff_mult: int = 4, dropout: float = 0.1, conv_pos_kernel: int = 31):
        super().__init__()
        self.in_proj = nn.Linear(latent_dim + cond_dim, d_model)
        self.conv_pos = ConvPositionEmbedding(d_model, conv_pos_kernel)
        self.time_emb = TimestepEmbedding(d_model)
        self.rope = RotaryEmbedding(d_model // n_heads)
        self.blocks = nn.ModuleList([DiTBlock(d_model, n_heads, ff_mult, dropout) for _ in range(depth)])
        self.final = FinalLayer(d_model, latent_dim)

    def forward(self, z: Tensor, t: Tensor, cond: Tensor, mask: Tensor) -> Tensor:
        """`z (B, C, L)`, `t (B,)`, `cond (B, D_c, L)`, `mask (B, L)` -> velocity `(B, C, L)`."""
        x = self.in_proj(torch.cat([z, cond], dim=1).transpose(1, 2)) * mask[..., None]
        x = x + self.conv_pos(x, mask)
        c = self.time_emb(t)
        rope = self.rope(x.shape[1], x.device)
        for block in self.blocks:
            x = block(x, c, mask, rope)
        return (self.final(x, c) * mask[..., None]).transpose(1, 2)


class LatentFlowTTS(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        n_mels: int = 96,
        latent_dim: int = 32,
        compression: int = 4,
        encoder: dict | None = None,
        duration_predictor: dict | None = None,
        decoder: dict | None = None,
        sigma_min: float = 1e-4,
        p_uncond: float = 0.1,
    ):
        super().__init__()
        encoder, duration_predictor, decoder = dict(encoder or {}), dict(duration_predictor or {}), dict(decoder or {})
        self.n_mels, self.latent_dim, self.compression = n_mels, latent_dim, compression
        self.sigma_min, self.p_uncond = sigma_min, p_uncond
        self.encoder = TextEncoder(n_vocab, n_mels, **encoder)
        hidden = self.encoder.hidden
        self.duration_predictor = DurationPredictor(hidden, **duration_predictor)
        self.decoder = LatentDiT(latent_dim, hidden, **decoder)

    # ------------------------------------------------------------------ alignment
    @torch.no_grad()
    def align(self, mu_mel: Tensor, x_mask: Tensor, mel: Tensor, mel_mask: Tensor) -> Tensor:
        """Hard monotonic path `(B, N, T)` of tokens to mel frames (MAS on the mel prior).

        Always fp32: the log-likelihoods are in the hundreds, where bf16 rounds to whole units
        and would make the search pick paths at random among near-ties."""
        with torch.autocast(mel.device.type, enabled=False):
            log_prior = gaussian_log_likelihood(mu_mel.float(), mel.float())
        return maximum_path(log_prior, x_mask[:, :, None] & mel_mask[:, None, :]).to(mu_mel.dtype)

    def expand(self, h: Tensor, path: Tensor, mel_mask: Tensor) -> Tensor:
        """Text features at the latent rate: `h (B, N, D)`, mel path `(B, N, T)` -> `(B, D, T / r)`."""
        return torch.matmul(h.transpose(1, 2), pool_path(path, mel_mask, self.compression))

    # ------------------------------------------------------------------ training
    def training_step(self, batch: dict, noise: Tensor | None = None, t: Tensor | None = None,
                      cond_drop: bool = True) -> dict:
        x, x_lengths = batch["x"], batch["x_lengths"]
        mel, mel_lengths = batch["mel"], batch["mel_lengths"]
        z, z_lengths = batch["z"], batch["z_lengths"]
        if mel.shape[-1] != z.shape[-1] * self.compression:
            raise ValueError(f"mel frames {mel.shape[-1]} != {self.compression} x latent frames {z.shape[-1]}")

        h, mu_mel, x_mask = self.encoder(x, x_lengths)
        mel_mask = sequence_mask(mel_lengths, mel.shape[-1])
        z_mask = sequence_mask(z_lengths, z.shape[-1])
        path = self.align(mu_mel, x_mask, mel, mel_mask)

        # durations: MSE between predicted and MAS log-durations (predictor sees detached h)
        log_dur = self.duration_predictor(h.detach(), x_mask)
        log_dur_target = torch.log(1e-8 + path.sum(-1)) * x_mask
        dur_loss = ((log_dur - log_dur_target).square() * x_mask).sum() / x_mask.sum()

        # prior: NLL of the mel under the expanded token means (what MAS maximised)
        mu_y = torch.matmul(mu_mel, path)
        m = mel_mask[:, None, :].to(mel.dtype)
        prior_loss = (0.5 * ((mel - mu_y).square() + math.log(2 * math.pi)) * m).sum() / (m.sum() * self.n_mels)

        # flow matching on the latents, conditioned on the text expanded along the alignment
        cond = self.expand(h, path, mel_mask)
        if cond_drop and self.p_uncond > 0:
            keep = (torch.rand(z.shape[0], device=z.device) >= self.p_uncond).to(cond.dtype)
            cond = cond * keep[:, None, None]
        fm_loss = self.flow_loss(z, z_mask, cond, noise, t)

        loss = dur_loss + prior_loss + fm_loss
        return {"loss": loss, "fm": fm_loss, "prior": prior_loss, "dur": dur_loss}

    def flow_loss(self, z1: Tensor, mask: Tensor, cond: Tensor, noise: Tensor | None = None, t: Tensor | None = None):
        m = mask[:, None, :].to(z1.dtype)
        z0 = torch.randn_like(z1) if noise is None else noise
        if t is None:
            t = torch.rand(z1.shape[0], device=z1.device, dtype=z1.dtype)
        tt = t[:, None, None]
        zt = (1 - (1 - self.sigma_min) * tt) * z0 + tt * z1
        target = z1 - (1 - self.sigma_min) * z0
        v = self.decoder(zt * m, t, cond, mask)
        return ((v - target).square() * m).sum() / (m.sum() * self.latent_dim)

    # ------------------------------------------------------------------ inference
    def velocity(self, z: Tensor, t: Tensor, cond: Tensor, mask: Tensor, guidance_scale: float = 1.0) -> Tensor:
        if guidance_scale == 1.0:
            return self.decoder(z, t, cond, mask)
        v = self.decoder(torch.cat([z, z]), torch.cat([t, t]), torch.cat([cond, torch.zeros_like(cond)]),
                         torch.cat([mask, mask]))
        v_cond, v_uncond = v.chunk(2)
        return v_uncond + guidance_scale * (v_cond - v_uncond)

    @torch.no_grad()
    def synthesise(self, x: Tensor, x_lengths: Tensor, n_steps: int = 32, temperature: float = 0.667,
                   length_scale: float = 1.0, guidance_scale: float = 1.0, durations: Tensor | None = None,
                   generator: torch.Generator | None = None) -> dict:
        """Text -> normalised latents.

        `durations` `(B, N)` mel-frame durations override the predictor (e.g. the MAS durations
        of a reference recording, to judge the decoder independently of the duration model).
        Returns `z (B, C, L)`, `z_lengths`, `mel_lengths` (frames the decoded mel should be cut
        to), `durations`, `path (B, N, T)` and the prior means expanded to `mu_mel_y (B, n_mels, T)`.
        """
        h, mu_mel, x_mask = self.encoder(x, x_lengths)
        if durations is None:
            # round and clamp at 1 (NOTES.md §4), not Matcha's ceil: with blanks most tokens last
            # 1-3 mel frames, and ceil's upward bias made speech ~20 % too long.
            log_dur = self.duration_predictor(h, x_mask)
            durations = torch.round(torch.exp(log_dur) * length_scale).clamp_min(1) * x_mask
        durations = durations.to(h.dtype) * x_mask
        mel_lengths = durations.sum(1).clamp_min(1).long()
        z_lengths = -(-mel_lengths // self.compression)
        L = int(z_lengths.max())
        T = L * self.compression
        mel_mask = sequence_mask(mel_lengths, T)
        z_mask = sequence_mask(z_lengths, L)
        path = generate_path(durations, (x_mask[:, :, None] & mel_mask[:, None, :]).to(h.dtype))
        cond = self.expand(h, path, mel_mask)

        m = z_mask[:, None, :].to(h.dtype)
        shape = (x.shape[0], self.latent_dim, L)
        z = torch.randn(shape, generator=generator, device="cpu" if generator is not None else x.device)
        z = z.to(x.device, h.dtype) * temperature * m
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=h.dtype)
            z = (z + dt * self.velocity(z, t, cond, z_mask, guidance_scale)) * m

        return {
            "z": z,
            "z_lengths": z_lengths,
            "mel_lengths": mel_lengths,
            "durations": durations,
            "path": path,
            "mu_mel_y": torch.matmul(mu_mel, path),
        }
