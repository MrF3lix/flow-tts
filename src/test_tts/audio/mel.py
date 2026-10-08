"""Mel front-end matching the VocBulwark vocoder exactly.

The vocoder was trained on Whisper-style log-mels (`transformers.WhisperFeatureExtractor`):

    S      = |STFT(x; n_fft, hop, hann, center=True, reflect)|^2        (drop the last frame)
    M      = mel_filters(slaney, 0..sr/2)^T @ S
    L      = log10(max(M, 1e-10))
    L      = max(L, L.max() - 8)                                      (per utterance!)
    mel    = (L + 4) / 4

This module reimplements that in torch (the HF version is numpy): it runs on any device, works
on batches, and is differentiable, which later stages need for mel-domain energies (NOTES.md §8).
`tests/test_mel.py` checks it against the HF extractor to float tolerance.

The `L.max() - 8` clamp depends on the *whole* utterance, so always extract the full clip and
crop the mel afterwards; cropping the waveform first changes every frame.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from test_tts.data.lengths import sequence_mask


def load_wav(path, sample_rate: int | None = None) -> tuple[Tensor, int]:
    """Read a wav as mono float32 `(samples,)` in [-1, 1]; resample when `sample_rate` differs."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(np.ascontiguousarray(audio.T)).mean(0)
    if sample_rate is not None and sr != sample_rate:
        import torchaudio.functional as AF  # pylint: disable=import-outside-toplevel

        wav = AF.resample(wav, sr, sample_rate)
        sr = sample_rate
    return wav, sr


class WhisperMel(nn.Module):
    """Whisper log-mel spectrogram, `(samples,)`/`(B, samples)` -> `(n_mels, T)`/`(B, n_mels, T)`.

    `T = samples // hop_length`. Stateless apart from the filterbank and window buffers, so it
    can live in a dataset (CPU) or on the training device.
    """

    def __init__(self, sample_rate: int = 24000, n_fft: int = 1024, n_mels: int = 96, hop_length: int = 256):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length

        # Take the filterbank from the HF extractor itself so the two can never drift apart.
        from transformers import WhisperFeatureExtractor  # pylint: disable=import-outside-toplevel

        fe = WhisperFeatureExtractor(
            sampling_rate=sample_rate, n_fft=n_fft, feature_size=n_mels, hop_length=hop_length
        )
        filters = torch.as_tensor(np.asarray(fe.mel_filters), dtype=torch.float32)  # (n_freq, n_mels)
        self.register_buffer("mel_filters", filters.T.contiguous(), persistent=False)  # (n_mels, n_freq)
        self.register_buffer("window", torch.hann_window(n_fft, periodic=True), persistent=False)

    @property
    def frames_per_second(self) -> float:
        return self.sample_rate / self.hop_length

    def num_frames(self, num_samples: Tensor | int):
        return num_samples // self.hop_length

    def forward(self, wav: Tensor, lengths: Tensor | None = None) -> Tensor:
        """`wav`: `(samples,)` or `(B, samples)`, zero-padded; `lengths`: `(B,)` valid samples.

        With padding, the per-utterance noise-floor clamp is computed over valid frames only,
        and padded frames are returned as zeros.
        """
        squeeze = wav.dim() == 1
        if squeeze:
            wav = wav.unsqueeze(0)
        wav = wav.to(self.window.dtype)
        n_frames_out = wav.shape[-1] // self.hop_length

        if lengths is not None:
            # Reproduce the single-utterance result for every item: `center=True` reflect-pads
            # each signal at *its own* end, whereas a zero-padded batch would continue with
            # zeros there. Append room for the reflection and fill it per item; the frames we
            # keep then never see the batch-level padding.
            half = self.n_fft // 2
            lengths = lengths.to(wav.device)
            wav = F.pad(wav, (0, half))
            for i, n in enumerate(lengths.tolist()):
                if n > half + 1:
                    wav[i, n : n + half] = wav[i, n - 1 - half : n - 1].flip(0)

        spec = torch.stft(
            wav,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = spec.real.square() + spec.imag.square()  # (B, n_freq, T + 1)
        power = power[..., :n_frames_out]  # drop the last frame (Whisper) / the reflection room
        mel = torch.matmul(self.mel_filters, power)  # (B, n_mels, T)
        log_spec = torch.log10(mel.clamp(min=1e-10))

        if lengths is None:
            peak = log_spec.amax(dim=(1, 2), keepdim=True)
            log_spec = torch.maximum(log_spec, peak - 8.0)
        else:
            frame_lengths = self.num_frames(lengths)
            mask = sequence_mask(frame_lengths, log_spec.shape[-1]).unsqueeze(1)  # (B, 1, T)
            peak = log_spec.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)
            log_spec = torch.maximum(log_spec, peak - 8.0)
            log_spec = log_spec.masked_fill(~mask, 0.0)

        out = (log_spec + 4.0) / 4.0
        if lengths is not None:
            out = out.masked_fill(~mask, 0.0)
        return out.squeeze(0) if squeeze else out
