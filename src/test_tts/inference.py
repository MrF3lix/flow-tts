"""Text -> waveform with a trained latent flow model, its VAE and the VocBulwark vocoder.

    from test_tts.inference import Synthesizer
    tts = Synthesizer("logs/train_fm/be_fm/runs/<date>/checkpoints/best.pt")
    out = tts("Grüezi mitenand, das isch e Test.", seed=0)
    out["wav"], out["sample_rate"], out["mel"], out["path"]
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import hydra
import torch

from test_tts.data.latents import load_vae
from test_tts.text.tokenizer import CharTokenizer
from test_tts.training.utils import load_checkpoint, model_from_checkpoint, resolve_device

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000  # VocBulwark output rate
HOP_LENGTH = 256  # mel hop of the Whisper front-end


class Synthesizer:
    def __init__(self, ckpt_path: str | Path, device: str = "auto", vocoder_device: str | None = None,
                 speaker_embedding: str | Path | None = None, use_ema: bool = True):
        self.device = resolve_device(device)
        ckpt = load_checkpoint(ckpt_path)
        self.cfg = ckpt["cfg"]
        self.step = ckpt["step"]
        self.model = model_from_checkpoint(ckpt, self.device, use_ema=use_ema)
        self.tokenizer = CharTokenizer.from_state(ckpt["tokenizer"])
        self.vae, vae_ckpt = load_vae(ckpt["vae_ckpt"], self.device)
        if vae_ckpt["step"] != ckpt.get("vae_step", vae_ckpt["step"]):
            log.warning("VAE at %s is now at step %d, but the flow model was trained on its step-%d latents",
                        ckpt["vae_ckpt"], vae_ckpt["step"], ckpt["vae_step"])
        self.mel_mean, self.mel_std = float(ckpt["mel_stats"]["mean"]), float(ckpt["mel_stats"]["std"])
        self.vocoder_cfg = dict(self.cfg["vocoder"])
        if vocoder_device or self.vocoder_cfg.get("device", "auto") == "auto":
            self.vocoder_cfg["device"] = vocoder_device or str(self.device)
        if speaker_embedding is not None:
            self.vocoder_cfg["speaker_embedding"] = str(speaker_embedding)
        self._vocoder = None

    @property
    def vocoder(self):
        if self._vocoder is None:
            self._vocoder = hydra.utils.instantiate(self.vocoder_cfg)
        return self._vocoder

    @torch.no_grad()
    def __call__(self, text: str, n_steps: int = 32, temperature: float = 0.667, length_scale: float = 1.0,
                 guidance_scale: float = 1.0, seed: int | None = None, vocode: bool = True) -> dict:
        unknown = self.tokenizer.unknown_chars(text)
        if unknown:
            log.warning("characters not in the training vocabulary are mapped to <unk>: %s", "".join(sorted(unknown)))
        ids = torch.tensor([self.tokenizer.encode(text)], device=self.device)
        x_lengths = torch.tensor([ids.shape[1]], device=self.device)
        generator = torch.Generator().manual_seed(seed) if seed is not None else None

        t0 = time.time()
        out = self.model.synthesise(ids, x_lengths, n_steps=n_steps, temperature=temperature,
                                    length_scale=length_scale, guidance_scale=guidance_scale, generator=generator)
        T = int(out["mel_lengths"][0])
        mel = self.vae.decode(out["z"], out["z_lengths"], normalized=True)[0, :, :T] * self.mel_std + self.mel_mean
        t_acoustic = time.time() - t0
        wav, t_vocoder = None, 0.0
        if vocode:
            t0 = time.time()
            wav = self.vocoder(mel)[0].cpu().numpy()
            t_vocoder = time.time() - t0
        seconds = T * HOP_LENGTH / SAMPLE_RATE
        return {
            "text": text,
            "clean_text": self.tokenizer.clean(text),
            "wav": wav,
            "sample_rate": SAMPLE_RATE,
            "mel": mel.cpu(),
            "z": out["z"][0, :, : int(out["z_lengths"][0])].cpu(),
            "durations": out["durations"][0, : ids.shape[1]].cpu(),
            "path": out["path"][0, :, :T].cpu(),
            "seconds": seconds,
            "rtf_acoustic": t_acoustic / seconds,
            "rtf_vocoder": t_vocoder / seconds,
        }
