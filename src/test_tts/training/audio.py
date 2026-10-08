"""Vocode mels for listening during training (VocBulwark). Loads lazily and disables itself on
any failure, so a Hub hiccup or a slow vocoder never kills a training run."""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


class AudioLogger:
    def __init__(self, vocoder_cfg: DictConfig, denormalize, enabled: bool, device):
        self.cfg = OmegaConf.to_container(vocoder_cfg, resolve=True)
        if self.cfg.get("device", "auto") == "auto":
            self.cfg["device"] = str(device)
        self.denormalize = denormalize
        self.enabled = enabled
        self.vocoder = None
        self.logged = set()

    def _load(self):
        if self.vocoder is None and self.enabled:
            try:
                self.vocoder = hydra.utils.instantiate(self.cfg)
                log.info("Vocoder ready on %s", self.cfg["device"])
            except Exception as exc:  # noqa: BLE001
                self.enabled = False
                log.warning("Audio samples disabled, vocoder failed to load: %s", exc)
        return self.vocoder

    def __call__(self, logger, step: int, mels: dict, once: bool = False):
        """`mels`: key -> normalised mel `(n_mels, T)`. With `once`, keys logged before are skipped."""
        if not self.enabled or self._load() is None:
            return
        try:
            for key, mel in mels.items():
                if once and key in self.logged:
                    continue
                audio = self.vocoder(self.denormalize(mel))[0].cpu().numpy()
                logger.log_audio(key, audio, step, self.vocoder.sample_rate)
                self.logged.add(key)
        except Exception as exc:  # noqa: BLE001
            self.enabled = False
            log.warning("Audio samples disabled after an error: %s", exc)
