"""Train the mel VAE (NOTES.md §3, test plan step 2).

    uv run tts-train-vae experiment=be_vae
    uv run tts-train-vae experiment=be_vae debug=smoke            # 20-step end-to-end check
    uv run tts-train-vae experiment=be_vae ckpt_path=logs/.../checkpoints/last.pt

Checkpoints land in `<run dir>/checkpoints/{last,best}.pt`; `best.pt` has the lowest validation
L1 of the EMA weights. Afterwards run `tts-vae-latent-stats` to fit the latent normalisation and
`tts-vae-reconstruct` to listen.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from test_tts.training.audio import AudioLogger  # noqa: E402
from test_tts.training.loggers import build_logger  # noqa: E402
from test_tts.training.trainer import Trainer  # noqa: E402
from test_tts.training.utils import count_parameters, plot_mel, resolve_device, seed_everything  # noqa: E402

CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs")
log = logging.getLogger(__name__)


def make_validate(loader, device, logger, n_samples, audio, audio_every):
    @torch.no_grad()
    def validate(model, step, n_val):
        sums = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        n_frames = 0
        originals, reconstructions = [], []
        for batch in loader:
            mel, lengths = batch["mel"].to(device), batch["lengths"].to(device)
            out = model(mel, lengths, sample=False)
            frames = int(lengths.sum())
            for key in sums:
                sums[key] += float(out[key]) * frames
            n_frames += frames
            for i in range(mel.shape[0]):
                if len(originals) < n_samples:
                    n = int(lengths[i])
                    originals.append(mel[i, :, :n].cpu())
                    reconstructions.append(out["mel_hat"][i, :, :n].cpu())
        for i, (orig, recon) in enumerate(zip(originals, reconstructions)):
            logger.log_image(f"mel/{i}", plot_mel(orig.numpy(), recon.numpy(), titles=["original", "reconstruction"]), step)
        if audio_every > 0 and n_val % audio_every == 0:
            audio(logger, step, {f"audio/original_{i}": m for i, m in enumerate(originals)}, once=True)
            audio(logger, step, {f"audio/reconstruction_{i}": m for i, m in enumerate(reconstructions)})
        return {f"val/{k}": v / max(n_frames, 1) for k, v in sums.items()}

    return validate


def train(cfg: DictConfig):
    seed_everything(cfg.seed)
    device = resolve_device(cfg.trainer.device)
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    log.info("Device: %s", device)

    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model).to(device)
    log.info("Parameters: %.2f M", count_parameters(model) / 1e6)
    optimizer = hydra.utils.instantiate(cfg.optimizer)(model.parameters())
    logger = build_logger(cfg)

    tcfg = cfg.trainer
    audio = AudioLogger(cfg.vocoder, datamodule.valid_set.denormalize, tcfg.audio_every_n_vals > 0, device)
    validate = make_validate(datamodule.valid_dataloader(), device, logger, tcfg.n_audio_samples, audio,
                             tcfg.audio_every_n_vals)
    trainer = Trainer(cfg, model, optimizer, datamodule.train_dataloader(), device, logger, validate,
                      extra_state={"mel_stats": OmegaConf.to_container(cfg.data.mel_stats)})
    if cfg.ckpt_path:
        trainer.resume(cfg.ckpt_path)
    best = trainer.fit()
    logger.close()
    return best


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="train_vae.yaml")
def main(cfg: DictConfig):
    return train(cfg)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
