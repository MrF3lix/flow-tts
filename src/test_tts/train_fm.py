"""Train the text-conditioned latent flow matching model (NOTES.md: the flow baseline that the
energy model will replace).

    uv run tts-train-fm experiment=be_fm
    uv run tts-train-fm experiment=be_fm debug=smoke        # 20 steps end to end
    uv run tts-train-fm experiment=be_fm debug=overfit      # memorise the 54 validation clips
    uv run tts-train-fm experiment=be_fm ckpt_path=logs/train_fm/be_fm/runs/<date>/checkpoints/last.pt

The frozen VAE (`vae_ckpt`) defines the latent space; its latents are cached on first use.
Validation logs fixed-noise losses, the oracle-duration mel L1 (generation with the real
durations, decoded through the VAE, against the real mel), the predicted/real length ratio,
plots (real mel, oracle-duration and predicted-duration synthesis, alignment) and audio.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import hydra
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from test_tts.data.latents import load_vae  # noqa: E402
from test_tts.data.lengths import sequence_mask  # noqa: E402
from test_tts.training.audio import AudioLogger  # noqa: E402
from test_tts.training.loggers import build_logger  # noqa: E402
from test_tts.training.trainer import Trainer, to_device  # noqa: E402
from test_tts.training.utils import count_parameters, load_checkpoint, plot_mel, resolve_device, seed_everything  # noqa: E402

CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs")
log = logging.getLogger(__name__)


def make_validate(loader, vae, device, logger, synth_cfg, n_samples, audio, audio_every):
    sample_batch = next(iter(loader))
    synth = OmegaConf.to_container(synth_cfg)

    @torch.no_grad()
    def validate(model, step, n_val):
        # 1) losses with fixed noise and t, so they are comparable across validations
        g = torch.Generator().manual_seed(1234)
        sums, n = defaultdict(float), 0
        for batch in loader:
            batch = to_device(batch, device)
            B = batch["z"].shape[0]
            noise = torch.randn(batch["z"].shape, generator=g).to(device)
            t = torch.rand(B, generator=g).to(device)
            out = model.training_step(batch, noise=noise, t=t, cond_drop=False)
            for k, v in out.items():
                sums[k] += float(v) * B
            n += B
        metrics = {f"val/{k}": v / n for k, v in sums.items()}

        # 2) synthesis of a few validation sentences
        b = to_device(sample_batch, device)
        l1s, ratios, gt, generated = [], [], {}, {}
        for i in range(min(n_samples, b["x"].shape[0])):
            xl, T = b["x_lengths"][i : i + 1], int(b["mel_lengths"][i])
            x = b["x"][i : i + 1, : int(xl)]
            t_pad = -(-T // model.compression) * model.compression
            mel = b["mel"][i : i + 1, :, :t_pad]
            _, mu_mel, x_mask = model.encoder(x, xl)
            path = model.align(mu_mel, x_mask, mel, sequence_mask(torch.tensor([T], device=device), t_pad))
            oracle = model.synthesise(x, xl, durations=path.sum(-1), generator=torch.Generator().manual_seed(i), **synth)
            pred = model.synthesise(x, xl, generator=torch.Generator().manual_seed(i), **synth)
            mel_oracle = vae.decode(oracle["z"], oracle["z_lengths"], normalized=True)[0, :, :T]
            mel_pred = vae.decode(pred["z"], pred["z_lengths"], normalized=True)[0, :, : int(pred["mel_lengths"][0])]
            l1s.append(float((mel_oracle - mel[0, :, :T]).abs().mean()))
            ratios.append(int(pred["mel_lengths"][0]) / T)
            image = plot_mel(mel[0, :, :T].cpu().numpy(), mel_oracle.cpu().numpy(), mel_pred.cpu().numpy(),
                             path[0, :, :T].cpu().numpy(),
                             titles=["real", "generated, real durations", "generated, predicted durations",
                                     "alignment (MAS): token x mel frame"])
            logger.log_image(f"synth/{i}", image, step)
            gt[f"audio/real_{i}"] = mel[0, :, :T].cpu()
            generated[f"audio/generated_{i}"] = mel_pred.cpu()
        metrics["val/oracle_mel_l1"] = sum(l1s) / len(l1s)
        metrics["val/length_ratio"] = sum(ratios) / len(ratios)
        if audio_every > 0 and n_val % audio_every == 0:
            audio(logger, step, gt, once=True)
            audio(logger, step, generated)
        return metrics

    return validate


def train(cfg: DictConfig):
    seed_everything(cfg.seed)
    device = resolve_device(cfg.trainer.device)
    log.info("Device: %s", device)

    resume = load_checkpoint(cfg.ckpt_path) if cfg.ckpt_path else None
    datamodule = hydra.utils.instantiate(cfg.data, tokenizer_state=resume["tokenizer"] if resume else None,
                                         _convert_="all")
    tokenizer = datamodule.tokenizer
    with open_dict(cfg):
        cfg.model.n_vocab = tokenizer.vocab_size
        cfg.model.n_mels = datamodule.n_mels
        cfg.model.latent_dim = datamodule.latent_dim
        cfg.model.compression = datamodule.compression
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    log.info("Vocabulary (%d): %s", tokenizer.vocab_size, "".join(tokenizer.symbols))
    log.info("Latents: %s (%d train / %d valid utterances)", datamodule.latent_dir, len(datamodule.train_set),
             len(datamodule.valid_set))
    out_dir = Path(cfg.paths.output_dir)
    (out_dir / "vocab.json").write_text(json.dumps(tokenizer.state_dict(), ensure_ascii=False, indent=1))

    model = hydra.utils.instantiate(cfg.model).to(device)
    log.info("Parameters: %.2f M (encoder %.2f, duration %.2f, decoder %.2f)", count_parameters(model) / 1e6,
             count_parameters(model.encoder) / 1e6, count_parameters(model.duration_predictor) / 1e6,
             count_parameters(model.decoder) / 1e6)
    optimizer = hydra.utils.instantiate(cfg.optimizer)(model.parameters())
    vae, vae_ckpt = load_vae(cfg.vae_ckpt, device)
    logger = build_logger(cfg)

    mean, std = float(datamodule.mel_stats["mean"]), float(datamodule.mel_stats["std"])
    tcfg = cfg.trainer
    audio = AudioLogger(cfg.vocoder, lambda m: m * std + mean, tcfg.audio_every_n_vals > 0, device)
    validate = make_validate(datamodule.valid_dataloader(), vae, device, logger, cfg.synthesis, tcfg.n_audio_samples,
                             audio, tcfg.audio_every_n_vals)
    extra = {"tokenizer": tokenizer.state_dict(), "vae_ckpt": str(Path(cfg.vae_ckpt).resolve()),
             "vae_step": vae_ckpt["step"], "mel_stats": datamodule.mel_stats}
    trainer = Trainer(cfg, model, optimizer, datamodule.train_dataloader(), device, logger, validate, extra)
    if cfg.ckpt_path:
        trainer.resume(cfg.ckpt_path)
    best = trainer.fit()
    logger.close()
    return best


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="train_fm.yaml")
def main(cfg: DictConfig):
    return train(cfg)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
