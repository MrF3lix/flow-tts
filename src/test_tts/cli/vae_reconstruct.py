"""Listening check for the mel VAE (NOTES.md test plan step 2).

    uv run tts-vae-reconstruct --ckpt logs/train_vae/be_vae/runs/<date>/checkpoints/best.pt --n 5

For each of the first `--n` validation utterances (or the given `--wav` files) this writes

    <stem>_original.wav     the recording, resampled to 24 kHz
    <stem>_copysynth.wav    ground-truth mel -> vocoder   (the vocoder's ceiling)
    <stem>_recon.wav        mel -> VAE -> mel -> vocoder  (what the energy model will decode)
    <stem>.png              both mels

and prints the per-utterance L1 in normalised mel space. Pass: recon is intelligible and the
speaker is preserved relative to copysynth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import hydra
import soundfile as sf
import torch
from omegaconf import OmegaConf

from test_tts.audio.mel import load_wav
from test_tts.data.filelist import parse_filelist
from test_tts.data.lengths import pad_to_multiple
from test_tts.training.utils import load_checkpoint, model_from_checkpoint, plot_mel, resolve_device


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--wav", type=Path, nargs="*", help="wav files; default: validation filelist of the run")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--out-dir", type=Path, default=None, help="default: <run dir>/reconstructions")
    p.add_argument("--speaker-embedding", type=Path, default=None, help="default: the run's vocoder config")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--sample", action="store_true", help="sample z instead of using the posterior mean")
    args = p.parse_args()

    device = resolve_device(args.device)
    ckpt = load_checkpoint(args.ckpt)
    cfg = OmegaConf.create(ckpt["cfg"])
    model = model_from_checkpoint(ckpt, device, use_ema=not args.no_ema)
    mel_fn = hydra.utils.instantiate(cfg.data.mel)
    mean, std = float(cfg.data.mel_stats.mean), float(cfg.data.mel_stats["std"])

    if args.wav:
        paths = args.wav
    else:
        paths = [u.path for u in parse_filelist(cfg.data.valid_filelist, cfg.data.get("root"))][: args.n]
    out_dir = args.out_dir or (args.ckpt.parent.parent / "reconstructions")
    out_dir.mkdir(parents=True, exist_ok=True)

    vcfg = OmegaConf.to_container(cfg.vocoder, resolve=True)
    if vcfg.get("device", "auto") == "auto":
        vcfg["device"] = str(device)
    if args.speaker_embedding:
        vcfg["speaker_embedding"] = str(args.speaker_embedding)
    vocoder = hydra.utils.instantiate(vcfg)
    sr = vocoder.sample_rate

    for path in paths:
        wav, _ = load_wav(path, mel_fn.sample_rate)
        with torch.no_grad():
            mel = mel_fn(wav)  # (n_mels, T), vocoder space
            x = ((mel - mean) / std).unsqueeze(0).to(device)
            t = x.shape[-1]
            t_pad = pad_to_multiple(t, model.compression)
            x = torch.nn.functional.pad(x, (0, t_pad - t))
            lengths = torch.tensor([t], device=device)
            out = model(x, lengths, sample=args.sample)
            mel_hat = out["mel_hat"][0, :, :t].cpu()
            l1 = float(out["recon"])
            recon = mel_hat * std + mean

            copysynth = vocoder(mel)[0].cpu().numpy()
            recon_audio = vocoder(recon)[0].cpu().numpy()

        stem = Path(path).stem
        sf.write(out_dir / f"{stem}_original.wav", wav.numpy(), sr)
        sf.write(out_dir / f"{stem}_copysynth.wav", copysynth, sr)
        sf.write(out_dir / f"{stem}_recon.wav", recon_audio, sr)
        image = plot_mel(mel.numpy(), recon.numpy(), titles=["original", f"VAE reconstruction (L1 {l1:.4f})"])
        import matplotlib.image  # pylint: disable=import-outside-toplevel

        matplotlib.image.imsave(out_dir / f"{stem}.png", image)
        print(f"{stem}: {t} frames, L1 (normalised mel) {l1:.4f}")

    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
