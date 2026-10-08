"""Generate wavs with the VocBulwark vocoder (the Test-TTS counterpart of `matcha-tts --vocoder vocbulwark`).

Inputs, any mix, each written as `<out-dir>/<stem>.wav` at 24 kHz:

    --wav a.wav b.wav        copy-synthesis: audio -> vocoder mel -> vocoder (the quality ceiling)
    --mel m.npy m.pt         a saved mel (n_mels, T) or (T, n_mels) in the vocoder's feature space;
                             add --normalized --ckpt <vae.pt> if it is in the VAE's normalised space
    --latent z.pt            a VAE latent (C, L) / (L, C) decoded with --ckpt <vae.pt> (normalised
                             latents, as produced by encode(..., normalize=True), unless --raw-latent)

Speaker conditioning (the vocoder cannot run without it), first match wins:

    --speaker-embedding e.pt    a [768] / [1, 768] embedding (default: data/be_speaker_embedding.pt)
    --reference-wav ref.wav     embed this clip with the speaker encoder
    --speaker-from-input        each --wav input supplies its own embedding (as in Matcha's notebook)

Examples:

    uv run tts-vocode --wav data/be/prepared/wav/ch_be_0000.wav --out-dir synth_output/copysynth
    uv run tts-vocode --mel data/cache/mels_be_vocbulwark/ch_be_0000.npy --out-dir synth_output
    uv run tts-vocode --latent z.pt --ckpt logs/.../checkpoints/best.pt --out-dir synth_output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from test_tts.audio.mel import WhisperMel, load_wav
from test_tts.audio.vocoder import VocBulwark
from test_tts.training.utils import load_checkpoint, model_from_checkpoint, resolve_device


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wav", type=Path, nargs="*", default=[], help="wavs for copy-synthesis")
    p.add_argument("--mel", type=Path, nargs="*", default=[], help="saved mels (.npy / .pt)")
    p.add_argument("--latent", type=Path, nargs="*", default=[], help="saved VAE latents (.pt / .npy)")
    p.add_argument("--ckpt", type=Path, default=None, help="VAE checkpoint, needed for --latent and --normalized")
    p.add_argument("--normalized", action="store_true", help="--mel inputs are in the VAE's normalised mel space")
    p.add_argument("--raw-latent", action="store_true", help="--latent inputs are un-normalised (raw encoder output)")
    p.add_argument("--speaker-embedding", type=Path, default=Path("data/be_speaker_embedding.pt"))
    p.add_argument("--reference-wav", type=Path, default=None)
    p.add_argument("--speaker-from-input", action="store_true")
    p.add_argument("--out-dir", type=Path, default=Path("synth_output/vocbulwark"))
    p.add_argument("--suffix", default="", help="appended to the output stem, e.g. _copysynth")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def load_array(path: Path) -> torch.Tensor:
    if path.suffix == ".npy":
        return torch.from_numpy(np.load(path))
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):  # e.g. {"mel": ...} / {"z": ...}
        obj = next(v for v in obj.values() if torch.is_tensor(v))
    return torch.as_tensor(obj)


def as_channels_first(x: torch.Tensor, n_channels: int) -> torch.Tensor:
    x = x.squeeze()
    if x.dim() != 2:
        raise ValueError(f"expected a 2-D array, got {tuple(x.shape)}")
    if x.shape[0] != n_channels and x.shape[1] == n_channels:
        x = x.T
    if x.shape[0] != n_channels:
        raise ValueError(f"expected {n_channels} channels, got {tuple(x.shape)}")
    return x.float().contiguous()


def main():
    args = parse_args()
    if not (args.wav or args.mel or args.latent):
        raise SystemExit("nothing to vocode: pass --wav, --mel and/or --latent")
    if (args.latent or args.normalized) and args.ckpt is None:
        raise SystemExit("--latent and --normalized need --ckpt <vae checkpoint>")

    device = resolve_device(args.device)
    need_encoder = args.reference_wav is not None or args.speaker_from_input
    vocoder = VocBulwark.load(
        None if need_encoder and args.speaker_from_input else args.speaker_embedding,
        device=str(device), load_speaker_encoder=need_encoder,
    )
    mel_fn = WhisperMel()
    sr = vocoder.sample_rate

    default_emb = None
    if args.reference_wav is not None:
        wav, wsr = load_wav(args.reference_wav)
        default_emb = vocoder.embed_speaker(wav, wsr)
        print(f"speaker embedding from {args.reference_wav}")

    vae, mean, std = None, 0.0, 1.0
    if args.ckpt is not None:
        ckpt = load_checkpoint(args.ckpt)
        vae = model_from_checkpoint(ckpt, device)
        mean, std = float(ckpt["mel_stats"]["mean"]), float(ckpt["mel_stats"]["std"])
        if args.latent and not args.raw_latent and not bool(vae.latent_stats_fitted):
            print("warning: checkpoint has no fitted latent stats (tts-vae-latent-stats); treating latents as raw")
            args.raw_latent = True

    jobs = []  # (stem, mel in vocoder space (n_mels, T), speaker embedding or None)
    for path in args.wav:
        wav, _ = load_wav(path, sr)
        emb = vocoder.embed_speaker(wav, sr) if args.speaker_from_input else default_emb
        jobs.append((path.stem, mel_fn(wav), emb))
    for path in args.mel:
        mel = as_channels_first(load_array(path), mel_fn.n_mels)
        if args.normalized:
            mel = mel * std + mean
        jobs.append((path.stem, mel, default_emb))
    for path in args.latent:
        z = as_channels_first(load_array(path), vae.latent_dim)
        with torch.no_grad():
            mel = vae.decode(z[None].to(device), normalized=not args.raw_latent)[0].cpu()
        jobs.append((path.stem, mel * std + mean, default_emb))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stem, mel, emb in jobs:
        with torch.no_grad():
            audio = vocoder(mel, emb)[0].cpu().numpy()
        out = args.out_dir / f"{stem}{args.suffix}.wav"
        sf.write(out, audio, sr)
        print(f"{out}  ({mel.shape[-1]} frames, {len(audio) / sr:.2f} s)")


if __name__ == "__main__":
    main()
