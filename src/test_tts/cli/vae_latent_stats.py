"""Fit the per-channel latent mean/std of a trained VAE and store them in the checkpoint.

    uv run tts-vae-latent-stats --ckpt logs/train_vae/be_vae/runs/<date>/checkpoints/best.pt

NOTES.md §2: normalise `z` to roughly unit variance per channel; the energy model sees
`encode(..., normalize=True)` and the decoder gets `decode(..., normalized=True)`. Statistics
are those of the posterior mean `mu` over all valid latent frames of the training set, which is
also what the energy model is trained on. The posterior std is reported for reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from test_tts.data.mel_dataset import MelDataModule
from test_tts.training.utils import load_checkpoint, model_from_checkpoint, resolve_device


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    device = resolve_device(args.device)
    ckpt = load_checkpoint(args.ckpt)
    cfg = OmegaConf.create(ckpt["cfg"])
    model = model_from_checkpoint(ckpt, device, use_ema=True)

    dcfg = OmegaConf.to_container(cfg.data, resolve=True)
    dcfg.update(segment_frames=None, num_workers=args.workers)
    datamodule: MelDataModule = hydra.utils.instantiate(dcfg)
    loader = datamodule._loader(datamodule.train_set, shuffle=False)  # pylint: disable=protected-access

    c = model.latent_dim
    total = torch.zeros(c, dtype=torch.float64)
    total_sq = torch.zeros(c, dtype=torch.float64)
    total_var = torch.zeros(c, dtype=torch.float64)
    n = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="encoding"):
            mel, lengths = batch["mel"].to(device), batch["lengths"].to(device)
            mu, logvar = model.encode(mel, lengths)
            z_lengths = model.latent_lengths(lengths)
            for i in range(mu.shape[0]):
                m = mu[i, :, : z_lengths[i]].cpu().double()
                total += m.sum(1)
                total_sq += m.square().sum(1)
                total_var += logvar[i, :, : z_lengths[i]].cpu().double().exp().sum(1)
                n += int(z_lengths[i])

    mean = total / n
    std = (total_sq / n - mean.square()).clamp_min(0).sqrt()
    posterior_std = (total_var / n).sqrt()
    print(f"{n} latent frames")
    print(f"mu mean   per channel: min {mean.min():.3f} max {mean.max():.3f}")
    print(f"mu std    per channel: min {std.min():.3f} max {std.max():.3f}  (overall {std.mean():.3f})")
    print(f"posterior std (mean over channels): {posterior_std.mean():.4f}  -> ratio to mu std {posterior_std.mean() / std.mean():.3f}")

    mean_f, std_f = mean.float().reshape(c, 1), std.float().reshape(c, 1)
    for key in ("model", "ema"):
        if ckpt.get(key) is not None:
            ckpt[key]["z_mean"] = mean_f.clone()
            ckpt[key]["z_std"] = std_f.clamp_min(1e-5).clone()
            ckpt[key]["latent_stats_fitted"] = torch.tensor(True)
    ckpt["latent_stats"] = {"mean": mean.tolist(), "std": std.tolist(), "posterior_std": posterior_std.tolist(), "n_frames": n}
    torch.save(ckpt, args.ckpt)
    stats_path = args.ckpt.with_name(args.ckpt.stem + "_latent_stats.json")
    stats_path.write_text(json.dumps(ckpt["latent_stats"], indent=2))
    print(f"updated {args.ckpt} (z_mean / z_std buffers), stats also in {stats_path}")


if __name__ == "__main__":
    main()
