"""Cache of VAE latents for the flow / energy models (`z_1` of NOTES.md §6).

One `.npy` per utterance holding `(2, C, L)` = posterior mean and log-variance, both in the
*normalised* latent space (`MelVAE.encode(..., normalize=True)`), plus `meta.json` with the
VAE identity, the latent and mel statistics, and every utterance's mel frame count.

The cache directory is keyed by VAE run and step (`<run>_step<k>`), so a retrained or resumed
VAE can never be mixed with latents of an older one. Building it takes about a minute on MPS.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import torch

from test_tts.data.filelist import parse_filelist
from test_tts.data.lengths import pad_to_multiple
from test_tts.data.mel_dataset import MelDataset
from test_tts.training.utils import load_checkpoint, model_from_checkpoint

log = logging.getLogger(__name__)


def vae_cache_key(vae_ckpt: str | Path, step: int) -> str:
    vae_ckpt = Path(vae_ckpt)
    run = vae_ckpt.parent.parent.name if vae_ckpt.parent.name == "checkpoints" else vae_ckpt.parent.name
    return f"{run}_{vae_ckpt.stem}_step{step}"


def load_vae(vae_ckpt: str | Path, device="cpu"):
    """`(vae, ckpt)` with EMA weights; refuses a VAE whose latent statistics are not fitted."""
    ckpt = load_checkpoint(vae_ckpt)
    vae = model_from_checkpoint(ckpt, device, use_ema=True)
    if not bool(vae.latent_stats_fitted):
        raise RuntimeError(f"{vae_ckpt} has no latent statistics; run: uv run tts-vae-latent-stats --ckpt {vae_ckpt}")
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae, ckpt


@torch.no_grad()
def ensure_latent_cache(vae_ckpt: str | Path, filelists: list[str | Path], mel_cache_dir: str | Path | None,
                        cache_root: str | Path, root: str | Path | None = None, device="cpu") -> tuple[Path, dict]:
    """Encode every utterance of `filelists` that is not cached yet; returns `(dir, meta)`."""
    vae, ckpt = load_vae(vae_ckpt, device)
    out_dir = Path(cache_root) / vae_cache_key(vae_ckpt, ckpt["step"])
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    lengths = meta.get("lengths", {})

    mel_fn = hydra.utils.instantiate(ckpt["cfg"]["data"]["mel"])
    stats = ckpt["mel_stats"]
    todo = []
    for filelist in filelists:
        ds = MelDataset(filelist, mel_fn, stats, None, 1, mel_cache_dir, root)
        todo += [(ds, i) for i, u in enumerate(ds.items) if u.path.stem not in lengths or not (out_dir / f"{u.path.stem}.npy").is_file()]
    if todo:
        log.info("Encoding %d utterances with %s -> %s", len(todo), vae_ckpt, out_dir)
        r = vae.compression
        for ds, i in todo:
            stem = ds.items[i].path.stem
            mel = ds.normalize(ds.raw_mel(i))
            T = mel.shape[-1]
            x = torch.nn.functional.pad(mel, (0, pad_to_multiple(T, r) - T))[None].to(device)
            mu, logvar = vae.encode(x, torch.tensor([T], device=device), normalize=True)
            L = -(-T // r)
            np.save(out_dir / f"{stem}.npy", torch.stack([mu[0, :, :L], logvar[0, :, :L]]).cpu().numpy().astype(np.float32))
            lengths[stem] = T

    meta = {
        "vae_ckpt": str(Path(vae_ckpt).resolve()),
        "vae_step": ckpt["step"],
        "latent_dim": vae.latent_dim,
        "compression": vae.compression,
        "mel_stats": stats,
        "z_mean": vae.z_mean.flatten().tolist(),
        "z_std": vae.z_std.flatten().tolist(),
        "lengths": lengths,
    }
    meta_path.write_text(json.dumps(meta, indent=1))
    return out_dir, meta


def filelist_stems(filelist, root=None) -> list[str]:
    return [u.path.stem for u in parse_filelist(filelist, root)]
