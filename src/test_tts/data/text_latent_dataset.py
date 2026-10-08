"""Paired data for the text-conditioned latent models: tokens, normalised mel, normalised latent.

The mel is only needed for alignment (MAS against the text encoder's mel prior); generation
happens on the latent. Mel front-end and normalisation are taken from the VAE checkpoint so
the latents and the mels always come from the same features.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from test_tts.data.latents import ensure_latent_cache, load_vae
from test_tts.data.mel_dataset import MelDataset
from test_tts.text.tokenizer import CharTokenizer
from test_tts.training.utils import resolve_device

log = logging.getLogger(__name__)


class TextLatentDataset(Dataset):
    def __init__(self, filelist, tokenizer: CharTokenizer, mel_fn, mel_stats: dict, mel_cache_dir, latent_dir: Path,
                 lengths: dict, compression: int, root=None, sample_latent: bool = False):
        self.mels = MelDataset(filelist, mel_fn, mel_stats, None, 1, mel_cache_dir, root)
        self.latent_dir = Path(latent_dir)
        self.compression = compression
        self.sample_latent = sample_latent
        self.items, self.tokens = [], []
        dropped = 0
        for i, utt in enumerate(self.mels.items):
            ids = tokenizer.encode(utt.text)
            if len(ids) > lengths[utt.path.stem]:
                dropped += 1  # MAS needs at least one mel frame per token
                continue
            self.items.append(i)
            self.tokens.append(torch.tensor(ids, dtype=torch.long))
        if dropped:
            log.warning("%s: dropped %d utterances with more tokens than mel frames", filelist, dropped)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        i = self.items[index]
        utt = self.mels.items[i]
        mel = self.mels.normalize(self.mels.raw_mel(i))
        lat = torch.from_numpy(np.load(self.latent_dir / f"{utt.path.stem}.npy"))
        mu, logvar = lat[0], lat[1]
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.sample_latent else mu
        if z.shape[-1] != -(-mel.shape[-1] // self.compression):
            raise RuntimeError(f"{utt.path}: latent has {z.shape[-1]} frames for {mel.shape[-1]} mel frames; stale cache?")
        return {"x": self.tokens[index], "mel": mel, "z": z, "text": utt.text, "path": str(utt.path)}


class TextLatentCollate:
    def __init__(self, compression: int):
        self.r = compression

    def __call__(self, batch: list[dict]) -> dict:
        B = len(batch)
        x_lengths = torch.tensor([b["x"].shape[0] for b in batch])
        mel_lengths = torch.tensor([b["mel"].shape[-1] for b in batch])
        z_lengths = torch.tensor([b["z"].shape[-1] for b in batch])
        L = int(z_lengths.max())
        x = torch.zeros(B, int(x_lengths.max()), dtype=torch.long)
        mel = torch.zeros(B, batch[0]["mel"].shape[0], L * self.r)
        z = torch.zeros(B, batch[0]["z"].shape[0], L)
        for i, b in enumerate(batch):
            x[i, : x_lengths[i]] = b["x"]
            mel[i, :, : mel_lengths[i]] = b["mel"]
            z[i, :, : z_lengths[i]] = b["z"]
        return {"x": x, "x_lengths": x_lengths, "mel": mel, "mel_lengths": mel_lengths, "z": z, "z_lengths": z_lengths,
                "texts": [b["text"] for b in batch], "paths": [b["path"] for b in batch]}


class TextLatentDataModule:
    def __init__(self, name: str, train_filelist: str, valid_filelist: str, vae_ckpt: str, batch_size: int,
                 num_workers: int, pin_memory: bool, cleaner: str, add_blank: bool, mel_cache_dir: str | None,
                 latent_cache_root: str, root: str | None = None, sample_latent: bool = False, seed: int = 0,
                 device: str = "auto", tokenizer_state: dict | None = None):
        self.name, self.batch_size, self.num_workers, self.pin_memory = name, batch_size, num_workers, pin_memory
        device = resolve_device(device)
        self.latent_dir, self.latent_meta = ensure_latent_cache(
            vae_ckpt, [train_filelist, valid_filelist], mel_cache_dir, latent_cache_root, root, device
        )
        _, vae_ckpt_dict = load_vae(vae_ckpt, "cpu")
        self.mel_cfg = vae_ckpt_dict["cfg"]["data"]["mel"]
        self.mel_stats = vae_ckpt_dict["mel_stats"]
        self.compression = self.latent_meta["compression"]
        self.latent_dim = self.latent_meta["latent_dim"]
        mel_fn = hydra.utils.instantiate(self.mel_cfg)
        self.n_mels = mel_fn.n_mels

        if tokenizer_state is not None:
            self.tokenizer = CharTokenizer.from_state(tokenizer_state)
        else:
            from test_tts.data.filelist import parse_filelist  # pylint: disable=import-outside-toplevel

            texts = [u.text for f in (train_filelist, valid_filelist) for u in parse_filelist(f, root)]
            self.tokenizer = CharTokenizer.build(texts, cleaner, add_blank)

        common = dict(tokenizer=self.tokenizer, mel_fn=mel_fn, mel_stats=self.mel_stats, mel_cache_dir=mel_cache_dir,
                      latent_dir=self.latent_dir, lengths=self.latent_meta["lengths"], compression=self.compression,
                      root=root)
        self.train_set = TextLatentDataset(train_filelist, sample_latent=sample_latent, **common)
        self.valid_set = TextLatentDataset(valid_filelist, sample_latent=False, **common)
        self.collate = TextLatentCollate(self.compression)
        self.seed = seed

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.num_workers > 0, drop_last=True,
                          collate_fn=self.collate)

    def valid_dataloader(self):
        return DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.num_workers > 0, collate_fn=self.collate)
