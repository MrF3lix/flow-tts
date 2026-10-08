"""Mel-only dataset for the VAE stage: wav -> Whisper mel -> normalised, with random crops."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from test_tts.audio.mel import WhisperMel, load_wav
from test_tts.data.filelist import parse_filelist
from test_tts.data.lengths import pad_to_multiple


class MelDataset(Dataset):
    """Returns `{"mel": (n_mels, T) normalised, "length": T, "path": str}`.

    * The whole utterance is always extracted (the Whisper noise-floor clamp is global), then
      cached as `.npy` under `cache_dir` if given, then cropped to `segment_frames` at random.
    * `mel_stats` = `{"mean", "std"}` scalars; None leaves the mel in the vocoder's space.
    """

    def __init__(
        self,
        filelist: str | Path,
        mel: WhisperMel,
        mel_stats: dict | None = None,
        segment_frames: int | None = None,
        multiple_of: int = 1,
        cache_dir: str | Path | None = None,
        root: str | Path | None = None,
        seed: int = 0,
    ):
        self.items = parse_filelist(filelist, root)
        self.mel = mel
        self.mean = float(mel_stats["mean"]) if mel_stats else 0.0
        self.std = float(mel_stats["std"]) if mel_stats else 1.0
        self.segment_frames = segment_frames
        self.multiple_of = multiple_of
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.rng = random.Random(seed)
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.items)

    def raw_mel(self, index: int) -> Tensor:
        """Un-normalised full-utterance mel `(n_mels, T)` in the vocoder's feature space."""
        path = self.items[index].path
        cache = self.cache_dir / f"{path.stem}.npy" if self.cache_dir is not None else None
        if cache is not None and cache.is_file():
            return torch.from_numpy(np.load(cache))
        wav, _ = load_wav(path, self.mel.sample_rate)
        with torch.no_grad():
            mel = self.mel(wav)
        if cache is not None:
            tmp = cache.with_suffix(f".{os.getpid()}.tmp.npy")
            np.save(tmp, mel.numpy())
            os.replace(tmp, cache)  # atomic, so parallel workers never read a half-written file
        return mel

    def normalize(self, mel: Tensor) -> Tensor:
        return (mel - self.mean) / self.std

    def denormalize(self, mel: Tensor) -> Tensor:
        return mel * self.std + self.mean

    def __getitem__(self, index: int) -> dict:
        mel = self.normalize(self.raw_mel(index))
        if self.segment_frames is not None and mel.shape[-1] > self.segment_frames:
            start = self.rng.randint(0, mel.shape[-1] - self.segment_frames)
            mel = mel[:, start : start + self.segment_frames]
        return {"mel": mel, "length": mel.shape[-1], "path": str(self.items[index].path)}


class MelCollate:
    """Picklable collate (macOS/Windows dataloader workers are spawned, lambdas cannot be sent)."""

    def __init__(self, multiple_of: int = 1):
        self.multiple_of = multiple_of

    def __call__(self, batch: list[dict]) -> dict:
        return collate_mels(batch, self.multiple_of)


def collate_mels(batch: list[dict], multiple_of: int = 1) -> dict:
    """Zero-pad to the longest item (rounded up to `multiple_of`); returns `mel (B, n_mels, T)`,
    `lengths (B,)`, `paths`."""
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    t_max = pad_to_multiple(int(lengths.max()), multiple_of)
    n_mels = batch[0]["mel"].shape[0]
    mel = torch.zeros(len(batch), n_mels, t_max)
    for i, item in enumerate(batch):
        mel[i, :, : item["length"]] = item["mel"]
    return {"mel": mel, "lengths": lengths, "paths": [item["path"] for item in batch]}


class MelDataModule:
    """Builds the train/valid datasets and loaders from the hydra `data` config."""

    def __init__(
        self,
        name: str,
        train_filelist: str,
        valid_filelist: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        mel: WhisperMel,
        mel_stats: dict | None,
        segment_frames: int | None,
        multiple_of: int,
        cache_dir: str | None = None,
        root: str | None = None,
        seed: int = 0,
    ):
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mel = mel
        self.mel_stats = mel_stats
        self.multiple_of = multiple_of
        self.train_set = MelDataset(
            train_filelist, mel, mel_stats, segment_frames, multiple_of, cache_dir, root, seed
        )
        # Validation: whole utterances, so the reconstruction metric is comparable across runs.
        self.valid_set = MelDataset(valid_filelist, mel, mel_stats, None, multiple_of, cache_dir, root, seed)

    def _loader(self, dataset, shuffle, batch_size=None):
        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            drop_last=shuffle,
            collate_fn=MelCollate(self.multiple_of),
        )

    def train_dataloader(self):
        return self._loader(self.train_set, shuffle=True)

    def valid_dataloader(self):
        # Whole utterances are long; halve the batch so validation fits where training fits.
        return self._loader(self.valid_set, shuffle=False, batch_size=max(1, self.batch_size // 2))
