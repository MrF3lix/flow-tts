"""Mean / std of the training mels for `data.mel_stats` in the data config.

    uv run tts-data-stats data=be_vocbulwark [--workers 4] [hydra overrides...]

Uses the configured mel front-end on whole utterances, so the numbers are exactly those the
training pipeline normalises with. Fills the mel cache as a side effect.
"""

from __future__ import annotations

import argparse

import hydra
from torch.utils.data import DataLoader
from tqdm import tqdm

from test_tts.cli._compose import compose_config
from test_tts.data.mel_dataset import MelDataset, collate_mels


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("overrides", nargs="*", help="hydra overrides, e.g. data=be_vocbulwark")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    cfg = compose_config(args.overrides)
    dcfg = cfg.data
    mel = hydra.utils.instantiate(dcfg.mel)
    dataset = MelDataset(dcfg.train_filelist, mel, mel_stats=None, segment_frames=None,
                         cache_dir=dcfg.get("cache_dir"), root=dcfg.get("root"))
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate_mels)

    total, total_sq, n = 0.0, 0.0, 0
    for batch in tqdm(loader, desc="mels"):
        m, lengths = batch["mel"].double(), batch["lengths"]
        for i in range(m.shape[0]):
            x = m[i, :, : lengths[i]]
            total += x.sum().item()
            total_sq += x.square().sum().item()
            n += x.numel()
    mean = total / n
    std = (total_sq / n - mean**2) ** 0.5
    print(f"\n{len(dataset)} utterances, {n / mel.n_mels / mel.frames_per_second / 3600:.2f} h\n")
    print(f"# paste into configs/data/{dcfg.name}.yaml\nmel_stats:\n  mean: {mean}\n  std: {std}")


if __name__ == "__main__":
    main()
