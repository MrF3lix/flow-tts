"""Average VocBulwark speaker embedding(s) from reference audio (the `s` of NOTES.md §2).

Single speaker, from a folder of wavs:

    uv run tts-speaker-embedding --wav-dir data/be/prepared/wav --n-clips 50 --out data/be_speaker_embedding.pt

Multi-speaker, from a `path|speaker_id|text` filelist -> `[N, 768]` table, row i = speaker i:

    uv run tts-speaker-embedding --filelist data/filelists/xy/train.txt --out data/xy_speaker_embeddings.pt
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from test_tts.audio.mel import load_wav
from test_tts.audio.vocoder import load_speaker_encoder
from test_tts.data.filelist import parse_filelist


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--wav-dir", type=Path)
    src.add_argument("--filelist", type=Path)
    p.add_argument("--n-clips", type=int, default=50)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--l2-normalise", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.wav_dir is not None:
        clips = sorted(args.wav_dir.glob("*.wav"))
        if not clips:
            raise SystemExit(f"no .wav files under {args.wav_dir}")
        groups = {0: clips[: args.n_clips]}
    else:
        groups = defaultdict(list)
        for utt in parse_filelist(args.filelist):
            if utt.speaker is None:
                raise SystemExit("filelist has no speaker column; use --wav-dir for a single speaker")
            groups[utt.speaker].append(utt.path)
        if sorted(groups) != list(range(len(groups))):
            raise SystemExit(f"speaker ids must be contiguous 0..N-1, got {sorted(groups)}")
        groups = {k: sorted(v)[: args.n_clips] for k, v in groups.items()}

    enc = load_speaker_encoder(args.device)
    sr = enc.config.raw_sample_rate
    rows = []
    for spk in sorted(groups):
        embs = []
        for path in groups[spk]:
            wav, _ = load_wav(path, sr)
            with torch.no_grad():
                embs.append(enc.embed(wav[None].to(args.device)).cpu())
        stacked = torch.cat(embs)
        emb = stacked.mean(0, keepdim=True)
        if args.l2_normalise:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        spread = (stacked - stacked.mean(0, keepdim=True)).norm(dim=-1)
        print(f"speaker {spk:3d}: {len(embs):3d} clips  norm={emb.norm():.4f}  spread mean={spread.mean():.4f} max={spread.max():.4f}")
        rows.append(emb)

    table = torch.cat(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(table, args.out)
    print(f"encoder input {sr} Hz, table {tuple(table.shape)} -> {args.out}")


if __name__ == "__main__":
    main()
