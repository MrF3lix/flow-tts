"""Write `path|text` (or `path|speaker|text`) train/val filelists from a metadata table.

Bernese corpus (`id|text|text`, no header, wavs at data/be/prepared/wav):

    uv run tts-build-filelist --metadata data/be/metadata.txt --audio-col 0 --text-col 1 \
        --audio-root data/be/prepared/wav --out-dir data/filelists/be

Audio paths are written as `--audio-root` + stem + `.wav`, relative to the project root, so the
filelists work on any machine with the same layout. Every file is opened once to check that it
exists and to drop clips outside [--min-seconds, --max-seconds].
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--delimiter", default="|")
    p.add_argument("--header", action="store_true", help="first row is a header; columns are then names")
    p.add_argument("--audio-col", required=True)
    p.add_argument("--text-col", required=True)
    p.add_argument("--speaker-col", default=None, help="omit for a single-speaker corpus")
    p.add_argument("--audio-root", type=Path, required=True)
    p.add_argument("--audio-ext", default=".wav")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--val-fraction", type=float, default=0.02)
    p.add_argument("--min-val-per-speaker", type=int, default=1)
    p.add_argument("--min-seconds", type=float, default=0.5)
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    with open(args.metadata, encoding="utf-8", newline="") as f:
        if args.header:
            rows = list(csv.DictReader(f, delimiter=args.delimiter, quoting=csv.QUOTE_NONE))
        else:
            rows = [{str(i): v for i, v in enumerate(r)} for r in csv.reader(f, delimiter=args.delimiter, quoting=csv.QUOTE_NONE) if r]
    print(f"read {len(rows)} rows from {args.metadata}")

    dropped = Counter()
    rates = Counter()
    seconds = 0.0
    per_speaker = defaultdict(list)
    for row in rows:
        text = row[args.text_col].strip()
        speaker = row[args.speaker_col].strip() if args.speaker_col else "speaker"
        path = Path(row[args.audio_col].strip())
        if not path.suffix:
            path = path.with_suffix(args.audio_ext)
        path = args.audio_root / path
        if not text:
            dropped["empty text"] += 1
            continue
        if not path.is_file():
            dropped["audio missing"] += 1
            continue
        info = sf.info(str(path))
        dur = info.frames / info.samplerate
        rates[(info.samplerate, info.channels)] += 1
        if dur < args.min_seconds:
            dropped[f"shorter than {args.min_seconds}s"] += 1
            continue
        if dur > args.max_seconds:
            dropped[f"longer than {args.max_seconds}s"] += 1
            continue
        seconds += dur
        per_speaker[speaker].append((path, text))

    speakers = sorted(per_speaker)
    speaker_ids = {name: i for i, name in enumerate(speakers)}
    train, val = [], []
    for name in speakers:
        items = per_speaker[name]
        random.shuffle(items)
        n_val = max(args.min_val_per_speaker, round(len(items) * args.val_fraction))
        val += [(speaker_ids[name], *it) for it in items[:n_val]]
        train += [(speaker_ids[name], *it) for it in items[n_val:]]
    random.shuffle(train)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, items in (("train", train), ("val", val)):
        with open(args.out_dir / f"{split}.txt", "w", encoding="utf-8") as f:
            for spk, path, text in items:
                f.write(f"{path.as_posix()}|{spk}|{text}\n" if args.speaker_col else f"{path.as_posix()}|{text}\n")
    if args.speaker_col:
        import json  # pylint: disable=import-outside-toplevel

        (args.out_dir / "speakers.json").write_text(json.dumps({i: n for n, i in speaker_ids.items()}, indent=2, ensure_ascii=False))

    print(f"kept {len(train)} train + {len(val)} val utterances, {seconds / 3600:.2f} h, {len(speakers)} speaker(s)")
    print(f"sample rates / channels: {dict(rates)}")
    if dropped:
        print(f"dropped: {dict(dropped)}")
    print(f"written: {args.out_dir / 'train.txt'}, {args.out_dir / 'val.txt'}")


if __name__ == "__main__":
    main()
