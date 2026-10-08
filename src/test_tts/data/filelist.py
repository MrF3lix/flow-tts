"""Filelists: one utterance per line, `path|text` or `path|speaker_id|text` (Matcha format)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Utterance:
    path: Path
    text: str
    speaker: int | None = None


def parse_filelist(filelist: str | Path, root: str | Path | None = None) -> list[Utterance]:
    """Relative audio paths are resolved against `root` (default: the filelist's project root,
    i.e. the current working directory)."""
    items = []
    with open(filelist, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 2:
                path, text, spk = parts[0], parts[1], None
            elif len(parts) >= 3:
                path, spk, text = parts[0], int(parts[1]), "|".join(parts[2:])
            else:
                raise ValueError(f"{filelist}: expected `path|text` or `path|speaker|text`, got {line[:80]!r}")
            path = Path(path)
            if root is not None and not path.is_absolute():
                path = Path(root) / path
            items.append(Utterance(path, text, spk))
    return items
