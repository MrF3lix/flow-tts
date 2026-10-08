"""Character tokenizer whose vocabulary is built from the training texts.

Ids: 0 = padding, 1 = blank, 2 = unknown, then the characters in sorted order. With
`add_blank=True` a blank is interspersed between all characters (Glow-TTS / Matcha), which
gives monotonic alignment search a place to put transitions and pauses.

The vocabulary is stored in every checkpoint (`state_dict()`), so inference never depends on
re-deriving it from a filelist.
"""

from __future__ import annotations

from collections.abc import Iterable

from test_tts.text.cleaners import get_cleaner

PAD, BLANK, UNK = "<pad>", "<blank>", "<unk>"
SPECIALS = [PAD, BLANK, UNK]


class CharTokenizer:
    def __init__(self, symbols: list[str], cleaner: str = "swiss_german", add_blank: bool = True):
        self.symbols = list(symbols)
        self.cleaner_name = cleaner
        self.cleaner = get_cleaner(cleaner)
        self.add_blank = add_blank
        self.vocab = SPECIALS + self.symbols
        self.index = {s: i for i, s in enumerate(self.vocab)}
        self.pad_id, self.blank_id, self.unk_id = 0, 1, 2

    @classmethod
    def build(cls, texts: Iterable[str], cleaner: str = "swiss_german", add_blank: bool = True) -> "CharTokenizer":
        clean = get_cleaner(cleaner)
        chars = sorted({c for t in texts for c in clean(t)})
        return cls(chars, cleaner, add_blank)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def clean(self, text: str) -> str:
        return self.cleaner(text)

    def encode(self, text: str) -> list[int]:
        ids = [self.index.get(c, self.unk_id) for c in self.clean(text)]
        if self.add_blank:
            out = [self.blank_id] * (2 * len(ids) + 1)
            out[1::2] = ids
            ids = out
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.vocab[i] for i in ids if i > self.unk_id or i == self.unk_id)

    def unknown_chars(self, text: str) -> set[str]:
        return {c for c in self.clean(text) if c not in self.index}

    def state_dict(self) -> dict:
        return {"symbols": self.symbols, "cleaner": self.cleaner_name, "add_blank": self.add_blank}

    @classmethod
    def from_state(cls, state: dict) -> "CharTokenizer":
        return cls(state["symbols"], state["cleaner"], state["add_blank"])
