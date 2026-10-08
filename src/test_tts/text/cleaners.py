"""Text normalisation for character-level TTS (no phonemiser, NOTES.md §1).

`swiss_german` follows the Matcha-TTS fork's `swiss_german_cleaners`: NFC, numbers spelled out
in German, lowercase, typographic variants folded onto plain characters, brackets dropped,
whitespace collapsed. Umlauts are kept: the model learns grapheme-to-sound directly. All
quotation marks are folded onto `"` so the rare typographic ones do not get their own token.
"""

from __future__ import annotations

import re
import unicodedata

from test_tts.text.numbers_de import normalize_numbers_de

_whitespace_re = re.compile(r"\s+")
_brackets_re = re.compile(r"[\[\]\(\)\{\}]")

_equivalents = [
    ("’", "'"), ("‘", "'"), ("‚", "'"), ("´", "'"), ("`", "'"),  # apostrophes
    ("«", '"'), ("»", '"'), ("“", '"'), ("”", '"'), ("„", '"'),  # « » “ ” „
    ("‹", '"'), ("›", '"'),  # ‹ ›
    ("–", "-"), ("—", "-"), ("‑", "-"),  # dashes, non-breaking hyphen
    (" ", " "),  # no-break space
    ("­", ""),  # soft hyphen
    ("​", ""),  # zero-width space
    ("ß", "ss"),  # Swiss Standard German has no ß
    ("%", " prozent"),
    ("&", " und "),
    ("/", " "),
]


def swiss_german(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = normalize_numbers_de(text)
    text = text.lower()
    for src, dst in _equivalents:
        text = text.replace(src, dst)
    text = _brackets_re.sub("", text)
    text = _whitespace_re.sub(" ", text)
    return text.strip()


def basic(text: str) -> str:
    """Lowercase + whitespace only, for corpora that are already normalised."""
    return _whitespace_re.sub(" ", unicodedata.normalize("NFC", text).lower()).strip()


CLEANERS = {"swiss_german": swiss_german, "basic": basic}


def get_cleaner(name: str):
    try:
        return CLEANERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown cleaner {name!r}, available: {sorted(CLEANERS)}") from exc
