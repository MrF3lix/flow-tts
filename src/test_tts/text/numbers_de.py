"""Spell out numbers in German text, for character-level TTS without a phonemiser.

Ported from the Matcha-TTS fork (matcha/text/numbers_de.py). Handles the forms found in Swiss
news text: Swiss thousands separators (50'000), decimals with point or comma (26.9 / 26,9),
years (1999 -> neunzehnhundertneunundneunzig), ordinals before a month or "Jahrhundert"
(3. April -> dritten April), then any remaining integer.
"""

import re

from num2words import num2words

_MONTHS = "januar|februar|märz|april|mai|juni|juli|august|september|oktober|november|dezember|jahrhundert"

_thousands_re = re.compile(r"\d{1,3}(?:['’  ]\d{3})+")
_decimal_re = re.compile(r"(\d+)[.,](\d+)")
_year_re = re.compile(r"\b(1[1-9]\d{2}|20\d{2})\b")
_ordinal_re = re.compile(rf"\b(\d{{1,2}})\.(?=\s+(?:{_MONTHS})\b)", re.IGNORECASE)
_number_re = re.compile(r"\d+")


def _de(n, **kw):
    return num2words(n, lang="de", **kw)


def _expand_thousands(m):
    return re.sub(r"['’  ]", "", m.group(0))


def _expand_decimal(m):
    # "26.9" -> "sechsundzwanzig Komma neun": digits after the separator are read one by one.
    frac = " ".join(_de(int(d)) for d in m.group(2))
    return f"{_de(int(m.group(1)))} Komma {frac}"


def _expand_year(m):
    return _de(int(m.group(1)), to="year")


def _expand_ordinal(m):
    # Dative form, the case a date has after "am": "am dritten April".
    return _de(int(m.group(1)), to="ordinal") + "n"


def _expand_number(m):
    return _de(int(m.group(0)))


def normalize_numbers_de(text: str) -> str:
    text = _thousands_re.sub(_expand_thousands, text)
    text = _decimal_re.sub(_expand_decimal, text)
    text = _year_re.sub(_expand_year, text)
    text = _ordinal_re.sub(_expand_ordinal, text)
    text = _number_re.sub(_expand_number, text)
    return text
