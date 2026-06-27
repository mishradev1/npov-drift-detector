"""A small, dependency-free, deterministic sentence splitter.

We avoid heavyweight NLP tokenizers (and their data downloads) so the splitter
is reproducible and CI needs no network. It is heuristic: it splits on
sentence-final punctuation followed by whitespace and a capital/quote/digit,
after protecting a GENERIC list of English abbreviations (no topic-specific
terms). Paragraph/newline breaks are treated as hard sentence boundaries.
"""

from __future__ import annotations

import re

# Generic English abbreviations whose trailing/internal periods must NOT end a
# sentence. Deliberately topic-agnostic.
_ABBREVIATIONS = [
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr.",
    "Prof.",
    "Sr.",
    "Jr.",
    "St.",
    "Mt.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
    "cf.",
    "al.",
    "approx.",
    "No.",
    "Vol.",
    "pp.",
    "fig.",
    "Fig.",
    "Inc.",
    "Ltd.",
    "Co.",
    "U.S.",
    "U.K.",
    "U.N.",
    "a.k.a.",
]

_PLACEHOLDER = "\x00"

# Split after . ! ? (optionally followed by a closing quote/bracket) when the
# next non-space char starts a new sentence (capital, digit, or opening quote).
_SPLIT_RE = re.compile(r'(?<=[.!?])["\')\]]?\s+(?=["\'(\[]?[A-Z0-9])')
_WS_RE = re.compile(r"[ \t]+")


def _protect(text: str) -> str:
    for abbr in _ABBREVIATIONS:
        text = text.replace(abbr, abbr.replace(".", _PLACEHOLDER))
    return text


def _restore(text: str) -> str:
    return text.replace(_PLACEHOLDER, ".")


def _split_chunk(chunk: str) -> list[str]:
    if not chunk:
        return []
    protected = _protect(chunk)
    parts = _SPLIT_RE.split(protected)
    return [_restore(p).strip() for p in parts if p.strip()]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (newlines are hard boundaries)."""
    if not text:
        return []
    sentences: list[str] = []
    for line in re.split(r"[\r\n]+", text):
        line = _WS_RE.sub(" ", line).strip()
        if line:
            sentences.extend(_split_chunk(line))
    return sentences
