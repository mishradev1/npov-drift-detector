"""Shared helpers for extracting analysable text from a snapshot."""

from __future__ import annotations

from ..models import RevisionContent
from ..stance.sentences import split_sentences


def body_sections(snapshot: RevisionContent):
    """Non-boilerplate sections (References/External links/... excluded)."""
    return [s for s in snapshot.sections if not s.is_boilerplate]


def body_text(snapshot: RevisionContent) -> str:
    return "\n".join(s.text for s in body_sections(snapshot))


def body_sentences(snapshot: RevisionContent, min_words: int = 5) -> list[str]:
    return [s for s in split_sentences(body_text(snapshot)) if len(s.split()) >= min_words]
