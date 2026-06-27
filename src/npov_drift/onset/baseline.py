"""Mature-baseline selection.

Per the WMF "choice of baseline" guidance, an article's own stub-era revisions
are an invalid baseline. We take the baseline as the first snapshot at which the
article body crosses a maturity word floor (optionally informed by the
type-matched reference profile's typical mature size). Everything downstream
measures departure from this point, and changepoint detection runs only on the
mature window.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import RevisionContent


@dataclass
class Baseline:
    index: int  # index into the snapshot list
    timestamp: str
    body_words: int


def select_mature_baseline(
    snapshots: list[RevisionContent],
    min_words: int = 800,
) -> Baseline | None:
    """First snapshot whose non-boilerplate body reaches ``min_words``.

    Returns None if the article never matures (so callers can report "no
    baseline" rather than baselining on a stub).
    """
    for i, snap in enumerate(snapshots):
        if snap.body_word_count() >= min_words:
            return Baseline(index=i, timestamp=snap.timestamp, body_words=snap.body_word_count())
    return None
