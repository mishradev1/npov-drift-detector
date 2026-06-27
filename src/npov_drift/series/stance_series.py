"""Signal A: stance balance over time.

Applies the Phase 2 stance classifier per snapshot and records the stance
distribution (favorable/critical/neutral fractions, balance, and the self-gating
active flag). The directional quantity later phases watch for drift is
``dist.balance`` (favorable minus critical), tracked across snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ..ingest.sampling import _evenly_spaced_indices
from ..models import RevisionContent
from ..stance.aggregate import StanceDistribution, is_stance_active, stance_distribution
from ..stance.base import StanceClassifier


@dataclass
class StanceSeriesPoint:
    revid: int
    timestamp: str
    dist: StanceDistribution
    active: bool


def _sample(seq: list[str], k: Optional[int]) -> list[str]:
    if not k or len(seq) <= k:
        return seq
    return [seq[i] for i in _evenly_spaced_indices(len(seq), k)]


def stance_balance_series(
    snapshots: list[RevisionContent],
    classifier: StanceClassifier,
    topic: str,
    sentence_fn: Callable[[RevisionContent], list[str]],
    max_sentences_per_snapshot: Optional[int] = 120,
) -> list[StanceSeriesPoint]:
    points: list[StanceSeriesPoint] = []
    for snap in snapshots:
        sentences = _sample(sentence_fn(snap), max_sentences_per_snapshot)
        results = classifier.classify(sentences, topic)
        dist = stance_distribution(results)
        points.append(
            StanceSeriesPoint(
                revid=snap.revid,
                timestamp=snap.timestamp,
                dist=dist,
                active=is_stance_active(dist),
            )
        )
    return points
