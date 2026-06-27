"""Secondary signal: directional section embedding drift (all articles).

For each section (matched by heading across snapshots) we embed its text per
snapshot and measure DIRECTIONALITY, not just magnitude:

    path_length      = sum of consecutive step distances
    net_displacement = distance from first to last embedding
    straightness     = net_displacement / path_length   in [0, 1]

High straightness + non-trivial net displacement = a section moving slowly but
consistently in one direction (drift no human notices edit-by-edit), as opposed
to back-and-forth churn (low straightness). Boilerplate sections are excluded.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ..embedding import SentenceEncoder
from ..models import RevisionContent


@dataclass
class SectionDrift:
    heading: str
    occurrences: int
    path_length: float
    net_displacement: float
    straightness: float


def _straightness(vectors: np.ndarray) -> tuple[float, float, float]:
    if len(vectors) < 2:
        return 0.0, 0.0, 0.0
    steps = np.linalg.norm(np.diff(vectors, axis=0), axis=1)
    path = float(steps.sum())
    net = float(np.linalg.norm(vectors[-1] - vectors[0]))
    straightness = net / path if path > 1e-12 else 0.0
    return path, net, straightness


def section_drift(
    snapshots: list[RevisionContent],
    encoder: SentenceEncoder,
    min_words: int = 10,
) -> list[SectionDrift]:
    """Per-section directional drift, sorted by (straightness x net) descending."""
    items: list[tuple[str, int, str]] = []  # (heading, snapshot_index, text)
    for i, snap in enumerate(snapshots):
        for sec in snap.sections:
            if sec.is_boilerplate or sec.word_count < min_words:
                continue
            items.append((sec.heading or "(lead)", i, sec.text))
    if not items:
        return []

    vecs = encoder.encode([t for _, _, t in items])
    by_heading: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for (heading, idx, _), v in zip(items, vecs):
        by_heading[heading].append((idx, v))

    out: list[SectionDrift] = []
    for heading, seq in by_heading.items():
        if len(seq) < 2:
            continue
        seq.sort(key=lambda x: x[0])
        matrix = np.array([v for _, v in seq])
        path, net, s = _straightness(matrix)
        out.append(SectionDrift(heading, len(seq), path, net, s))

    out.sort(key=lambda d: d.straightness * d.net_displacement, reverse=True)
    return out
