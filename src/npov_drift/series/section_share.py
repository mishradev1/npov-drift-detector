"""Signal C: due-weight / section-proportion drift (pure; works on ALL articles).

Tracks each section's share of the article body (non-boilerplate) over time, and
the aggregate share of alternative-view sections. Shrinking alt-view share is a
due-weight drift candidate. No ML, no network -- just word counts.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import RevisionContent


@dataclass
class SectionSharePoint:
    revid: int
    timestamp: str
    total_body_words: int
    alt_view_share: float  # aggregate share of alt-view sections within the body
    shares: dict[str, float]  # heading -> share of body words


def section_share_series(snapshots: list[RevisionContent]) -> list[SectionSharePoint]:
    points: list[SectionSharePoint] = []
    for snap in snapshots:
        body = [s for s in snap.sections if not s.is_boilerplate]
        total = sum(s.word_count for s in body)
        shares: dict[str, float] = {}
        alt_words = 0
        for s in body:
            if s.word_count <= 0:
                continue
            heading = s.heading or "(lead)"
            share = s.word_count / total if total else 0.0
            shares[heading] = shares.get(heading, 0.0) + share
            if s.is_alt_view:
                alt_words += s.word_count
        points.append(
            SectionSharePoint(
                revid=snap.revid,
                timestamp=snap.timestamp,
                total_body_words=total,
                alt_view_share=(alt_words / total if total else 0.0),
                shares=shares,
            )
        )
    return points
