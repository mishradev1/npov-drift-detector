"""Choose which revisions to fetch *content* for.

Metadata is fetched for every revision (cheap). Content is large, so we sample.
The default strategy samples one snapshot per calendar month (the WMF guidance
is to "measure trends, not moments"), always keeping the first and last
revisions, and falls back to even spacing when there are more candidate months
than ``max_snapshots``.

All functions here are pure and deterministic so they are trivially unit-tested
with synthetic revisions (no network).
"""

from __future__ import annotations

from typing import Iterable

from ..models import RevisionMeta


def _evenly_spaced_indices(n: int, k: int) -> list[int]:
    """k indices in [0, n-1], including the endpoints, as evenly spaced as
    integer rounding allows, de-duplicated while preserving order."""
    if k <= 0 or n <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k == 1:
        return [0]
    raw = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    seen: set[int] = set()
    out: list[int] = []
    for idx in raw:
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def select_snapshot_revids(
    revisions: Iterable[RevisionMeta],
    max_snapshots: int = 24,
    strategy: str = "monthly",
) -> list[int]:
    """Return the revids to fetch content for, chronologically.

    * If there are no more revisions than ``max_snapshots``, take them all.
    * ``monthly``: keep the last revision of each calendar month, plus the very
      first and last revisions overall; if that still exceeds ``max_snapshots``,
      thin the set down with even spacing.
    * ``evenly``: just pick ``max_snapshots`` evenly spaced revisions by index.
    """
    revs = sorted(revisions, key=lambda r: (r.timestamp, r.revid))
    if not revs:
        return []
    if len(revs) <= max_snapshots:
        return [r.revid for r in revs]

    if strategy == "evenly":
        idxs = _evenly_spaced_indices(len(revs), max_snapshots)
        return [revs[i].revid for i in idxs]

    if strategy != "monthly":
        raise ValueError(f"unknown sampling strategy: {strategy!r}")

    # One per calendar month (last revision wins because revs is ascending).
    by_month: dict[str, RevisionMeta] = {}
    for r in revs:
        by_month[r.timestamp[:7]] = r  # "YYYY-MM"

    chosen: dict[int, RevisionMeta] = {r.revid: r for r in by_month.values()}
    # Always anchor on the true endpoints (baseline-relevant).
    chosen[revs[0].revid] = revs[0]
    chosen[revs[-1].revid] = revs[-1]

    candidates = sorted(chosen.values(), key=lambda r: (r.timestamp, r.revid))
    if len(candidates) <= max_snapshots:
        return [c.revid for c in candidates]

    idxs = _evenly_spaced_indices(len(candidates), max_snapshots)
    return [candidates[i].revid for i in idxs]
