"""Assemble the full drift report for an article (view-agnostic, testable).

Produces the spec's output sections: active-signals summary, onset estimate,
trajectory series, section directional-drift map, due-weight section-share
changes, key edits in the drift window, and a hedged plain-language statement.
Crucially, the statement is ALWAYS a hedged "candidate for human review", never
a determination of bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional
from urllib.parse import quote

from ..embedding import SentenceEncoder
from ..models import ArticleHistory
from ..onset.detect import OnsetReport, detect_drift_onset
from ..series.section_drift import SectionDrift, section_drift
from ..series.section_share import SectionSharePoint, section_share_series
from ..stance.base import StanceClassifier


@dataclass
class DriftReport:
    title: str
    pageid: Optional[int]
    bucket: str
    contested_prior: Optional[float]
    n_revisions: int
    n_editors: int
    date_span: tuple[Optional[str], Optional[str]]
    active_signals: dict[str, bool]
    onset: OnsetReport
    share_series: list[SectionSharePoint]
    section_drifts: list[SectionDrift]
    key_edits: list[dict]
    hedged_statement: str
    reference_noise_floor: Optional[dict] = field(default=None)


def _diff_url(title: str, revid: int) -> str:
    return f"https://en.wikipedia.org/w/index.php?title={quote(title.replace(' ', '_'))}&diff=prev&oldid={revid}"


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _key_edits(hist: ArticleHistory, onset_ts: Optional[str], window_days: int = 120, top_n: int = 15) -> list[dict]:
    """Revisions in the drift window, ranked by absolute size change.

    The window is +/- ``window_days`` around the estimated onset. Size deltas are
    computed against each revision's chronological predecessor.
    """
    if onset_ts is None or not hist.revisions:
        return []
    center = _parse(onset_ts)
    lo, hi = center - timedelta(days=window_days), center + timedelta(days=window_days)

    revs = sorted(hist.revisions, key=lambda r: r.timestamp)
    edits = []
    prev_size = 0
    for r in revs:
        delta = r.size - prev_size
        prev_size = r.size
        if lo <= _parse(r.timestamp) <= hi:
            edits.append(
                {
                    "revid": r.revid,
                    "timestamp": r.timestamp,
                    "user": r.user or "(hidden)",
                    "size_delta": delta,
                    "comment": (r.comment or "")[:140],
                    "diff_url": _diff_url(hist.title, r.revid),
                }
            )
    edits.sort(key=lambda e: abs(e["size_delta"]), reverse=True)
    return edits[:top_n]


def _hedged_statement(report: OnsetReport, bucket: str) -> str:
    disclaimer = (
        " This is a candidate flagged for human review, not a determination that "
        "the article is biased or violates NPOV."
    )
    if report.consensus_timestamp is None:
        return ("No directional drift onset was detected in the analysed window." + disclaimer)
    when = report.consensus_timestamp[:10]
    if report.viewpoint_active:
        lead = (
            f"This article's balance of perspectives appears to have begun shifting "
            f"around {when}; review the edits in that window below."
        )
    else:
        lead = (
            f"No viewpoint-balance drift was measurable (the stance signal is "
            f"inactive or was not run); a structural/semantic content shift was "
            f"estimated around {when}. Review the edits in that window below."
        )
    agree = f" ({report.agreement}/{len(report.signals)} signals agree on the timing.)"
    return lead + agree + disclaimer


def build_drift_report(
    hist: ArticleHistory,
    *,
    encoder: Optional[SentenceEncoder] = None,
    classifier: Optional[StanceClassifier] = None,
    topic: Optional[str] = None,
    sentence_fn: Optional[Callable] = None,
    min_words: int = 800,
    reference_profile: Optional[dict] = None,
) -> DriftReport:
    onset = detect_drift_onset(
        hist.snapshots,
        encoder=encoder,
        classifier=classifier,
        topic=topic,
        sentence_fn=sentence_fn,
        min_words=min_words,
    )

    active_signals = {
        "due_weight": True,
        "semantic": encoder is not None and "semantic_departure" in onset.signals,
        "stance": onset.viewpoint_active,
    }

    bucket = hist.article_type.bucket if hist.article_type else "unknown"
    contested_prior = hist.article_type.contested_prior if hist.article_type else None

    ref_noise = None
    if reference_profile and bucket in reference_profile:
        ref_noise = reference_profile[bucket].get("noise_floor")

    return DriftReport(
        title=hist.title,
        pageid=hist.pageid,
        bucket=bucket,
        contested_prior=contested_prior,
        n_revisions=len(hist.revisions),
        n_editors=hist.num_editors(),
        date_span=hist.date_span(),
        active_signals=active_signals,
        onset=onset,
        share_series=section_share_series(hist.snapshots),
        section_drifts=section_drift(hist.snapshots, encoder) if encoder is not None else [],
        key_edits=_key_edits(hist, onset.consensus_timestamp),
        hedged_statement=_hedged_statement(onset, bucket),
        reference_noise_floor=ref_noise,
    )
