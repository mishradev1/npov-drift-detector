"""Orchestrate onset detection across drift signals.

Builds 1-D drift series from an article's snapshots (relative to its mature
baseline), finds each signal's onset (largest level shift within the mature
window), and reconciles them into a consensus onset with an agreement count.

Signals:
  * due_weight_departure  -- total-variation of section shares vs baseline (pure)
  * alt_view_share        -- aggregate alt-view section share (pure)
  * semantic_departure    -- 1 - cos(snapshot doc-embedding, baseline)   (encoder)
  * stance_balance        -- favorable-minus-critical balance per snapshot (classifier)

If no viewpoint (stance) signal is active, the report says so and rests on the
structure/semantic signals -- never forcing a viewpoint verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..embedding import SentenceEncoder
from ..models import RevisionContent
from ..series.section_share import section_share_series
from ..series.stance_series import stance_balance_series
from ..stance.base import StanceClassifier
from .baseline import select_mature_baseline
from .changepoint import detect_changepoint_meanshift


@dataclass
class Onset:
    signal: str
    index: int
    timestamp: str
    effect: float


@dataclass
class OnsetReport:
    baseline_timestamp: Optional[str]
    signals: list[str]
    onsets: list[Onset]
    consensus_timestamp: Optional[str]
    agreement: int
    note: str
    viewpoint_active: bool = False
    series: dict[str, list[float]] = field(default_factory=dict)
    timestamps: list[str] = field(default_factory=list)


def _tv(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def reconcile(onsets: list[Onset], tol_index: int = 2) -> tuple[Optional[str], int]:
    """Cluster onsets by snapshot proximity; the largest cluster wins.

    Consensus timestamp is the EARLIEST in the winning cluster (onset = start).
    """
    if not onsets:
        return None, 0
    items = sorted(onsets, key=lambda o: o.index)
    best: list[Onset] = []
    for anchor in items:
        cluster = [o for o in items if abs(o.index - anchor.index) <= tol_index]
        if len(cluster) > len(best):
            best = cluster
    consensus = min(o.timestamp for o in best)
    return consensus, len(best)


def _semantic_departure(snapshots, encoder: SentenceEncoder, baseline_idx: int) -> Optional[list[float]]:
    texts: list[str] = []
    spans: list[tuple[int, int]] = []
    for snap in snapshots:
        secs = [s.text for s in snap.sections if not s.is_boilerplate and s.word_count >= 10]
        spans.append((len(texts), len(texts) + len(secs)))
        texts.extend(secs)
    if not texts:
        return None
    vecs = encoder.encode(texts)

    snap_vecs: list[Optional[np.ndarray]] = []
    for a, b in spans:
        if b > a:
            v = vecs[a:b].mean(axis=0)
            norm = np.linalg.norm(v)
            snap_vecs.append(v / norm if norm > 0 else v)
        else:
            snap_vecs.append(None)

    base = snap_vecs[baseline_idx]
    if base is None:
        return None
    return [0.0 if v is None else float(1.0 - float(np.dot(v, base))) for v in snap_vecs]


def detect_drift_onset(
    snapshots: list[RevisionContent],
    *,
    encoder: Optional[SentenceEncoder] = None,
    classifier: Optional[StanceClassifier] = None,
    topic: Optional[str] = None,
    sentence_fn: Optional[Callable[[RevisionContent], list[str]]] = None,
    min_words: int = 800,
    max_sentences_per_snapshot: int = 60,
) -> OnsetReport:
    if len(snapshots) < 3:
        return OnsetReport(None, [], [], None, 0, "too few snapshots for onset detection")

    baseline = select_mature_baseline(snapshots, min_words=min_words)
    if baseline is None:
        return OnsetReport(None, [], [], None, 0, "article never reached the maturity floor; no baseline")
    bi = baseline.index

    points = section_share_series(snapshots)
    timestamps = [p.timestamp for p in points]
    base_shares = points[bi].shares

    series: dict[str, list[float]] = {
        "due_weight_departure": [_tv(p.shares, base_shares) for p in points],
        "alt_view_share": [p.alt_view_share for p in points],
    }

    if encoder is not None:
        sem = _semantic_departure(snapshots, encoder, bi)
        if sem is not None:
            series["semantic_departure"] = sem

    viewpoint_active = False
    if classifier is not None and topic is not None and sentence_fn is not None:
        sp = stance_balance_series(
            snapshots, classifier, topic, sentence_fn, max_sentences_per_snapshot=max_sentences_per_snapshot
        )
        series["stance_balance"] = [p.dist.balance for p in sp]
        viewpoint_active = any(p.active for p in sp)

    # Onset per signal: largest level shift within the MATURE window only.
    onsets: list[Onset] = []
    for name, values in series.items():
        cp = detect_changepoint_meanshift(values[bi:])
        if cp is not None:
            idx, effect = cp
            global_idx = bi + idx
            onsets.append(Onset(name, global_idx, timestamps[global_idx], effect))

    consensus, agreement = reconcile(onsets)

    if classifier is None:
        note = "viewpoint (stance) signal not run; structure/semantic drift only"
    elif not viewpoint_active:
        note = "no viewpoint drift measurable (stance signal inactive); structure/semantic drift only"
    else:
        note = "viewpoint (stance) signal active"

    return OnsetReport(
        baseline_timestamp=baseline.timestamp,
        signals=list(series.keys()),
        onsets=onsets,
        consensus_timestamp=consensus,
        agreement=agreement,
        note=note,
        viewpoint_active=viewpoint_active,
        series=series,
        timestamps=timestamps,
    )
