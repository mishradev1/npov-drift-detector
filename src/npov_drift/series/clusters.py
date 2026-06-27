"""Signal B: unsupervised perspective-cluster balance (topic-agnostic).

We embed sentences and cluster them into latent perspective/aspect groups. To
keep cluster identity stable across revisions (the spec's "align clusters across
revisions"), we fit the clustering ONCE on the pooled sentences from all
snapshots, then assign each snapshot's sentences to those fixed clusters. This
is alignment-by-construction and avoids fragile per-revision re-labelling.

We then track concentration via the Herfindahl-Hirschman Index (HHI) of cluster
shares: HHI rising over time means one perspective is crowding out the others.
Clusters are latent aspect groups, NOT labelled viewpoints -- we report share
shifts, not what each cluster "means".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from sklearn.cluster import KMeans

from ..embedding import SentenceEncoder
from ..ingest.sampling import _evenly_spaced_indices
from ..models import RevisionContent


@dataclass
class ClusterPoint:
    revid: int
    timestamp: str
    n: int
    shares: list[float]
    hhi: float


def hhi(shares) -> float:
    """Herfindahl-Hirschman Index = sum of squared shares, in [1/k, 1]."""
    return float(sum(float(x) * float(x) for x in shares))


def _sample(seq: list[str], k: Optional[int]) -> list[str]:
    if not k or len(seq) <= k:
        return seq
    return [seq[i] for i in _evenly_spaced_indices(len(seq), k)]


def perspective_balance_series(
    snapshots: list[RevisionContent],
    encoder: SentenceEncoder,
    sentence_fn: Callable[[RevisionContent], list[str]],
    k: int = 5,
    random_state: int = 0,
    max_sentences_per_snapshot: Optional[int] = 200,
) -> tuple[list[ClusterPoint], Optional[np.ndarray]]:
    """Return (per-snapshot cluster shares + HHI, fitted cluster centroids)."""
    per_snapshot = [_sample(sentence_fn(s), max_sentences_per_snapshot) for s in snapshots]
    all_sentences = [s for group in per_snapshot for s in group]
    if len(all_sentences) < k:
        return [], None

    emb = encoder.encode(all_sentences)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(emb)

    points: list[ClusterPoint] = []
    pos = 0
    for snap, group in zip(snapshots, per_snapshot):
        n = len(group)
        if n == 0:
            points.append(ClusterPoint(snap.revid, snap.timestamp, 0, [0.0] * k, 0.0))
            continue
        labels = km.predict(emb[pos : pos + n])
        pos += n
        counts = np.bincount(labels, minlength=k)
        shares = (counts / counts.sum()).tolist()
        points.append(ClusterPoint(snap.revid, snap.timestamp, n, shares, hhi(shares)))

    return points, km.cluster_centers_
