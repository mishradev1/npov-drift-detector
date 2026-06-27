"""Coarse article-type classification for baseline bucketing.

We map Wikimedia ``articletopic`` predictions (ORES, falling back to Lift Wing)
onto a small set of coarse buckets used later to compare an article against a
type-matched Featured/Good-Article reference distribution.

IMPORTANT: ``contested_prior`` is only a prior for choosing the comparison
bucket and setting expectations. It is NOT a verdict and is NOT used to declare
any article biased. Whether the viewpoint signal is actually active is decided
later from data (stance dispersion), per the build spec.
"""

from __future__ import annotations

from typing import Optional

from .. import config
from ..models import ArticleType

# Ordered (prefix, bucket). More specific prefixes must come first so that,
# e.g., "STEM.Medicine & Health" maps to "medicine" before the generic "STEM".
#
# NOTE: "Geography.Regions.*" is deliberately NOT here. Those labels describe
# which region a topic is *associated* with (they fire for any article that
# mentions places, e.g. "Capital punishment" lights up Asia/Africa/Europe), so
# they are a locational axis, not an article-type signal. Only the real
# "Geography.Geographical" label maps to the geography bucket (see
# _bucket_for_label, which drops Regions.* first).
_BUCKET_RULES: list[tuple[str, str]] = [
    ("History and Society.Politics and government", "politics"),
    ("History and Society.History", "history"),
    ("History and Society.Military and warfare", "history"),
    ("History and Society.Society", "society"),
    ("History and Society.Education", "society"),
    ("History and Society.Business and economics", "society"),
    ("History and Society.Transportation", "society"),
    ("Geography", "geography"),  # only reached for Geography.Geographical
    ("Culture.Biography", "biography"),
    ("Culture.Sports", "sports"),
    ("Culture.Philosophy and religion", "religion"),
    ("Culture.Media", "arts"),
    ("Culture.Visual arts", "arts"),
    ("Culture.Performing arts", "arts"),
    ("Culture.Literature", "arts"),
    ("Culture.Linguistics", "arts"),
    ("Culture.Food and drink", "society"),
    ("Culture.Internet culture", "society"),
    ("STEM.Medicine", "medicine"),
    ("STEM", "science"),
]

# Soft prior on how likely a bucket is to host genuine viewpoint contestation.
# Used only to choose expectations / baselines, never as a bias judgement.
CONTESTED_PRIOR: dict[str, float] = {
    "politics": 0.90,
    "religion": 0.85,
    "society": 0.60,
    "medicine": 0.60,
    "history": 0.55,
    "biography": 0.45,
    "unknown": 0.40,
    "arts": 0.30,
    "sports": 0.20,
    "geography": 0.15,
    "science": 0.10,
}

# Lower rank == more likely contested; used only as a deterministic tiebreak.
_PRIOR_RANK: dict[str, int] = {b: i for i, b in enumerate(CONTESTED_PRIOR)}


def _bucket_for_label(label: str) -> Optional[str]:
    # Regions tags are a locational axis, not an article type (see _BUCKET_RULES).
    if label.startswith("Geography.Regions"):
        return None
    for prefix, bucket in _BUCKET_RULES:
        if label.startswith(prefix):
            return bucket
    return None


def map_topics_to_bucket(scores: dict[str, float]) -> tuple[str, float]:
    """Pick the coarse bucket of the single strongest *substantive* topic label.

    articletopic emits independent multi-label probabilities, and the areas have
    very uneven label granularity (e.g. "History and Society" is split across
    Politics/Society/History/..., while geography is essentially one label).
    Summing mass per bucket therefore biases toward finely-split areas, so we
    instead take the bucket of the highest-probability label, after dropping the
    locational Regions.* tags. Returns (bucket, contested_prior).
    """
    ranked = []
    for label, prob in scores.items():
        bucket = _bucket_for_label(label)
        if bucket is not None:
            ranked.append((float(prob), -_PRIOR_RANK[bucket], bucket))

    if not ranked:
        return "unknown", CONTESTED_PRIOR["unknown"]

    # Highest probability wins; ties go to the more-contested bucket.
    _, _, bucket = max(ranked)
    return bucket, CONTESTED_PRIOR[bucket]


def _extract_probabilities(data: dict) -> Optional[dict[str, float]]:
    """Find the topic->probability dict in an ORES/Lift Wing response.

    Both services nest the result differently across versions, so we search
    recursively for the first dict stored under a "probability" key.
    """
    if isinstance(data, dict):
        if "probability" in data and isinstance(data["probability"], dict):
            return {k: float(v) for k, v in data["probability"].items()}
        for v in data.values():
            found = _extract_probabilities(v)
            if found is not None:
                return found
    elif isinstance(data, list):
        for v in data:
            found = _extract_probabilities(v)
            if found is not None:
                return found
    return None


def classify_article_type(client, revid: int) -> ArticleType:
    """Classify via ORES, then Lift Wing, then fall back to 'unknown'."""
    # 1) ORES (classic)
    try:
        url = config.ORES_ARTICLETOPIC_URL.format(revid=revid)
        data = client.get_external_json(url)
        scores = _extract_probabilities(data)
        if scores:
            bucket, prior = map_topics_to_bucket(scores)
            return ArticleType(bucket=bucket, scores=scores, method="ores", contested_prior=prior)
    except Exception:
        pass

    # 2) Lift Wing (modern)
    try:
        data = client.get_external_json(
            config.LIFTWING_ARTICLETOPIC_URL,
            method="POST",
            json_body={"rev_id": int(revid)},
        )
        scores = _extract_probabilities(data)
        if scores:
            bucket, prior = map_topics_to_bucket(scores)
            return ArticleType(
                bucket=bucket, scores=scores, method="liftwing", contested_prior=prior
            )
    except Exception:
        pass

    # 3) Unavailable -> neutral default (unknown), explicitly flagged.
    return ArticleType(
        bucket="unknown",
        scores={},
        method="unavailable",
        contested_prior=CONTESTED_PRIOR["unknown"],
    )
