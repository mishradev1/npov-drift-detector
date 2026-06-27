"""Orchestrate building the reference profile from real articles (due-weight)."""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from ..ingest.pipeline import ingest_article
from ..series.section_share import section_share_series
from .profile import ArticleSignalSummary, TypeProfile, build_type_profiles, summarize_article


def build_due_weight_reference(
    client,
    corpus: list[tuple[str, str]],
    max_snapshots: int = 10,
    maturity_min_words: int = 800,
    on_article=None,
) -> tuple[list[ArticleSignalSummary], dict[str, TypeProfile]]:
    """Ingest each reference (title, bucket) and compute the due-weight profile.

    ``classify_type=False`` because the bucket is already known, saving an
    articletopic call per article.
    """
    summaries: list[ArticleSignalSummary] = []
    for title, bucket in corpus:
        hist = ingest_article(client, title, max_snapshots=max_snapshots, classify_type=False)
        points = section_share_series(hist.snapshots)
        summary = summarize_article(title, bucket, points, maturity_min_words=maturity_min_words)
        if on_article is not None:
            on_article(title, bucket, summary)
        if summary is not None:
            summaries.append(summary)

    return summaries, build_type_profiles(summaries)


def profile_to_dict(profiles: dict[str, TypeProfile]) -> dict:
    return {bucket: asdict(p) for bucket, p in profiles.items()}
