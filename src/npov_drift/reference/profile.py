"""Per-type normal profile + noise floor from reference-article signal series.

Pure and deterministic (numpy only): given each reference article's due-weight
section-share series, summarize it, then aggregate per topic bucket into:

  * a NORMAL PROFILE -- distribution (mean/std/percentiles) of mature-state
    metrics (alt-view share, lead share, section count, share concentration);
  * a NOISE FLOOR -- how much stable articles of this type naturally move
    between consecutive mature snapshots (total-variation of section shares).

A target article whose movement exceeds the type noise floor (p90) is a drift
candidate; one whose mature state sits far outside the normal profile is a
due-weight anomaly candidate. These calibrate the provisional thresholds from
Phases 2-3.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ..series.section_share import SectionSharePoint


@dataclass
class Stat:
    n: int
    mean: float
    std: float
    p10: float
    p50: float
    p90: float


def stat(values) -> Stat:
    a = np.asarray([float(v) for v in values], dtype=float)
    if a.size == 0:
        return Stat(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return Stat(
        n=int(a.size),
        mean=float(a.mean()),
        std=float(a.std()),
        p10=float(np.percentile(a, 10)),
        p50=float(np.percentile(a, 50)),
        p90=float(np.percentile(a, 90)),
    )


@dataclass
class ArticleSignalSummary:
    title: str
    bucket: str
    n_mature_snapshots: int
    # mature-state metrics (means over the mature window)
    alt_view_share: float
    lead_share: float
    n_sections: float
    share_hhi: float
    # natural movement between consecutive mature snapshots
    tv_step_p50: float  # median total-variation of section shares per step
    tv_step_p90: float
    alt_step_p50: float  # median |delta alt-view share| per step


@dataclass
class TypeProfile:
    bucket: str
    n_articles: int
    metrics: dict[str, Stat]  # normal profile of mature-state metrics
    noise_floor: dict[str, dict[str, float]]  # metric -> {"p50":.., "p90":..}


def _tv(a: dict[str, float], b: dict[str, float]) -> float:
    """Total-variation distance between two section-share distributions in [0,1]."""
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def _hhi(shares: dict[str, float]) -> float:
    return sum(v * v for v in shares.values())


def summarize_article(
    title: str,
    bucket: str,
    points: list[SectionSharePoint],
    maturity_min_words: int = 800,
) -> ArticleSignalSummary | None:
    """Summarize one article's due-weight series over its MATURE window.

    Returns None if the article never reaches the maturity word floor (so stubs
    never enter the reference profile -- the WMF baseline caution).
    """
    mature = [p for p in points if p.total_body_words >= maturity_min_words]
    if not mature:
        return None

    alt = [p.alt_view_share for p in mature]
    lead = [p.shares.get("(lead)", 0.0) for p in mature]
    nsec = [len(p.shares) for p in mature]
    hhi = [_hhi(p.shares) for p in mature]

    tv_steps = [_tv(mature[i - 1].shares, mature[i].shares) for i in range(1, len(mature))]
    alt_steps = [abs(mature[i].alt_view_share - mature[i - 1].alt_view_share) for i in range(1, len(mature))]

    def med(xs):
        return float(np.percentile(xs, 50)) if xs else 0.0

    def p90(xs):
        return float(np.percentile(xs, 90)) if xs else 0.0

    return ArticleSignalSummary(
        title=title,
        bucket=bucket,
        n_mature_snapshots=len(mature),
        alt_view_share=float(np.mean(alt)),
        lead_share=float(np.mean(lead)),
        n_sections=float(np.mean(nsec)),
        share_hhi=float(np.mean(hhi)),
        tv_step_p50=med(tv_steps),
        tv_step_p90=p90(tv_steps),
        alt_step_p50=med(alt_steps),
    )


def build_type_profiles(summaries: list[ArticleSignalSummary]) -> dict[str, TypeProfile]:
    by_bucket: dict[str, list[ArticleSignalSummary]] = defaultdict(list)
    for s in summaries:
        by_bucket[s.bucket].append(s)

    profiles: dict[str, TypeProfile] = {}
    for bucket, arts in by_bucket.items():
        metrics = {
            "alt_view_share": stat([a.alt_view_share for a in arts]),
            "lead_share": stat([a.lead_share for a in arts]),
            "n_sections": stat([a.n_sections for a in arts]),
            "share_hhi": stat([a.share_hhi for a in arts]),
        }
        noise_floor = {
            # Type noise floor: aggregate per-article step magnitudes. p90 is the
            # "stable articles rarely move more than this per step" threshold.
            "share_tv_step": {
                "p50": float(np.percentile([a.tv_step_p50 for a in arts], 50)),
                "p90": float(np.percentile([a.tv_step_p90 for a in arts], 90)),
            },
            "alt_view_share_step": {
                "p90": float(np.percentile([a.alt_step_p50 for a in arts], 90)),
            },
        }
        profiles[bucket] = TypeProfile(bucket=bucket, n_articles=len(arts), metrics=metrics, noise_floor=noise_floor)
    return profiles
