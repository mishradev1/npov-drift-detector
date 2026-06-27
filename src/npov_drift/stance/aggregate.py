"""Aggregate per-sentence stance into a revision-level distribution, and decide
whether the stance signal is ACTIVE for the article (the self-gating test).

The gate implements the spec's rule: "if stance dispersion is near-zero, mark
the stance signal INACTIVE for that article." Factual topics (chemistry, math)
produce almost entirely neutral sentences, so the favorable/critical mass is
near zero and the signal correctly goes quiet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .base import StanceLabel, StanceResult

# Defaults; these are PROVISIONAL placeholders to be CALIBRATED against the
# Featured/Good Article reference corpus noise floor in Phase 4 (per the spec's
# "learned thresholds"). The real-data Phase 2 demo showed why: a zero-shot
# stance model assigns a false stance to ~15-22% of purely factual sentences, so
# any gate on raw non-neutral fraction must sit above that floor.
DEFAULT_MIN_SIDE_FRAC = 0.10
DEFAULT_MIN_SENTENCES = 20


@dataclass
class StanceDistribution:
    n: int
    favorable: int
    critical: int
    neutral: int

    @property
    def favorable_frac(self) -> float:
        return self.favorable / self.n if self.n else 0.0

    @property
    def critical_frac(self) -> float:
        return self.critical / self.n if self.n else 0.0

    @property
    def neutral_frac(self) -> float:
        return self.neutral / self.n if self.n else 0.0

    @property
    def nonneutral_frac(self) -> float:
        """Share of sentences taking ANY stance toward the topic."""
        return (self.favorable + self.critical) / self.n if self.n else 0.0

    @property
    def balance(self) -> float:
        """Favorable-vs-critical balance in [-1, 1].

        +1 = all favorable, -1 = all critical, 0 = evenly balanced or none.
        This is the directional quantity later phases track over time.
        """
        denom = self.favorable + self.critical
        return (self.favorable - self.critical) / denom if denom else 0.0

    @property
    def entropy(self) -> float:
        """Shannon entropy over the 3 classes, normalised to [0, 1]."""
        if not self.n:
            return 0.0
        ps = [self.favorable_frac, self.critical_frac, self.neutral_frac]
        h = -sum(p * math.log(p) for p in ps if p > 0)
        return h / math.log(3)


def stance_distribution(results: list[StanceResult]) -> StanceDistribution:
    fav = sum(1 for r in results if r.label is StanceLabel.FAVORABLE)
    crit = sum(1 for r in results if r.label is StanceLabel.CRITICAL)
    neu = sum(1 for r in results if r.label is StanceLabel.NEUTRAL)
    return StanceDistribution(n=len(results), favorable=fav, critical=crit, neutral=neu)


def is_stance_active(
    dist: StanceDistribution,
    min_side_frac: float = DEFAULT_MIN_SIDE_FRAC,
    min_sentences: int = DEFAULT_MIN_SENTENCES,
) -> bool:
    """Self-gating: is the stance signal meaningful for this article?

    Viewpoint *balance* requires at least TWO perspectives to be expressed, so
    we require a minimum presence of BOTH favorable and critical stance (plus a
    minimum sample size). This is robust to the stance model's main failure
    mode: on single-perspective factual prose its false positives cluster on one
    side (e.g. utility facts read as 'favorable'), producing a one-sided, not a
    balanced, distribution -- which correctly stays INACTIVE.

    A genuinely contested article that has already drifted fully to one side is
    a known edge case: it is gated from a mature/representative window (Phase 5),
    where a healthy contested article still shows both sides.
    """
    if dist.n < min_sentences:
        return False
    return dist.favorable_frac >= min_side_frac and dist.critical_frac >= min_side_frac
