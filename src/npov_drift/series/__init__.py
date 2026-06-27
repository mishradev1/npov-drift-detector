"""Time-series drift signals computed over an article's sampled snapshots.

Four signals (see the build spec):
  * A. stance balance over time           -> stance_series
  * B. perspective-cluster balance         -> clusters
  * C. due-weight / section-share drift    -> section_share   (pure, all articles)
  * (secondary) directional section drift  -> section_drift
"""

from .clusters import ClusterPoint, hhi, perspective_balance_series
from .section_drift import SectionDrift, section_drift
from .section_share import SectionSharePoint, section_share_series
from .stance_series import StanceSeriesPoint, stance_balance_series
from .util import body_sentences, body_text

__all__ = [
    "section_share_series",
    "SectionSharePoint",
    "section_drift",
    "SectionDrift",
    "perspective_balance_series",
    "ClusterPoint",
    "hhi",
    "stance_balance_series",
    "StanceSeriesPoint",
    "body_text",
    "body_sentences",
]
