"""Featured/Good-Article reference corpus and per-type normal profiles.

The reference corpus gives each topic bucket a *normal profile* (what a neutral,
mature article of this type looks like) and a *noise floor* (how much stable
articles of this type naturally move between revisions). Drift is later judged as
a departure from this type-matched reference, so thresholds are calibrated from
data rather than hand-picked (WMF "choice of baseline").
"""

from .profile import (
    ArticleSignalSummary,
    Stat,
    TypeProfile,
    build_type_profiles,
    summarize_article,
)

__all__ = [
    "ArticleSignalSummary",
    "TypeProfile",
    "Stat",
    "summarize_article",
    "build_type_profiles",
]
