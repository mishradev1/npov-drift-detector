"""Validation against human {{POV}} maintenance tags + matched controls.

Ground truth comes from the revision that ADDS an NPOV-dispute template (per
Wiki-Reliability's labelling scheme). We detect that tag-add date directly from
an article's history, then ask: did our estimated drift onset PRECEDE the human
tag? Matched, never-tagged controls should not be flagged.
"""

from .evaluate import ValidationRecord, summarize
from .pov_tags import (
    POV_TEMPLATE_RE,
    POV_TEMPLATES,
    fetch_pov_tagged_titles,
    find_first_tagged_index,
    find_pov_tag_onset,
    has_pov_tag,
)

__all__ = [
    "POV_TEMPLATE_RE",
    "POV_TEMPLATES",
    "has_pov_tag",
    "fetch_pov_tagged_titles",
    "find_first_tagged_index",
    "find_pov_tag_onset",
    "ValidationRecord",
    "summarize",
]
