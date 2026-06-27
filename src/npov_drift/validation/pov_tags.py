"""Detect NPOV-dispute maintenance templates and when they were first added.

The tag-add revision is the positive label (Wiki-Reliability). For a currently
tagged article the tag is present at the latest revision and (usually) was added
once and kept, so a binary search over the chronological history finds the first
tagged revision in O(log N) content fetches.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from ..ingest.revisions import fetch_revision_content, fetch_revision_metadata

# Article/section-level NPOV-dispute templates. ``N?POV\b`` covers {{POV}},
# {{NPOV}}, {{POV-section}}, {{POV-check}}, {{POV lead}}, etc.; we add the other
# common neutrality-dispute banners.
POV_TEMPLATE_RE = re.compile(r"\{\{\s*(?:N?POV|Neutrality|Unbalanced)\b", re.IGNORECASE)


def has_pov_tag(wikitext: str) -> bool:
    return bool(POV_TEMPLATE_RE.search(wikitext or ""))


# Currently-tagged articles are found by template transclusion (the NPOV-dispute
# tracking *categories* hold dated subcats, not articles, so list=embeddedin on
# the templates themselves is the reliable source).
POV_TEMPLATES = [
    "Template:POV",
    "Template:NPOV",
    "Template:POV section",
    "Template:Unbalanced",
    "Template:Neutrality",
]


def fetch_pov_tagged_titles(client, templates=None, max_per_template: int = 200) -> list[str]:
    """Mainspace titles currently transcluding an NPOV-dispute template (deduped)."""
    templates = templates or POV_TEMPLATES
    titles: list[str] = []
    for tmpl in templates:
        cont: dict = {}
        got = 0
        while True:
            params = {
                "action": "query",
                "list": "embeddedin",
                "eititle": tmpl,
                "einamespace": 0,
                "eilimit": 500,
                **cont,
            }
            data = client.get_json(params)
            for m in data.get("query", {}).get("embeddedin", []):
                titles.append(m["title"])
                got += 1
                if got >= max_per_template:
                    break
            cont = data.get("continue") or {}
            if not cont or got >= max_per_template:
                break
    seen: set[str] = set()
    out: list[str] = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def find_first_tagged_index(n: int, is_tagged: Callable[[int], bool]) -> Optional[int]:
    """Smallest i in [0, n) with ``is_tagged(i)`` true, assuming once-true-stays-true.

    Returns None if the article is not tagged at the latest revision (index n-1),
    so we only assert a tag-add date for articles that are actually tagged.
    """
    if n <= 0 or not is_tagged(n - 1):
        return None
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if is_tagged(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def find_pov_tag_onset(client, title: str) -> Optional[dict]:
    """Find the first revision bearing an NPOV-dispute tag.

    Returns {"revid", "timestamp", "index", "n_revisions"} or None if the article
    is not currently tagged / has no revisions.
    """
    _pageid, _norm, revisions = fetch_revision_metadata(client, title)
    if not revisions:
        return None
    revids = [r.revid for r in revisions]
    timestamps = [r.timestamp for r in revisions]

    cache: dict[int, bool] = {}

    def is_tagged(i: int) -> bool:
        if i not in cache:
            content = fetch_revision_content(client, [revids[i]])
            wikitext = content.get(revids[i], {}).get("wikitext", "")
            cache[i] = has_pov_tag(wikitext)
        return cache[i]

    first = find_first_tagged_index(len(revids), is_tagged)
    if first is None:
        return None
    return {
        "revid": revids[first],
        "timestamp": timestamps[first],
        "index": first,
        "n_revisions": len(revids),
    }
