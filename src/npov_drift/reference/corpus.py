"""Fetch and bucket the Featured/Good-Article reference corpus.

Featured Articles meet the NPOV criterion, so they serve as the "neutral, mature"
reference. We pull category members, then bucket each by topic type using the
SAME Phase-1 classifier (so target and reference are bucketed identically).
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Optional

from ..ingest.article_type import classify_article_type

FEATURED_CATEGORY = "Category:Featured articles"
GOOD_CATEGORY = "Category:Good articles"


def fetch_category_members(
    client,
    category: str,
    namespace: int = 0,
    max_members: Optional[int] = None,
) -> list[str]:
    """Return article titles in a category (mainspace by default), paginated."""
    members: list[str] = []
    cont: dict = {}
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmnamespace": namespace,
            "cmlimit": 500,
            **cont,
        }
        data = client.get_json(params)
        for m in data.get("query", {}).get("categorymembers", []):
            members.append(m["title"])
            if max_members is not None and len(members) >= max_members:
                return members
        cont = data.get("continue") or {}
        if not cont:
            break
    return members


def latest_revid(client, title: str) -> Optional[int]:
    data = client.get_json(
        {"action": "query", "prop": "revisions", "titles": title, "rvprop": "ids", "rvlimit": 1}
    )
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return None
    revs = pages[0].get("revisions", [])
    return revs[0]["revid"] if revs else None


def classify_title(client, title: str) -> str:
    """Coarse topic bucket for a title (via latest revision + articletopic)."""
    revid = latest_revid(client, title)
    if revid is None:
        return "unknown"
    return classify_article_type(client, revid).bucket


def select_reference_corpus(
    client,
    categories: list[str],
    per_bucket: int = 5,
    max_classify: int = 60,
    seed: int = 0,
    classify_fn: Optional[Callable[[object, str], str]] = None,
) -> list[tuple[str, str]]:
    """Pick up to ``per_bucket`` reference articles per topic bucket.

    Deterministic given ``seed``: candidates are shuffled with a seeded RNG, then
    classified one by one until each bucket is filled or ``max_classify`` titles
    have been examined (bounds the number of classification calls).
    """
    classify = classify_fn or classify_title

    candidates: list[str] = []
    for cat in categories:
        candidates.extend(fetch_category_members(client, cat, max_members=4000))

    seen: set[str] = set()
    uniq: list[str] = []
    for t in candidates:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    random.Random(seed).shuffle(uniq)

    buckets: dict[str, list[str]] = defaultdict(list)
    examined = 0
    for title in uniq:
        if examined >= max_classify:
            break
        bucket = classify(client, title)
        examined += 1
        if len(buckets[bucket]) < per_bucket:
            buckets[bucket].append(title)

    return [(t, b) for b, titles in buckets.items() for t in titles]
