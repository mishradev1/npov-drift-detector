"""Fetch revision metadata (dense) and revision content (sampled)."""

from __future__ import annotations

from typing import Iterable, Optional

from .. import config
from ..models import RevisionContent, RevisionMeta
from .parsing import parse_sections, strip_to_text


class ArticleNotFound(RuntimeError):
    pass


def _parse_rev_meta(r: dict) -> RevisionMeta:
    """Build a RevisionMeta from a formatversion=2 revision dict.

    Suppressed/hidden fields are surfaced as ``None`` rather than guessed.
    """
    hidden_user = r.get("userhidden", False)
    return RevisionMeta(
        revid=r["revid"],
        parentid=r.get("parentid"),
        timestamp=r["timestamp"],
        user=None if hidden_user else r.get("user"),
        userid=None if hidden_user else r.get("userid"),
        anon=bool(r.get("anon", False)),
        minor=bool(r.get("minor", False)),
        size=r.get("size", 0),
        comment=None if r.get("commenthidden", False) else r.get("comment"),
        tags=list(r.get("tags", []) or []),
        sha1=None if r.get("sha1hidden", False) else r.get("sha1"),
    )


def fetch_revision_metadata(
    client,
    title: str,
    max_revisions: Optional[int] = None,
    rvlimit: int = config.REVISIONS_PER_REQUEST,
) -> tuple[Optional[int], str, list[RevisionMeta]]:
    """Return (pageid, normalized_title, revisions) oldest-first.

    Paginates with ``rvcontinue`` until the whole history (or ``max_revisions``)
    is collected.
    """
    revisions: list[RevisionMeta] = []
    pageid: Optional[int] = None
    norm_title = title
    cont: dict = {}

    while True:
        params = {
            "action": "query",
            "prop": "revisions",
            "titles": title,
            "rvprop": "ids|timestamp|user|userid|comment|size|tags|flags|sha1",
            "rvlimit": rvlimit,
            "rvdir": "newer",  # oldest -> newest (chronological)
            **cont,
        }
        data = client.get_json(params)
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            break
        page = pages[0]
        if page.get("missing"):
            raise ArticleNotFound(title)
        pageid = page.get("pageid", pageid)
        norm_title = page.get("title", norm_title)

        for r in page.get("revisions", []):
            revisions.append(_parse_rev_meta(r))
            if max_revisions is not None and len(revisions) >= max_revisions:
                return pageid, norm_title, revisions

        cont = data.get("continue") or {}
        if not cont:
            break

    return pageid, norm_title, revisions


def _chunks(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def fetch_revision_content(
    client,
    revids: Iterable[int],
    batch_size: int = config.CONTENT_BATCH_SIZE,
) -> dict[int, dict]:
    """Fetch wikitext for the given revids. Returns {revid: {timestamp, wikitext}}.

    Revisions the API cannot return (deleted/suppressed) are simply omitted.
    """
    revids = list(revids)
    out: dict[int, dict] = {}
    for batch in _chunks(revids, batch_size):
        params = {
            "action": "query",
            "prop": "revisions",
            "revids": "|".join(str(r) for r in batch),
            "rvprop": "ids|timestamp|content",
            "rvslots": "main",
        }
        data = client.get_json(params)
        for page in data.get("query", {}).get("pages", []):
            for r in page.get("revisions", []):
                slot = r.get("slots", {}).get("main", {})
                if slot.get("texthidden"):
                    continue
                out[r["revid"]] = {
                    "timestamp": r["timestamp"],
                    "wikitext": slot.get("content", "") or "",
                }
    return out


def build_snapshots(content_map: dict[int, dict]) -> list[RevisionContent]:
    """Parse fetched content into chronological RevisionContent snapshots."""
    snaps: list[RevisionContent] = []
    for revid, d in content_map.items():
        plain = strip_to_text(d["wikitext"])
        snaps.append(
            RevisionContent(
                revid=revid,
                timestamp=d["timestamp"],
                word_count=len(plain.split()),
                sections=parse_sections(d["wikitext"]),
                plain_text=plain,
            )
        )
    snaps.sort(key=lambda s: (s.timestamp, s.revid))
    return snaps
