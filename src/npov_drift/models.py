"""Core data models for ingested article history.

These are intentionally plain dataclasses (JSON-serialisable via ``asdict``) so
that ingested histories can be cached, inspected, and fed to later phases
without coupling to any analysis code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class RevisionMeta:
    """Dense, cheap-to-fetch metadata for a single revision.

    We fetch this for *every* revision (it is small). ``user``/``comment``/
    ``sha1`` may be ``None`` when the value was suppressed/hidden by an admin.
    """

    revid: int
    parentid: Optional[int]
    timestamp: str  # ISO-8601, e.g. "2021-03-04T12:34:56Z"
    user: Optional[str]
    userid: Optional[int]
    anon: bool
    minor: bool
    size: int  # bytes of wikitext
    comment: Optional[str]
    tags: list[str] = field(default_factory=list)
    sha1: Optional[str] = None


@dataclass
class Section:
    """One non-overlapping section of a revision's wikitext."""

    heading: str  # "" for the lead section
    level: int  # 0 for the lead, 2 for "==", 3 for "===", ...
    text: str  # plain text (wiki markup stripped)
    word_count: int
    char_count: int
    is_boilerplate: bool  # References / External links / etc.
    is_alt_view: bool  # Criticism / Controversy / Reception / ... (generic)


@dataclass
class RevisionContent:
    """A sampled content snapshot for one revision."""

    revid: int
    timestamp: str
    word_count: int
    sections: list[Section]
    plain_text: str = ""

    def body_word_count(self) -> int:
        """Words excluding boilerplate sections (the analytic 'body')."""
        return sum(s.word_count for s in self.sections if not s.is_boilerplate)


@dataclass
class ArticleType:
    """Coarse topic classification used for baseline bucketing.

    ``contested_prior`` is a SOFT PRIOR (0..1) used only to pick a comparison
    bucket and to set expectations; it is explicitly NOT a claim that the
    article is biased. Whether the viewpoint signal is actually active is
    decided later from data (stance dispersion), not from this number.
    """

    bucket: str  # politics | history | science | medicine | biography | geography | arts | sports | religion | society | unknown
    scores: dict[str, float]  # raw topic -> probability from the scorer
    method: str  # "ores" | "liftwing" | "unavailable"
    contested_prior: float


@dataclass
class ArticleHistory:
    """Everything ingested for one article in Phase 1."""

    title: str
    pageid: Optional[int]
    revisions: list[RevisionMeta]  # dense: all revisions, chronological
    snapshots: list[RevisionContent]  # sampled content, chronological
    article_type: Optional[ArticleType] = None

    # -- convenience summaries (used by the demo / later phases) -----------
    def date_span(self) -> tuple[Optional[str], Optional[str]]:
        if not self.revisions:
            return (None, None)
        return (self.revisions[0].timestamp, self.revisions[-1].timestamp)

    def num_editors(self) -> int:
        return len({r.user for r in self.revisions if r.user is not None})

    def to_dict(self) -> dict:
        return asdict(self)
