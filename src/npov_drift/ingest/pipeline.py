"""Phase 1 orchestration: ingest a full article history.

Combines: dense metadata fetch -> article-type classification -> content
snapshot sampling -> content fetch -> wikitext parsing -> ArticleHistory.
"""

from __future__ import annotations

from typing import Optional

from .. import config
from ..models import ArticleHistory
from .api_client import WikipediaClient
from .article_type import classify_article_type
from .cache import JsonCache
from .revisions import build_snapshots, fetch_revision_content, fetch_revision_metadata
from .sampling import select_snapshot_revids


def make_client(cache_dir=config.CACHE_DIR) -> WikipediaClient:
    """A ready-to-use cached client pointed at en.wikipedia."""
    return WikipediaClient(cache=JsonCache(cache_dir))


def ingest_article(
    client: WikipediaClient,
    title: str,
    max_revisions: Optional[int] = None,
    max_snapshots: int = 24,
    sampling_strategy: str = "monthly",
    classify_type: bool = True,
) -> ArticleHistory:
    """Ingest one article into an ArticleHistory.

    ``max_revisions`` caps the dense metadata (None = full history).
    ``max_snapshots`` caps how many revisions we fetch *content* for.
    """
    pageid, norm_title, revisions = fetch_revision_metadata(
        client, title, max_revisions=max_revisions
    )

    article_type = None
    if classify_type and revisions:
        article_type = classify_article_type(client, revisions[-1].revid)

    snapshot_revids = select_snapshot_revids(
        revisions, max_snapshots=max_snapshots, strategy=sampling_strategy
    )
    content_map = fetch_revision_content(client, snapshot_revids)
    snapshots = build_snapshots(content_map)

    return ArticleHistory(
        title=norm_title,
        pageid=pageid,
        revisions=revisions,
        snapshots=snapshots,
        article_type=article_type,
    )
