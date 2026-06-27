"""Phase 1 demo: ingest real Wikipedia articles and print a structured report.

Run (after `pip install -e .`):
    python scripts/phase1_demo.py
    python scripts/phase1_demo.py "Climate change" "Pythagorean theorem"
    python scripts/phase1_demo.py --max-snapshots 16 "Abortion"

Phase 1 is INGESTION ONLY. This prints what we ingested (revision metadata,
article type, sampled content structure). It makes NO drift or bias claims.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Allow running straight from the repo without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from npov_drift import config  # noqa: E402
from npov_drift.ingest.pipeline import ingest_article, make_client  # noqa: E402
from npov_drift.models import ArticleHistory, RevisionContent  # noqa: E402

# One clearly-contested + one clearly-factual article by default.
DEFAULT_ARTICLES = ["Capital punishment", "Sodium chloride"]


def _section_shares(snap: RevisionContent) -> list[tuple[str, int, float]]:
    """(heading, word_count, share-of-body) for non-boilerplate sections, desc."""
    body = [s for s in snap.sections if not s.is_boilerplate]
    total = sum(s.word_count for s in body) or 1
    rows = [
        (s.heading or "(lead)", s.word_count, s.word_count / total)
        for s in body
        if s.word_count > 0
    ]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def _fmt_date(ts: str | None) -> str:
    return (ts or "?")[:10]


def report(hist: ArticleHistory) -> None:
    title = hist.title
    bar = "=" * (len(title) + 8)
    print(f"\n{bar}\n=== {title} ===\n{bar}")
    print(f"pageid: {hist.pageid}")

    start, end = hist.date_span()
    print(
        f"revisions: {len(hist.revisions):,}  "
        f"({_fmt_date(start)} .. {_fmt_date(end)})  "
        f"editors: {hist.num_editors():,}"
    )

    at = hist.article_type
    if at is not None:
        print(
            f"article type: {at.bucket}  "
            f"(method={at.method}, contested_prior={at.contested_prior:.2f})"
        )
        if at.scores:
            top = sorted(at.scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
            print("   top topics: " + "; ".join(f"{k} {v:.2f}" for k, v in top))
    else:
        print("article type: (not classified)")

    print(f"content snapshots fetched: {len(hist.snapshots)}")
    if not hist.snapshots:
        return

    latest = hist.snapshots[-1]
    body_secs = [s for s in latest.sections if not s.is_boilerplate]
    alt = [s for s in latest.sections if s.is_alt_view]
    print(
        f"latest snapshot rev {latest.revid} ({_fmt_date(latest.timestamp)}): "
        f"{len(latest.sections)} sections, body words {latest.body_word_count():,}"
    )

    shares = _section_shares(latest)
    print(f"  viewpoint-relevant structure: {len(alt)} alt-view section(s) "
          f"/ {len(body_secs)} content section(s)")
    if alt:
        total_body = latest.body_word_count() or 1
        for s in alt:
            print(f"    alt-view: {s.heading!r}  {100 * s.word_count / total_body:4.1f}% of body")
    print("  top sections by body-word share:")
    for heading, wc, share in shares[:8]:
        print(f"     {100 * share:5.1f}%  {heading}  ({wc:,} words)")


def _summary_dict(hist: ArticleHistory) -> dict:
    """A compact, JSON-friendly summary (omits full plain_text blobs)."""
    snaps = []
    for s in hist.snapshots:
        snaps.append(
            {
                "revid": s.revid,
                "timestamp": s.timestamp,
                "word_count": s.word_count,
                "body_word_count": s.body_word_count(),
                "sections": [
                    {
                        "heading": sec.heading,
                        "level": sec.level,
                        "word_count": sec.word_count,
                        "is_boilerplate": sec.is_boilerplate,
                        "is_alt_view": sec.is_alt_view,
                    }
                    for sec in s.sections
                ],
            }
        )
    start, end = hist.date_span()
    return {
        "title": hist.title,
        "pageid": hist.pageid,
        "num_revisions": len(hist.revisions),
        "num_editors": hist.num_editors(),
        "date_span": [start, end],
        "article_type": asdict(hist.article_type) if hist.article_type else None,
        "num_snapshots": len(hist.snapshots),
        "snapshots": snaps,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1 ingestion demo.")
    ap.add_argument("titles", nargs="*", default=DEFAULT_ARTICLES,
                    help="Article titles (default: one contested + one factual).")
    ap.add_argument("--max-snapshots", type=int, default=24)
    ap.add_argument("--max-revisions", type=int, default=None,
                    help="Cap dense metadata (default: full history).")
    ap.add_argument("--no-type", action="store_true", help="Skip article-type classification.")
    ap.add_argument("--out-dir", default=str(config.OUT_DIR))
    args = ap.parse_args()

    titles = args.titles or DEFAULT_ARTICLES
    client = make_client()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Phase 1 = INGESTION ONLY. No drift or bias claims are made below.")
    print(f"Cache: {config.CACHE_DIR}")

    for title in titles:
        hist = ingest_article(
            client,
            title,
            max_revisions=args.max_revisions,
            max_snapshots=args.max_snapshots,
            classify_type=not args.no_type,
        )
        report(hist)
        out_path = out_dir / f"phase1_{hist.title.replace('/', '_')}.json"
        out_path.write_text(json.dumps(_summary_dict(hist), indent=2), encoding="utf-8")
        print(f"  -> wrote summary: {out_path}")


if __name__ == "__main__":
    main()
