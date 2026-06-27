"""Phase 5 demo: mature-baseline selection + changepoint onset on REAL articles.

For each article: pick the mature baseline (never the stub), build drift series
relative to it, detect each signal's onset, and reconcile into a consensus onset.

    python scripts/phase5_demo.py
    python scripts/phase5_demo.py --with-stance "Brexit" "Pythagorean theorem"

Uses the Phase 1 cache + MiniLM (semantic). Stance (--with-stance) is the slow
DeBERTa path and is off by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

from npov_drift.embedding import MiniLMEncoder  # noqa: E402
from npov_drift.ingest.pipeline import ingest_article, make_client  # noqa: E402
from npov_drift.ingest.sampling import _evenly_spaced_indices  # noqa: E402
from npov_drift.onset.detect import detect_drift_onset  # noqa: E402
from npov_drift.series.util import body_sentences  # noqa: E402
from npov_drift.stance.topic import topic_from_title  # noqa: E402

DEFAULT_ARTICLES = ["Capital punishment", "Sodium chloride"]


def subset(snaps, k):
    if len(snaps) <= k:
        return snaps
    return [snaps[i] for i in _evenly_spaced_indices(len(snaps), k)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("titles", nargs="*", default=DEFAULT_ARTICLES)
    ap.add_argument("--analyze-snapshots", type=int, default=14)
    ap.add_argument("--min-words", type=int, default=800)
    ap.add_argument("--with-stance", action="store_true")
    args = ap.parse_args()

    client = make_client()
    encoder = MiniLMEncoder()
    classifier = None
    if args.with_stance:
        from npov_drift.stance.nli import NLIStanceClassifier

        classifier = NLIStanceClassifier()

    for title in args.titles or DEFAULT_ARTICLES:
        hist = ingest_article(client, title)
        snaps = subset(hist.snapshots, args.analyze_snapshots)
        report = detect_drift_onset(
            snaps,
            encoder=encoder,
            classifier=classifier,
            topic=topic_from_title(hist.title),
            sentence_fn=lambda s: body_sentences(s, min_words=5),
            min_words=args.min_words,
        )
        bucket = hist.article_type.bucket if hist.article_type else "?"
        print(f"\n=== {hist.title} (type={bucket}, {len(snaps)} analyzed snapshots) ===")
        print(f"  mature baseline: {report.baseline_timestamp}")
        print(f"  signals: {', '.join(report.signals)}")
        print("  per-signal onset (largest level shift in the mature window):")
        for o in sorted(report.onsets, key=lambda o: o.timestamp):
            print(f"      {o.timestamp[:10]}  {o.signal}  (effect={o.effect:.3f})")
        print(f"  >>> consensus onset: {report.consensus_timestamp} "
              f"(agreement: {report.agreement}/{len(report.signals)} signals)")
        print(f"  note: {report.note}")


if __name__ == "__main__":
    main()
