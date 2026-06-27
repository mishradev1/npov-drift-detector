"""Phase 3 demo: drift time-series on REAL articles.

Shows, over each article's sampled history:
  * C  due-weight section-share drift   (pure; printed first, instant)
  * secondary  directional section embedding drift   (MiniLM, CPU)
  * B  perspective-cluster concentration (HHI)        (MiniLM, CPU)
  * A  stance-balance series             (opt-in via --with-stance; DeBERTa, slow)

    python scripts/phase3_demo.py
    python scripts/phase3_demo.py --with-stance "Climate change" "Pythagorean theorem"

Uses the Phase 1 ingestion cache. MiniLM downloads ~80 MB on first use.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # utf-8 + stream output
except Exception:
    pass

from npov_drift.embedding import MiniLMEncoder  # noqa: E402
from npov_drift.ingest.pipeline import ingest_article, make_client  # noqa: E402
from npov_drift.ingest.sampling import _evenly_spaced_indices  # noqa: E402
from npov_drift.series.clusters import perspective_balance_series  # noqa: E402
from npov_drift.series.section_drift import section_drift  # noqa: E402
from npov_drift.series.section_share import section_share_series  # noqa: E402
from npov_drift.series.stance_series import stance_balance_series  # noqa: E402
from npov_drift.series.util import body_sentences  # noqa: E402
from npov_drift.stance.topic import topic_from_title  # noqa: E402

DEFAULT_ARTICLES = ["Capital punishment", "Sodium chloride"]


def subset(snaps, k):
    if len(snaps) <= k:
        return snaps
    return [snaps[i] for i in _evenly_spaced_indices(len(snaps), k)]


def pick_baseline(points, min_words=500):
    """Crude mature-baseline proxy (Phase 5 does this properly): earliest
    snapshot above a word floor, so we never baseline on a stub."""
    for p in points:
        if p.total_body_words >= min_words:
            return p
    return points[0]


def due_weight_report(title, snaps):
    ss = section_share_series(snaps)
    base = pick_baseline(ss)
    last = ss[-1]
    print(f"  [C due-weight] body words {base.total_body_words:,} ({base.timestamp[:10]}) "
          f"-> {last.total_body_words:,} ({last.timestamp[:10]})")
    print(f"      alt-view (by generic heading name) share: "
          f"{base.alt_view_share:.1%} -> {last.alt_view_share:.1%}")
    common = set(base.shares) & set(last.shares)
    movers = sorted(((last.shares[h] - base.shares[h], h) for h in common), key=lambda x: x[0])
    if movers:
        print("      biggest share losers (mature baseline -> latest):")
        for delta, h in movers[:3]:
            print(f"        {base.shares[h]:5.1%} -> {last.shares[h]:5.1%}  ({delta:+.1%})  {h}")
        print("      biggest share gainers:")
        for delta, h in reversed(movers[-3:]):
            print(f"        {base.shares[h]:5.1%} -> {last.shares[h]:5.1%}  ({delta:+.1%})  {h}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("titles", nargs="*", default=DEFAULT_ARTICLES)
    ap.add_argument("--analyze-snapshots", type=int, default=10,
                    help="cap snapshots used for the MiniLM signals (compute bound)")
    ap.add_argument("--k", type=int, default=5, help="perspective clusters")
    ap.add_argument("--with-stance", action="store_true", help="also run the (slow) stance series")
    args = ap.parse_args()

    client = make_client()
    histories = [(t, ingest_article(client, t)) for t in (args.titles or DEFAULT_ARTICLES)]

    # --- Signal C first: pure, instant, full history ----------------------
    print("== Signal C: due-weight section-share drift (pure, full history) ==")
    for title, hist in histories:
        print(f"\n--- {hist.title} (type={hist.article_type.bucket if hist.article_type else '?'}, "
              f"{len(hist.snapshots)} snapshots) ---")
        if hist.snapshots:
            due_weight_report(title, hist.snapshots)

    # --- MiniLM signals on a bounded snapshot subset -----------------------
    print(f"\n== Signals (secondary) directional drift + (B) cluster concentration "
          f"[MiniLM, <= {args.analyze_snapshots} snapshots] ==")
    encoder = MiniLMEncoder()
    for title, hist in histories:
        snaps = subset(hist.snapshots, args.analyze_snapshots)
        if len(snaps) < 2:
            continue
        print(f"\n--- {hist.title} ---")
        drifts = section_drift(snaps, encoder, min_words=15)
        print("  [secondary] most directionally-drifting sections (straightness x net):")
        for d in drifts[:5]:
            print(f"      straightness={d.straightness:.2f}  net={d.net_displacement:.3f}  "
                  f"occ={d.occurrences}  {d.heading}")

        pts, _ = perspective_balance_series(
            snaps, encoder, lambda s: body_sentences(s, min_words=5), k=args.k,
            max_sentences_per_snapshot=60,
        )
        if pts:
            # Require enough sentences for a meaningful distribution, so stub-era
            # snapshots (n~1 -> trivially HHI 1.0) do not pollute the trajectory.
            live = [p for p in pts if p.n >= 20]
            if live:
                print(f"  [B] perspective-cluster concentration (HHI, k={args.k}; "
                      f"1/k={1/args.k:.2f}=even .. 1=one cluster):")
                print(f"      {live[0].hhi:.2f} ({live[0].timestamp[:10]}) -> "
                      f"{live[-1].hhi:.2f} ({live[-1].timestamp[:10]})")

    # --- Signal A: stance balance over time (opt-in, slow) -----------------
    if args.with_stance:
        from npov_drift.stance.nli import NLIStanceClassifier

        print("\n== Signal A: stance-balance series [DeBERTa, slow] ==")
        clf = NLIStanceClassifier()
        for title, hist in histories:
            snaps = subset(hist.snapshots, args.analyze_snapshots)
            pts = stance_balance_series(
                snaps, clf, topic_from_title(hist.title),
                lambda s: body_sentences(s, min_words=5), max_sentences_per_snapshot=40,
            )
            print(f"\n--- {hist.title} ---")
            for p in pts:
                flag = "ACTIVE" if p.active else "inactive"
                print(f"      {p.timestamp[:10]}  balance={p.dist.balance:+.2f}  "
                      f"fav={p.dist.favorable_frac:.0%} crit={p.dist.critical_frac:.0%} "
                      f"neu={p.dist.neutral_frac:.0%}  [{flag}]")


if __name__ == "__main__":
    main()
