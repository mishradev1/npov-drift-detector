"""Phase 4: build the Featured-Article reference profile (due-weight signal).

Fetches Featured Articles, buckets them by topic type, ingests a sampled history
for each, and computes a per-type NORMAL PROFILE + NOISE FLOOR. Persists the
profile to data/out/reference_profile.json.

    python scripts/phase4_build_reference.py --per-bucket 3 --max-classify 24

Everything is cached, so a interrupted/resumed build re-uses prior work. Only the
cheap due-weight (section-share) signal is calibrated here; stance/embedding
noise floors need a longer batch run (documented in the README).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

from npov_drift import config  # noqa: E402
from npov_drift.ingest.pipeline import make_client  # noqa: E402
from npov_drift.reference.build import build_due_weight_reference, profile_to_dict  # noqa: E402
from npov_drift.reference.corpus import FEATURED_CATEGORY, GOOD_CATEGORY, select_reference_corpus  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bucket", type=int, default=3)
    ap.add_argument("--max-classify", type=int, default=24)
    ap.add_argument("--max-snapshots", type=int, default=8)
    ap.add_argument("--maturity-min-words", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-good", action="store_true", help="also draw from Category:Good articles")
    args = ap.parse_args()

    client = make_client()
    categories = [FEATURED_CATEGORY] + ([GOOD_CATEGORY] if args.include_good else [])

    print(f"Selecting reference corpus (per_bucket={args.per_bucket}, max_classify={args.max_classify})...")
    corpus = select_reference_corpus(
        client, categories, per_bucket=args.per_bucket, max_classify=args.max_classify, seed=args.seed
    )
    print(f"Selected {len(corpus)} reference articles:")
    for title, bucket in corpus:
        print(f"  [{bucket:>10}] {title}")

    def progress(title, bucket, summary):
        n = summary.n_mature_snapshots if summary else 0
        print(f"  ingested [{bucket:>10}] {title}  (mature snapshots: {n})")

    print("\nIngesting + summarizing (cached)...")
    summaries, profiles = build_due_weight_reference(
        client, corpus, max_snapshots=args.max_snapshots,
        maturity_min_words=args.maturity_min_words, on_article=progress,
    )

    print(f"\n== Per-type NORMAL PROFILE + NOISE FLOOR (n_articles per bucket) ==")
    for bucket, p in sorted(profiles.items()):
        avs = p.metrics["alt_view_share"]
        lead = p.metrics["lead_share"]
        nsec = p.metrics["n_sections"]
        nf = p.noise_floor["share_tv_step"]
        print(f"\n  {bucket}  (n={p.n_articles})")
        print(f"    alt-view share:  mean={avs.mean:.1%}  p10-p90=[{avs.p10:.1%},{avs.p90:.1%}]")
        print(f"    lead share:      mean={lead.mean:.1%}")
        print(f"    #sections:       mean={nsec.mean:.1f}")
        print(f"    NOISE FLOOR (section-share total-variation per step): "
              f"p50={nf['p50']:.3f}  p90={nf['p90']:.3f}")

    out = config.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "reference_profile.json"
    out_path.write_text(
        json.dumps(
            {
                "corpus": [{"title": t, "bucket": b} for t, b in corpus],
                "n_summarized": len(summaries),
                "profiles": profile_to_dict(profiles),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n-> wrote {out_path}")


if __name__ == "__main__":
    main()
