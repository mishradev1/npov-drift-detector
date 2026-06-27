"""Phase 6: validate onset detection against {{POV}}-tagged articles + controls.

For each currently NPOV-tagged article: find when the tag was first added, run
onset detection on its history, and check whether the estimated onset PRECEDES
the human tag. Matched, never-tagged Featured Articles serve as controls.

    python scripts/phase6_validate.py --n-pov 3 --n-control 3

Bounded + cached + MiniLM (stance off). This is a small, self-contained proof of
the validation pipeline, NOT the full Wiki-Reliability evaluation (see README).
"""

from __future__ import annotations

import argparse
import random
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
from npov_drift.reference.corpus import FEATURED_CATEGORY, fetch_category_members  # noqa: E402
from npov_drift.validation.evaluate import ValidationRecord, days_between, summarize  # noqa: E402
from npov_drift.validation.pov_tags import fetch_pov_tagged_titles, find_pov_tag_onset  # noqa: E402


def subset(snaps, k):
    return snaps if len(snaps) <= k else [snaps[i] for i in _evenly_spaced_indices(len(snaps), k)]


def onset_of(snaps, encoder):
    report = detect_drift_onset(snaps, encoder=encoder, min_words=800)
    effect = max((o.effect for o in report.onsets), default=0.0)
    return report.consensus_timestamp, effect, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pov", type=int, default=3)
    ap.add_argument("--n-control", type=int, default=3)
    ap.add_argument("--analyze-snapshots", type=int, default=12)
    ap.add_argument("--effect-threshold", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    client = make_client()
    encoder = MiniLMEncoder()
    rng = random.Random(args.seed)

    pov_members = fetch_pov_tagged_titles(client, max_per_template=200)
    print(f"Currently NPOV-tagged articles found (via template transclusion): {len(pov_members)}")
    rng.shuffle(pov_members)
    fa_members = fetch_category_members(client, FEATURED_CATEGORY, max_members=4000)
    rng.shuffle(fa_members)

    records: list[ValidationRecord] = []

    print("\n-- POV-tagged articles --")
    taken = 0
    for title in pov_members:
        if taken >= args.n_pov:
            break
        tag = find_pov_tag_onset(client, title)
        if tag is None:
            continue  # not actually tagged at latest rev / no history
        hist = ingest_article(client, title, classify_type=False)
        snaps = subset(hist.snapshots, args.analyze_snapshots)
        if len(snaps) < 3:
            continue
        onset_ts, effect, _ = onset_of(snaps, encoder)
        lead = days_between(onset_ts, tag["timestamp"]) if onset_ts else None
        records.append(ValidationRecord(title, "?", True, tag["timestamp"], onset_ts, effect))
        precede = (lead is not None and lead >= 0)
        print(f"  {title}")
        print(f"      tag added: {tag['timestamp'][:10]}   onset: {onset_ts[:10] if onset_ts else 'none'}"
              f"   effect={effect:.3f}   precedes_tag={precede}"
              + (f"  (lead {lead:.0f}d)" if lead is not None else ""))
        taken += 1

    print("\n-- Control Featured Articles (untagged) --")
    taken = 0
    for title in fa_members:
        if taken >= args.n_control:
            break
        hist = ingest_article(client, title, classify_type=False)
        snaps = subset(hist.snapshots, args.analyze_snapshots)
        if len(snaps) < 3:
            continue
        onset_ts, effect, _ = onset_of(snaps, encoder)
        records.append(ValidationRecord(title, "?", False, None, onset_ts, effect))
        print(f"  {title}: onset={onset_ts[:10] if onset_ts else 'none'}  effect={effect:.3f}")
        taken += 1

    print("\n== Validation summary ==")
    m = summarize(records, effect_threshold=args.effect_threshold)
    for key, value in m.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
