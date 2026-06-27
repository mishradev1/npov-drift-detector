"""Phase 2 demo: run stance classification + the self-gating test on REAL
article snapshots (one contested + one factual) and show that the stance signal
activates on the contested article and goes INACTIVE on the factual one.

    python scripts/phase2_demo.py
    python scripts/phase2_demo.py "Climate change" "Pythagorean theorem"

Uses the Phase 1 ingestion cache, so only the stance model runs live (CPU).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from npov_drift.ingest.pipeline import ingest_article, make_client  # noqa: E402
from npov_drift.ingest.sampling import _evenly_spaced_indices  # noqa: E402
from npov_drift.stance.aggregate import is_stance_active, stance_distribution  # noqa: E402
from npov_drift.stance.base import StanceLabel  # noqa: E402
from npov_drift.stance.nli import DEFAULT_MODEL, NLIStanceClassifier  # noqa: E402
from npov_drift.stance.sentences import split_sentences  # noqa: E402
from npov_drift.stance.topic import topic_from_title  # noqa: E402

DEFAULT_ARTICLES = ["Capital punishment", "Sodium chloride"]


def body_sentences(snapshot, min_words: int = 5) -> list[str]:
    text = "\n".join(s.text for s in snapshot.sections if not s.is_boilerplate)
    return [s for s in split_sentences(text) if len(s.split()) >= min_words]


def sample(seq: list[str], k: int) -> list[str]:
    if len(seq) <= k:
        return seq
    return [seq[i] for i in _evenly_spaced_indices(len(seq), k)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("titles", nargs="*", default=DEFAULT_ARTICLES)
    ap.add_argument("--max-sentences", type=int, default=120)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    client = make_client()
    clf = NLIStanceClassifier(model_name=args.model)

    print("Phase 2: stance toward topic + self-gating, on the latest snapshot.")
    print(f"Model: {args.model} (CPU); up to {args.max_sentences} sentences/article.\n")

    for title in args.titles or DEFAULT_ARTICLES:
        hist = ingest_article(client, title)
        if not hist.snapshots:
            print(f"{title}: no snapshots; skipping.")
            continue
        topic = topic_from_title(hist.title)
        latest = hist.snapshots[-1]
        sents = sample(body_sentences(latest), args.max_sentences)

        results = clf.classify(sents, topic)
        dist = stance_distribution(results)
        active = is_stance_active(dist)

        bucket = hist.article_type.bucket if hist.article_type else "?"
        print(f"=== {hist.title} ===  (type={bucket}, topic hypothesis target={topic!r})")
        print(f"  rev {latest.revid} @ {latest.timestamp[:10]}; classified {dist.n} sentences")
        print(
            f"  favorable={dist.favorable} ({dist.favorable_frac:.0%})  "
            f"critical={dist.critical} ({dist.critical_frac:.0%})  "
            f"neutral={dist.neutral} ({dist.neutral_frac:.0%})"
        )
        print(
            f"  non-neutral fraction={dist.nonneutral_frac:.2f}  "
            f"balance(fav-crit)={dist.balance:+.2f}  entropy={dist.entropy:.2f}"
        )
        verdict = "ACTIVE" if active else "INACTIVE (falls back to semantic/structure drift)"
        print(f"  >>> stance signal: {verdict}")

        # A couple of qualitative examples per non-neutral class.
        for lab in (StanceLabel.FAVORABLE, StanceLabel.CRITICAL):
            exs = [r for r in results if r.label is lab][:2]
            for r in exs:
                print(f"      [{lab.value}] fav={r.scores['favorable']:.2f} "
                      f"crit={r.scores['critical']:.2f}  {r.sentence[:90]}")
        print()


if __name__ == "__main__":
    main()
