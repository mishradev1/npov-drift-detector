"""Phase 2 validation: run the stance classifier on the hand-labeled gold set
and report accuracy, per-class metrics, a confusion matrix, the hard-subset
accuracy, and a threshold-sensitivity sweep.

    python scripts/phase2_validate.py
    python scripts/phase2_validate.py --model facebook/bart-large-mnli

Honesty: the gold set is an AI-authored, single-annotator probe set (see
docs/stance_labeling_rubric.md). These numbers describe behavior on curated
probes, NOT in-the-wild accuracy, and no human inter-annotator agreement is
claimed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from npov_drift.stance.nli import DEFAULT_MODEL, NLIStanceClassifier  # noqa: E402

LABELS = ["favorable", "critical", "neutral"]
GOLD_PATH = Path(__file__).resolve().parents[1] / "data" / "labeled" / "stance_gold.jsonl"


def load_gold(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def confusion(y_true, y_pred):
    idx = {l: i for i, l in enumerate(LABELS)}
    m = [[0] * len(LABELS) for _ in LABELS]
    for t, p in zip(y_true, y_pred):
        m[idx[t]][idx[p]] += 1
    return m


def per_class_prf(y_true, y_pred):
    out = {}
    for l in LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == l and p == l)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != l and p == l)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == l and p != l)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[l] = (prec, rec, f1, tp + fn)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--gold", default=str(GOLD_PATH))
    args = ap.parse_args()

    items = load_gold(Path(args.gold))
    print(f"Loaded {len(items)} labeled sentences from {args.gold}")
    print(f"Model: {args.model}  (CPU)\n")

    clf = NLIStanceClassifier(model_name=args.model)

    # Batch by topic (the hypothesis template embeds the topic).
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_topic[it["topic"]].append(it)

    scores: dict[str, dict] = {}
    try:
        for topic, group in by_topic.items():
            res = clf.classify([g["sentence"] for g in group], topic)
            for g, r in zip(group, res):
                scores[g["id"]] = r.scores
    except Exception as exc:  # pragma: no cover - model load/runtime issues
        print(f"ERROR running model: {exc}")
        print("If this is a DeBERTa tokenizer/load issue, retry with:")
        print("  python scripts/phase2_validate.py --model facebook/bart-large-mnli")
        sys.exit(1)

    def pred_of(it: dict) -> str:
        return clf.decide_from_scores(scores[it["id"]]).value

    def fmt_scores(s: dict) -> str:
        return " ".join(f"{k[:3]}={v:.2f}" for k, v in s.items())

    y_true = [it["gold"] for it in items]
    y_pred = [pred_of(it) for it in items]

    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(items)
    prf = per_class_prf(y_true, y_pred)
    macro_f1 = sum(v[2] for v in prf.values()) / len(prf)

    print(f"Overall accuracy: {acc:.3f}    macro-F1: {macro_f1:.3f}")
    print(f"(scheme={clf.scheme}, activation={clf.activation_threshold}, margin={clf.margin})\n")

    print(f"{'class':<12}{'prec':>7}{'recall':>8}{'f1':>7}{'n':>5}")
    for l in LABELS:
        prec, rec, f1, n = prf[l]
        print(f"{l:<12}{prec:>7.2f}{rec:>8.2f}{f1:>7.2f}{n:>5}")

    print("\nconfusion matrix (rows=gold, cols=pred):")
    print(f"{'':<12}" + "".join(f"{l[:4]:>8}" for l in LABELS))
    m = confusion(y_true, y_pred)
    for i, l in enumerate(LABELS):
        print(f"{l:<12}" + "".join(f"{m[i][j]:>8}" for j in range(len(LABELS))))

    # Hard subset (neutral-negative/positive-fact probes).
    hard = [it for it in items if it.get("hard")]
    if hard:
        hard_acc = sum(1 for it in hard if pred_of(it) == it["gold"]) / len(hard)
        print(f"\nHARD subset (neutral fact vs stance), n={len(hard)}: accuracy {hard_acc:.3f}")

    # Misclassifications (most informative; hard ones flagged).
    print("\nmisclassified:")
    for it in items:
        if pred_of(it) != it["gold"]:
            tag = " [HARD]" if it.get("hard") else ""
            print(
                f"  {it['id']} gold={it['gold']:<9} pred={pred_of(it):<9} "
                f"({fmt_scores(scores[it['id']])}){tag}  {it['sentence'][:66]}"
            )

    # Sensitivity sweep (re-uses cached scores; no re-running of the model).
    print("\nmargin sweep (accuracy):")
    for mg in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        tmp = NLIStanceClassifier(scheme=clf.scheme, activation_threshold=clf.activation_threshold, margin=mg)
        a = sum(1 for it in items if tmp.decide_from_scores(scores[it["id"]]).value == it["gold"]) / len(items)
        hard_a = (
            sum(1 for it in hard if tmp.decide_from_scores(scores[it["id"]]).value == it["gold"]) / len(hard)
            if hard
            else 0.0
        )
        print(f"  margin={mg:<5} overall={a:.3f}  hard={hard_a:.3f}")


if __name__ == "__main__":
    main()
