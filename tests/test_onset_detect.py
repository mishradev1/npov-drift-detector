from npov_drift.models import RevisionContent, Section
from npov_drift.onset.detect import Onset, detect_drift_onset, reconcile


def test_reconcile_consensus_is_largest_cluster_earliest():
    onsets = [
        Onset("a", 3, "2010-01-01", 0.5),
        Onset("b", 4, "2011-01-01", 0.4),
        Onset("c", 10, "2020-01-01", 0.3),
    ]
    ts, agree = reconcile(onsets, tol_index=2)
    assert agree == 2
    assert ts == "2010-01-01"  # earliest in winning cluster


def test_reconcile_empty():
    assert reconcile([]) == (None, 0)


def _snap(i, sections, words):
    return RevisionContent(
        revid=i, timestamp=f"20{i:02d}-01-01", word_count=words, sections=sections, plain_text=""
    )


def _sec(h, w, alt=False):
    return Section(h, 2, "x " * w, w, 2 * w, False, alt)


def test_due_weight_onset_after_baseline():
    # Stub, then a stable mature structure, then a structural shift at 2005.
    snaps = [
        _snap(1, [_sec("", 50)], 50),  # stub (below maturity floor)
        _snap(2, [_sec("", 400), _sec("History", 400), _sec("Reception", 200, alt=True)], 1000),
        _snap(3, [_sec("", 400), _sec("History", 400), _sec("Reception", 200, alt=True)], 1000),
        _snap(4, [_sec("", 400), _sec("History", 400), _sec("Reception", 200, alt=True)], 1000),
        _snap(5, [_sec("", 400), _sec("History", 600), _sec("Reception", 0, alt=True)], 1000),  # Reception removed
        _snap(6, [_sec("", 400), _sec("History", 600)], 1000),
    ]
    report = detect_drift_onset(snaps, min_words=800)
    # Baseline is the first mature snapshot (2002), not the stub.
    assert report.baseline_timestamp.startswith("2002")
    assert "due_weight_departure" in report.signals
    # Some structural-shift onset is detected at/after the change (snap 5/6).
    dw = [o for o in report.onsets if o.signal == "due_weight_departure"]
    assert dw and dw[0].index >= 4
    assert "structure/semantic drift only" in report.note  # no stance classifier supplied
