"""Tests for the NLI decision rule only (no model load, no torch needed).

This is where the WMF "neutral description of a negative fact" caution lives:
low entailment for both stance hypotheses -> NEUTRAL.
"""

from npov_drift.stance.base import StanceLabel
from npov_drift.stance.nli import NLIStanceClassifier


def make_clf():
    return NLIStanceClassifier(activation_threshold=0.55, margin=0.10)


def test_clear_favorable_and_critical():
    clf = make_clf()
    assert clf._decide(0.90, 0.10) is StanceLabel.FAVORABLE
    assert clf._decide(0.10, 0.90) is StanceLabel.CRITICAL


def test_neutral_negative_fact_stays_neutral():
    # A neutral statement of a negative fact entails NEITHER stance hypothesis
    # strongly -> NEUTRAL (not CRITICAL).
    clf = make_clf()
    assert clf._decide(0.18, 0.30) is StanceLabel.NEUTRAL


def test_both_high_but_ambiguous_is_neutral():
    clf = make_clf()
    assert clf._decide(0.80, 0.75) is StanceLabel.NEUTRAL  # within margin


def test_threshold_and_margin_respected():
    clf = make_clf()
    assert clf._decide(0.60, 0.20) is StanceLabel.FAVORABLE  # above threshold, clear
    assert clf._decide(0.54, 0.00) is StanceLabel.NEUTRAL  # just below threshold


# --- 3-way scheme (default): explicit neutral hypothesis -------------------
def make_clf3():
    return NLIStanceClassifier(scheme="3way", margin=0.10)


def test_3way_neutral_hypothesis_wins():
    # The neutral-negative-fact case: neutral hypothesis scores highest.
    assert make_clf3()._decide3(0.20, 0.30, 0.50) is StanceLabel.NEUTRAL


def test_3way_stance_must_beat_neutral_by_margin():
    clf = make_clf3()
    assert clf._decide3(0.45, 0.10, 0.40) is StanceLabel.NEUTRAL  # beats neutral by only 0.05
    assert clf._decide3(0.55, 0.10, 0.40) is StanceLabel.FAVORABLE  # beats neutral by 0.15
    assert clf._decide3(0.10, 0.70, 0.20) is StanceLabel.CRITICAL


def test_decide_from_scores_dispatches_by_scheme():
    clf3 = NLIStanceClassifier(scheme="3way", margin=0.10)
    assert (
        clf3.decide_from_scores({"favorable": 0.6, "critical": 0.1, "neutral": 0.3})
        is StanceLabel.FAVORABLE
    )
    clf2 = NLIStanceClassifier(scheme="2hyp", activation_threshold=0.55, margin=0.10)
    assert clf2.decide_from_scores({"favorable": 0.9, "critical": 0.1}) is StanceLabel.FAVORABLE
