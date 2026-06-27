import math

from npov_drift.stance.aggregate import (
    is_stance_active,
    stance_distribution,
)
from npov_drift.stance.base import StanceLabel, StanceResult


def make(labels: list[StanceLabel]) -> list[StanceResult]:
    return [StanceResult(sentence="x", label=l) for l in labels]


def dist_from_counts(fav: int, crit: int, neu: int):
    labels = (
        [StanceLabel.FAVORABLE] * fav
        + [StanceLabel.CRITICAL] * crit
        + [StanceLabel.NEUTRAL] * neu
    )
    return stance_distribution(make(labels))


def test_counts_and_fractions():
    d = dist_from_counts(3, 1, 6)
    assert (d.n, d.favorable, d.critical, d.neutral) == (10, 3, 1, 6)
    assert d.nonneutral_frac == 0.4
    assert d.balance == (3 - 1) / (3 + 1)  # 0.5


def test_balance_and_entropy_edges():
    allneu = dist_from_counts(0, 0, 5)
    assert allneu.balance == 0.0
    assert allneu.entropy == 0.0
    even = dist_from_counts(1, 1, 0)
    assert even.balance == 0.0
    assert math.isclose(even.entropy, math.log(2) / math.log(3), rel_tol=1e-9)


def test_gate_inactive_when_all_neutral():
    d = dist_from_counts(0, 0, 100)  # factual article
    assert not is_stance_active(d)


def test_gate_active_when_both_sides_present():
    d = dist_from_counts(20, 15, 65)  # contested article: both sides present
    assert is_stance_active(d)


def test_gate_inactive_when_one_sided():
    # Like a factual article whose classifier false-positives cluster on one
    # side: lots of "favorable" noise but almost no "critical" -> not balanced.
    d = dist_from_counts(20, 2, 78)  # crit_frac = 0.02 < 0.10
    assert not is_stance_active(d)


def test_gate_inactive_on_small_sample():
    d = dist_from_counts(3, 2, 0)  # only 5 sentences -> too few to trust
    assert not is_stance_active(d)
