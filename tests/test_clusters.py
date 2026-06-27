from npov_drift.embedding import FakeEncoder
from npov_drift.models import RevisionContent
from npov_drift.series.clusters import hhi, perspective_balance_series
from npov_drift.stance.sentences import split_sentences


def test_hhi():
    assert abs(hhi([0.5, 0.5]) - 0.5) < 1e-9
    assert abs(hhi([1.0, 0.0]) - 1.0) < 1e-9
    assert abs(hhi([1 / 3, 1 / 3, 1 / 3]) - 1 / 3) < 1e-6


def _snap(i, sentences):
    return RevisionContent(
        revid=i, timestamp=f"20{i:02d}-01-01", word_count=0, sections=[], plain_text="\n".join(sentences)
    )


def _sfn(snap):
    return split_sentences(snap.plain_text)


def test_concentration_increases_when_one_cluster_crowds_out():
    enc = FakeEncoder()
    group_a = ["alpha beta gamma", "alpha beta delta", "alpha gamma delta"]
    group_b = ["red green blue", "red green yellow", "green blue yellow"]
    balanced = _snap(1, group_a + group_b)
    one_sided = _snap(2, group_a * 3 + ["alpha beta gamma"])  # group B crowded out

    points, centers = perspective_balance_series(
        [balanced, one_sided], enc, _sfn, k=2, max_sentences_per_snapshot=None
    )
    assert centers is not None
    assert len(points) == 2
    # Balanced snapshot ~ even split (HHI ~0.5); one-sided ~ concentrated (HHI ->1).
    assert points[1].hhi > points[0].hhi
    assert points[0].hhi < 0.75
