from npov_drift.stance.base import StanceLabel
from npov_drift.stance.stub import KeywordStanceStub


def test_stub_labels():
    clf = KeywordStanceStub()
    res = clf.classify(
        [
            "Supporters praised the landmark law.",
            "Critics condemned the cruel policy.",
            "The compound is a white crystalline solid.",
        ],
        topic="the law",
    )
    assert [r.label for r in res] == [
        StanceLabel.FAVORABLE,
        StanceLabel.CRITICAL,
        StanceLabel.NEUTRAL,
    ]


def test_stub_tie_is_neutral():
    clf = KeywordStanceStub()
    (r,) = clf.classify(["It was praised but also criticized."], topic="x")
    assert r.label is StanceLabel.NEUTRAL  # 1 favorable cue vs 1 critical cue
