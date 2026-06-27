from npov_drift.models import RevisionContent, Section
from npov_drift.series.stance_series import stance_balance_series
from npov_drift.series.util import body_sentences
from npov_drift.stance.stub import KeywordStanceStub


def _snap(i, text):
    sec = Section("", 0, text, len(text.split()), len(text), False, False)
    return RevisionContent(revid=i, timestamp=f"20{i:02d}-01-01", word_count=0, sections=[sec], plain_text=text)


def _sfn(snap):
    return body_sentences(snap, min_words=1)


def test_stance_series_tracks_balance_drift():
    s1 = _snap(1, "Supporters praised the landmark law. Critics condemned the cruel law.")
    s2 = _snap(2, "Critics condemned the cruel law. Opponents denounced the harmful flawed law.")

    pts = stance_balance_series([s1, s2], KeywordStanceStub(), "the law", _sfn, max_sentences_per_snapshot=None)

    assert (pts[0].dist.favorable, pts[0].dist.critical) == (1, 1)
    assert (pts[1].dist.favorable, pts[1].dist.critical) == (0, 2)
    # Balance moved from neutral (0.0) toward critical (-1.0).
    assert pts[1].dist.balance < pts[0].dist.balance
