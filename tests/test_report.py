from npov_drift.dashboard.report import build_drift_report
from npov_drift.models import ArticleHistory, RevisionContent, RevisionMeta, Section


def sec(h, w, alt=False, boiler=False):
    return Section(h, 2 if h else 0, "x " * w, w, 2 * w, boiler, alt)


def snap(revid, ts, sections):
    return RevisionContent(
        revid=revid, timestamp=ts, word_count=sum(s.word_count for s in sections), sections=sections, plain_text=""
    )


def rev(revid, ts, size, user):
    return RevisionMeta(
        revid=revid, parentid=None, timestamp=ts, user=user, userid=1, anon=False, minor=False, size=size, comment="edit"
    )


def make_hist():
    mature = [sec("", 400), sec("History", 400), sec("Reception", 200, alt=True)]
    shifted = [sec("", 400), sec("History", 600)]  # Reception removed
    snaps = [
        snap(1, "2001-01-01T00:00:00Z", [sec("", 50)]),  # stub
        snap(2, "2002-01-01T00:00:00Z", mature),
        snap(3, "2003-01-01T00:00:00Z", mature),
        snap(4, "2004-01-01T00:00:00Z", mature),
        snap(5, "2005-01-01T00:00:00Z", shifted),  # onset here
        snap(6, "2006-01-01T00:00:00Z", shifted),
    ]
    revisions = [
        rev(101, "2002-01-01T00:00:00Z", 1000, "A"),
        rev(102, "2004-12-15T00:00:00Z", 1100, "B"),  # in window
        rev(103, "2005-01-10T00:00:00Z", 600, "C"),  # in window: big -500 delta
        rev(104, "2005-02-01T00:00:00Z", 650, "D"),  # in window
        rev(105, "2006-01-01T00:00:00Z", 700, "E"),  # outside window
    ]
    return ArticleHistory(title="Test Article", pageid=42, revisions=revisions, snapshots=snaps, article_type=None)


def test_build_drift_report_structure_and_disclaimer():
    report = build_drift_report(make_hist(), min_words=800)

    assert report.active_signals == {"due_weight": True, "semantic": False, "stance": False}
    assert report.onset.consensus_timestamp.startswith("2005")
    assert len(report.share_series) == 6
    assert report.section_drifts == []  # no encoder supplied

    # Hedged: a human-review candidate, never a bias verdict.
    s = report.hedged_statement
    assert "candidate flagged for human review" in s
    assert "not a determination that the article is biased" in s


def test_key_edits_window_and_ranking():
    report = build_drift_report(make_hist(), min_words=800)
    revids = [e["revid"] for e in report.key_edits]
    assert revids  # non-empty
    assert 101 not in revids and 105 not in revids  # outside +/-120d window
    assert revids[0] == 103  # largest |size delta| (-500) ranked first
    assert "oldid=103" in report.key_edits[0]["diff_url"]
