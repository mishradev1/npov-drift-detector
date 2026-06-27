from npov_drift.models import RevisionContent, Section
from npov_drift.series.section_share import section_share_series


def sec(heading, words, alt=False, boiler=False, level=2):
    return Section(
        heading=heading,
        level=level,
        text="word " * words,
        word_count=words,
        char_count=5 * words,
        is_boilerplate=boiler,
        is_alt_view=alt,
    )


def snap(revid, ts, sections):
    return RevisionContent(
        revid=revid,
        timestamp=ts,
        word_count=sum(s.word_count for s in sections),
        sections=sections,
        plain_text="",
    )


def test_alt_view_share_trajectory_and_boilerplate_excluded():
    s1 = snap(1, "2020-01-01", [
        sec("", 50), sec("History", 50), sec("Criticism", 100, alt=True), sec("References", 200, boiler=True)
    ])
    s2 = snap(2, "2021-01-01", [
        sec("", 50), sec("History", 150), sec("Criticism", 20, alt=True), sec("References", 200, boiler=True)
    ])
    pts = section_share_series([s1, s2])

    # References (boilerplate) excluded -> body totals are 200 and 220.
    assert pts[0].total_body_words == 200
    assert pts[1].total_body_words == 220
    assert abs(pts[0].alt_view_share - 0.5) < 1e-9  # 100/200
    assert abs(pts[1].alt_view_share - (20 / 220)) < 1e-9
    assert pts[0].alt_view_share > pts[1].alt_view_share  # criticism shrank
    assert abs(pts[0].shares["Criticism"] - 0.5) < 1e-9
    assert "References" not in pts[0].shares
