from npov_drift.reference.profile import build_type_profiles, summarize_article
from npov_drift.series.section_share import SectionSharePoint


def pt(words, shares, alt):
    return SectionSharePoint(revid=0, timestamp="t", total_body_words=words, alt_view_share=alt, shares=shares)


def test_summarize_article_hand_computed():
    pts = [
        pt(1000, {"(lead)": 0.4, "A": 0.4, "Crit": 0.2}, alt=0.2),
        pt(1000, {"(lead)": 0.4, "A": 0.3, "Crit": 0.3}, alt=0.3),
    ]
    s = summarize_article("X", "politics", pts, maturity_min_words=800)
    assert s is not None
    assert s.n_mature_snapshots == 2
    assert abs(s.alt_view_share - 0.25) < 1e-9
    assert abs(s.lead_share - 0.4) < 1e-9
    assert abs(s.n_sections - 3) < 1e-9
    assert abs(s.share_hhi - 0.35) < 1e-9  # (0.36 + 0.34) / 2
    assert abs(s.tv_step_p50 - 0.1) < 1e-9  # 0.5*(0+0.1+0.1)
    assert abs(s.alt_step_p50 - 0.1) < 1e-9


def test_stub_only_article_excluded():
    pts = [pt(100, {"(lead)": 1.0}, alt=0.0), pt(200, {"(lead)": 1.0}, alt=0.0)]
    assert summarize_article("Stub", "science", pts, maturity_min_words=800) is None


def test_build_type_profiles_aggregates():
    a1 = summarize_article("A", "science", [pt(1000, {"(lead)": 0.5, "B": 0.5}, 0.1)], 800)
    a2 = summarize_article("B", "science", [pt(1000, {"(lead)": 0.5, "B": 0.5}, 0.3)], 800)
    profiles = build_type_profiles([a1, a2])
    assert set(profiles) == {"science"}
    p = profiles["science"]
    assert p.n_articles == 2
    assert abs(p.metrics["alt_view_share"].mean - 0.2) < 1e-9
    assert abs(p.metrics["alt_view_share"].p50 - 0.2) < 1e-9
