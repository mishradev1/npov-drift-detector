from npov_drift.models import RevisionContent, Section
from npov_drift.onset.baseline import select_mature_baseline


def snap(i, words):
    sec = Section("", 0, "x " * words, words, 2 * words, False, False)
    return RevisionContent(revid=i, timestamp=f"20{i:02d}-01-01", word_count=words, sections=[sec], plain_text="")


def test_skips_stub_picks_first_mature():
    snaps = [snap(1, 50), snap(2, 200), snap(3, 900), snap(4, 1500)]
    b = select_mature_baseline(snaps, min_words=800)
    assert b is not None
    assert b.index == 2
    assert b.timestamp.startswith("2003")
    assert b.body_words == 900


def test_never_mature_returns_none():
    assert select_mature_baseline([snap(1, 50), snap(2, 100)], min_words=800) is None
