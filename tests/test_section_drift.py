import numpy as np

from npov_drift.embedding import FakeEncoder
from npov_drift.models import RevisionContent, Section
from npov_drift.series.section_drift import _straightness, section_drift


def test_straightness_pure():
    line = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], float)
    path, net, s = _straightness(line)
    assert abs(path - 3) < 1e-9 and abs(net - 3) < 1e-9 and abs(s - 1.0) < 1e-9

    osc = np.array([[0, 0], [1, 0], [0, 0], [1, 0]], float)
    path, net, s = _straightness(osc)
    assert abs(path - 3) < 1e-9 and abs(net - 1) < 1e-9 and abs(s - 1 / 3) < 1e-9

    assert _straightness(np.array([[0, 0]], float)) == (0.0, 0.0, 0.0)


def _sec(heading, text, boiler=False):
    wc = len(text.split())
    return Section(heading, 2, text, wc, len(text), boiler, False)


def _snap(i, sections):
    return RevisionContent(revid=i, timestamp=f"20{i:02d}-01-01", word_count=0, sections=sections, plain_text="")


def test_directional_section_beats_churn_and_excludes_boilerplate():
    enc = FakeEncoder()
    snaps = [
        _snap(1, [
            _sec("Drifting", "alpha beta gamma delta"),
            _sec("Churn", "red green red green"),
            _sec("References", "cite cite cite cite", boiler=True),
        ]),
        _snap(2, [
            _sec("Drifting", "beta gamma delta epsilon zeta"),
            _sec("Churn", "blue yellow blue yellow"),
            _sec("References", "cite cite cite cite", boiler=True),
        ]),
        _snap(3, [
            _sec("Drifting", "gamma delta epsilon zeta eta theta"),
            _sec("Churn", "red green red green"),  # returns to snap-1 text
            _sec("References", "cite cite cite cite", boiler=True),
        ]),
    ]
    drifts = {d.heading: d for d in section_drift(snaps, enc, min_words=2)}
    assert "Drifting" in drifts and "Churn" in drifts
    assert "References" not in drifts  # boilerplate excluded
    # Churn returns to its start -> ~zero net displacement -> low straightness.
    assert drifts["Drifting"].straightness > drifts["Churn"].straightness
    assert drifts["Churn"].net_displacement < 1e-9
