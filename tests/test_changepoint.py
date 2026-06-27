import pytest

from npov_drift.onset.changepoint import detect_changepoint_meanshift, detect_changepoints_pelt


def test_step_series_meanshift():
    cp = detect_changepoint_meanshift([0, 0, 0, 1, 1, 1], min_size=1)
    assert cp is not None
    idx, effect = cp
    assert idx == 3
    assert abs(effect - 1.0) < 1e-9


def test_flat_series_is_none():
    assert detect_changepoint_meanshift([0.5, 0.5, 0.5, 0.5]) is None


def test_too_short_is_none():
    assert detect_changepoint_meanshift([1.0]) is None


def test_pelt_detects_step_if_available():
    pytest.importorskip("ruptures")
    bkps = detect_changepoints_pelt([0, 0, 0, 0, 5, 5, 5, 5], penalty=1.0)
    assert any(2 <= b <= 6 for b in bkps)
