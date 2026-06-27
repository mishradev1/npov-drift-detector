from npov_drift.validation.evaluate import ValidationRecord, days_between, summarize


def test_days_between():
    assert abs(days_between("2007-01-01T00:00:00Z", "2008-01-01T00:00:00Z") - 365) < 1


def test_summarize_metrics():
    recs = [
        # POV article: onset clearly precedes the tag (1 year lead)
        ValidationRecord("A", "politics", True, "2008-01-01T00:00:00Z", "2007-01-01T00:00:00Z", 0.5),
        # POV article: onset AFTER the tag (flagged but does not precede)
        ValidationRecord("B", "politics", True, "2010-01-01T00:00:00Z", "2010-06-01T00:00:00Z", 0.5),
        # POV article: no onset detected
        ValidationRecord("C", "science", True, "2009-01-01T00:00:00Z", None, 0.0),
        # control flagged above threshold (false positive)
        ValidationRecord("Ctrl1", "politics", False, None, "2015-01-01T00:00:00Z", 0.5),
        # control below threshold (not flagged)
        ValidationRecord("Ctrl2", "science", False, None, "2016-01-01T00:00:00Z", 0.01),
    ]
    m = summarize(recs, effect_threshold=0.1)
    assert m["n_pov"] == 3
    assert m["n_detected"] == 2  # A and B clear the threshold; C has no onset
    assert m["n_precede"] == 1  # only A precedes its tag
    assert abs(m["median_lead_days"] - 365) < 2
    assert m["n_controls"] == 2
    assert m["n_control_flagged"] == 1  # Ctrl1 (0.5) yes, Ctrl2 (0.01) no
    assert abs(m["control_fp_rate"] - 0.5) < 1e-9
