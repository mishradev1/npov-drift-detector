"""Mature-baseline selection and changepoint onset detection.

Estimates WHEN an article's trajectory departed from its mature baseline, by
running changepoint detection on the drift signal series and reconciling the
onsets across signals. Per the spec, this never uses the stub era as a baseline.
"""

from .baseline import Baseline, select_mature_baseline
from .changepoint import detect_changepoint_meanshift, detect_changepoints_pelt
from .detect import Onset, OnsetReport, detect_drift_onset, reconcile

__all__ = [
    "Baseline",
    "select_mature_baseline",
    "detect_changepoint_meanshift",
    "detect_changepoints_pelt",
    "Onset",
    "OnsetReport",
    "detect_drift_onset",
    "reconcile",
]
