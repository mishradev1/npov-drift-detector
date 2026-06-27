"""Dashboard data assembly (view-agnostic).

``report.build_drift_report`` turns an ingested article into everything the
Streamlit UI displays. Keeping it separate from Streamlit makes the substance
unit-testable and runnable headless.
"""

from .report import DriftReport, build_drift_report

__all__ = ["DriftReport", "build_drift_report"]
