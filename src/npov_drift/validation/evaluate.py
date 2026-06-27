"""Validation metrics (pure): does the estimated onset precede the human tag?

We treat an onset as "flagged" only if its effect size clears ``effect_threshold``
(ideally the type noise floor from Phase 4). For POV-tagged articles we measure
how often a flagged onset PRECEDES the tag and the lead time; for controls we
measure the false-positive (flagged-anyway) rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from statistics import median
from typing import Optional


@dataclass
class ValidationRecord:
    title: str
    type_bucket: str
    is_pov: bool
    tag_timestamp: Optional[str] = None
    onset_timestamp: Optional[str] = None
    onset_effect: float = 0.0


def days_between(earlier: str, later: str) -> float:
    a = datetime.fromisoformat(earlier.replace("Z", "+00:00"))
    b = datetime.fromisoformat(later.replace("Z", "+00:00"))
    return (b - a).total_seconds() / 86400.0


def summarize(records: list[ValidationRecord], effect_threshold: float = 0.0) -> dict:
    pov = [r for r in records if r.is_pov and r.tag_timestamp]
    controls = [r for r in records if not r.is_pov]

    detected: list[ValidationRecord] = []
    leads: list[float] = []
    for r in pov:
        flagged = r.onset_timestamp is not None and r.onset_effect >= effect_threshold
        if not flagged:
            continue
        detected.append(r)
        lead = days_between(r.onset_timestamp, r.tag_timestamp)  # >=0 => onset precedes tag
        if lead >= 0:
            leads.append(lead)

    flagged_controls = [
        r for r in controls if r.onset_timestamp is not None and r.onset_effect >= effect_threshold
    ]

    return {
        "effect_threshold": effect_threshold,
        "n_pov": len(pov),
        "n_detected": len(detected),
        "n_precede": len(leads),
        "detection_rate": (len(detected) / len(pov)) if pov else 0.0,
        "precede_rate": (len(leads) / len(pov)) if pov else 0.0,
        "median_lead_days": (median(leads) if leads else None),
        "n_controls": len(controls),
        "n_control_flagged": len(flagged_controls),
        "control_fp_rate": (len(flagged_controls) / len(controls)) if controls else 0.0,
    }
