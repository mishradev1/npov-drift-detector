"""Changepoint detection on a 1-D drift series.

Two detectors:
  * ``detect_changepoint_meanshift`` -- a deterministic, dependency-light (numpy
    only) single-best level-shift detector. Used as the per-signal ONSET (the
    moment the level changes most), so the core onset logic is reproducible and
    testable without ruptures.
  * ``detect_changepoints_pelt`` -- ruptures PELT (rbf), per the spec, for the
    fuller set of changepoints. Lazily imported.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def detect_changepoint_meanshift(
    values, min_size: int = 2, min_effect: float = 1e-9
) -> Optional[tuple[int, float]]:
    """Return (changepoint_index, effect) of the largest mean level shift.

    ``changepoint_index`` is the start of the right segment. ``effect`` is the
    absolute difference of segment means. Returns None if the series is too
    short or essentially flat.
    """
    a = np.asarray(values, dtype=float)
    n = a.size
    if n < 2 * min_size:
        return None
    best_t: Optional[int] = None
    best_eff = -1.0
    for t in range(min_size, n - min_size + 1):
        effect = abs(float(a[t:].mean()) - float(a[:t].mean()))
        if effect > best_eff:
            best_eff = effect
            best_t = t
    if best_t is None or best_eff < min_effect:
        return None
    return best_t, float(best_eff)


def detect_changepoints_pelt(
    values, penalty: float = 3.0, model: str = "rbf", min_size: int = 2
) -> list[int]:
    """ruptures PELT changepoint indices (excluding the trailing end index)."""
    import ruptures as rpt  # lazy

    sig = np.asarray(values, dtype=float).reshape(-1, 1)
    if sig.shape[0] < 2 * min_size:
        return []
    algo = rpt.Pelt(model=model, min_size=min_size).fit(sig)
    bkps = algo.predict(pen=penalty)
    return [b for b in bkps if b < sig.shape[0]]
