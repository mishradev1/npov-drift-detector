"""A deterministic, dependency-free stance stub.

This exists so the test suite and offline demos can exercise the whole stance
pipeline without torch/transformers or a model download. It is a crude
keyword counter and is NOT suitable for real analysis -- the real classifier is
``nli.NLIStanceClassifier``.
"""

from __future__ import annotations

from .base import StanceLabel, StanceResult

# Lowercased cue words. These are generic stance cues, used only by the stub.
_FAVORABLE_CUES = {
    "praised",
    "praise",
    "beneficial",
    "successful",
    "effective",
    "acclaimed",
    "celebrated",
    "supporters",
    "endorsed",
    "landmark",
    "hailed",
    "admired",
}
_CRITICAL_CUES = {
    "criticized",
    "criticised",
    "condemned",
    "denounced",
    "harmful",
    "cruel",
    "inhumane",
    "accused",
    "failed",
    "dangerous",
    "controversial",
    "opponents",
    "flawed",
    "discredited",
}


class KeywordStanceStub:
    """Counts favorable/critical cue words; ties and zeros -> neutral."""

    def __init__(
        self,
        favorable_cues: set[str] | None = None,
        critical_cues: set[str] | None = None,
    ):
        self.favorable_cues = favorable_cues or set(_FAVORABLE_CUES)
        self.critical_cues = critical_cues or set(_CRITICAL_CUES)

    def _classify_one(self, sentence: str) -> StanceResult:
        tokens = {t.strip(".,;:!?\"'()[]").lower() for t in sentence.split()}
        fav = len(tokens & self.favorable_cues)
        crit = len(tokens & self.critical_cues)
        if fav > crit:
            label = StanceLabel.FAVORABLE
        elif crit > fav:
            label = StanceLabel.CRITICAL
        else:
            label = StanceLabel.NEUTRAL
        return StanceResult(
            sentence=sentence,
            label=label,
            scores={"favorable": float(fav), "critical": float(crit)},
        )

    def classify(self, sentences: list[str], topic: str) -> list[StanceResult]:
        # topic is ignored by the stub; signature matches the real classifier.
        return [self._classify_one(s) for s in sentences]
