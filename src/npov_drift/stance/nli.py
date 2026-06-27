"""Zero-shot NLI stance classifier (DeBERTa-MNLI), CPU-only.

We frame stance as natural-language inference against hypotheses about the
TOPIC. Two schemes are supported:

* ``"3way"`` (default, recommended): a single-label (softmax) choice between
  three competing hypotheses ::

      favorable: "This text is favorable toward {topic}."
      critical:  "This text is critical of {topic}."
      neutral:   "This text is a neutral, factual description of {topic}."

  The explicit neutral hypothesis is what lets a *neutral description of a
  negative fact* ("the drug can cause nausea") land on NEUTRAL instead of being
  dragged to CRITICAL -- the WMF caution. A stance label is only assigned if it
  beats the neutral hypothesis by ``margin``.

* ``"2hyp"``: two independent (multi_label) hypotheses for favorable/critical,
  with NEUTRAL inferred when neither is entailed. Kept for comparison; it badly
  conflates negative *facts* with critical *stance* (see Phase 2 validation),
  which is precisely why ``"3way"`` is the default.

transformers/torch are imported lazily so importing this module never requires
the ML stack to be installed.
"""

from __future__ import annotations

from .base import StanceLabel, StanceResult

DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# 3-way hypotheses, substituted into "This text {} {topic}."
_3WAY = {
    "favorable": "is favorable toward",
    "critical": "is critical of",
    "neutral": "is a neutral, factual description of",
}
_3WAY_INV = {v: k for k, v in _3WAY.items()}

# 2-hyp keys
_FAVORABLE_KEY = "favorable toward"
_CRITICAL_KEY = "critical of"


class NLIStanceClassifier:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        scheme: str = "3way",
        activation_threshold: float = 0.55,  # used by the 2hyp scheme only
        margin: float = 0.10,
        batch_size: int = 8,
        device: int = -1,  # CPU
    ):
        if scheme not in ("3way", "2hyp"):
            raise ValueError(f"unknown scheme {scheme!r}")
        self.model_name = model_name
        self.scheme = scheme
        self.activation_threshold = activation_threshold
        self.margin = margin
        self.batch_size = batch_size
        self.device = device
        self._pipe = None

    def _ensure_pipe(self):
        if self._pipe is None:
            from transformers import pipeline  # lazy import

            self._pipe = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=self.device,
            )
        return self._pipe

    # -- decision rules ----------------------------------------------------
    def _decide(self, fav: float, crit: float) -> StanceLabel:
        """2hyp rule: neutral unless one hypothesis is clearly entailed."""
        if max(fav, crit) < self.activation_threshold:
            return StanceLabel.NEUTRAL
        if abs(fav - crit) < self.margin:
            return StanceLabel.NEUTRAL
        return StanceLabel.FAVORABLE if fav > crit else StanceLabel.CRITICAL

    def _decide3(self, fav: float, crit: float, neu: float) -> StanceLabel:
        """3way rule: a stance must beat the neutral hypothesis by ``margin``."""
        stance_label, stance = (
            (StanceLabel.FAVORABLE, fav) if fav >= crit else (StanceLabel.CRITICAL, crit)
        )
        if stance - neu < self.margin:
            return StanceLabel.NEUTRAL
        return stance_label

    def decide_from_scores(self, scores: dict[str, float]) -> StanceLabel:
        if self.scheme == "2hyp":
            return self._decide(scores["favorable"], scores["critical"])
        return self._decide3(scores["favorable"], scores["critical"], scores["neutral"])

    # -- classification ----------------------------------------------------
    def classify(self, sentences: list[str], topic: str) -> list[StanceResult]:
        if not sentences:
            return []
        pipe = self._ensure_pipe()

        if self.scheme == "2hyp":
            template = "This statement is {} " + topic + "."
            raw = pipe(
                sentences,
                candidate_labels=[_FAVORABLE_KEY, _CRITICAL_KEY],
                hypothesis_template=template,
                multi_label=True,
                batch_size=self.batch_size,
            )
        else:
            template = "This text {} " + topic + "."
            raw = pipe(
                sentences,
                candidate_labels=list(_3WAY.values()),
                hypothesis_template=template,
                multi_label=False,  # softmax: the 3 hypotheses compete
                batch_size=self.batch_size,
            )

        if isinstance(raw, dict):  # single-input convenience shape
            raw = [raw]

        results: list[StanceResult] = []
        for sent, out in zip(sentences, raw):
            label_scores = dict(zip(out["labels"], out["scores"]))
            if self.scheme == "2hyp":
                scores = {
                    "favorable": float(label_scores.get(_FAVORABLE_KEY, 0.0)),
                    "critical": float(label_scores.get(_CRITICAL_KEY, 0.0)),
                }
            else:
                mapped = {_3WAY_INV[lab]: float(sc) for lab, sc in label_scores.items()}
                scores = {
                    "favorable": mapped.get("favorable", 0.0),
                    "critical": mapped.get("critical", 0.0),
                    "neutral": mapped.get("neutral", 0.0),
                }
            results.append(
                StanceResult(
                    sentence=sent,
                    label=self.decide_from_scores(scores),
                    scores=scores,
                )
            )
        return results
